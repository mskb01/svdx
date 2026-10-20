
# ============================================================
# 1. inferV()
# ============================================================


# inferV() performs conditional inference on feature loadings V.
# Key principle: Utest is predicted using metadata only, then Ytest
# is regressed on Utest + Ztest. This avoids double dipping.

.inferV_get_zformula <- function(fit) {
  zf <- fit$fitAtOptCV$trainModel$mod$z.formula %||%
    fit$fitAtOptCV$trainModel$zformula %||%
    fit$zformula
  
  if (is.null(zf)) {
    stop("Could not locate zformula inside fit. Please pass zformula explicitly.", call. = FALSE)
  }
  
  zf
}


.normalize_i_family <- function(family, D, dataset_names = NULL) {
  if (is.null(family)) return(NULL)
  dataset_names <- dataset_names %||% paste0("dataset_", seq_len(D))
  specs <- .svdx_family_spec_list(family, dataset_names)
  family <- setNames(lapply(specs, `[[`, "family"), dataset_names)
  if (all(vapply(family, is.null, logical(1)))) NULL else family
}

.normalize_i_offsets <- function(offset, D, dataset_names, arg_name) {
  if (is.null(offset)) {
    out <- vector("list", D)
    names(out) <- dataset_names
    return(out)
  }
  
  if (!is.list(offset)) {
    if (D == 1) {
      out <- list(offset)
      names(out) <- dataset_names
      return(out)
    }
    stop(arg_name, " must be NULL or a list of length equal to the number of datasets.", call. = FALSE)
  }
  
  if (length(offset) != D) {
    stop(arg_name, " must be NULL or a list of length equal to the number of datasets.", call. = FALSE)
  }
  
  names(offset) <- names(offset) %||% dataset_names
  offset
}

.inferV_get_izformula <- function(fit, zformula = NULL) {
  if (!is.null(zformula)) return(zformula)
  
  zf <- fit$fitAtOptCV$trainModel$mod$z.formula %||%
    fit$fitAtOptCV$trainModel$zformula %||%
    fit$fitAtOptCV$zformula %||%
    fit$zformula
  
  if (is.null(zf)) {
    stop("Could not locate zformula list inside iSSANOVASVD fit. Please pass zformula explicitly.", call. = FALSE)
  }
  
  zf
}



.inferV_formula_label <- function(f) {
  paste(deparse(f), collapse = "")
}

.inferV_formula_matches <- function(train_zformula, test_zformula) {
  identical(.inferV_formula_label(train_zformula), .inferV_formula_label(test_zformula))
}

.inferV_get_zeta <- function(fit, dataset = NULL) {
  zeta <- fit$fitAtOptCV$Zeta %||% fit$Zeta
  if (is.null(zeta)) {
    stop("z_adjustment='fitted' requires fitted SVDX nuisance coefficients, but Zeta was not found in the fit.", call. = FALSE)
  }
  if (!is.null(dataset) && is.list(zeta)) {
    if (!dataset %in% names(zeta)) {
      stop("z_adjustment='fitted' could not locate fitted nuisance coefficients for dataset '", dataset, "'.", call. = FALSE)
    }
    zeta <- zeta[[dataset]]
  }
  if (is.list(zeta)) {
    stop("z_adjustment='fitted' expected a matrix of nuisance coefficients but found a list. For integrated fits, a dataset must be supplied.", call. = FALSE)
  }
  as.matrix(zeta)
}

.inferV_fitted_z_adjustment <- function(fit, newzdata, train_zformula, test_zformula, Ytest, dataset = NULL) {
  if (!.inferV_formula_matches(train_zformula, test_zformula)) {
    stop(
      "z_adjustment='fitted' requires test zformula to match the training zformula used to estimate nuisance effects.",
      call. = FALSE
    )
  }

  Ztest <- stats::model.matrix(test_zformula, data = newzdata)
  zeta <- .inferV_get_zeta(fit, dataset = dataset)

  if (ncol(zeta) != ncol(Ztest)) {
    stop(
      "z_adjustment='fitted' found incompatible nuisance design dimensions: test Z has ",
      ncol(Ztest), " column(s), but fitted Zeta has ", ncol(zeta), " column(s).",
      call. = FALSE
    )
  }

  zeta_cols <- colnames(zeta)
  ztest_cols <- colnames(Ztest)
  if (!is.null(zeta_cols) && !is.null(ztest_cols)) {
    if (!identical(zeta_cols, ztest_cols)) {
      stop(
        "z_adjustment='fitted' requires test Z design columns to match fitted Zeta columns when names are available.",
        call. = FALSE
      )
    }
  }

  if (nrow(zeta) != nrow(Ytest)) {
    stop(
      "z_adjustment='fitted' found incompatible feature dimensions: Ytest has ",
      nrow(Ytest), " feature(s), but fitted Zeta has ", nrow(zeta), " row(s).",
      call. = FALSE
    )
  }

  eta <- zeta %*% t(Ztest)
  rownames(eta) <- rownames(Ytest)
  colnames(eta) <- colnames(Ytest)

  list(Ztest = Ztest, eta = eta)
}


.inferV_predict_U <- function(fit, newdata, include = NULL) {
  if (is.null(include)) {
    U <- predict(fit, newdata)
  } else {
    U <- predict(fit, newdata, include = include)
  }
  
  U <- as.matrix(U)
  colnames(U) <- colnames(U) %||% .make_component_names(ncol(U))
  U
}

.inferV_make_design <- function(Utest, newdata, newzdata, zformula = NULL, include_intercept = FALSE) {
  Utest <- as.matrix(Utest)
  colnames(Utest) <- colnames(Utest) %||% .make_component_names(ncol(Utest))
  
  if (!is.null(zformula)) {
    Ztest <- model.matrix(zformula, data = newzdata)
  } else {
    Ztest <- NULL
  }
  
  if (!include_intercept && !is.null(Ztest) && "(Intercept)" %in% colnames(Ztest)) {
    Ztest <- Ztest[, colnames(Ztest) != "(Intercept)", drop = FALSE]
  }
  
  design <- if (is.null(Ztest)) {
    Utest
  } else {
    cbind(Utest, Ztest)
  }
  
  list(
    design = design,
    Utest = Utest,
    Ztest = Ztest
  )
}

.inferV_gaussian_lmfit <- function(Ytest, design, component_cols, row_offset = NULL, col_offset = NULL) {
  .require_namespace("limma")
  
  Yfit <- Ytest
  
  if (!is.null(row_offset)) {
    Yfit <- sweep(Yfit, 1, row_offset, "-")
  }
  
  if (!is.null(col_offset)) {
    Yfit <- sweep(Yfit, 2, col_offset, "-")
  }
  
  ft <- limma::lmFit(Yfit, design)
  eb <- limma::eBayes(ft)
  
  coef <- eb$coefficients[, component_cols, drop = FALSE]
  stat <- eb$t[, component_cols, drop = FALSE]
  pval <- eb$p.value[, component_cols, drop = FALSE]
  
  se <- coef / stat
  se[!is.finite(se)] <- NA_real_
  
  qval <- apply(pval, 2, .safe_bh)
  qval <- as.matrix(qval)
  colnames(qval) <- colnames(pval)
  rownames(qval) <- rownames(pval)

  # Omnibus feature-level test across all estimated SVDX components.
  # For Gaussian fits this is limma's moderated F-test over the component
  # coefficients only; nuisance Z terms are retained in the model but are not
  # included in the tested coefficient set.
  global_stat <- rep(NA_real_, nrow(Yfit))
  global_p <- rep(NA_real_, nrow(Yfit))
  names(global_stat) <- names(global_p) <- rownames(Yfit)

  if (length(component_cols) == 1L) {
    # With one tested component, limma reports a moderated t-test rather than
    # an omnibus F column. The one-component omnibus hypothesis is identical
    # to the component-specific hypothesis, with F = t^2.
    global_stat <- as.numeric(eb$t[, component_cols])^2
    global_p <- as.numeric(eb$p.value[, component_cols])
    names(global_stat) <- names(global_p) <- rownames(Yfit)
  } else {
    global_tab <- limma::topTable(
      eb,
      coef = component_cols,
      number = Inf,
      sort.by = "none"
    )
    if (!all(c("F", "P.Value") %in% colnames(global_tab))) {
      stop(
        "Gaussian omnibus inference did not return the expected F and P.Value columns.",
        call. = FALSE
      )
    }
    hits <- match(rownames(Yfit), rownames(global_tab))
    ok <- !is.na(hits)
    global_stat[ok] <- global_tab$F[hits[ok]]
    global_p[ok] <- global_tab$P.Value[hits[ok]]
  }
  
  list(
    fit = ft,
    eb = eb,
    coef = coef,
    se = se,
    stat = stat,
    pval = pval,
    qval = qval,
    global_stat = global_stat,
    global_p = global_p
  )
}

.inferV_glm_featurewise <- function(
    Ytest,
    design,
    component_cols,
    family = stats::poisson(),
    global_test = c("lrt", "quasi_f"),
    row_offset = NULL,
    col_offset = NULL,
    matrix_offset = NULL,
    verbose = FALSE
) {
  global_test <- match.arg(global_test)
  p <- nrow(Ytest)
  K <- length(component_cols)
  
  coef <- se <- stat <- pval <- matrix(NA_real_, nrow = p, ncol = K)
  colnames(coef) <- colnames(se) <- colnames(stat) <- colnames(pval) <- component_cols
  rownames(coef) <- rownames(se) <- rownames(stat) <- rownames(pval) <- rownames(Ytest)

  global_stat <- rep(NA_real_, p)
  global_p <- rep(NA_real_, p)
  names(global_stat) <- names(global_p) <- rownames(Ytest)
  noncomponent_cols <- setdiff(colnames(design), component_cols)
  
  base_offset <- if (is.null(col_offset)) rep(0, ncol(Ytest)) else as.numeric(col_offset)
  
  for (i in seq_len(p)) {
    if (verbose && i %% 100 == 0) message("Fitting feature ", i, " / ", p)
    
    off_i <- base_offset
    if (!is.null(row_offset)) off_i <- off_i + row_offset[i]
    if (!is.null(matrix_offset)) off_i <- off_i + as.numeric(matrix_offset[i, ])
    
    dat <- as.data.frame(design)
    dat$.y <- as.numeric(Ytest[i, ])
    dat$.offset <- off_i
    
    fit_i <- tryCatch(
      stats::glm(.y ~ . - 1 + offset(.offset), data = dat, family = family),
      error = function(e) NULL
    )
    
    if (is.null(fit_i)) next

    # Omnibus feature-level test across all SVDX component coefficients.
    # For Poisson GLM fits this is a likelihood-ratio test comparing the full model
    # to a reduced model; for quasi-Poisson fits it is an analysis-of-deviance
    # F-test scaled by the estimated dispersion, comparing the full model
    # to the same reduced model that retains nuisance Z terms but removes all
    # component columns. The resulting p-value has the same role as the
    # Gaussian moderated F-test but is not literally F-distributed.
    red_dat <- dat[, c(noncomponent_cols, ".y", ".offset"), drop = FALSE]
    red_formula <- if (length(noncomponent_cols) > 0L) {
      stats::as.formula(".y ~ . - 1 + offset(.offset)")
    } else {
      stats::as.formula(".y ~ 0 + offset(.offset)")
    }
    fit_red <- tryCatch(
      stats::glm(red_formula, data = red_dat, family = family),
      error = function(e) NULL
    )
    if (!is.null(fit_red)) {
      df_diff <- fit_red$df.residual - fit_i$df.residual
      dev_diff <- fit_red$deviance - fit_i$deviance
      if (is.finite(df_diff) && df_diff > 0 && is.finite(dev_diff)) {
        dev_diff <- max(0, dev_diff)
        if (identical(global_test, "quasi_f")) {
          phi <- tryCatch(summary(fit_i)$dispersion, error = function(e) NA_real_)
          if (is.finite(phi) && phi > 0 && is.finite(fit_i$df.residual) && fit_i$df.residual > 0) {
            F_stat <- (dev_diff / df_diff) / phi
            if (is.finite(F_stat)) {
              global_stat[i] <- F_stat
              global_p[i] <- stats::pf(F_stat, df1 = df_diff, df2 = fit_i$df.residual, lower.tail = FALSE)
            }
          }
        } else {
          global_stat[i] <- dev_diff
          global_p[i] <- stats::pchisq(dev_diff, df = df_diff, lower.tail = FALSE)
        }
      }
    }
    
    sm <- summary(fit_i)$coefficients
    keep <- intersect(component_cols, rownames(sm))
    if (length(keep) == 0) next
    
    idx <- match(keep, component_cols)
    coef[i, idx] <- sm[keep, "Estimate"]
    se[i, idx] <- sm[keep, "Std. Error"]
    
    stat_col <- if ("z value" %in% colnames(sm)) "z value" else if ("t value" %in% colnames(sm)) "t value" else NA_character_
    p_col <- grep("Pr\\(", colnames(sm), value = TRUE)[1]
    
    if (!is.na(stat_col)) stat[i, idx] <- sm[keep, stat_col]
    if (!is.na(p_col)) pval[i, idx] <- sm[keep, p_col]
  }
  
  qval <- apply(pval, 2, .safe_bh)
  qval <- as.matrix(qval)
  colnames(qval) <- colnames(pval)
  rownames(qval) <- rownames(pval)
  
  list(
    fit = NULL,
    eb = NULL,
    coef = coef,
    se = se,
    stat = stat,
    pval = pval,
    qval = qval,
    global_stat = global_stat,
    global_p = global_p
  )
}

.make_feature_results <- function(coef, se, stat, pval, qval) {
  feature <- rownames(coef) %||% paste0("feature_", seq_len(nrow(coef)))
  component <- colnames(coef) %||% paste0("U", seq_len(ncol(coef)))
  
  expand.grid(
    feature = feature,
    component = component,
    stringsAsFactors = FALSE
  ) |>
    dplyr::mutate(
      coef = as.vector(coef),
      se = as.vector(se),
      stat = as.vector(stat),
      pval = as.vector(pval),
      qval = as.vector(qval)
    )
}

.make_union_results <- function(coef, stat, pval, qval = NULL, global_stat = NULL, global_p = NULL, alpha = 0.05) {
  feature <- rownames(coef) %||% paste0("feature_", seq_len(nrow(coef)))
  components <- colnames(coef) %||% paste0("U", seq_len(ncol(coef)))
  
  K_eff <- ncol(pval)
  min_p <- apply(pval, 1, function(x) {
    if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE)
  })
  union_p <- pmin(1, K_eff * min_p) # adjust for multiplicity across estimated components.
  union_q <- .safe_bh(union_p)

  if (is.null(global_p)) {
    global_p <- rep(NA_real_, length(feature))
    names(global_p) <- feature
  }
  if (is.null(global_stat)) {
    global_stat <- rep(NA_real_, length(feature))
    names(global_stat) <- feature
  }
  if (!is.null(names(global_p))) {
    global_p <- as.numeric(global_p[match(feature, names(global_p))])
  } else {
    global_p <- as.numeric(global_p)
  }
  if (!is.null(names(global_stat))) {
    global_stat <- as.numeric(global_stat[match(feature, names(global_stat))])
  } else {
    global_stat <- as.numeric(global_stat)
  }
  if (length(global_p) != length(feature)) global_p <- rep(NA_real_, length(feature))
  if (length(global_stat) != length(feature)) global_stat <- rep(NA_real_, length(feature))
  global_q <- .safe_bh(global_p)
  
  best_idx <- apply(abs(stat), 1, function(x) {
    if (all(is.na(x))) NA_integer_ else which.max(x)
  })
  
  max_abs_stat <- apply(abs(stat), 1, function(x) {
    if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
  })
  
  best_component <- ifelse(is.na(best_idx), NA_character_, components[best_idx])
  
  best_coef <- rep(NA_real_, nrow(coef))
  best_stat <- rep(NA_real_, nrow(coef))
  for (i in seq_len(nrow(coef))) {
    if (!is.na(best_idx[i])) {
      best_coef[i] <- coef[i, best_idx[i]]
      best_stat[i] <- stat[i, best_idx[i]]
    }
  }
  
  tibble::tibble(
    feature = feature,
    best_component = best_component,
    best_coef = best_coef,
    best_stat = best_stat,
    max_abs_stat = max_abs_stat,
    min_p = min_p,
    union_p = union_p,
    union_q = union_q,
    global_stat = global_stat,
    global_p = global_p,
    global_q = global_q,
    significant = union_q < alpha,
    global_significant = global_q < alpha
  )
}

inferV.SSANOVASVD <- function(
    fit,
    Ytest,
    newdata,
    newzdata = newdata, 
    zformula = NULL,
    family = NULL,
    alpha = 0.05,
    include_intercept = FALSE,
    row_offset = NULL,
    col_offset = NULL,
    z_adjustment = c("fitted", "reestimate"),
    return_design = TRUE,
    verbose = FALSE,
    ...
) {
  if (is.null(family)) {
    family <- "gaussian"
  } else {
    family <- match.arg(family, c("gaussian", "poisson", "quasipoisson"))
  }
  z_adjustment <- match.arg(z_adjustment)
  
  
  Ytest <- as.matrix(Ytest)
  rownames(Ytest) <- rownames(Ytest) %||% .make_feature_names(Ytest)
  .svdx_validate_nonnegative_for_family(Ytest, family, context = "Ytest")
  
  if (identical(z_adjustment, "fitted")) {
    train_zformula <- .inferV_get_zformula(fit)
    zformula <- zformula %||% train_zformula
  } else {
    train_zformula <- NULL
    zformula <- zformula %||% .inferV_get_zformula(fit)
  }
  
  Utest <- .inferV_predict_U(fit, newdata = newdata)
  component_cols <- colnames(Utest)

  fitted_z <- NULL
  if (identical(z_adjustment, "fitted")) {
    message("Adjusting data for SVDX estimated nuisance effects.")
    fitted_z <- .inferV_fitted_z_adjustment(
      fit = fit,
      newzdata = newzdata,
      train_zformula = train_zformula,
      test_zformula = zformula,
      Ytest = Ytest
    )
  }
  
  design_obj <- if (identical(z_adjustment, "fitted")) {
    list(design = Utest, Utest = Utest, Ztest = fitted_z$Ztest)
  } else {
    .inferV_make_design(
      Utest = Utest,
      newdata = newdata,
      newzdata = newzdata, 
      zformula = zformula,
      include_intercept = include_intercept
    )
  }

  Y_for_inference <- if (identical(z_adjustment, "fitted")) Ytest - fitted_z$eta else Ytest
  matrix_offset <- if (identical(z_adjustment, "fitted")) fitted_z$eta else NULL
  
  if (family == "gaussian") {
    fit_obj <- .inferV_gaussian_lmfit(
      Ytest = Y_for_inference,
      design = design_obj$design,
      component_cols = component_cols,
      row_offset = row_offset,
      col_offset = col_offset
    )
  } else {
    glm_family <- if (identical(family, "quasipoisson")) stats::quasipoisson() else stats::poisson()
    global_test <- if (identical(family, "quasipoisson")) "quasi_f" else "lrt"
    fit_obj <- .inferV_glm_featurewise(
      Ytest = Ytest,
      design = design_obj$design,
      component_cols = component_cols,
      family = glm_family,
      global_test = global_test,
      row_offset = row_offset,
      col_offset = col_offset,
      matrix_offset = matrix_offset,
      verbose = verbose
    )
  }
  
  feature_results <- .make_feature_results(
    coef = fit_obj$coef,
    se = fit_obj$se,
    stat = fit_obj$stat,
    pval = fit_obj$pval,
    qval = fit_obj$qval
  )
  
  union_results <- .make_union_results(
    coef = fit_obj$coef,
    stat = fit_obj$stat,
    pval = fit_obj$pval,
    qval = fit_obj$qval,
    global_stat = fit_obj$global_stat,
    global_p = fit_obj$global_p,
    alpha = alpha
  )
  
  out <- list(
    coef = fit_obj$coef,
    se = fit_obj$se,
    stat = fit_obj$stat,
    pval = fit_obj$pval,
    qval = fit_obj$qval,
    global_stat = fit_obj$global_stat,
    global_p = fit_obj$global_p,
    global_q = union_results$global_q,
    feature_results = feature_results,
    union_results = union_results,
    Utest = Utest,
    Ztest = design_obj$Ztest,
    design = if (return_design) design_obj$design else NULL,
    limma_fit = fit_obj$fit,
    limma_ebayes = fit_obj$eb,
    family = family,
    z_adjustment = z_adjustment,
    alpha = alpha,
    call = match.call()
  )
  
  class(out) <- "svdxInference"
  out
}

#' @export
inferV.iSSANOVASVD <- function(
    fit,
    Ytest,
    newdata,
    newzdata,
    zformula = NULL,
    family = NULL,
    alpha = 0.05,
    include_intercept = FALSE,
    row_offset = NULL,
    col_offset = NULL,
    z_adjustment = c("fitted", "reestimate"),
    return_design = TRUE,
    verbose = FALSE,
    ...
) {
  if (!is.list(Ytest)) {
    stop("For iSSANOVASVD, Ytest must be a list of feature x sample matrices.", call. = FALSE)
  }
  
  D <- length(Ytest)
  dataset_names <- names(Ytest) %||% paste0("dataset_", seq_len(D))
  names(Ytest) <- dataset_names
  
  if (!is.list(newzdata) || length(newzdata) != D) {
    stop("newzdata must be a list of length equal to length(Ytest).", call. = FALSE)
  }
  names(newzdata) <- names(newzdata) %||% dataset_names
  
  z_adjustment <- match.arg(z_adjustment)
  
  if (identical(z_adjustment, "fitted")) {
    train_zformula <- .inferV_get_izformula(fit, zformula = NULL)
    if (!is.list(train_zformula) || length(train_zformula) != D) {
      stop("Training zformula must be a list of length equal to length(Ytest).", call. = FALSE)
    }
    names(train_zformula) <- names(train_zformula) %||% dataset_names
    zformula <- if (is.null(zformula)) train_zformula else zformula
  } else {
    train_zformula <- NULL
    zformula <- .inferV_get_izformula(fit, zformula = zformula)
  }
  
  if (!is.list(zformula) || length(zformula) != D) {
    stop("zformula must be a list of length equal to length(Ytest).", call. = FALSE)
  }
  names(zformula) <- names(zformula) %||% dataset_names
  
  family <- .normalize_i_family(family, D, dataset_names = dataset_names)
  row_offset <- .normalize_i_offsets(row_offset, D, dataset_names, "row_offset")
  col_offset <- .normalize_i_offsets(col_offset, D, dataset_names, "col_offset")
  
  Utest <- .inferV_predict_U(fit, newdata = newdata)
  component_cols <- colnames(Utest)
  
  dataset_results <- lapply(seq_len(D), function(d) {
    fam_d <- if (is.null(family)) "gaussian" else family[[d]]
    
    Yd <- as.matrix(Ytest[[d]])
    rownames(Yd) <- rownames(Yd) %||% .make_feature_names(Yd)
    .svdx_validate_nonnegative_for_family(Yd, fam_d, context = paste0("Ytest[['", dataset_names[[d]], "']]"))
    
    fitted_z <- NULL
    if (identical(z_adjustment, "fitted")) {
      message("Adjusting data for SVDX estimated nuisance effects.")
      fitted_z <- .inferV_fitted_z_adjustment(
        fit = fit,
        newzdata = newzdata[[d]],
        train_zformula = train_zformula[[d]],
        test_zformula = zformula[[d]],
        Ytest = Yd,
        dataset = dataset_names[d]
      )
    }

    design_obj <- if (identical(z_adjustment, "fitted")) {
      list(design = Utest, Utest = Utest, Ztest = fitted_z$Ztest)
    } else {
      .inferV_make_design(
        Utest = Utest,
        newdata = newdata,
        newzdata = newzdata[[d]],
        zformula = zformula[[d]],
        include_intercept = include_intercept
      )
    }

    Yd_for_inference <- if (identical(z_adjustment, "fitted")) Yd - fitted_z$eta else Yd
    matrix_offset <- if (identical(z_adjustment, "fitted")) fitted_z$eta else NULL
    
    fit_obj <- if (identical(fam_d, "gaussian")) {
      .inferV_gaussian_lmfit(
        Ytest = Yd_for_inference,
        design = design_obj$design,
        component_cols = component_cols,
        row_offset = row_offset[[d]],
        col_offset = col_offset[[d]]
      )
    } else {
      glm_family <- if (identical(fam_d, "quasipoisson")) stats::quasipoisson() else stats::poisson()
      global_test <- if (identical(fam_d, "quasipoisson")) "quasi_f" else "lrt"
      .inferV_glm_featurewise(
        Ytest = Yd,
        design = design_obj$design,
        component_cols = component_cols,
        family = glm_family,
        global_test = global_test,
        row_offset = row_offset[[d]],
        col_offset = col_offset[[d]],
        matrix_offset = matrix_offset,
        verbose = verbose
      )
    }
    
    feature_results <- .make_feature_results(
      coef = fit_obj$coef,
      se = fit_obj$se,
      stat = fit_obj$stat,
      pval = fit_obj$pval,
      qval = fit_obj$qval
    )
    
    union_results <- .make_union_results(
      coef = fit_obj$coef,
      stat = fit_obj$stat,
      pval = fit_obj$pval,
      qval = fit_obj$qval,
      global_stat = fit_obj$global_stat,
      global_p = fit_obj$global_p,
      alpha = alpha
    )
    
    out_d <- list(
      coef = fit_obj$coef,
      se = fit_obj$se,
      stat = fit_obj$stat,
      pval = fit_obj$pval,
      qval = fit_obj$qval,
      global_stat = fit_obj$global_stat,
      global_p = fit_obj$global_p,
      global_q = union_results$global_q,
      feature_results = feature_results,
      union_results = union_results,
      Utest = Utest,
      Ztest = design_obj$Ztest,
      design = if (return_design) design_obj$design else NULL,
      limma_fit = fit_obj$fit,
      limma_ebayes = fit_obj$eb,
      family = fam_d,
      z_adjustment = z_adjustment,
      alpha = alpha,
      dataset = dataset_names[d],
      call = match.call()
    )
    
    class(out_d) <- "svdxInference"
    out_d
  })
  
  names(dataset_results) <- dataset_names
  
  out <- list(
    dataset_results = dataset_results,
    shared_Utest = Utest,
    alpha = alpha,
    family = family,
    z_adjustment = z_adjustment,
    datasets = dataset_names,
    call = match.call()
  )
  
  class(out) <- "isvdxInference"
  out
}
