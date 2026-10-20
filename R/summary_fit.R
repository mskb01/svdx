#summary_fit.R

# ============================================================
# 3. Training summary: relevant xformula terms only, plus x/z diagnostics
# ============================================================

.svdx_summary_todos <- c(
  "TODO: summary for family svdx",
  "TODO: summary for family isvdx/iSSANOVASVD",
  "TODO: package documentation"
)

.skip_family_term_summary <- function(object_name = "fit") {
  warning(
    "Term/component response summaries for family-based ",
    object_name,
    " are not yet implemented. Skipping term summaries. ",
    "TODO: implement summary for family svdx and family isvdx/iSSANOVASVD.",
    call. = FALSE
  )
}

.get_x_terms <- function(fit) {
  labs <- fit$fitAtOptCV$trainModel$mod$terms$labels %||%
    fit$fitAtOptCV$trainModel$mod$mod$terms$labels
  
  labs <- labs[!labs %in% c("offset")]
  labs
}

.get_z_terms <- function(fit) {
  zf <- fit$fitAtOptCV$trainModel$mod$z.formula %||% fit$zformula
  if (is.null(zf)) return(character())
  c("(Intercept)", attr(terms(zf), "term.labels"))
}

.predict_terms_array <- function(fit, newdata, terms, include_intercept = FALSE) {
  pred_list <- list()
  
  for (term in terms) {
    if (term %in% c("1", "(Intercept)", "offset")) next
    
    include <- if (include_intercept) c("1", term) else term
    Uterm <- tryCatch(
      predict(fit, newdata, include = include),
      error = function(e) NULL
    )
    
    if (is.null(Uterm)) next
    Uterm <- as.matrix(Uterm)
    colnames(Uterm) <- colnames(Uterm) %||% .make_component_names(ncol(Uterm))
    pred_list[[term]] <- Uterm
  }
  
  pred_list
}


.has_family <- function(family) {
  !is.null(family)
}

.comp_response_matrix_no_family_from_parts <- function(
    Y,
    vmat,
    Z = NULL,
    Zeta = NULL,
    rowOffsets = NULL,
    colOffsets = NULL
) {
  n <- ncol(Y)
  
  vtY <- crossprod(vmat, Y)
  
  vtZetaZ <- if (!is.null(Z) && !is.null(Zeta)) {
    tcrossprod(crossprod(vmat, Zeta), Z)
  } else {
    matrix(0, nrow = ncol(vmat), ncol = n)
  }
  
  vtRowOffs <- if (!is.null(rowOffsets)) {
    ro <- crossprod(vmat, rowOffsets)
    do.call(cbind, lapply(seq_len(n), function(i) ro))
  } else {
    matrix(0, nrow = ncol(vmat), ncol = n)
  }
  
  vtColOffs <- if (!is.null(colOffsets)) {
    colOffsetsMat <- do.call(
      rbind,
      lapply(seq_len(ncol(vmat)), function(i) colOffsets)
    )
    
    sweep(
      colOffsetsMat,
      1,
      crossprod(vmat, rep(1, nrow(Y))),
      "*"
    )
  } else {
    matrix(0, nrow = ncol(vmat), ncol = n)
  }
  
  t(vtY - vtZetaZ - vtRowOffs - vtColOffs)
}

.component_response_matrix_no_family_single <- function(fit, Y) {
  # The "responses" against which each of u1, ..., uK are fitted.
  #
  # C++ analogue:
  #   arma::rowvec vtY = (vk.t() * Y).eval()
  #   vtY = vtY
  #       - arma::dot(vk, ones_p) * colOffsets.t()
  #       - arma::dot(vk, rowOffsets) * ones_n.t()
  #   arma::mat Zetav = (Zeta * vk).eval()
  #   arma::mat vv = (vmatOthers.t() * vk).eval()
  #   w_hat = vtY.t() - umatOthers * vv - Z * Zetav
  #
  # Returns n x K matrix of working responses y_k before
  # subtracting previous component contributions.
  
  with(fit$fitAtOptCV, {
    .comp_response_matrix_no_family_from_parts(
      Y = Y,
      vmat = vmat,
      Z = trainModel$Z %||% Z,
      Zeta = Zeta,
      rowOffsets = rowOffsets,
      colOffsets = colOffsets
    )
  })
}

.component_response_adjust_previous_no_family <- function(
    yMat,
    umat,
    vmat,
    kprime
) {
  # Reconstructs the response vector when Y has been adjusted to:
  #
  #   Y - sum_previous_components u_prev v_prev^T
  
  y <- yMat[, kprime]
  
  if (kprime > 1) {
    adj <- rowSums(
      do.call(
        cbind,
        lapply(seq_len(kprime - 1), function(kk) {
          sum(vmat[, kprime] * vmat[, kk]) * umat[, kk, drop = FALSE]
        })
      )
    )
    
    y <- y - adj
  }
  
  y
}

.component_response_adjust_previous_no_family_single <- function(
    yMat,
    umat,
    vmat,
    kprime
) {
  .component_response_adjust_previous_no_family(
    yMat = yMat,
    umat = umat,
    vmat = vmat,
    kprime = kprime
  )
}

.component_response_matrix_no_family_integrated <- function(fit, Y) {
  # Integrated analogue.
  #
  # C++ analogue:
  #
  # this->w_hat = 0 * this->w_hat;
  #
  # for d = 1,...,D:
  #   vtYd = vk[d].t() * Yd
  #   vtYd = vtYd
  #       - dot(vk[d], ones_pd[d]) * colOffsets_d.t()
  #       - dot(vk[d], rowOffsets_d) * ones_n.t()
  #   Zetavd = Zeta[d] * vk[d]
  #   vv = vmatOthers[d].t() * vk[d]
  #   this->w_hat += vtYd.t() - umatOthers * vv - Z[d] * Zetavd
  #
  # Returns n x K matrix of summed working responses across datasets,
  # before subtracting previous component contributions.
  
  with(fit$fitAtOptCV, {
    
    Zlist <- trainModel$Z %||% Z
    
    Reduce(
      "+",
      lapply(seq_along(Y), function(d) {
        .comp_response_matrix_no_family_from_parts(
          Y = Y[[d]],
          vmat = vmat[[d]],
          Z = Zlist[[d]],
          Zeta = Zeta[[d]],
          rowOffsets = rowOffsets[[d]],
          colOffsets = colOffsets[[d]]
        )
      })
    )
  })
}

.component_response_adjust_previous_no_family_integrated <- function(
    yMat,
    umat,
    vmat,
    kprime
) {
  # Reconstructs the integrated response vector from:
  #
  #   sum_d response_d,k
  #
  # after accounting for previous components:
  #
  #   sum_d sum_prev (v_prev[d]' v_k[d]) u_prev
  
  y <- yMat[, kprime]
  
  if (kprime > 1) {
    adj <- Reduce(
      "+",
      lapply(seq_along(vmat), function(d) {
        rowSums(
          do.call(
            cbind,
            lapply(seq_len(kprime - 1), function(kk) {
              sum(vmat[[d]][, kprime] * vmat[[d]][, kk]) *
                umat[, kk, drop = FALSE]
            })
          )
        )
      })
    )
    
    y <- y - adj
  }
  
  y
}


.gss_style_component_summary <- function(
    y,
    yhat,
    term_mat,
    w = NULL,
    eps = 1e-12
) {
  stopifnot(nrow(term_mat) == length(y))
  
  res <- y - yhat
  
  comp <- cbind(term_mat, yhat = yhat, y = y, e = res)
  
  if (!is.null(w)) {
    comp <- sqrt(w) * comp - outer(sqrt(w), apply(w * comp, 2, sum)) / sum(w)
  } else {
    comp <- sweep(comp, 2, colMeans(comp, na.rm = TRUE), "-")
  }
  
  term_names <- colnames(term_mat)
  
  comp1 <- comp[, c(term_names, "yhat"), drop = FALSE]
  
  decom <- as.numeric(crossprod(comp1, comp1[, "yhat"]))
  names(decom) <- colnames(comp1)
  pi <- decom[term_names] / decom["yhat"]
  
  S <- crossprod(comp)
  d <- sqrt(diag(S))
  R <- t(S / d) / d
  R[!is.finite(R)] <- NA_real_
  
  norm <- apply(comp, 2, function(x) sqrt(sum(x^2, na.rm = TRUE)))
  
  R_terms <- R[term_names, term_names, drop = FALSE]
  if (anyNA(R_terms) || qr(R_terms)$rank < ncol(R_terms)) {
    kappa <- rep(Inf, length(term_names))
  } else {
    kappa <- sqrt(diag(solve(R_terms)))
  }
  names(kappa) <- term_names
  
  max_abs_cos_other <- nearest_term <- rep(NA_real_, length(term_names))
  names(max_abs_cos_other) <- names(nearest_term) <- term_names
  
  nearest_term_chr <- rep(NA_character_, length(term_names))
  names(nearest_term_chr) <- term_names
  
  for (tt in term_names) {
    others <- setdiff(term_names, tt)
    if (length(others) > 0) {
      vals <- abs(R_terms[tt, others])
      ix <- which.max(vals)
      max_abs_cos_other[tt] <- vals[ix]
      nearest_term_chr[tt] <- others[ix]
    }
  }
  
  r_squared <- if (is.null(w)) {
    stats::var(yhat, na.rm = TRUE) / stats::var(y, na.rm = TRUE)
  } else {
    ybar <- sum(w * y, na.rm = TRUE) / sum(w, na.rm = TRUE)
    yhatbar <- sum(w * yhat, na.rm = TRUE) / sum(w, na.rm = TRUE)
    sum(w * (yhat - yhatbar)^2, na.rm = TRUE) / sum(w * (y - ybar)^2, na.rm = TRUE)
  }
  
  rss <- if (is.null(w)) sum(res^2, na.rm = TRUE) else sum(w * res^2, na.rm = TRUE)
  
  term_effects <- tibble::tibble(
    term = term_names,
    cos_y = as.numeric(R[term_names, "y"]),
    cos_e = as.numeric(R[term_names, "e"]),
    norm = as.numeric(norm[term_names]),
    pi = as.numeric(pi),
    kappa = as.numeric(kappa),
    max_abs_cos_other = as.numeric(max_abs_cos_other),
    nearest_term = nearest_term_chr
  )
  
  list(
    term_effects = term_effects,
    cosine_matrix = R_terms,
    r_squared = r_squared,
    rss = rss,
    pi_sum = sum(pi, na.rm = TRUE)
  )
}

featurewise_Z_U_cosines_all_terms <- function(
    Z,
    Zeta,
    u_terms,
    zformula = NULL,
    z_assign = attr(Z, "assign"),
    z_term_names = NULL,
    include_intercept = FALSE,
    return_distribution = TRUE,
    eps = 1e-12
) {
  stopifnot(is.matrix(Z))
  stopifnot(is.matrix(Zeta))
  stopifnot(ncol(Z) == ncol(Zeta))
  stopifnot(is.list(u_terms))
  
  if (is.null(names(u_terms))) names(u_terms) <- paste0("u_term_", seq_along(u_terms))
  if (is.null(z_assign)) stop("z_assign is NULL.", call. = FALSE)
  
  if (is.null(z_term_names)) {
    if (!is.null(zformula)) {
      z_term_names <- c("(Intercept)", attr(terms(zformula), "term.labels"))
    } else {
      z_term_names <- paste0("z_term_", sort(unique(z_assign)))
    }
  }
  
  z_term_ids <- sort(unique(z_assign))
  if (!include_intercept) z_term_ids <- setdiff(z_term_ids, 0L)
  
  summary_list <- list()
  distribution_list <- list()
  
  for (zt in z_term_ids) {
    z_cols <- which(z_assign == zt)
    if (length(z_cols) == 0) next
    
    z_name <- if ((zt + 1L) <= length(z_term_names)) z_term_names[zt + 1L] else paste0("z_term_", zt)
    
    Zsub <- Z[, z_cols, drop = FALSE]
    Zetasub <- Zeta[, z_cols, drop = FALSE]
    
    Gz <- crossprod(Zsub)
    z_norm <- sqrt(rowSums((Zetasub %*% Gz) * Zetasub))
    z_norm[z_norm < eps | !is.finite(z_norm)] <- NA_real_
    
    for (ut in names(u_terms)) {
      u <- as.numeric(u_terms[[ut]])
      if (length(u) != nrow(Z)) stop("u_terms element has wrong length: ", ut, call. = FALSE)
      
      u <- u - mean(u, na.rm = TRUE)
      u_norm <- sqrt(sum(u^2, na.rm = TRUE))
      
      if (u_norm < eps || !is.finite(u_norm)) {
        cosines <- rep(NA_real_, nrow(Zeta))
      } else {
        a <- as.numeric(crossprod(Zsub, u))
        numer <- as.numeric(Zetasub %*% a)
        cosines <- numer / (u_norm * z_norm)
      }
      
      abs_cos <- abs(cosines)
      
      summary_list[[length(summary_list) + 1L]] <- tibble::tibble(
        u_term = ut,
        z_term = z_name,
        n_features = length(abs_cos),
        n_finite = sum(is.finite(abs_cos)),
        mean_abs_cos = mean(abs_cos, na.rm = TRUE),
        median_abs_cos = stats::median(abs_cos, na.rm = TRUE),
        q90_abs_cos = as.numeric(stats::quantile(abs_cos, 0.90, na.rm = TRUE)),
        q95_abs_cos = as.numeric(stats::quantile(abs_cos, 0.95, na.rm = TRUE)),
        q99_abs_cos = as.numeric(stats::quantile(abs_cos, 0.99, na.rm = TRUE)),
        max_abs_cos = max(abs_cos, na.rm = TRUE)
      )
      
      if (return_distribution) {
        distribution_list[[length(distribution_list) + 1L]] <- tibble::tibble(
          u_term = ut,
          z_term = z_name,
          feature = seq_len(nrow(Zeta)),
          cosine = cosines,
          abs_cosine = abs_cos
        )
      }
    }
  }
  
  out <- list(summary = dplyr::bind_rows(summary_list))
  if (return_distribution) out$distribution <- dplyr::bind_rows(distribution_list)
  out
}

.normalize_summary_family <- function(family) {
  if (is.null(family)) return(NULL)
  
  if (length(family) == 1 && is.na(family)) return(NULL)
  
  if (is.list(family)) {
    empty <- vapply(
      family,
      function(x) is.null(x) || (length(x) == 1 && is.na(x)),
      logical(1)
    )
    if (all(empty)) return(NULL)
  }
  
  family
}


summary_svdx_fit <- function(
    fit,
    Y,
    newdata = NULL,
    return_conf_distribution = FALSE,
    ...
) {
  
  family <- family <- .normalize_summary_family(
    fit$fitAtOptCV$family %||%
      fit$fitAtOptCV$trainModel$family %||%
      fit$family
  )
  
  if (!is.null(family)) {
    .skip_family_term_summary("SSANOVASVD")
    
    out <- list(
      components = NULL,
      term_effects = NULL,
      cosine_matrices = NULL,
      xz_confounding = NULL,
      xz_confounding_distribution = NULL,
      todos = .svdx_summary_todos,
      call = fit$call
    )
    
    class(out) <- "summary.SSANOVASVD"
    return(out)
  }
  
  object <- fit$fitAtOptCV$trainModel$mod
  w <- model.weights(object$mf)
  mf <- newdata %||% object$mf
  
  x_terms <- .get_x_terms(fit)
  x_terms <- x_terms[!x_terms %in% c("1", "(Intercept)", "offset")]
  
  term_preds <- .predict_terms_array(fit, mf, x_terms, include_intercept = FALSE)
  
  Ufull <- predict(fit, mf)
  Ufull <- as.matrix(Ufull)
  colnames(Ufull) <- colnames(Ufull) %||% .make_component_names(ncol(Ufull))
  
  yMat <- .component_response_matrix_no_family_single(fit, Y)
  umat <- fit$fitAtOptCV$umat
  vmat <- fit$fitAtOptCV$vmat
  
  component_rows <- list()
  term_rows <- list()
  cosine_mats <- list()
  
  for (k in seq_len(ncol(Ufull))) {
    component <- colnames(Ufull)[k]
    y <- .component_response_adjust_previous_no_family_single(
      yMat = yMat,
      umat = umat,
      vmat = vmat,
      kprime = k
    )
    yhat <- Ufull[, k]
    
    term_mat <- do.call(cbind, lapply(names(term_preds), function(tt) term_preds[[tt]][, k]))
    colnames(term_mat) <- names(term_preds)
    
    gs <- .gss_style_component_summary(y = y, yhat = yhat, term_mat = term_mat, w = w)
    
    term_rows[[k]] <- gs$term_effects |>
      dplyr::mutate(component = component, .before = 1)
    
    component_rows[[k]] <- tibble::tibble(
      component = component,
      r_squared = gs$r_squared,
      rss = gs$rss,
      pi_sum = gs$pi_sum,
      max_kappa = max(gs$term_effects$kappa, na.rm = TRUE),
      dominant_term = gs$term_effects$term[which.max(abs(gs$term_effects$pi))],
      dominant_pi = gs$term_effects$pi[which.max(abs(gs$term_effects$pi))]
    )
    
    cosine_mats[[component]] <- gs$cosine_matrix
  }
  
  term_effects <- dplyr::bind_rows(term_rows)
  components <- dplyr::bind_rows(component_rows)
  
  # x/z feature-wise diagnostics, component-specific.
  confounding <- NULL
  confounding_distribution <- NULL
  
  Z <- fit$fitAtOptCV$trainModel$Z %||% fit$fitAtOptCV$Z
  Zeta <- fit$fitAtOptCV$Zeta
  zformula <- fit$fitAtOptCV$trainModel$mod$z.formula %||% fit$zformula
  
  if (!is.null(Z) && !is.null(Zeta) && length(x_terms) > 0) {
    conf_summaries <- list()
    conf_distributions <- list()
    
    for (k in seq_len(ncol(Ufull))) {
      component <- colnames(Ufull)[k]
      
      u_terms_k <- lapply(names(term_preds), function(tt) term_preds[[tt]][, k])
      names(u_terms_k) <- names(term_preds)
      
      cres <- featurewise_Z_U_cosines_all_terms(
        Z = as.matrix(Z),
        Zeta = as.matrix(Zeta),
        u_terms = u_terms_k,
        zformula = zformula,
        include_intercept = FALSE,
        return_distribution = return_conf_distribution
      )
      
      conf_summaries[[k]] <- cres$summary |>
        dplyr::mutate(component = component, .before = 1)
      
      if (return_conf_distribution && !is.null(cres$distribution)) {
        conf_distributions[[k]] <- cres$distribution |>
          dplyr::mutate(component = component, .before = 1)
      }
    }
    
    confounding <- dplyr::bind_rows(conf_summaries)
    if (return_conf_distribution) confounding_distribution <- dplyr::bind_rows(conf_distributions)
  }
  
  out <- list(
    components = components,
    term_effects = term_effects,
    term_cosine_matrices = cosine_mats,
    xz_confounding = confounding,
    xz_confounding_distribution = confounding_distribution,
    call = match.call()
  )
  
  class(out) <- "summary.SSANOVASVD"
  out
}

summary.SSANOVASVD <- function(object, Y, ...) {
  summary_svdx_fit(object, Y = Y, ...)
}

