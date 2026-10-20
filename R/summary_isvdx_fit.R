# ============================================================
# Summary for fitted integrated SSANOVA-SVD objects
# ============================================================

.get_i_vmat_list <- function(fit) {
  vlist <- fit$fitAtOptCV$vmat %||% fit$vmat
  
  if (!is.list(vlist)) {
    stop("Expected list-valued vmat for iSSANOVASVD.", call. = FALSE)
  }
  
  names(vlist) <- names(vlist) %||% paste0("dataset_", seq_along(vlist))
  vlist
}

.get_i_Z_list <- function(fit) {
  Z <- fit$fitAtOptCV$trainModel$Z %||%
    fit$fitAtOptCV$Z %||%
    fit$Z
  
  if (is.null(Z)) return(NULL)
  
  if (!is.list(Z)) {
    stop("Expected list-valued Z for iSSANOVASVD.", call. = FALSE)
  }
  
  names(Z) <- names(Z) %||% paste0("dataset_", seq_along(Z))
  Z
}

.get_i_Zeta_list <- function(fit) {
  Zeta <- fit$fitAtOptCV$Zeta %||% fit$Zeta
  
  if (is.null(Zeta)) return(NULL)
  
  if (!is.list(Zeta)) {
    stop("Expected list-valued Zeta for iSSANOVASVD.", call. = FALSE)
  }
  
  names(Zeta) <- names(Zeta) %||% paste0("dataset_", seq_along(Zeta))
  Zeta
}

.get_i_zformula_list <- function(fit) {
  zf <- fit$fitAtOptCV$trainModel$zformula %||%
    fit$fitAtOptCV$zformula %||%
    fit$zformula
  
  if (is.null(zf)) return(NULL)
  
  if (!is.list(zf)) {
    stop("Expected list-valued zformula for iSSANOVASVD.", call. = FALSE)
  }
  
  names(zf) <- names(zf) %||% paste0("dataset_", seq_along(zf))
  zf
}

.get_i_family <-function(fit) {
  .normalize_summary_family(
    fit$fitAtOptCV$family %||%
      fit$fitAtOptCV$trainModel$family
  )
}

.summary_i_loadings <- function(vlist) {
  dplyr::bind_rows(lapply(names(vlist), function(ds) {
    V <- as.matrix(vlist[[ds]])
    colnames(V) <- colnames(V) %||% .make_component_names(ncol(V))
    
    feature_names <- rownames(V) %||% paste0("feature_", seq_len(nrow(V)))
    
    dplyr::bind_rows(lapply(seq_len(ncol(V)), function(k) {
      v <- V[, k]
      
      tibble::tibble(
        dataset = ds,
        component = colnames(V)[k],
        n_features = nrow(V),
        n_nonzero = sum(abs(v) > 1e-8, na.rm = TRUE),
        prop_nonzero = mean(abs(v) > 1e-8, na.rm = TRUE),
        loading_norm = sqrt(sum(v^2, na.rm = TRUE)),
        max_abs_loading = max(abs(v), na.rm = TRUE),
        mean_abs_loading = mean(abs(v), na.rm = TRUE),
        median_abs_loading = stats::median(abs(v), na.rm = TRUE),
        top_feature = feature_names[which.max(abs(v))],
        top_loading = v[which.max(abs(v))]
      )
    }))
  }))
}

#' Summarize a fitted integrated SSANOVA-SVD object
#'
#' Shared U/x-term diagnostics are computed using the integrated
#' non-family component response. Dataset-specific loading and
#' x/z confounding diagnostics are also returned.
#'
#' Family-based summaries are intentionally skipped for now because
#' they require feature-wise working weights at the converged fit.
#'
#' @export
summary.iSSANOVASVD <- function(
    object,
    Y,
    newdata = NULL,
    return_conf_distribution = FALSE,
    ...
) {
  fit <- object
  
  family <- .get_i_family(fit)
  
  vlist <- .get_i_vmat_list(fit)
  dataset_loadings <- .summary_i_loadings(vlist)
  
  if (!is.null(family)) {
    .skip_family_term_summary("iSSANOVASVD")
    
    out <- list(
      components = NULL,
      term_effects = NULL,
      dataset_term_effects = NULL,
      term_cosine_matrices = NULL,
      dataset_loadings = dataset_loadings,
      xz_confounding = NULL,
      xz_confounding_distribution = NULL,
      todos = .svdx_summary_todos,
      call = fit$call
    )
    
    class(out) <- "summary.iSSANOVASVD"
    return(out)
  }
  
  if (missing(Y) || is.null(Y)) {
    stop("Y must be supplied for summary.iSSANOVASVD().", call. = FALSE)
  }
  
  if (!is.list(Y)) {
    stop("For iSSANOVASVD, Y must be a list of feature x sample matrices.", call. = FALSE)
  }
  
  names(Y) <- names(Y) %||% names(vlist)
  if (!is.null(names(vlist)) && !is.null(names(Y))) {
    Y <- Y[names(vlist)]
  }
  
  
  
  mf <- newdata %||%
    fit$fitAtOptCV$trainModel$mod$mf %||%
    fit$fitAtOptCV$trainModel$mod$model
  
  if (is.null(mf)) {
    stop("Could not locate model frame. Please supply newdata.", call. = FALSE)
  }
  
  x_terms <- .get_x_terms(fit)
  x_terms <- x_terms[!x_terms %in% c("1", "(Intercept)", "offset")]
  
  term_preds <- .predict_terms_array(
    fit = fit,
    newdata = mf,
    terms = x_terms,
    include_intercept = FALSE
  )
  
  Ufull <- predict(fit, mf)
  Ufull <- as.matrix(Ufull)
  colnames(Ufull) <- colnames(Ufull) %||% .make_component_names(ncol(Ufull))
  
  yMat <- .component_response_matrix_no_family_integrated(
    fit = fit,
    Y = Y
  )
  
  component_rows <- list()
  term_rows <- list()
  cosine_mats <- list()
  
  for (k in seq_len(ncol(Ufull))) {
    component <- colnames(Ufull)[k]
    
    y <- .component_response_adjust_previous_no_family_integrated(
      yMat = yMat,
      umat = Ufull,
      vmat = vlist,
      kprime = k
    )
    
    yhat <- Ufull[, k]
    
    term_mat <- do.call(
      cbind,
      lapply(names(term_preds), function(tt) {
        term_preds[[tt]][, k]
      })
    )
    colnames(term_mat) <- names(term_preds)
    
    gs <- .gss_style_component_summary(
      y = y,
      yhat = yhat,
      term_mat = term_mat,
      w = NULL
    )
    
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
  
  components <- dplyr::bind_rows(component_rows)
  term_effects <- dplyr::bind_rows(term_rows)
  
  Zlist <- .get_i_Z_list(fit)
  Zetalist <- .get_i_Zeta_list(fit)
  zflist <- .get_i_zformula_list(fit)
  
  xz_confounding <- NULL
  xz_confounding_distribution <- NULL
  
  if (!is.null(Zlist) && !is.null(Zetalist) && length(x_terms) > 0) {
    conf_summaries <- list()
    conf_distributions <- list()
    
    for (ds in names(Zlist)) {
      if (is.null(Zlist[[ds]]) || is.null(Zetalist[[ds]])) next
      
      for (k in seq_len(ncol(Ufull))) {
        component <- colnames(Ufull)[k]
        
        u_terms_k <- lapply(names(term_preds), function(tt) {
          term_preds[[tt]][, k]
        })
        names(u_terms_k) <- names(term_preds)
        
        cres <- featurewise_Z_U_cosines_all_terms(
          Z = as.matrix(Zlist[[ds]]),
          Zeta = as.matrix(Zetalist[[ds]]),
          u_terms = u_terms_k,
          zformula = zflist[[ds]] %||% NULL,
          return_distribution = return_conf_distribution
        )
        
        if (!is.null(cres$summary) && nrow(cres$summary) > 0) {
          conf_summaries[[length(conf_summaries) + 1L]] <-
            cres$summary |>
            dplyr::mutate(
              dataset = ds,
              component = component,
              .before = 1
            )
        }
        
        if (
          return_conf_distribution &&
          !is.null(cres$distribution) &&
          nrow(cres$distribution) > 0
        ) {
          conf_distributions[[length(conf_distributions) + 1L]] <-
            cres$distribution |>
            dplyr::mutate(
              dataset = ds,
              component = component,
              .before = 1
            )
        }
      }
    }
    
    xz_confounding <- dplyr::bind_rows(conf_summaries)
    
    if (return_conf_distribution) {
      xz_confounding_distribution <- dplyr::bind_rows(conf_distributions)
    }
  }
  
  out <- list(
    components = components,
    term_effects = term_effects,
    term_cosine_matrices = cosine_mats,
    dataset_loadings = dataset_loadings,
    xz_confounding = xz_confounding,
    xz_confounding_distribution = xz_confounding_distribution,
    call = fit$call
  )
  
  class(out) <- "summary.iSSANOVASVD"
  out
}

#' @export
print.summary.iSSANOVASVD <- function(x, ...) {
  cat("\nIntegrated SSANOVA-SVD fit summary\n\n")
  
  if (!is.null(x$components)) {
    cat("Component summaries:\n")
    print(x$components)
  } else {
    cat("Component summaries: not available.\n")
  }
  
  cat("\nDataset loading summaries:\n")
  print(x$dataset_loadings)
  
  if (!is.null(x$xz_confounding) && nrow(x$xz_confounding) > 0) {
    cat("\nX/Z confounding diagnostics available in $xz_confounding.\n")
  }
  
  if (!is.null(x$todos)) {
    cat("\nTODOs:\n")
    cat(paste0(" - ", x$todos, collapse = "\n"))
    cat("\n")
  }
  
  invisible(x)
}

print.summary.SSANOVASVD <- function(x, ...) {
  cat("\nSSANOVA-SVD fit summary\n\n")
  
  if (!is.null(x$components))
    print(x$components)
  
  invisible(x)
}
