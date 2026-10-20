# ============================================================
# 4. Summary after inference
# ============================================================

.svdx_safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

.svdx_safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  max(x)
}

summary.svdxInference <- function(object, alpha = object$alpha %||% 0.05, n_top = 10, ...) {
  ur <- object$union_results
  fr <- object$feature_results
  
  component_summary <- fr |>
    dplyr::group_by(component) |>
    dplyr::summarise(
      n_significant = sum(qval < alpha, na.rm = TRUE),
      min_p = .svdx_safe_min(pval),
      max_abs_stat = .svdx_safe_max(abs(stat)),
      .groups = "drop"
    )
  
  union_summary <- tibble::tibble(
    n_features = nrow(ur),
    n_union_significant = sum(ur$union_q < alpha, na.rm = TRUE),
    n_global_significant = if ("global_q" %in% names(ur)) sum(ur$global_q < alpha, na.rm = TRUE) else NA_integer_,
    min_p = .svdx_safe_min(ur$min_p),
    min_union_q = .svdx_safe_min(ur$union_q),
    min_global_q = if ("global_q" %in% names(ur)) .svdx_safe_min(ur$global_q) else NA_real_,
    max_abs_stat = .svdx_safe_max(ur$max_abs_stat)
  )
  
  top_union <- ur |>
    dplyr::arrange(union_q, min_p) |>
    dplyr::slice_head(n = n_top)

  top_global <- if ("global_q" %in% names(ur)) {
    ur |>
      dplyr::arrange(global_q, global_p) |>
      dplyr::slice_head(n = n_top)
  } else {
    NULL
  }
  
  out <- list(
    union_summary = union_summary,
    component_summary = component_summary,
    top_union = top_union,
    top_global = top_global,
    alpha = alpha,
    call = object$call
  )
  
  class(out) <- "summary.svdxInference"
  out
}

