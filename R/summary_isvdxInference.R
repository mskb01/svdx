# ============================================================
# summary.isvdxInference
# ============================================================

#' Summarize an integrated SSANOVA-SVD inference object
#'
#' Applies summary.svdxInference() to each dataset-specific
#' inference result and combines the results.
#'
#' @param object An isvdxInference object
#' @param alpha Significance threshold
#' @param n_top Number of top features to retain per dataset
#' @param ...
#'
#' @export
summary.isvdxInference <- function(
    object,
    alpha = object$alpha %||% 0.05,
    n_top = 10,
    ...
) {
  
  dataset_names <- names(object$dataset_results)
  
  dataset_summaries <- lapply(dataset_names, function(ds) {
    
    s <- summary(
      object$dataset_results[[ds]],
      alpha = alpha,
      n_top = n_top,
      ...
    )
    
    list(
      dataset = ds,
      summary = s
    )
  })
  
  union_summary <- dplyr::bind_rows(
    lapply(dataset_summaries, function(x) {
      
      out <- x$summary$union_summary
      
      if (!is.null(out) && nrow(out) > 0) {
        out$dataset <- x$dataset
      }
      
      out
    })
  )
  
  component_summary <- dplyr::bind_rows(
    lapply(dataset_summaries, function(x) {
      
      out <- x$summary$component_summary
      
      if (!is.null(out) && nrow(out) > 0) {
        out$dataset <- x$dataset
      }
      
      out
    })
  )
  
  top_union <- dplyr::bind_rows(
    lapply(dataset_summaries, function(x) {
      
      out <- x$summary$top_union
      
      if (!is.null(out) && nrow(out) > 0) {
        out$dataset <- x$dataset
      }
      
      out
    })
  )

  top_global <- dplyr::bind_rows(
    lapply(dataset_summaries, function(x) {
      out <- x$summary$top_global
      if (!is.null(out) && nrow(out) > 0) {
        out$dataset <- x$dataset
      }
      out
    })
  )
  
  out <- list(
    dataset_summary = dataset_summaries,
    union_summary = union_summary,
    component_summary = component_summary,
    top_union = top_union,
    top_global = top_global,
    alpha = alpha,
    call = object$call
  )
  
  class(out) <- "summary.isvdxInference"
  
  out
}

# ============================================================
# print.summary.isvdxInference
# ============================================================

#' @export
print.summary.isvdxInference <- function(x, ...) {
  
  cat("\n")
  cat("Integrated SSANOVA-SVD inference summary\n")
  cat("\n")
  
  datasets <- unique(x$union_summary$dataset)
  
  cat("Datasets:\n")
  cat(paste(datasets, collapse = ", "))
  cat("\n\n")
  
  if (
    !is.null(x$union_summary) &&
    nrow(x$union_summary) > 0
  ) {
    
    cols <- intersect(
      c(
        "dataset",
        "n_features",
        "n_significant",
        "prop_significant"
      ),
      colnames(x$union_summary)
    )
    
    print(
      x$union_summary[, cols, drop = FALSE]
    )
  }
  
  invisible(x)
}

