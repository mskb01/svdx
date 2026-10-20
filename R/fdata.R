# ============================================================
# Feature metadata helpers
# ============================================================
# Lightweight feature annotation object used to carry display labels and
# enrichment identifiers without imposing column-name conventions on users.

#' Construct SVDX feature metadata
#'
#' @param fdata Data frame with one row per feature.
#' @param feature_col Column in `fdata` containing the feature identifiers used
#'   as row names of the input matrix/matrices.
#' @param label_col Optional column in `fdata` containing display labels for
#'   plots, such as gene symbols.
#' @param enrichment_col Optional column in `fdata` containing identifiers used
#'   for enrichment, such as gene symbols, Entrez/NCBI IDs, or Ensembl IDs.
#'
#' @return A data frame with class `svdx_fdata`. The input columns are preserved;
#'   semantic mappings are stored as attributes.
#' @export
as_svdx_fdata <- function(fdata, feature_col, label_col = NULL, enrichment_col = NULL) {
  if (is.list(fdata) && !is.data.frame(fdata) && !inherits(fdata, "svdx_fdata")) {
    if (!length(fdata)) stop("fdata list must contain at least one data frame.", call. = FALSE)
    dataset_names <- names(fdata)
    if (is.null(dataset_names) || length(dataset_names) != length(fdata) || any(!nzchar(dataset_names))) {
      dataset_names <- paste0("d", seq_along(fdata))
      names(fdata) <- dataset_names
    }
    if (anyDuplicated(dataset_names)) {
      stop("Integrated fdata dataset names must be unique.", call. = FALSE)
    }
    if (!all(vapply(fdata, is.data.frame, logical(1)))) {
      stop("Each element of integrated fdata must be a data frame.", call. = FALSE)
    }

    resolve_map_arg <- function(arg, arg_name, required = FALSE) {
      if (is.null(arg)) {
        if (required) stop(arg_name, " must be supplied.", call. = FALSE)
        vals <- vector("list", length(fdata))
        names(vals) <- names(fdata)
        return(vals)
      }
      if (is.list(arg) && !is.data.frame(arg)) {
        vals <- arg
      } else {
        vals <- as.list(arg)
      }
      if (length(vals) == 1L) vals <- rep(vals, length(fdata))
      if (length(vals) != length(fdata)) {
        stop(arg_name, " must be length 1 or have one entry per dataset.", call. = FALSE)
      }
      if (!is.null(names(vals)) && all(nzchar(names(vals)))) {
        missing_names <- setdiff(names(fdata), names(vals))
        if (length(missing_names) > 0L) {
          stop(arg_name, " is missing dataset(s): ", paste(missing_names, collapse = ", "), call. = FALSE)
        }
        vals <- vals[names(fdata)]
      } else {
        names(vals) <- names(fdata)
      }
      vals
    }

    feature_cols <- resolve_map_arg(feature_col, "feature_col", required = TRUE)
    label_cols <- resolve_map_arg(label_col, "label_col", required = FALSE)
    enrichment_cols <- resolve_map_arg(enrichment_col, "enrichment_col", required = FALSE)

    out <- stats::setNames(vector("list", length(fdata)), names(fdata))
    for (nm in names(fdata)) {
      out[[nm]] <- as_svdx_fdata(
        fdata[[nm]],
        feature_col = feature_cols[[nm]],
        label_col = label_cols[[nm]],
        enrichment_col = enrichment_cols[[nm]]
      )
    }
    class(out) <- unique(c("svdx_fdata_list", class(out)))
    return(out)
  }

  if (!is.data.frame(fdata)) stop("fdata must be a data frame, or a named list of data frames for integrated workflows.", call. = FALSE)
  if (missing(feature_col) || is.null(feature_col) || length(feature_col) != 1L) {
    stop("feature_col must name one column in fdata.", call. = FALSE)
  }

  required <- c(feature_col, label_col, enrichment_col)
  required <- required[!is.null(required) & !is.na(required)]
  missing_cols <- setdiff(required, colnames(fdata))
  if (length(missing_cols) > 0L) {
    stop("fdata column(s) not found: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  features <- as.character(fdata[[feature_col]])
  if (any(is.na(features) | !nzchar(features))) {
    stop("feature_col contains missing or empty feature identifiers.", call. = FALSE)
  }
  if (anyDuplicated(features)) {
    stop("feature_col must contain unique feature identifiers.", call. = FALSE)
  }

  attr(fdata, "svdx_feature_col") <- feature_col
  attr(fdata, "svdx_label_col") <- label_col
  attr(fdata, "svdx_enrichment_col") <- enrichment_col
  class(fdata) <- unique(c("svdx_fdata", class(fdata)))
  fdata
}

#' @export
is_svdx_fdata <- function(x) inherits(x, "svdx_fdata")

.svdx_fdata_get_col <- function(fdata, attr_name, required = FALSE, purpose = "") {
  if (is.null(fdata)) {
    if (isTRUE(required)) stop("fdata is required", if (nzchar(purpose)) paste0(" for ", purpose) else "", ".", call. = FALSE)
    return(NULL)
  }
  col <- attr(fdata, attr_name, exact = TRUE)
  if (is.null(col) || !nzchar(col)) {
    if (isTRUE(required)) {
      stop("fdata does not define ", attr_name, if (nzchar(purpose)) paste0(" required for ", purpose) else "", ".", call. = FALSE)
    }
    return(NULL)
  }
  if (!col %in% colnames(fdata)) {
    stop("fdata attribute ", attr_name, " refers to missing column: ", col, call. = FALSE)
  }
  as.character(fdata[[col]])
}

.svdx_fdata_feature <- function(fdata, required = TRUE) {
  .svdx_fdata_get_col(fdata, "svdx_feature_col", required = required, purpose = "feature matching")
}

.svdx_fdata_label <- function(fdata, required = FALSE) {
  .svdx_fdata_get_col(fdata, "svdx_label_col", required = required, purpose = "plot labels")
}

.svdx_fdata_enrichment <- function(fdata, required = FALSE) {
  .svdx_fdata_get_col(fdata, "svdx_enrichment_col", required = required, purpose = "enrichment")
}

.svdx_fdata_to_row_labels <- function(fdata) {
  if (is.null(fdata)) return(NULL)
  features <- .svdx_fdata_feature(fdata, required = TRUE)
  labels <- .svdx_fdata_label(fdata, required = FALSE)
  if (is.null(labels)) return(NULL)
  data.frame(feature = features, label = labels, stringsAsFactors = FALSE)
}

.svdx_fdata_to_enrichment_id_map <- function(fdata, required = TRUE) {
  if (is.null(fdata)) {
    if (isTRUE(required)) stop("fdata is required to derive enrichment_id_map.", call. = FALSE)
    return(NULL)
  }
  features <- .svdx_fdata_feature(fdata, required = TRUE)
  ids <- .svdx_fdata_enrichment(fdata, required = required)
  if (is.null(ids)) return(NULL)
  data.frame(feature = features, enrichment_id = ids, stringsAsFactors = FALSE)
}

.svdx_resolve_fdata_for_dataset <- function(fdata, dataset = NULL, dataset_names = NULL, required = FALSE) {
  if (is.null(fdata)) {
    if (isTRUE(required)) stop("fdata is required.", call. = FALSE)
    return(NULL)
  }
  if (is.list(fdata) && !is.data.frame(fdata) && !inherits(fdata, "svdx_fdata")) {
    if (is.null(dataset)) stop("dataset must be supplied when fdata is a list.", call. = FALSE)
    if (is.numeric(dataset)) {
      if (dataset < 1L || dataset > length(fdata)) stop("dataset index is out of range for fdata.", call. = FALSE)
      return(fdata[[dataset]])
    }
    if (!dataset %in% names(fdata)) {
      stop("fdata does not contain dataset: ", dataset, call. = FALSE)
    }
    return(fdata[[dataset]])
  }
  fdata
}

.svdx_validate_fdata_matches_features <- function(fdata, features, context = "Y") {
  if (is.null(fdata)) return(invisible(NULL))
  f <- .svdx_fdata_feature(fdata, required = TRUE)
  missing <- setdiff(features, f)
  if (length(missing) > 0L) {
    stop("fdata is missing feature annotations for ", length(missing), " feature(s) in ", context, ".", call. = FALSE)
  }
  invisible(NULL)
}

#' @export
print.svdx_fdata <- function(x, ...) {
  cat("svdx_fdata object\n")
  cat("=================\n")
  cat("Features: ", nrow(x), "\n", sep = "")
  cat("Feature column: ", attr(x, "svdx_feature_col", exact = TRUE) %||% "<unset>", "\n", sep = "")
  cat("Label column: ", attr(x, "svdx_label_col", exact = TRUE) %||% "<unset>", "\n", sep = "")
  cat("Enrichment column: ", attr(x, "svdx_enrichment_col", exact = TRUE) %||% "<unset>", "\n", sep = "")
  invisible(x)
}

#' @export
summary.svdx_fdata <- function(object, ...) {
  out <- list(
    n_features = nrow(object),
    feature_col = attr(object, "svdx_feature_col", exact = TRUE),
    label_col = attr(object, "svdx_label_col", exact = TRUE),
    enrichment_col = attr(object, "svdx_enrichment_col", exact = TRUE),
    columns = colnames(object)
  )
  class(out) <- "summary.svdx_fdata"
  out
}

#' @export
print.summary.svdx_fdata <- function(x, ...) {
  cat("svdx_fdata summary\n")
  cat("==================\n")
  cat("Features: ", x$n_features, "\n", sep = "")
  cat("Feature column: ", x$feature_col %||% "<unset>", "\n", sep = "")
  cat("Label column: ", x$label_col %||% "<unset>", "\n", sep = "")
  cat("Enrichment column: ", x$enrichment_col %||% "<unset>", "\n", sep = "")
  invisible(x)
}

.svdx_subset_fdata <- function(fdata, features) {
  if (is.null(fdata)) return(NULL)
  f <- .svdx_fdata_feature(fdata, required = TRUE)
  idx <- match(features, f)
  if (anyNA(idx)) {
    stop("fdata is missing feature annotations for selected features.", call. = FALSE)
  }
  out <- fdata[idx, , drop = FALSE]
  attr(out, "svdx_feature_col") <- attr(fdata, "svdx_feature_col", exact = TRUE)
  attr(out, "svdx_label_col") <- attr(fdata, "svdx_label_col", exact = TRUE)
  attr(out, "svdx_enrichment_col") <- attr(fdata, "svdx_enrichment_col", exact = TRUE)
  class(out) <- unique(c("svdx_fdata", setdiff(class(out), "svdx_fdata")))
  rownames(out) <- NULL
  out
}


.svdx_format_family <- function(family) {
  if (is.null(family)) return("NULL")
  if (is.list(family)) {
    vals <- vapply(family, function(x) if (is.null(x)) "NULL" else as.character(x)[1], character(1))
    return(paste(paste(names(vals), vals, sep = "="), collapse = ", "))
  }
  as.character(family)[1]
}

.svdx_format_description <- function(x) {
  if (is.null(x)) return("<not recorded>")
  if (is.list(x)) {
    vals <- vapply(x, function(y) if (is.null(y)) "<not recorded>" else as.character(y)[1], character(1))
    return(paste(paste(names(vals), vals, sep = "="), collapse = "; "))
  }
  as.character(x)[1]
}


.svdx_format_modelType <- function(modelType) {
  if (is.null(modelType) || !length(modelType)) return("<none>")
  vals <- vapply(names(modelType), function(nm) {
    spec <- modelType[[nm]]
    if (is.character(spec) && length(spec) == 1L) {
      paste0(nm, "=", spec)
    } else if (is.list(spec) && length(spec) >= 2L && is.character(spec[[1L]]) && is.numeric(spec[[2L]]) && length(spec[[2L]]) == 2L) {
      paste0(nm, "=", spec[[1L]], " [", paste(signif(spec[[2L]], 4), collapse = ", "), "]")
    } else if (is.list(spec) && length(spec) >= 1L && is.character(spec[[1L]])) {
      paste0(nm, "=", spec[[1L]])
    } else {
      paste0(nm, "=<custom>")
    }
  }, character(1))
  paste(vals, collapse = "; ")
}

# ============================================================
# Prepared SVDX data object
# ============================================================

#' Test whether an object is prepared SVDX data
#'
#' @param x Object to test.
#' @export
is_svdx_data <- function(x) inherits(x, "svdx_data")

#' @export
print.svdx_data <- function(x, ...) {
  cat("svdx_data object\n")
  cat("================\n")
  cat("Integrated: ", if (isTRUE(x$integrated)) "TRUE" else "FALSE", "\n", sep = "")
  cat("Requested family: ", .svdx_format_family(x$family), "\n", sep = "")
  cat("Fit family: ", .svdx_format_family(x$fit_family), "\n", sep = "")
  cat("Inference family: ", .svdx_format_family(x$inference_family), "\n", sep = "")
  cat("Fit objective: ", .svdx_format_description(x$fit_description), "\n", sep = "")
  if (isTRUE(x$integrated)) {
    cat("Datasets: ", paste(names(x$Y), collapse = ", "), "\n", sep = "")
    cat("Samples: ", nrow(x$xdata), "\n", sep = "")
    cat("Features by dataset:\n")
    print(vapply(x$Y, nrow, integer(1)))
  } else {
    cat("Features: ", nrow(x$Y), "\n", sep = "")
    cat("Samples: ", ncol(x$Y), "\n", sep = "")
  }
  cat("xformula: ", paste(deparse(x$xformula), collapse = " "), "\n", sep = "")
  cat("zformula: ", if (isTRUE(x$integrated)) "<dataset-specific/list>" else paste(deparse(x$zformula), collapse = " "), "\n", sep = "")
  invisible(x)
}

#' @export
summary.svdx_data <- function(object, ...) {
  out <- list(
    integrated = isTRUE(object$integrated),
    family = object$family,
    fit_family = object$fit_family,
    inference_family = object$inference_family,
    fit_description = object$fit_description,
    n_samples = if (isTRUE(object$integrated)) nrow(object$xdata) else ncol(object$Y),
    n_features = if (isTRUE(object$integrated)) vapply(object$Y, nrow, integer(1)) else nrow(object$Y),
    datasets = if (isTRUE(object$integrated)) names(object$Y) else NULL,
    xformula = object$xformula,
    zformula = object$zformula,
    modelType = object$modelType,
    x_term_info = object$x_term_info,
    z_term_info = object$z_term_info,
    has_fdata = !is.null(object$fdata),
    has_row_offset = !is.null(object$row_offset),
    has_col_offset = !is.null(object$col_offset),
    normalization_factors = object$normalization_factors,
    row_center = object$row_center,
    row_scale = object$row_scale
  )
  class(out) <- "summary.svdx_data"
  out
}

#' @export
print.summary.svdx_data <- function(x, ...) {
  cat("svdx_data summary\n")
  cat("=================\n")
  cat("Integrated: ", x$integrated, "\n", sep = "")
  cat("Requested family: ", .svdx_format_family(x$family), "\n", sep = "")
  cat("Fit family: ", .svdx_format_family(x$fit_family), "\n", sep = "")
  cat("Inference family: ", .svdx_format_family(x$inference_family), "\n", sep = "")
  cat("Fit objective: ", .svdx_format_description(x$fit_description), "\n", sep = "")
  cat("Samples: ", x$n_samples, "\n", sep = "")
  if (isTRUE(x$integrated)) {
    cat("Datasets: ", paste(x$datasets, collapse = ", "), "\n", sep = "")
    cat("Features by dataset:\n")
    print(x$n_features)
  } else {
    cat("Features: ", x$n_features, "\n", sep = "")
  }
  cat("fdata: ", if (isTRUE(x$has_fdata)) "present" else "absent", "\n", sep = "")
  cat("row_offset: ", if (isTRUE(x$has_row_offset)) "present" else "absent", "\n", sep = "")
  cat("col_offset: ", if (isTRUE(x$has_col_offset)) "present" else "absent", "\n", sep = "")
  invisible(x)
}

#' @export
as.list.svdx_data <- function(x, ...) {
  unclass(x)
}
