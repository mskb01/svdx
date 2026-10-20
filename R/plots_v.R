#plots_v.R

# Move V plotting utilities here:
#   plotVLoadings()
#   plotVHeatmap()
#   plotVInferenceHeatmap()
#   plotVInferenceDotplot()
#   plotVLoadingInferenceScatter()



.resolve_cols <- function(Y, cols = NULL, warn_duplicates = TRUE) {
  if (is.null(cols)) return(seq_len(ncol(Y)))
  
  if (is.character(cols)) {
    idx <- match(cols, colnames(Y))
    if (any(is.na(idx))) stop("Some cols were not found in colnames(Y).", call. = FALSE)
  } else if (is.logical(cols)) {
    idx <- which(cols)
  } else {
    idx <- as.integer(cols)
  }
  
  if (anyDuplicated(idx)) {
    if (warn_duplicates) warning("Duplicate columns supplied; duplicates removed.", call. = FALSE)
    idx <- unique(idx)
  }
  
  idx
}

.order_metadata_rows <- function(data, order_by = NULL) {
  if (is.null(order_by) || length(order_by) == 0) return(seq_len(nrow(data)))
  
  missing <- setdiff(order_by, colnames(data))
  if (length(missing) > 0) {
    stop("order_by variable(s) not found in data: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  
  do.call(order, data[order_by])
}


.resolve_heatmap_row_labels <- function(row_labels, features) {
  if (is.null(row_labels)) return(features)

  if (is.vector(row_labels) && !is.list(row_labels)) {
    if (is.null(names(row_labels))) {
      if (length(row_labels) != length(features)) {
        stop("Unnamed row_labels must have the same length as the displayed features.", call. = FALSE)
      }
      return(as.character(row_labels))
    }

    out <- features
    hits <- match(features, names(row_labels))
    matched <- !is.na(hits)
    out[matched] <- as.character(row_labels[hits[matched]])
    return(out)
  }

  if (is.data.frame(row_labels)) {
    if (ncol(row_labels) < 2L) {
      stop("row_labels as a data frame must have at least two columns: feature and display label.", call. = FALSE)
    }
    map <- row_labels[, 1:2, drop = FALSE]
    names(map) <- c("feature", "label")
    if (anyDuplicated(map$feature)) {
      stop("row_labels feature column must not contain duplicate feature IDs.", call. = FALSE)
    }
    label_vec <- stats::setNames(as.character(map$label), as.character(map$feature))
    return(.resolve_heatmap_row_labels(label_vec, features))
  }

  stop("row_labels must be NULL, a named vector, an unnamed vector matching displayed features, or a two-column data frame.", call. = FALSE)
}

.normalize_feature_label_map <- function(row_labels, features = NULL) {
  if (is.null(row_labels)) return(NULL)

  if (is.vector(row_labels) && !is.list(row_labels)) {
    if (is.null(names(row_labels))) {
      if (is.null(features) || length(row_labels) != length(features)) {
        stop(
          "Unnamed row_labels must have the same length as features for feature profile plots.",
          call. = FALSE
        )
      }
      return(data.frame(
        feature = as.character(features),
        label = as.character(row_labels),
        stringsAsFactors = FALSE
      ))
    }

    return(data.frame(
      feature = as.character(names(row_labels)),
      label = as.character(row_labels),
      stringsAsFactors = FALSE
    ))
  }

  if (is.data.frame(row_labels)) {
    if (ncol(row_labels) < 2L) {
      stop("row_labels as a data frame must have at least two columns: feature and display label.", call. = FALSE)
    }
    out <- row_labels[, 1:2, drop = FALSE]
    names(out) <- c("feature", "label")
    out$feature <- as.character(out$feature)
    out$label <- as.character(out$label)
    return(out)
  }

  stop("row_labels must be NULL, a named vector, an unnamed vector matching features, or a two-column data frame.", call. = FALSE)
}


.resolve_row_labels_from_fdata <- function(row_labels = NULL, fdata = NULL, dataset = NULL) {
  if (!is.null(row_labels)) return(row_labels)
  fdata <- .svdx_resolve_fdata_for_dataset(fdata, dataset = dataset, required = FALSE)
  .svdx_fdata_to_row_labels(fdata)
}

.resolve_workflow_fdata <- function(object, fdata = NULL, dataset = NULL) {
  if (!is.null(fdata)) return(.svdx_resolve_fdata_for_dataset(fdata, dataset = dataset, required = FALSE))
  if (inherits(object, c("svdxWorkflow", "isvdxWorkflow")) && !is.null(object$fdata)) {
    return(.svdx_resolve_fdata_for_dataset(object$fdata, dataset = dataset, required = FALSE))
  }
  NULL
}

.resolve_annotation_colors <- function(df, reference_df = NULL) {
  if (is.null(df) || ncol(df) == 0L) return(NULL)
  if (is.null(reference_df)) reference_df <- df

  out <- list()
  for (nm in colnames(df)) {
    x <- df[[nm]]
    ref <- if (nm %in% colnames(reference_df)) reference_df[[nm]] else x

    if (is.numeric(x) || is.integer(x)) {
      rng <- range(ref, na.rm = TRUE)
      if (length(rng) != 2L || any(!is.finite(rng))) next
      if (identical(rng[1], rng[2])) {
        delta <- if (rng[1] == 0) 0.5 else abs(rng[1]) * 0.05
        rng <- rng + c(-delta, delta)
      }
      if (requireNamespace("circlize", quietly = TRUE)) {
        out[[nm]] <- circlize::colorRamp2(
          rng,
          grDevices::hcl.colors(2L, palette = "BluYl")
        )
      }
      next
    }

    lev <- if (is.factor(ref)) {
      levels(ref)
    } else {
      sort(unique(as.character(ref[!is.na(ref)])))
    }
    if (length(lev) == 0L) next
    cols <- grDevices::hcl.colors(length(lev), palette = "Dark 3")
    out[[nm]] <- stats::setNames(cols, lev)
  }
  if (length(out) == 0L) NULL else out
}

.resolve_profile_features <- function(features, feature_names, row_labels = NULL) {
  if (missing(features) || is.null(features) || length(features) == 0L) {
    stop("features must contain at least one feature ID or display label.", call. = FALSE)
  }

  requested <- as.character(features)
  label_map <- .normalize_feature_label_map(row_labels, features = NULL)

  if (!is.null(label_map)) {
    if (anyDuplicated(label_map$feature)) {
      stop("row_labels feature column must not contain duplicate feature IDs.", call. = FALSE)
    }
  }

  resolved <- requested
  matched_feature <- requested %in% feature_names
  unresolved <- !matched_feature

  if (any(unresolved) && !is.null(label_map)) {
    dup_labels <- unique(label_map$label[duplicated(label_map$label)])
    requested_labels <- intersect(requested[unresolved], label_map$label)
    ambiguous <- intersect(requested_labels, dup_labels)
    if (length(ambiguous) > 0L) {
      stop(
        "Feature label(s) map to multiple features: ",
        paste(ambiguous, collapse = ", "),
        ". Please specify the original feature IDs or provide unique labels.",
        call. = FALSE
      )
    }

    hits <- match(requested[unresolved], label_map$label)
    has_hit <- !is.na(hits)
    unresolved_idx <- which(unresolved)
    resolved[unresolved_idx[has_hit]] <- label_map$feature[hits[has_hit]]
  }

  missing <- setdiff(resolved, feature_names)
  if (length(missing) > 0L) {
    stop(
      "Some requested features/labels were not found in rownames(Y) or row_labels: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  labels <- resolved
  if (!is.null(label_map)) {
    hits <- match(resolved, label_map$feature)
    use <- !is.na(hits)
    labels[use] <- label_map$label[hits[use]]
  }

  list(feature = resolved, label = labels)
}


.resolve_feature_profile_inference <- function(object, dataset = NULL) {
  if (inherits(object, "svdxWorkflow")) {
    return(object$inf)
  }

  if (inherits(object, "isvdxWorkflow")) {
    if (is.null(dataset)) {
      stop("dataset must be supplied for inference-driven feature selection from integrated workflows.", call. = FALSE)
    }
    return(.resolve_isvdx_workflow_dataset_inference(object, dataset))
  }

  if (inherits(object, "svdxInference")) {
    return(object)
  }

  if (inherits(object, "isvdxInference")) {
    if (is.null(dataset)) {
      stop("dataset must be supplied for inference-driven feature selection from integrated inference objects.", call. = FALSE)
    }
    return(.resolve_isvdx_inference_dataset(object, dataset))
  }

  NULL
}

.resolve_profile_features_from_inference <- function(
    object,
    feature_names,
    dataset = NULL,
    inference_scope = c("component", "union", "global"),
    component = NULL,
    select_by = c("q", "p", "stat"),
    top_n = NULL,
    direction = c("absolute", "positive", "negative"),
    row_labels = NULL
) {
  inference_scope <- match.arg(inference_scope)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  if (is.null(top_n) || length(top_n) != 1L || is.na(top_n) || top_n < 1L) {
    stop("top_n must be a positive integer when features is not supplied.", call. = FALSE)
  }
  top_n <- as.integer(top_n)

  inf <- .resolve_feature_profile_inference(object, dataset = dataset)
  if (is.null(inf)) {
    stop(
      "features must be supplied unless object is an SVDX workflow or inference object with inference results.",
      call. = FALSE
    )
  }

  selected <- .select_features_from_inference(
    inf,
    inference_scope = inference_scope,
    component = component,
    select_by = select_by,
    direction = direction,
    n_features = top_n
  )
  selected <- selected[!is.na(selected)]
  if (length(selected) == 0L) {
    message("Feature profile skipped: inference feature selection returned no valid features.")
    return(list(feature = character(), label = character()))
  }

  missing <- setdiff(selected, feature_names)
  if (length(missing) > 0L) {
    message("Feature profile skipped: inference-selected features were not found in the plotted Y matrix.")
    return(list(feature = character(), label = character()))
  }

  labels <- selected
  label_map <- .normalize_feature_label_map(row_labels, features = NULL)
  if (!is.null(label_map)) {
    if (anyDuplicated(label_map$feature)) {
      stop("row_labels feature column must not contain duplicate feature IDs.", call. = FALSE)
    }
    hits <- match(selected, label_map$feature)
    use <- !is.na(hits)
    labels[use] <- label_map$label[hits[use]]
  }

  list(feature = selected, label = labels)
}


.protect_profile_metadata_names <- function(meta, reserved) {
  nms <- colnames(meta)
  if (is.null(nms)) nms <- paste0("V", seq_len(ncol(meta)))

  if (anyDuplicated(nms)) {
    nms <- make.unique(nms, sep = "_")
  }

  hits <- nms %in% reserved
  if (any(hits)) {
    nms[hits] <- paste0("data_", nms[hits])
    nms <- make.unique(nms, sep = "_")
  }

  colnames(meta) <- nms
  meta
}

.align_profile_data <- function(Y, data) {
  if (is.null(data)) stop("data must be supplied for feature profile plots.", call. = FALSE)
  data <- as.data.frame(data)

  if (nrow(data) != ncol(Y)) {
    if (!is.null(colnames(Y)) && !is.null(rownames(data)) && all(colnames(Y) %in% rownames(data))) {
      data <- data[colnames(Y), , drop = FALSE]
    } else {
      stop("nrow(data) must equal ncol(Y), or rownames(data) must contain colnames(Y).", call. = FALSE)
    }
  } else if (!is.null(colnames(Y)) && !is.null(rownames(data)) && all(colnames(Y) %in% rownames(data))) {
    data <- data[colnames(Y), , drop = FALSE]
  }

  data
}

.resolve_feature_profile_input <- function(object, Y = NULL, data = NULL, dataset = NULL, source = c("auto", "train", "test", "all", "passed")) {
  source <- match.arg(source)

  if (source == "passed" || (!is.null(Y) && !is.null(data))) {
    Y <- .resolve_dataset_matrix(Y, dataset)
    return(list(Y = as.matrix(Y), data = as.data.frame(data), source = "passed"))
  }

  if (is.matrix(object) || is.data.frame(object)) {
    if (is.null(data)) stop("data must be supplied when object is a matrix/data frame.", call. = FALSE)
    return(list(Y = as.matrix(object), data = as.data.frame(data), source = "passed"))
  }

  choose_source <- function(obj, dataset = NULL) {
    if (source != "auto") return(source)
    if (inherits(obj, "isvdxWorkflow")) {
      if (!is.null(dataset) && !is.null(obj$Ytest) && !is.null(obj$data_test) &&
          !is.null(obj$Ytest[[dataset]])) return("test")
      if (!is.null(dataset) && !is.null(obj$Ytrain) && !is.null(obj$data_train) &&
          !is.null(obj$Ytrain[[dataset]])) return("train")
      if (!is.null(dataset) && !is.null(obj$Y) && !is.null(obj$data) &&
          !is.null(obj$Y[[dataset]])) return("all")
      stop("Workflow object does not contain dataset-specific Y/data; rerun with return_design=TRUE or pass Y and data directly.", call. = FALSE)
    }
    if (!is.null(obj$Ytest) && !is.null(obj$data_test)) return("test")
    if (!is.null(obj$Ytrain) && !is.null(obj$data_train)) return("train")
    if (!is.null(obj$Y) && !is.null(obj$data)) return("all")
    stop("Workflow object does not contain Y/data; rerun with return_design=TRUE or pass Y and data directly.", call. = FALSE)
  }

  if (inherits(object, "svdxWorkflow")) {
    src <- choose_source(object)
    if (src == "test") return(list(Y = as.matrix(object$Ytest), data = as.data.frame(object$data_test), source = src))
    if (src == "train") return(list(Y = as.matrix(object$Ytrain), data = as.data.frame(object$data_train), source = src))
    if (src == "all") return(list(Y = as.matrix(object$Y), data = as.data.frame(object$data), source = src))
  }

  if (inherits(object, "isvdxWorkflow")) {
    if (is.null(dataset)) stop("dataset must be supplied for integrated feature profile plots.", call. = FALSE)
    src <- choose_source(object, dataset = dataset)
    if (src == "test") return(list(Y = as.matrix(.resolve_dataset_matrix(object$Ytest, dataset)), data = as.data.frame(object$data_test), source = src))
    if (src == "train") return(list(Y = as.matrix(.resolve_dataset_matrix(object$Ytrain, dataset)), data = as.data.frame(object$data_train), source = src))
    if (src == "all") return(list(Y = as.matrix(.resolve_dataset_matrix(object$Y, dataset)), data = as.data.frame(object$data), source = src))
  }

  if (inherits(object, "svdxInference") || inherits(object, "isvdxInference")) {
    stop("Y and data must be supplied when plotting feature profiles from inference objects.", call. = FALSE)
  }

  stop("object must be a matrix/data frame, svdxWorkflow, isvdxWorkflow, svdxInference, or isvdxInference for feature profile plots. For inference objects, supply Y and data.", call. = FALSE)
}

.scale_profile_values <- function(df, scale = c("raw", "zscore")) {
  scale <- match.arg(scale)
  if (scale == "raw") return(df)

  df |>
    dplyr::group_by(.data$feature) |>
    dplyr::mutate(
      value = {
        s <- stats::sd(.data$value, na.rm = TRUE)
        if (!is.finite(s) || s <= 0) .data$value - mean(.data$value, na.rm = TRUE)
        else (.data$value - mean(.data$value, na.rm = TRUE)) / s
      }
    ) |>
    dplyr::ungroup()
}

#' Plot selected feature profiles against a sample-level covariate
#'
#' Plot a small set of selected features from a data matrix, SVDX workflow,
#' or integrated SVDX workflow against a user-specified x-axis. This is meant
#' as a close-up companion to V heatmaps: users can inspect a handful of genes
#' or other features as raw points, grouped trajectories, or summarized line plots.
#'
#' @param object A feature x sample matrix/data frame, an `svdxWorkflow`,
#'   `isvdxWorkflow`, `svdxInference`, or `isvdxInference` object. For
#'   inference objects, `Y` and `data` must be supplied because inference
#'   objects do not contain expression matrices.
#' @param Y Optional feature x sample matrix. If supplied with `data`, this is
#'   used instead of extracting train/test/all data from a workflow.
#' @param data Sample-level metadata corresponding to columns of `Y`.
#' @param features Optional character vector of feature IDs to plot. If
#'   `row_labels` is supplied, entries may also match the display labels in the
#'   second column of `row_labels`, provided labels are unique. If `features` is
#'   omitted, `top_n` features are selected from inference results using
#'   `component` and `select_by`.
#' @param x Column name in `data` to use as the x-axis. Use `x = "sample"`
#'   to plot samples on the x-axis.
#' @param component For `inference_scope = "component"`, a component number.
#'   When `features` is supplied explicitly, this may be length one or a numeric
#'   vector aligned with `features`; `NA` suppresses component-specific
#'   significance for that feature. Scalar component labels remain accepted for
#'   backward compatibility. When `features` is omitted, `component` must have
#'   length one.
#' @param inference_scope Inference result scope used for automatic feature
#'   selection when `features` is omitted: `"component"`, `"union"`, or
#'   `"global"`.
#' @param select_by Generic statistic used within `inference_scope` for
#'   automatic feature selection when `features` is omitted: `"q"`, `"p"`,
#'   or `"stat"`.
#' @param top_n Number of inference-selected features to plot when `features`
#'   is omitted.
#' @param direction Direction for statistic-based selection. Used when
#'   `select_by = "stat"`.
#' @param color Optional column name in the plotting data used for color. Use
#'   `color = "feature"` to color by selected feature/display label.
#' @param group Optional column name used for line grouping. If `line = TRUE`
#'   and `group` is `NULL`, grouping is inferred from `color` when possible,
#'   otherwise from feature.
#' @param facet Faceting variable. Use `facet = "feature"` to facet by feature
#'   display label, or `facet = NULL` to overlay selected features.
#' @param dataset Dataset name or index for integrated workflows or list-valued
#'   `Y` inputs.
#' @param source For workflow objects, which matrix to plot: `"test"`,
#'   `"train"`, `"all"`, or `"auto"`. `"auto"` prefers test data when
#'   available. Use `"passed"` when supplying `Y` and `data` directly.
#' @param row_labels Optional named vector or two-column data frame mapping
#'   original feature IDs to display labels. For a data frame, column names are
#'   ignored; the first column is the original feature ID and the second column
#'   is the display label. Explicit `row_labels` override labels derived from
#'   `fdata`.
#' @param fdata Optional `svdx_fdata` object. If supplied, its label column is
#'   used for display labels unless `row_labels` is supplied explicitly. For
#'   integrated workflows this may also be a named list and is resolved by
#'   `dataset`.
#' @param summarize `"none"` to show individual samples, or `"mean"`/`"median"`
#'   to aggregate by feature, x, and any color/group variables before plotting.
#' @param scale `"raw"` for original values or `"zscore"` for plotting-time
#'   row-wise z-scores across displayed samples.
#' @param show_significance Logical; if `TRUE`, annotate features significant at
#'   the requested inference scope and FDR threshold. Defaults to `TRUE`.
#' @param line Logical; draw connecting lines.
#' @param points Logical; draw points.
#' @param alpha Point/line alpha.
#' @param point_size Point size.
#' @param line_width Line width.
#' @return A `ggplot` object.
#' @export
plotFeatureProfiles <- function(
    object,
    Y = NULL,
    data = NULL,
    features = NULL,
    x,
    inference_scope = c("component", "union", "global"),
    component = NULL,
    select_by = c("q", "p", "stat"),
    top_n = NULL,
    direction = c("absolute", "positive", "negative"),
    color = NULL,
    group = NULL,
    facet = "feature",
    dataset = NULL,
    source = c("auto", "train", "test", "all", "passed"),
    row_labels = NULL,
    fdata = NULL,
    summarize = c("none", "mean", "median"),
    show_se = TRUE,
    scale = c("raw", "zscore"),
    line = FALSE,
    points = TRUE,
    alpha = 0.7,
    point_size = 1.8,
    line_width = 0.7,
    se_width = 0.15,
    fdr = 0.1,
    show_significance = TRUE
) {
  .require_namespace("ggplot2")

  source <- match.arg(source)
  inference_scope <- match.arg(inference_scope)
  summarize <- match.arg(summarize)
  scale <- match.arg(scale)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  inp <- .resolve_feature_profile_input(
    object = object,
    Y = Y,
    data = data,
    dataset = dataset,
    source = source
  )

  Y_use <- inp$Y
  data_use <- .align_profile_data(Y_use, inp$data)

  fdata_use <- .resolve_workflow_fdata(object, fdata = fdata, dataset = dataset)
  row_labels <- .resolve_row_labels_from_fdata(row_labels = row_labels, fdata = fdata_use, dataset = dataset)

  feature_names <- rownames(Y_use) %||% paste0("feature_", seq_len(nrow(Y_use)))
  if (is.null(rownames(Y_use))) rownames(Y_use) <- feature_names

  explicit_features <- !is.null(features) && length(features) > 0L
  if (!explicit_features && inference_scope == "component" && length(component) > 1L) {
    stop("component must have length one when features is not supplied.", call. = FALSE)
  }

  if (explicit_features) {
    resolved <- .resolve_profile_features(features, feature_names, row_labels = row_labels)
  } else {
    resolved <- .resolve_profile_features_from_inference(
      object = object,
      feature_names = feature_names,
      dataset = dataset,
      inference_scope = inference_scope,
      component = component,
      select_by = select_by,
      top_n = top_n,
      direction = direction,
      row_labels = row_labels
    )
  }
  if (length(resolved$feature) == 0L) {
    message("Feature profile skipped: no valid features were selected.")
    return(NULL)
  }

  inf_for_fdr <- .resolve_feature_profile_inference(object, dataset = dataset)
  if (isTRUE(show_significance) && !is.null(inf_for_fdr)) {
    fdr_status <- .get_feature_profile_fdr_status(
      inf_for_fdr,
      features = resolved$feature,
      inference_scope = inference_scope,
      component = component,
      fdr = fdr,
      component_vector_allowed = explicit_features
    )
    if (!is.null(fdr_status)) {
      mark <- fdr_status$significant[match(resolved$feature, fdr_status$feature)]
      mark[is.na(mark)] <- FALSE
      resolved$label[mark] <- paste0(resolved$label[mark], " *")
    }
  }

  Y_sel <- Y_use[resolved$feature, , drop = FALSE]

  sample_names <- colnames(Y_sel) %||% paste0("sample_", seq_len(ncol(Y_sel)))
  if (is.null(colnames(Y_sel))) colnames(Y_sel) <- sample_names

  plot_df <- data.frame(
    feature = rep(resolved$feature, each = ncol(Y_sel)),
    feature_label = rep(resolved$label, each = ncol(Y_sel)),
    sample = rep(sample_names, times = length(resolved$feature)),
    value = as.vector(t(Y_sel)),
    stringsAsFactors = FALSE
  )

  meta <- data_use[rep(seq_len(nrow(data_use)), times = length(resolved$feature)), , drop = FALSE]
  rownames(meta) <- NULL
  meta <- .protect_profile_metadata_names(
    meta,
    reserved = c("feature", "feature_label", "sample", "value")
  )
  plot_df <- cbind(plot_df, meta)
  if (anyDuplicated(colnames(plot_df))) {
    stop("Internal error: feature-profile plotting data has duplicate columns after metadata protection.", call. = FALSE)
  }
  plot_df <- .scale_profile_values(plot_df, scale = scale)

  if (!x %in% colnames(plot_df)) stop("x variable not found in data: ", x, call. = FALSE)

  aes_var <- function(v) {
    if (is.null(v)) return(NULL)
    if (identical(v, "feature")) return("feature_label")
    v
  }

  color_var <- aes_var(color)
  group_var <- aes_var(group)
  facet_var <- aes_var(facet)

  for (v in c(color_var, group_var, facet_var)) {
    if (!is.null(v) && !v %in% colnames(plot_df)) {
      stop("Plot variable not found in data: ", v, call. = FALSE)
    }
  }

  if (summarize != "none") {
    by_vars <- unique(c("feature", "feature_label", x, color_var, group_var, facet_var))
    by_vars <- by_vars[!is.na(by_vars) & nzchar(by_vars)]

    plot_df <- plot_df |>
      dplyr::group_by(dplyr::across(dplyr::all_of(by_vars))) |>
      dplyr::summarise(
        n = sum(!is.na(.data$value)),
        se = if (summarize == "mean") {
          stats::sd(.data$value, na.rm = TRUE) / sqrt(sum(!is.na(.data$value)))
        } else {
          NA_real_
        },
        value = if (summarize == "mean") mean(.data$value, na.rm = TRUE) else stats::median(.data$value, na.rm = TRUE),
        .groups = "drop"
      )
    plot_df$se[!is.finite(plot_df$se)] <- NA_real_
  }

  if (is.null(group_var)) {
    if (!is.null(color_var)) group_var <- color_var
    else group_var <- "feature_label"
  }

  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = .data[[x]], y = .data$value))

  if (isTRUE(line)) {
    if (!is.null(color_var)) {
      p <- p + ggplot2::geom_line(
        ggplot2::aes(color = .data[[color_var]], group = interaction(.data[[group_var]], .data$feature_label, drop = TRUE)),
        alpha = alpha,
        linewidth = line_width
      )
    } else {
      p <- p + ggplot2::geom_line(
        ggplot2::aes(group = interaction(.data[[group_var]], .data$feature_label, drop = TRUE)),
        alpha = alpha,
        linewidth = line_width
      )
    }
  }

  if (isTRUE(points)) {
    if (!is.null(color_var)) {
      p <- p + ggplot2::geom_point(
        ggplot2::aes(color = .data[[color_var]]),
        alpha = alpha,
        size = point_size
      )
    } else {
      p <- p + ggplot2::geom_point(alpha = alpha, size = point_size)
    }
  }

  if (isTRUE(show_se) && identical(summarize, "mean")) {
    if (!"se" %in% colnames(plot_df)) {
      stop("Internal error: show_se=TRUE but summarized feature-profile data does not contain an se column.", call. = FALSE)
    }

    if (all(is.na(plot_df$se))) {
      message("Feature-profile SE bars were not drawn because all standard errors are NA.")
    } else if (!is.null(color_var)) {
      p <- p + ggplot2::geom_errorbar(
        ggplot2::aes(
          ymin = .data$value - .data$se,
          ymax = .data$value + .data$se,
          color = .data[[color_var]],
          group = interaction(.data[[group_var]], .data$feature_label, drop = TRUE)
        ),
        width = se_width,
        alpha = alpha,
        linewidth = line_width
      )
    } else {
      p <- p + ggplot2::geom_errorbar(
        ggplot2::aes(
          ymin = .data$value - .data$se,
          ymax = .data$value + .data$se,
          group = interaction(.data[[group_var]], .data$feature_label, drop = TRUE)
        ),
        width = se_width,
        alpha = alpha,
        linewidth = line_width
      )
    }
  }

  if (!is.null(facet_var)) {
    p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", facet_var)), scales = "free_y")
  }

  p <- p +
    ggplot2::labs(
      x = x,
      y = ifelse(scale == "zscore", "Feature z-score", "Feature value"),
      color = color %||% NULL
    ) +
    ggplot2::theme_bw()

  if (is.factor(plot_df[[x]]) || is.character(plot_df[[x]])) {
    p <- p +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(
          angle = 90,
          hjust = 1,
          vjust = 0.5
        )
      )
  }

  p
}

.select_features_from_loadings <- function(
    fit,
    component = 1,
    direction = c("absolute", "positive", "negative", "both"),
    n_features = 50, 
    dataset = NULL
) {
  direction <- match.arg(direction)
  vmat <- .get_vmat(fit, dataset=dataset)
  colnames(vmat) <- colnames(vmat) %||% .make_component_names(ncol(vmat))
  k <- .resolve_component_index(component, colnames(vmat))
  v <- vmat[, k]
  
  if (direction == "absolute") {
    idx <- order(abs(v), decreasing = TRUE)
  } else if (direction == "positive") {
    idx <- order(v, decreasing = TRUE)
  } else if (direction == "negative") {
    idx <- order(v, decreasing = FALSE)
  } else {
    n_each <- ceiling(n_features / 2)
    idx <- unique(c(order(v, decreasing = TRUE)[seq_len(min(n_each, length(v)))],
                    order(v, decreasing = FALSE)[seq_len(min(n_each, length(v)))]))
  }
  
  head(idx, n_features)
}

.resolve_inference_component_label <- function(inf, component) {
  if (identical(component, "union")) return("union")

  comps <- unique(inf$feature_results$component)
  comps <- comps[!is.na(comps)]

  if (is.numeric(component)) {
    if (length(component) != 1L || component < 1L || component > length(comps)) {
      stop("component index is out of range for inference results.", call. = FALSE)
    }
    return(comps[[as.integer(component)]])
  }

  component_chr <- as.character(component)
  if (component_chr %in% comps) return(component_chr)

  component_num <- suppressWarnings(as.integer(component_chr))
  if (!is.na(component_num) && component_num >= 1L && component_num <= length(comps)) {
    return(comps[[component_num]])
  }

  stop("component was not found in inference results: ", component_chr, call. = FALSE)
}

.inference_components <- function(inf) {
  comps <- unique(as.character(inf$feature_results$component))
  comps[!is.na(comps) & nzchar(comps)]
}

.select_features_from_inference <- function(
    inf,
    inference_scope = c("component", "union", "global"),
    component = NULL,
    select_by = c("q", "p", "stat"),
    direction = c("absolute", "positive", "negative"),
    n_features = 50
) {
  inference_scope <- match.arg(inference_scope)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  if (is.null(n_features) || length(n_features) != 1L || is.na(n_features) || n_features < 1L) {
    stop("n_features must be a positive integer.", call. = FALSE)
  }
  n_features <- as.integer(n_features)

  if (inference_scope %in% c("union", "global")) {
    ur <- inf$union_results
    needed <- switch(
      paste(inference_scope, select_by, sep = "_"),
      union_q = "union_q",
      union_p = "union_p",
      union_stat = "max_abs_stat",
      global_q = "global_q",
      global_p = "global_p",
      global_stat = "global_stat"
    )
    if (is.null(ur) || nrow(ur) == 0L || !needed %in% names(ur)) {
      message("Inference feature selection skipped: statistic '", needed, "' is not available.")
      return(character())
    }

    if (inference_scope == "union" && select_by == "q") {
      idx <- order(ur$union_q, ur$min_p, na.last = NA)
    } else if (inference_scope == "union" && select_by == "p") {
      idx <- order(ur$union_p, ur$min_p, na.last = NA)
    } else if (inference_scope == "union" && select_by == "stat") {
      idx <- order(ur$max_abs_stat, decreasing = TRUE, na.last = NA)
    } else if (inference_scope == "global" && select_by == "q") {
      idx <- order(ur$global_q, ur$global_p, na.last = NA)
    } else if (inference_scope == "global" && select_by == "p") {
      idx <- order(ur$global_p, ur$global_q, na.last = NA)
    } else {
      idx <- order(abs(ur$global_stat), decreasing = TRUE, na.last = NA)
    }

    selected <- head(as.character(ur$feature[idx]), n_features)
    selected <- selected[!is.na(selected)]
    if (length(selected) == 0L) {
      message("Inference feature selection skipped: no valid features were available for ", inference_scope, "-level statistic '", select_by, "'.")
      return(character())
    }
    return(selected)
  }

  if (is.null(component)) {
    component <- 1L
  }
  component_label <- .resolve_inference_component_label(inf, component)
  fr <- inf$feature_results |>
    dplyr::filter(.data$component == !!component_label)

  if (nrow(fr) == 0L) {
    message("Inference feature selection skipped: no results found for component '", component_label, "'.")
    return(character())
  }

  needed <- switch(select_by, q = "qval", p = "pval", stat = "stat")
  if (!needed %in% names(fr)) {
    message("Inference feature selection skipped: statistic '", needed, "' is not available.")
    return(character())
  }

  if (select_by %in% c("q", "p")) {
    idx <- order(fr[[needed]], na.last = NA)
  } else {
    stat <- fr$stat
    if (direction == "absolute") idx <- order(abs(stat), decreasing = TRUE, na.last = NA)
    else if (direction == "positive") idx <- order(stat, decreasing = TRUE, na.last = NA)
    else idx <- order(stat, decreasing = FALSE, na.last = NA)
  }

  selected <- head(as.character(fr$feature[idx]), n_features)
  selected <- selected[!is.na(selected)]
  if (length(selected) == 0L) {
    message("Inference feature selection skipped: no valid features were available for component '", component_label, "'.")
    return(character())
  }

  selected
}


.get_feature_profile_fdr_status <- function(
    inf,
    features,
    inference_scope = c("component", "union", "global"),
    component = NULL,
    fdr = 0.1,
    component_vector_allowed = TRUE
) {
  inference_scope <- match.arg(inference_scope)
  features <- as.character(features)
  if (length(fdr) != 1L || is.na(fdr) || !is.finite(fdr) || fdr <= 0 || fdr >= 1) {
    stop("fdr must be a single number between 0 and 1.", call. = FALSE)
  }

  if (inference_scope != "component") {
    return(.get_inference_fdr_status(
      inf, features = features, inference_scope = inference_scope,
      component = NULL, fdr = fdr
    ))
  }

  if (is.null(component)) component <- 1L
  if (length(component) == 1L) {
    if (is.na(component)) {
      return(data.frame(
        feature = features, qval = NA_real_, significant = FALSE,
        stringsAsFactors = FALSE
      ))
    }
    return(.get_inference_fdr_status(
      inf, features = features, inference_scope = "component",
      component = component, fdr = fdr
    ))
  }

  if (!isTRUE(component_vector_allowed)) {
    stop("component must have length one when features is not supplied.", call. = FALSE)
  }
  if (length(component) != length(features)) {
    stop("For component-wise feature profiles, component must have length one or length(features).", call. = FALSE)
  }
  if (!is.numeric(component)) {
    stop("A vector-valued component must contain component numbers or NA.", call. = FALSE)
  }
  non_na <- !is.na(component)
  if (any(component[non_na] < 1 | component[non_na] != as.integer(component[non_na]))) {
    stop("Component numbers must be positive integers or NA.", call. = FALSE)
  }

  fr <- inf$feature_results
  if (is.null(fr) || nrow(fr) == 0L || !all(c("feature", "component", "qval") %in% names(fr))) return(NULL)

  q <- rep(NA_real_, length(features))
  for (i in which(non_na)) {
    component_label <- .resolve_inference_component_label(inf, as.integer(component[i]))
    hit <- which(as.character(fr$feature) == features[i] & as.character(fr$component) == component_label)
    if (length(hit) > 0L) q[i] <- fr$qval[hit[1L]]
  }

  data.frame(
    feature = features,
    qval = q,
    significant = !is.na(q) & q <= fdr,
    stringsAsFactors = FALSE
  )
}

.compact_inference_component_label <- function(inf, component) {
  lab <- .resolve_inference_component_label(inf, component)
  lab <- sub("^IR", "", as.character(lab))
  gsub("_", "", lab, fixed = TRUE)
}

.feature_profile_scope_display_label <- function(inf, inference_scope, component = NULL) {
  inference_scope <- match.arg(inference_scope, c("component", "union", "global"))
  if (inference_scope == "union") return("Union")
  if (inference_scope == "global") return("Global")

  if (is.null(component)) component <- 1L
  component <- component[!is.na(component)]
  if (length(component) == 0L) return("Component")

  if (length(component) == 1L) {
    return(.compact_inference_component_label(inf, component))
  }

  if (!is.numeric(component)) {
    stop("A vector-valued component must contain component numbers or NA.", call. = FALSE)
  }
  component <- unique(as.integer(component))
  if (length(component) == 1L) {
    return(.compact_inference_component_label(inf, component))
  }
  "Component per feature"
}

.get_inference_fdr_status <- function(
    inf,
    features,
    inference_scope = c("component", "union", "global"),
    component = NULL,
    fdr = 0.1
) {
  inference_scope <- match.arg(inference_scope)
  if (is.null(inf) || is.null(features) || length(features) == 0L) return(NULL)
  if (length(fdr) != 1L || is.na(fdr) || !is.finite(fdr) || fdr <= 0 || fdr >= 1) {
    stop("fdr must be a single number between 0 and 1.", call. = FALSE)
  }

  features <- as.character(features)
  if (inference_scope %in% c("union", "global")) {
    ur <- inf$union_results
    qcol <- if (inference_scope == "union") "union_q" else "global_q"
    if (is.null(ur) || nrow(ur) == 0L || !all(c("feature", qcol) %in% names(ur))) return(NULL)
    idx <- match(features, as.character(ur$feature))
    q <- ur[[qcol]][idx]
  } else {
    if (is.null(component)) component <- 1L
    component_label <- .resolve_inference_component_label(inf, component)
    fr <- inf$feature_results
    if (is.null(fr) || nrow(fr) == 0L || !all(c("feature", "component", "qval") %in% names(fr))) return(NULL)
    fr <- fr[as.character(fr$component) == component_label, , drop = FALSE]
    idx <- match(features, as.character(fr$feature))
    q <- fr$qval[idx]
  }

  data.frame(
    feature = features,
    qval = q,
    significant = !is.na(q) & q <= fdr,
    stringsAsFactors = FALSE
  )
}

.inference_scope_display_label <- function(inf, inference_scope, component = NULL) {
  inference_scope <- match.arg(inference_scope, c("component", "union", "global"))
  if (inference_scope == "union") return("Union")
  if (inference_scope == "global") return("Global")

  if (is.null(component)) component <- 1L
  .compact_inference_component_label(inf, component)
}

.resolve_named_list_dataset <- function(x, dataset, what = "object") {
  if (is.null(dataset)) {
    stop("dataset must be supplied for integrated ", what, ".", call. = FALSE)
  }
  d <- if (is.numeric(dataset)) dataset else match(dataset, names(x))
  if (is.na(d) || d < 1L || d > length(x)) {
    stop(
      "Unknown dataset: ", dataset, ". Available datasets are: ",
      paste(names(x), collapse = ", "),
      call. = FALSE
    )
  }
  x[[d]]
}

.resolve_isvdx_inference_dataset <- function(object, dataset) {
  .resolve_named_list_dataset(object$dataset_results, dataset, what = "isvdxInference objects")
}

.resolve_isvdx_workflow_dataset_inference <- function(object, dataset) {
  .resolve_named_list_dataset(object$inf$dataset_results, dataset, what = "isvdxWorkflow inference plots")
}

.resolve_dataset_matrix <- function(Y, dataset) {
  if (is.list(Y) && !is.data.frame(Y)) {
    return(.resolve_named_list_dataset(Y, dataset, what = "dataset-specific Y matrices"))
  }
  Y
}

.scale_rows <- function(M, eps = 1e-8) {
  mu <- rowMeans(M, na.rm = TRUE)
  sd <- apply(M, 1, stats::sd, na.rm = TRUE)
  sd[sd < eps | !is.finite(sd)] <- 1
  sweep(sweep(M, 1, mu, "-"), 1, sd, "/")
}


.merge_plot_metadata <- function(data, zdata = NULL) {
  if (is.null(zdata)) return(data)
  if (!is.data.frame(data)) data <- as.data.frame(data)
  if (!is.data.frame(zdata)) zdata <- as.data.frame(zdata)
  if (nrow(data) != nrow(zdata)) {
    stop("data and zdata must have the same number of rows for heatmap metadata.", call. = FALSE)
  }
  if (!is.null(rownames(data)) && !is.null(rownames(zdata)) && !identical(rownames(data), rownames(zdata))) {
    if (setequal(rownames(data), rownames(zdata))) {
      zdata <- zdata[rownames(data), , drop = FALSE]
    } else {
      stop("rownames(data) and rownames(zdata) must match for heatmap metadata.", call. = FALSE)
    }
  }
  z_keep <- setdiff(names(zdata), names(data))
  if (!length(z_keep)) return(data)
  cbind(data, zdata[, z_keep, drop = FALSE])
}

.prepare_heatmap_matrix <- function(
    Y,
    data,
    features,
    cols = NULL,
    order_by = NULL,
    scale_rows = TRUE
) {
  cols_idx <- .resolve_cols(Y, cols)
  Y_use <- Y[features, cols_idx, drop = FALSE]
  data_use <- data[cols_idx, , drop = FALSE]
  
  ord <- .order_metadata_rows(data_use, order_by = order_by)
  Y_use <- Y_use[, ord, drop = FALSE]
  data_use <- data_use[ord, , drop = FALSE]
  
  if (scale_rows) Y_use <- .scale_rows(Y_use)
  
  list(Y = Y_use, data = data_use)
}




# ------------------------------------------------------------
# V plots
# ------------------------------------------------------------

#' Plot the largest loadings for a component
#'
#' Displays the largest positive, negative, or absolute loadings from a
#' selected component.
#'
#' @param fit A fitted SSANOVASVD or iSSANOVASVD object.
#' @param component Component to visualize.
#' @param n Number of features to display.
#' @param direction Loading-selection direction.
#' @param dataset Dataset to visualize for integrated models.
#'   May be either a dataset name (e.g. `"rna"`) or dataset index.
#'   Ignored for single-dataset SSANOVASVD fits.
#'
#' @return A ggplot object.
#'
#' @export
plotVLoadings <- function(
    fit,
    component = 1,
    n = 20,
    direction = c("both", "absolute", "positive", "negative"),
    dataset = NULL,
    row_labels = NULL,
    fdata = NULL
) {
  direction <- match.arg(direction)
  fdata_use <- .svdx_resolve_fdata_for_dataset(fdata, dataset = dataset, required = FALSE)
  row_labels <- .resolve_row_labels_from_fdata(row_labels = row_labels, fdata = fdata_use, dataset = dataset)

  vmat <- .get_vmat(fit, dataset = dataset)
  colnames(vmat) <- colnames(vmat) %||% .make_component_names(ncol(vmat))

  feature <- rownames(vmat) %||% paste0("feature_", seq_len(nrow(vmat)))

  k <- .resolve_component_index(component, colnames(vmat))
  comp <- colnames(vmat)[k]
  v <- vmat[, k]

  if (direction == "both") {
    n_each <- ceiling(n / 2)
    idx <- unique(c(order(v, decreasing = TRUE)[seq_len(min(n_each, length(v)))],
                    order(v, decreasing = FALSE)[seq_len(min(n_each, length(v)))]))
  } else if (direction == "absolute") {
    idx <- order(abs(v), decreasing = TRUE)[seq_len(min(n, length(v)))]
  } else if (direction == "positive") {
    idx <- order(v, decreasing = TRUE)[seq_len(min(n, length(v)))]
  } else {
    idx <- order(v, decreasing = FALSE)[seq_len(min(n, length(v)))]
  }

  feature_id <- feature[idx]
  feature_label <- .resolve_heatmap_row_labels(row_labels, feature_id)
  feature_key <- paste0("feature_", seq_along(feature_id))
  label_map <- stats::setNames(feature_label, feature_key)

  df <- tibble::tibble(
    feature = feature_id,
    feature_label = feature_label,
    feature_key = feature_key,
    loading = v[idx]
  )
  df$feature_key <- factor(df$feature_key, levels = df$feature_key[order(df$loading)])

  ggplot2::ggplot(df, ggplot2::aes(x = feature_key, y = loading)) +
    ggplot2::geom_col() +
    ggplot2::scale_x_discrete(labels = label_map) +
    ggplot2::coord_flip() +
    ggplot2::theme_bw() +
    ggplot2::labs(x = NULL, y = "Loading", title = paste("Top loadings:", comp))
}

#' Plot a feature heatmap for a component
#'
#' Displays a heatmap of features selected from a component loading vector or
#' inference result.
#'
#' @param object A fitted SSANOVASVD/iSSANOVASVD object or inference object.
#' @param Y Feature-by-sample matrix.
#' @param data Sample metadata.
#' @param component Component to visualize.
#' @param dataset Dataset to visualize for integrated models.
#'   May be either a dataset name (e.g. `"rna"`) or dataset index.
#'   Ignored for single-dataset SSANOVASVD fits.
#' @param n_features Number of features to display.
#' @param order_by Metadata variables used to order samples.
#' @param annotate Metadata variables displayed as annotations. If `NULL`,
#'   `order_by` is also used for annotation, preserving the default behavior.
#'   Set to `FALSE` to suppress annotations explicitly.
#' @param row_labels Optional row display labels. May be a named vector mapping
#'   feature IDs to labels, an unnamed vector with one label per displayed
#'   feature, or a two-column data frame whose first column contains feature
#'   IDs and second column contains labels. Explicit `row_labels` override labels
#'   derived from `fdata`. This only changes heatmap row display names and does
#'   not modify the input object or matrix.
#' @param fdata Optional `svdx_fdata` object, or named list of such objects for
#'   integrated objects. Used to derive row labels when `row_labels` is omitted.
#'
#' @return A ComplexHeatmap object.
#'
#' @details
#' Features may be selected using loading magnitudes or inference statistics.
#' For inference-driven selection, `inference_scope` chooses the component,
#' union, or global result table and `select_by` chooses the generic statistic
#' (`"q"`, `"p"`, or `"stat"`) within that scope.
#'
#' @param fdr FDR threshold for the automatic inference significance row annotation. Defaults to 0.10.
#' @param show_significance Logical; if `TRUE`, show the scope-specific FDR row
#'   annotation when inference results are available. Defaults to `TRUE`.
#' @export
plotVHeatmap <- function(
    object,
    Y,
    data,
    zdata = NULL,
    dataset=NULL,
    inference_scope = c("component", "union", "global"),
    component = 1,
    select_by = c("loading", "q", "p", "stat"),
    direction = c("absolute", "positive", "negative", "both"),
    n_features = 50,
    features = NULL,
    cols = NULL,
    order_by = NULL,
    annotate = NULL,
    scale_rows = TRUE,
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    show_feature_names = TRUE,
    show_sample_names = FALSE,
    row_labels = NULL,
    fdata = NULL,
    max_cols = 500,
    fdr = 0.1,
    show_significance = TRUE
) {
  .require_namespace("ComplexHeatmap")
  
  inference_scope <- match.arg(inference_scope)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  inference_annotation_requested <- !is.null(features) ||
    !identical(select_by, "loading") ||
    inherits(object, c("svdxInference", "isvdxInference"))

  inf_for_fdr <- NULL
  if (inherits(object, "svdxInference")) {
    inf_for_fdr <- object
  } else if (inherits(object, "isvdxInference")) {
    inf_for_fdr <- .resolve_isvdx_inference_dataset(object, dataset)
  } else if (inherits(object, "svdxWorkflow")) {
    inf_for_fdr <- object$inf
  } else if (inherits(object, "isvdxWorkflow") && !is.null(object$inf)) {
    inf_for_fdr <- .resolve_isvdx_workflow_dataset_inference(object, dataset)
  }

  fdata_use <- .resolve_workflow_fdata(object, fdata = fdata, dataset = dataset)
  row_labels <- .resolve_row_labels_from_fdata(row_labels = row_labels, fdata = fdata_use, dataset = dataset)

  if (inherits(object, "isvdxInference")) {
    object <- .resolve_isvdx_inference_dataset(object, dataset)
    Y <- .resolve_dataset_matrix(Y, dataset)
  } else if (inherits(object, "isvdxWorkflow")) {
    if (identical(select_by, "loading")) {
      Y <- .resolve_dataset_matrix(Y, dataset)
    } else {
      if (is.null(object$inf) || is.null(object$inf$dataset_results)) {
        message("V heatmap skipped: inference results unavailable.")
        return(NULL)
      }
      object <- .resolve_isvdx_workflow_dataset_inference(object, dataset)
      Y <- .resolve_dataset_matrix(Y, dataset)
    }
  }

  Y <- as.matrix(Y)
  data <- .merge_plot_metadata(data, zdata = zdata)
  
  cols_idx <- .resolve_cols(Y, cols)
  if (length(cols_idx) > max_cols) {
    warning("Heatmap contains ", length(cols_idx), " samples. Consider using cols= or set max_cols=Inf.", call. = FALSE)
  }
  
  feature_names <- rownames(Y) %||% paste0("feature_", seq_len(nrow(Y)))
  
  if (is.null(features)) {
    if (inherits(object, c("SSANOVASVD", "iSSANOVASVD"))) {
      idx <- .select_features_from_loadings(
        object,
        component = component %||% 1,
        direction = direction,
        n_features = n_features,
        dataset = dataset
      )
      features <- feature_names[idx]
    } else if (inherits(object, "svdxInference")) {
      if (identical(select_by, "loading")) {
        inference_scope <- "union"
        select_by <- "q"
      }
      features <- .select_features_from_inference(
        object,
        inference_scope = inference_scope,
        component = component,
        select_by = select_by,
        direction = direction,
        n_features = n_features
      )
    } else if (inherits(object, "svdxWorkflow")) {
      if (select_by == "loading") {
        idx <- .select_features_from_loadings(object$fit, component = component %||% 1, direction = direction, n_features = n_features)
        features <- feature_names[idx]
      } else {
        if (is.null(object$inf)) {
          message("V heatmap skipped: inference results unavailable.")
          return(NULL)
        }
        features <- .select_features_from_inference(
          object$inf,
          inference_scope = inference_scope,
          component = component,
          select_by = select_by,
          direction = direction,
          n_features = n_features
        )
      }
    } else if (inherits(object, "isvdxWorkflow")) {
      if (select_by == "loading") {
        idx <- .select_features_from_loadings(
          object$fit,
          component = component %||% 1,
          direction = direction,
          n_features = n_features,
          dataset = dataset
        )
        features <- feature_names[idx]
      } else {
        features <- .select_features_from_inference(
          object,
          inference_scope = inference_scope,
          component = component,
          select_by = select_by,
          direction = direction,
          n_features = n_features
        )
      }
    } else {
      stop("object must be SSANOVASVD, iSSANOVASVD, svdxInference, isvdxInference, svdxWorkflow, or isvdxWorkflow.", call. = FALSE)
    }
  }
  
  features <- features[!is.na(features)]
  if (length(features) == 0L) {
    message("V heatmap skipped: no valid features were selected.")
    return(NULL)
  }
  fidx <- match(features, feature_names)
  if (any(is.na(fidx))) {
    message("V heatmap skipped: selected features were not found in rownames(Y).")
    return(NULL)
  }
  
  prep <- .prepare_heatmap_matrix(
    Y = Y,
    data = data,
    features = fidx,
    cols = cols_idx,
    order_by = order_by,
    scale_rows = scale_rows
  )
  
  mat <- prep$Y
  rownames(mat) <- .resolve_heatmap_row_labels(row_labels, features)
  
  suppress_annotation <- identical(annotate, FALSE)
  if (suppress_annotation) {
    annotate <- NULL
  } else if (is.null(annotate) && !is.null(order_by)) {
    annotate <- order_by
  }

  ha <- NULL
  if (!is.null(annotate)) {
    missing <- setdiff(annotate, colnames(prep$data))
    if (length(missing) > 0) stop("annotate variable(s) not found in data: ", paste(missing, collapse = ", "), call. = FALSE)
    ann_df <- prep$data[, annotate, drop = FALSE]
    ann_ref_df <- data[, annotate, drop = FALSE]
    ha <- ComplexHeatmap::HeatmapAnnotation(
      df = ann_df,
      col = .resolve_annotation_colors(ann_df, reference_df = ann_ref_df)
    )
  }
  
  fdr_ha <- NULL
  if (isTRUE(show_significance) && isTRUE(inference_annotation_requested) && !is.null(inf_for_fdr)) {
    fdr_status <- .get_inference_fdr_status(
      inf_for_fdr,
      features = features,
      inference_scope = inference_scope,
      component = component,
      fdr = fdr
    )
    if (!is.null(fdr_status)) {
      sig <- fdr_status$significant[match(features, fdr_status$feature)]
      sig[is.na(sig)] <- FALSE
      sig <- factor(sig, levels = c(FALSE, TRUE), labels = c("FALSE", "TRUE"))
      ann_df <- data.frame(FDR = sig, check.names = FALSE)
      ann_cols <- list(FDR = c("FALSE" = "#E6E6E6", "TRUE" = "#4C78A8"))
      fdr_ha <- ComplexHeatmap::rowAnnotation(
        df = ann_df,
        col = ann_cols,
        annotation_name_side = "top",
        annotation_name_rot = 0,
        annotation_name_gp = grid::gpar(fontsize = 8),
        annotation_legend_param = list(
          FDR = list(
            title = sprintf(
              "FDR <= %.2f | %s",
              fdr,
              .inference_scope_display_label(inf_for_fdr, inference_scope, component)
            )
          )
        )
      )
    }
  }

  ComplexHeatmap::Heatmap(
    mat,
    name = ifelse(scale_rows, "z", "Y"),
    top_annotation = ha,
    left_annotation = fdr_ha,
    cluster_rows = cluster_rows,
    cluster_columns = cluster_cols,
    show_row_names = show_feature_names,
    show_column_names = show_sample_names
  )
}

#' Plot feature x component inference heatmap
#' @param fdr FDR threshold for the automatic scope-specific significance row annotation. Defaults to 0.10.
#' @param show_significance Logical; if `TRUE`, show the scope-specific FDR row
#'   annotation. Defaults to `TRUE`.
#' @export
plotVInferenceHeatmap <- function(
    inf,
    features = NULL,
    n_features = 50,
    value = c("stat", "coef", "neg_log10_q", "signed_neg_log10_q"),
    inference_scope = c("union", "global", "component"),
    component = NULL,
    select_by = c("q", "p", "stat"),
    direction = c("absolute", "positive", "negative"),
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    show_feature_names = TRUE,
    fdata = NULL,
    row_labels = NULL,
    use_feature_labels = TRUE,
    dataset = NULL,
    fdr = 0.1,
    show_significance = TRUE
) {
  .require_namespace("ComplexHeatmap")
  value <- match.arg(value)
  inference_scope <- match.arg(inference_scope)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  if (inference_scope == "component" && is.null(component)) {
    comps <- .inference_components(inf)
    out <- lapply(comps, function(comp) {
      plotVInferenceHeatmap(
        inf,
        features = features,
        n_features = n_features,
        value = value,
        inference_scope = "component",
        component = comp,
        select_by = select_by,
        direction = direction,
        cluster_rows = cluster_rows,
        cluster_cols = cluster_cols,
        show_feature_names = show_feature_names,
        fdata = fdata,
        row_labels = row_labels,
        use_feature_labels = use_feature_labels,
        dataset = dataset,
        fdr = fdr,
        show_significance = show_significance
      )
    })
    names(out) <- comps
    return(out)
  }

  if (!isTRUE(use_feature_labels)) {
    row_labels <- NULL
    fdata <- NULL
  }
  row_labels <- .resolve_row_labels_from_fdata(row_labels = row_labels, fdata = fdata, dataset = dataset)

  if (is.null(features)) {
    features <- .select_features_from_inference(
      inf,
      inference_scope = inference_scope,
      component = component,
      select_by = select_by,
      direction = direction,
      n_features = n_features
    )
  }

  features <- features[!is.na(features)]
  if (length(features) == 0L) {
    message("Inference heatmap skipped: no valid features were selected.")
    return(NULL)
  }

  fr <- inf$feature_results |>
    dplyr::filter(.data$feature %in% features)

  if (inference_scope == "component") {
    component_label <- .resolve_inference_component_label(inf, component)
    fr <- fr |>
      dplyr::filter(.data$component == !!component_label)
  }

  if (nrow(fr) == 0L) {
    message("Inference heatmap skipped: selected features have no component-level inference results.")
    return(NULL)
  }
  fr$value <- switch(
    value,
    stat = fr$stat,
    coef = fr$coef,
    neg_log10_q = -log10(pmax(fr$qval, .Machine$double.xmin)),
    signed_neg_log10_q = sign(fr$stat) * -log10(pmax(fr$qval, .Machine$double.xmin))
  )

  mat <- fr |>
    dplyr::select(feature, component, value) |>
    tidyr::pivot_wider(names_from = component, values_from = value) |>
    as.data.frame()

  feature_ids <- as.character(mat$feature)
  rownames(mat) <- .resolve_heatmap_row_labels(row_labels, feature_ids)
  mat$feature <- NULL
  mat <- as.matrix(mat)

  fdr_ha <- NULL
  if (isTRUE(show_significance)) {
    fdr_status <- .get_inference_fdr_status(
      inf,
      features = feature_ids,
      inference_scope = inference_scope,
      component = component,
      fdr = fdr
    )
    if (!is.null(fdr_status)) {
      sig <- fdr_status$significant[match(feature_ids, fdr_status$feature)]
      sig[is.na(sig)] <- FALSE
      sig <- factor(sig, levels = c(FALSE, TRUE), labels = c("FALSE", "TRUE"))
      ann_df <- data.frame(FDR = sig, check.names = FALSE)
      ann_cols <- list(FDR = c("FALSE" = "#E6E6E6", "TRUE" = "#4C78A8"))
      fdr_ha <- ComplexHeatmap::rowAnnotation(
        df = ann_df,
        col = ann_cols,
        annotation_name_side = "top",
        annotation_name_rot = 0,
        annotation_name_gp = grid::gpar(fontsize = 8),
        annotation_legend_param = list(
          FDR = list(
            title = sprintf(
              "FDR <= %.2f | %s",
              fdr,
              .inference_scope_display_label(inf, inference_scope, component)
            )
          )
        )
      )
    }
  }

  ComplexHeatmap::Heatmap(
    mat,
    name = value,
    left_annotation = fdr_ha,
    cluster_rows = cluster_rows,
    cluster_columns = cluster_cols,
    show_row_names = show_feature_names
  )
}

#' Plot feature x component inference dotplot
#' @param fdr FDR threshold used to distinguish significant component-level dots. Defaults to 0.10.
#' @export
plotVInferenceDotplot <- function(
    inf,
    features = NULL,
    n_features = 30,
    inference_scope = c("union", "global", "component"),
    component = NULL,
    select_by = c("q", "p", "stat"),
    direction = c("absolute", "positive", "negative"),
    fdata = NULL,
    row_labels = NULL,
    use_feature_labels = TRUE,
    dataset = NULL,
    fdr = 0.1
) {
  inference_scope <- match.arg(inference_scope)
  select_by <- match.arg(select_by)
  direction <- match.arg(direction)

  if (inference_scope == "component" && is.null(component)) {
    comps <- .inference_components(inf)
    out <- lapply(comps, function(comp) {
      plotVInferenceDotplot(
        inf,
        features = features,
        n_features = n_features,
        inference_scope = "component",
        component = comp,
        select_by = select_by,
        direction = direction,
        fdata = fdata,
        row_labels = row_labels,
        use_feature_labels = use_feature_labels,
        dataset = dataset,
        fdr = fdr
      )
    })
    names(out) <- comps
    return(out)
  }

  if (!isTRUE(use_feature_labels)) {
    row_labels <- NULL
    fdata <- NULL
  }
  row_labels <- .resolve_row_labels_from_fdata(row_labels = row_labels, fdata = fdata, dataset = dataset)

  if (is.null(features)) {
    features <- .select_features_from_inference(
      inf,
      inference_scope = inference_scope,
      component = component,
      select_by = select_by,
      direction = direction,
      n_features = n_features
    )
  }

  features <- features[!is.na(features)]
  if (length(features) == 0L) {
    message("Inference dotplot skipped: no valid features were selected.")
    return(NULL)
  }

  feature_labels <- .resolve_heatmap_row_labels(row_labels, features)
  feature_labels <- make.unique(as.character(feature_labels))
  feature_label_map <- stats::setNames(feature_labels, features)
  feature_label_levels <- rev(unname(feature_label_map[features]))

  df <- inf$feature_results |>
    dplyr::filter(.data$feature %in% features) |>
    dplyr::mutate(
      neg_log10_q = -log10(pmax(qval, .Machine$double.xmin)),
      significant = !is.na(qval) & qval <= fdr,
      feature_id = as.character(feature),
      feature_label = unname(feature_label_map[feature_id]),
      feature_label = ifelse(is.na(feature_label) | !nzchar(feature_label), feature_id, feature_label),
      feature_label = factor(feature_label, levels = feature_label_levels)
    )

  if (inference_scope == "component") {
    component_label <- .resolve_inference_component_label(inf, component)
    df <- df |>
      dplyr::filter(.data$component == !!component_label)
  }

  if (nrow(df) == 0L) {
    message("Inference dotplot skipped: selected features have no component-level inference results.")
    return(NULL)
  }

  ggplot2::ggplot(df, ggplot2::aes(x = component, y = feature_label, color = stat, size = neg_log10_q, shape = significant)) +
    ggplot2::geom_point(alpha = 0.8) +
    ggplot2::scale_shape_manual(
      values = c(`TRUE` = 16, `FALSE` = 1),
      breaks = c(TRUE, FALSE),
      labels = c(sprintf("FDR <= %.2f", fdr), sprintf("FDR > %.2f", fdr))
    ) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "Component", y = NULL, color = "Statistic", size = "-log10(q)", shape = NULL, title = "Feature-component inference")
}

#' Plot training loadings versus test inference statistic/coefficient
#' @export
plotVLoadingInferenceScatter <- function(
    fit,
    inf,
    dataset=NULL, 
    component = 1,
    y = c("stat", "coef", "neg_log10_q"),
    label_top = 0
) {
  y <- match.arg(y)
  
  vmat <- .get_vmat(fit, dataset=dataset)
  colnames(vmat) <- colnames(vmat) %||% .make_component_names(ncol(vmat))
  comp_names <- colnames(vmat)
  k <- .resolve_component_index(component, comp_names)
  comp <- comp_names[k]
  feature <- rownames(vmat) %||% paste0("feature_", seq_len(nrow(vmat)))
  
  fr <- inf$feature_results |>
    dplyr::filter(component == !!comp)
  
  df <- tibble::tibble(feature = feature, loading = vmat[, k]) |>
    dplyr::inner_join(fr, by = "feature") |>
    dplyr::mutate(
      yval = dplyr::case_when(
        y == "stat" ~ stat,
        y == "coef" ~ coef,
        y == "neg_log10_q" ~ -log10(pmax(qval, .Machine$double.xmin))
      )
    )
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = loading, y = yval)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "Training V loading", y = y, title = paste("Training loading vs test inference:", comp))
  
  if (label_top > 0 && requireNamespace("ggrepel", quietly = TRUE)) {
    lab <- df |>
      dplyr::arrange(dplyr::desc(abs(yval))) |>
      dplyr::slice_head(n = label_top)
    p <- p + ggrepel::geom_text_repel(data = lab, ggplot2::aes(label = feature), size = 3)
  }
  
  p
}

