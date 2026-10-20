# ============================================================
# Workflow report generators
# ============================================================

.report_message <- function(..., verbose = TRUE) {
  if (!isTRUE(verbose)) return(invisible(NULL))
  
  cat(
    format(Sys.time(), "%H:%M:%S"),
    " | ",
    paste0(..., collapse = ""),
    "\n"
  )
  
  invisible(NULL)
}

.report_dir_create <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  invisible(path)
}

.report_prepare_outdir <- function(outdir, overwrite = TRUE) {
  if (!is.character(outdir) || length(outdir) != 1L || is.na(outdir) || !nzchar(outdir)) {
    stop("`outdir` must be a single non-empty character string.", call. = FALSE)
  }

  outdir <- path.expand(outdir)

  if (dir.exists(outdir)) {
    if (isTRUE(overwrite)) {
      unlink(outdir, recursive = TRUE, force = TRUE)
      if (dir.exists(outdir)) {
        stop("Failed to remove existing output directory: ", outdir, call. = FALSE)
      }
    } else {
      stop(
        "Output directory already exists: ", outdir, ". ",
        "Use overwrite = TRUE or choose another outdir.",
        call. = FALSE
      )
    }
  }

  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  if (!dir.exists(outdir)) {
    stop("Failed to create output directory: ", outdir, call. = FALSE)
  }

  test_file <- tempfile(tmpdir = outdir)
  can_write <- tryCatch({
    writeLines("svdx report write test", test_file)
    unlink(test_file, force = TRUE)
    TRUE
  }, error = function(e) FALSE)

  if (!isTRUE(can_write)) {
    stop("Output directory is not writable: ", outdir, call. = FALSE)
  }

  normalizePath(outdir, mustWork = TRUE)
}

.report_validate_outdir_preflight <- function(outdir, overwrite = TRUE) {
  if (!is.character(outdir) || length(outdir) != 1L || is.na(outdir) || !nzchar(outdir)) {
    stop("`outdir` must be a single non-empty character string.", call. = FALSE)
  }

  outdir <- path.expand(outdir)

  if (dir.exists(outdir) && !isTRUE(overwrite)) {
    stop(
      "Output directory already exists: ", outdir, ". ",
      "Use overwrite = TRUE or choose another outdir.",
      call. = FALSE
    )
  }

  invisible(outdir)
}

.report_save_rds <- function(x, file) {
  saveRDS(x, file = file)
  invisible(file)
}


.report_save_tsv <- function(x, file) {
  .report_dir_create(dirname(file))
  utils::write.table(
    x,
    file = file,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = TRUE,
    na = "NA"
  )
  invisible(file)
}

.report_feature_annotation <- function(features, fdata = NULL) {
  features <- as.character(features)
  out <- data.frame(feature_id = features, stringsAsFactors = FALSE)

  if (is.null(fdata)) {
    out$feature_label <- NA_character_
    return(out)
  }

  fd_feature <- .svdx_fdata_feature(fdata, required = TRUE)
  fd_label <- .svdx_fdata_label(fdata, required = FALSE)
  idx <- match(features, fd_feature)
  out$feature_label <- if (is.null(fd_label)) NA_character_ else as.character(fd_label[idx])
  out
}

.report_annotate_inference_table <- function(tbl, fdata = NULL, alpha = 0.1) {
  if (is.null(tbl)) return(NULL)
  tbl <- as.data.frame(tbl, stringsAsFactors = FALSE)
  if (!"feature" %in% names(tbl)) return(tbl)

  ann <- .report_feature_annotation(tbl$feature, fdata = fdata)
  tbl$feature_id <- ann$feature_id
  tbl$feature_label <- ann$feature_label
  tbl$feature <- NULL
  tbl <- tbl[, c("feature_id", "feature_label", setdiff(names(tbl), c("feature_id", "feature_label"))), drop = FALSE]

  if ("qval" %in% names(tbl)) {
    tbl$significant_fdr <- !is.na(tbl$qval) & tbl$qval <= alpha
    tbl$fdr_threshold <- alpha
  } else if ("global_q" %in% names(tbl)) {
    tbl$significant_fdr <- !is.na(tbl$global_q) & tbl$global_q <= alpha
    tbl$fdr_threshold <- alpha
  }
  tbl
}

.report_enrichment_overlap_annotations <- function(x, fdata = NULL) {
  x <- as.character(x)
  if (is.null(fdata) || !length(x)) {
    return(list(feature_ids = rep(NA_character_, length(x)), feature_labels = rep(NA_character_, length(x))))
  }

  fd_feature <- .svdx_fdata_feature(fdata, required = TRUE)
  fd_label <- .svdx_fdata_label(fdata, required = FALSE)
  fd_enrichment <- .svdx_fdata_enrichment(fdata, required = FALSE)
  if (is.null(fd_enrichment)) fd_enrichment <- fd_feature

  convert_one <- function(z, values) {
    if (is.na(z) || !nzchar(z)) return(NA_character_)
    ids <- unlist(strsplit(z, "[;/]", perl = TRUE), use.names = FALSE)
    ids <- trimws(ids)
    ids <- ids[nzchar(ids)]
    if (!length(ids)) return(NA_character_)
    idx <- match(ids, fd_enrichment)
    vals <- values[idx]
    vals <- vals[!is.na(vals) & nzchar(vals)]
    if (!length(vals)) return(NA_character_)
    paste(unique(vals), collapse = ";")
  }

  list(
    feature_ids = vapply(x, convert_one, character(1), values = fd_feature),
    feature_labels = if (is.null(fd_label)) rep(NA_character_, length(x)) else vapply(x, convert_one, character(1), values = fd_label)
  )
}

.report_flatten_enrichment <- function(enr, fdata = NULL) {
  if (is.null(enr)) return(NULL)
  tbl <- .svdx_enrichment_results(enr)
  if (is.null(tbl)) return(NULL)
  tbl <- as.data.frame(tbl, stringsAsFactors = FALSE)

  # Ensure list columns are representable in a flat text file.
  for (nm in names(tbl)) {
    if (is.list(tbl[[nm]])) tbl[[nm]] <- .svdx_collapse_features(tbl[[nm]])
  }

  if ("overlap_features" %in% names(tbl)) {
    ann <- .report_enrichment_overlap_annotations(tbl$overlap_features, fdata = fdata)
    tbl$overlap_feature_ids <- ann$feature_ids
    tbl$overlap_feature_labels <- ann$feature_labels
  }
  tbl
}

.report_save_sample_split_tsv <- function(wf, file) {
  train <- if (!is.null(wf$data_train)) rownames(wf$data_train) else NULL
  test <- if (!is.null(wf$data_test)) rownames(wf$data_test) else NULL
  if (is.null(train)) train <- character(0)
  if (is.null(test)) test <- character(0)
  tbl <- data.frame(
    sample_id = c(as.character(train), as.character(test)),
    split = c(rep("train", length(train)), rep("test", length(test))),
    stringsAsFactors = FALSE
  )
  .report_save_tsv(tbl, file)
}

.report_export_inference_tables <- function(inf, fdata, tables_dir, components, alpha = 0.1) {
  if (is.null(inf) || is.null(inf$feature_results) || !nrow(inf$feature_results)) return(character(0))
  .report_dir_create(tables_dir)
  saved <- character(0)

  component_tables <- list()
  for (component in components) {
    comp_name <- tryCatch(.report_get_inference_component_name(inf, component), error = function(e) NULL)
    if (is.null(comp_name)) next
    tbl <- inf$feature_results[inf$feature_results$component == comp_name, , drop = FALSE]
    tbl <- .report_annotate_inference_table(tbl, fdata = fdata, alpha = alpha)
    if (!nrow(tbl)) next
    component_tables[[as.character(comp_name)]] <- tbl
    f <- file.path(tables_dir, paste0("inference_", .report_component_label(component), ".tsv"))
    .report_save_tsv(tbl, f)
    saved <- c(saved, f)
  }

  if (length(component_tables)) {
    all_tbl <- do.call(rbind, component_tables)
    rownames(all_tbl) <- NULL
    f <- file.path(tables_dir, "inference_allComponents.tsv")
    .report_save_tsv(all_tbl, f)
    saved <- c(saved, f)
  }

  if (!is.null(inf$union_results) && nrow(inf$union_results) && all(c("global_p", "global_q") %in% names(inf$union_results))) {
    keep <- intersect(c("feature", "global_stat", "global_p", "global_q"), names(inf$union_results))
    tbl <- inf$union_results[, keep, drop = FALSE]
    tbl <- .report_annotate_inference_table(tbl, fdata = fdata, alpha = alpha)
    f <- file.path(tables_dir, "inference_global.tsv")
    .report_save_tsv(tbl, f)
    saved <- c(saved, f)
  }

  saved
}

.report_save_plot_safe <- function(expr, file, width = 10, height = 7, dpi = 300, verbose = TRUE) {
  p <- tryCatch(expr, error = function(e) e)
  
  if (inherits(p, "error")) {
    .report_message("Skipping plot: ", file, verbose = verbose)
    .report_message("Reason: ", p$message, verbose = verbose)
    return(NULL)
  }

  if (is.null(p)) {
    .report_message("Skipping plot: ", file, verbose = verbose)
    .report_message("Reason: plot function returned no drawable object.", verbose = verbose)
    return(NULL)
  }

  if (is.list(p) && !inherits(p, "ggplot")) {
    nms <- names(p)
    if (is.null(nms)) nms <- rep("", length(p))

    ext <- tools::file_ext(file)
    stem <- if (nzchar(ext)) tools::file_path_sans_ext(file) else file
    dot_ext <- if (nzchar(ext)) paste0(".", ext) else ""

    saved <- list()
    for (ii in seq_along(p)) {
      suffix <- nms[[ii]]
      if (is.null(suffix) || !nzchar(suffix)) suffix <- paste0("plot_", ii)
      suffix <- gsub("[^A-Za-z0-9._-]+", "_", suffix)
      suffix <- gsub("^_+|_+$", "", suffix)
      if (!nzchar(suffix)) suffix <- paste0("plot_", ii)

      file_i <- paste0(stem, "_", suffix, dot_ext)
      saved[[file_i]] <- .report_save_plot_safe(
        p[[ii]], file = file_i, width = width, height = height, dpi = dpi, verbose = verbose
      )
    }

    saved <- Filter(Negate(is.null), saved)
    if (!length(saved)) return(NULL)
    return(saved)
  }
  
  if (inherits(p, "Heatmap") || inherits(p, "HeatmapList")) {
    grDevices::pdf(file, width = width, height = height)
    on.exit(grDevices::dev.off(), add = TRUE)
    ComplexHeatmap::draw(p)
    return(file)
  }
  
  ggplot2::ggsave(
    filename = file,
    plot = p,
    width = width,
    height = height,
    dpi = dpi,
    device = "pdf"
  )
  
  file
}

.report_get_n_components <- function(wf) {
  U <- tryCatch(
    predict(wf$fit, wf$data),
    error = function(e) NULL
  )
  
  if (!is.null(U)) {
    return(ncol(as.matrix(U)))
  }
  
  v <- wf$fit$fitAtOptCV$vmat %||% wf$fit$vmat
  
  if (is.list(v)) {
    return(ncol(v[[1]]))
  }
  
  ncol(v)
}

.report_resolve_components <- function(wf, components = NULL) {
  n_components <- .report_get_n_components(wf)
  
  if (is.null(components)) {
    components <- seq_len(n_components)
  }
  
  components <- as.integer(components)
  components <- components[is.finite(components)]
  components <- components[components >= 1 & components <= n_components]
  
  if (length(components) == 0) {
    stop("No valid components available to plot.", call. = FALSE)
  }
  
  list(
    components = components,
    n_components = n_components
  )
}

.report_get_inference_component_name <- function(inf, component) {
  comps <- unique(inf$feature_results$component)
  
  if (length(comps) == 0) {
    stop("Could not find components in inf$feature_results$component.", call. = FALSE)
  }
  
  if (is.character(component)) {
    if (!component %in% comps) {
      stop(
        "Unknown inference component: ", component,
        ". Available components are: ",
        paste(comps, collapse = ", "),
        call. = FALSE
      )
    }
    return(component)
  }
  
  component <- as.integer(component)
  
  if (!is.finite(component) || component < 1 || component > length(comps)) {
    stop(
      "Invalid inference component index: ", component,
      ". Available components are 1 through ", length(comps),
      call. = FALSE
    )
  }
  
  comps[component]
}


.report_inference_component_features <- function(inf, component, n_features = 50L) {
  if (is.null(inf) || is.null(inf$feature_results) || !nrow(inf$feature_results)) {
    return(character(0))
  }

  comp_name <- .report_get_inference_component_name(inf, component)
  fr <- inf$feature_results[inf$feature_results$component == comp_name, , drop = FALSE]
  if (!nrow(fr)) return(character(0))

  if ("qval" %in% names(fr)) {
    ord <- order(fr$qval, fr$pval %||% fr$qval, na.last = NA)
  } else if ("pval" %in% names(fr)) {
    ord <- order(fr$pval, na.last = NA)
  } else if ("stat" %in% names(fr)) {
    ord <- order(abs(fr$stat), decreasing = TRUE, na.last = NA)
  } else {
    ord <- seq_len(nrow(fr))
  }

  feats <- as.character(fr$feature[ord])
  feats <- feats[!is.na(feats) & nzchar(feats)]
  head(unique(feats), n_features)
}

.report_intersect_scopes <- function(scopes, allowed) {
  scopes <- .report_expand_choice(scopes, c("union", "component", "global", "all"))
  intersect(scopes, allowed)
}

.report_save_split <- function(wf, file) {
  split <- list(
    split = wf$split,
    train_samples = if (!is.null(wf$data_train)) rownames(wf$data_train) else NULL,
    test_samples = if (!is.null(wf$data_test)) rownames(wf$data_test) else NULL
  )
  
  saveRDS(split, file = file)
  invisible(file)
}

.report_component_label <- function(component) {
  paste0("component_", component)
}

.report_dataset_input <- function(x, dataset, dataset_names = NULL, arg = "input", list_kind = c("shared_list", "dataset_list")) {
  list_kind <- match.arg(list_kind)
  if (is.null(x)) return(NULL)
  if (!is.list(x) || is.data.frame(x)) return(x)

  nms <- names(x)
  if (!is.null(nms) && dataset %in% nms) return(x[[dataset]])

  if (identical(list_kind, "shared_list")) {
    return(x)
  }

  if (is.null(nms) || any(!nzchar(nms))) {
    stop(
      "For integrated report enrichment, `", arg, "` was supplied as an unnamed list. ",
      "Use a shared data frame/object, or provide a named list keyed by dataset.",
      call. = FALSE
    )
  }

  if (!is.null(dataset_names) && !all(dataset_names %in% nms)) {
    stop(
      "For integrated report enrichment, named list `", arg, "` must contain all dataset names: ",
      paste(dataset_names, collapse = ", "), ".",
      call. = FALSE
    )
  }

  NULL
}

.report_dataset_arg <- function(x, dataset, dataset_names = NULL, default = NULL) {
  if (is.null(x)) x <- default
  if (is.null(x)) return(NULL)

  if (is.list(x) && !is.data.frame(x)) {
    if (!is.null(names(x)) && dataset %in% names(x)) {
      return(x[[dataset]])
    }
    if (!is.null(dataset_names) && length(x) == length(dataset_names)) {
      names(x) <- dataset_names
      return(x[[dataset]])
    }
  }

  x
}

.report_dataset_inference_enrichment <- function(inference_enrichment, dataset, dataset_names = NULL) {
  val <- .report_dataset_arg(inference_enrichment, dataset, dataset_names, default = FALSE)
  isTRUE(val)
}

.report_safe_enrichment <- function(
    inf,
    feature_sets = NULL,
    alpha = 0.1,
    scope = "union",
    component = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    enrichment_term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01
) {
  enrichment_backend <- match.arg(enrichment_backend)
  enrichment_id_type <- match.arg(enrichment_id_type)

  tryCatch(
    enrichPathways(
      inf,
      feature_sets = feature_sets,
      alpha = alpha,
      scope = scope,
      component = component,
      enrichment_backend = enrichment_backend,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type,
      enrichment_score = enrichment_score,
      term2gene = enrichment_term2gene,
      enrichment_id_map = enrichment_id_map,
      enrichment_extra_args = enrichment_extra_args,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction
    ),
    error = function(e) {
      warning(
        "Report-time enrichment failed: ", e$message,
        call. = FALSE
      )
      NULL
    }
  )
}

.report_safe_integrated_enrichment <- function(
    wf,
    dataset,
    dataset_names = NULL,
    feature_sets = NULL,
    alpha = 0.1,
    scope = "union",
    component = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    enrichment_term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01
) {
  dataset_names <- dataset_names %||% wf$dataset_names %||% names(wf$Y)

  enrichment_backend <- match.arg(
    .report_dataset_arg(enrichment_backend, dataset, dataset_names, default = "auto"),
    c("auto", "internal", "clusterProfiler", "fgsea")
  )
  enrichment_id_type <- match.arg(
    .report_dataset_arg(enrichment_id_type, dataset, dataset_names, default = "symbol"),
    c("symbol", "entrez", "ensembl")
  )
  msigdb_db_species <- .report_dataset_arg(msigdb_db_species, dataset, dataset_names)
  msigdb_species <- .report_dataset_arg(msigdb_species, dataset, dataset_names)
  msigdb_collection <- .report_dataset_arg(msigdb_collection, dataset, dataset_names)
  msigdb_subcollection <- .report_dataset_arg(msigdb_subcollection, dataset, dataset_names)
  enrichment_score <- .report_dataset_arg(enrichment_score, dataset, dataset_names, default = "auto")
  enrichment_extra_args <- .report_dataset_input(enrichment_extra_args, dataset, dataset_names, arg = "enrichment_extra_args", list_kind = "shared_list") %||% list()
  min_feature_overlap_warn <- .report_dataset_arg(min_feature_overlap_warn, dataset, dataset_names, default = 5L)
  min_feature_overlap_fraction <- .report_dataset_arg(min_feature_overlap_fraction, dataset, dataset_names, default = 0.01)

  if (is.null(enrichment_id_map) && !is.null(wf$fdata)) {
    fd <- .svdx_resolve_fdata_for_dataset(wf$fdata, dataset = dataset, required = FALSE)
    enrichment_id_map <- .svdx_fdata_to_enrichment_id_map(fd, required = FALSE)
  }

  tryCatch(
    enrichPathways(
      wf,
      dataset = dataset,
      feature_sets = .report_dataset_input(feature_sets, dataset, dataset_names, arg = "feature_sets", list_kind = "shared_list"),
      alpha = alpha,
      scope = scope,
      component = component,
      enrichment_backend = enrichment_backend,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type,
      enrichment_score = enrichment_score,
      term2gene = .report_dataset_input(enrichment_term2gene, dataset, dataset_names, arg = "enrichment_term2gene", list_kind = "dataset_list"),
      enrichment_id_map = .report_dataset_input(enrichment_id_map, dataset, dataset_names, arg = "enrichment_id_map", list_kind = "dataset_list"),
      enrichment_extra_args = enrichment_extra_args,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction
    ),
    error = function(e) {
      warning(
        "Report-time enrichment failed for dataset '", dataset, "': ", e$message,
        call. = FALSE
      )
      NULL
    }
  )
}


.report_enrichment_include_union <- function(enrichment_scope) {
  any(.report_expand_choice(enrichment_scope, c("union", "component", "global", "all")) %in% "union")
}

.report_enrichment_include_global <- function(enrichment_scope) {
  any(.report_expand_choice(enrichment_scope, c("union", "component", "global", "all")) %in% "global")
}

.report_enrichment_include_component <- function(enrichment_scope) {
  any(.report_expand_choice(enrichment_scope, c("union", "component", "global", "all")) %in% "component")
}

.report_enrichment_components <- function(enrichment_scope, components) {
  if (!.report_enrichment_include_component(enrichment_scope)) {
    return(integer(0))
  }
  components <- as.integer(components)
  components[is.finite(components)]
}


.report_match_choice <- function(x, choices, arg) {
  if (length(x) != 1L || !x %in% choices) {
    stop(arg, " must be one of: ", paste(choices, collapse = ", "), call. = FALSE)
  }
  x
}

.report_expand_choice <- function(x, choices) {
  if (identical(x, "all")) setdiff(choices, "all") else x
}

.report_heatmap_annotate <- function(heatmap_order_by = NULL, heatmap_annotate = NULL) {
  if (is.null(heatmap_annotate) && !is.null(heatmap_order_by)) heatmap_order_by else heatmap_annotate
}

.report_expand_validated_choice <- function(x, choices, arg) {
  tryCatch(
    {
      x <- match.arg(x, choices)
          x
    },
    error = function(e) {
      stop("`", arg, "` must be one of: ", paste(choices, collapse = ", "), call. = FALSE)
    }
  )
}

.report_metadata_names <- function(data = NULL, zdata = NULL) {
  out <- character(0)
  if (!is.null(data)) out <- c(out, names(data))
  if (is.data.frame(zdata)) out <- c(out, names(zdata))
  if (is.list(zdata) && !is.data.frame(zdata)) {
    out <- c(out, unlist(lapply(zdata, names), use.names = FALSE))
  }
  unique(out)
}

.report_validate_metadata_vars <- function(vars, data = NULL, zdata = NULL, arg = "variable") {
  vars <- unique(stats::na.omit(as.character(vars %||% character(0))))
  vars <- vars[nzchar(vars)]
  if (length(vars) == 0) return(invisible(TRUE))

  available <- .report_metadata_names(data = data, zdata = zdata)
  missing_vars <- setdiff(vars, available)
  if (length(missing_vars) > 0) {
    stop(
      "Unknown `", arg, "` value(s): ", paste(missing_vars, collapse = ", "), ". ",
      "Available metadata variables are: ", paste(available, collapse = ", "),
      call. = FALSE
    )
  }

  invisible(TRUE)
}

.report_validate_train_fraction <- function(train_fraction) {
  .svdx_validate_train_fraction(train_fraction)
  invisible(TRUE)
}

.report_validate_enrichment_prereqs <- function(
    inference_enrichment,
    fdata = NULL,
    enrichment_id_map = NULL,
    feature_sets = NULL,
    enrichment_term2gene = NULL,
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    require_all_fdata_enrichment = TRUE
) {
  if (!isTRUE(inference_enrichment)) return(invisible(TRUE))

  if (is.null(enrichment_id_map) && is.null(fdata)) {
    stop(
      "`inference_enrichment = TRUE` requires either `enrichment_id_map` or `fdata` with an enrichment column.",
      call. = FALSE
    )
  }

  if (is.null(enrichment_id_map) && !is.null(fdata)) {
    if (is.list(fdata) && !is.data.frame(fdata) && !inherits(fdata, "svdx_fdata")) {
      maps <- lapply(fdata, .svdx_fdata_to_enrichment_id_map, required = isTRUE(require_all_fdata_enrichment))
      if (!isTRUE(require_all_fdata_enrichment) && !any(vapply(maps, Negate(is.null), logical(1)))) {
        stop(
          "`inference_enrichment = TRUE` requires at least one dataset-specific `fdata` object with an enrichment column, ",
          "or an explicit `enrichment_id_map`.",
          call. = FALSE
        )
      }
      invisible(maps)
    } else {
      .svdx_fdata_to_enrichment_id_map(fdata, required = TRUE)
    }
  }

  if (is.null(feature_sets) && is.null(enrichment_term2gene)) {
    missing_args <- c(
      if (is.null(msigdb_db_species)) "msigdb_db_species",
      if (is.null(msigdb_species)) "msigdb_species",
      if (is.null(msigdb_collection)) "msigdb_collection"
    )
    if (length(missing_args) > 0) {
      stop(
        "MSigDB enrichment requested but missing required argument(s): ",
        paste(missing_args, collapse = ", "), ". ",
        "Supply these before running the report, or provide `feature_sets`/`enrichment_term2gene`.",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}

.report_validate_common_args <- function(
    data,
    zdata,
    xaxis,
    yaxis = NULL,
    group = NULL,
    color = NULL,
    facet = NULL,
    heatmap_order_by = NULL,
    heatmap_annotate = NULL,
    train_fraction = 0.70,
    inference_enrichment = FALSE,
    fdata = NULL,
    enrichment_id_map = NULL,
    feature_sets = NULL,
    enrichment_term2gene = NULL,
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    require_all_fdata_enrichment = TRUE
) {
  .report_validate_train_fraction(train_fraction)
  .report_validate_metadata_vars(xaxis, data = data, zdata = zdata, arg = "xaxis")
  .report_validate_metadata_vars(yaxis, data = data, zdata = zdata, arg = "yaxis")
  .report_validate_metadata_vars(group, data = data, zdata = zdata, arg = "group")
  .report_validate_metadata_vars(color, data = data, zdata = zdata, arg = "color")
  .report_validate_metadata_vars(facet, data = data, zdata = zdata, arg = "facet")
  .report_validate_metadata_vars(heatmap_order_by, data = data, zdata = zdata, arg = "heatmap_order_by")
  .report_validate_metadata_vars(heatmap_annotate, data = data, zdata = zdata, arg = "heatmap_annotate")
  .report_validate_enrichment_prereqs(
    inference_enrichment = inference_enrichment,
    fdata = fdata,
    enrichment_id_map = enrichment_id_map,
    feature_sets = feature_sets,
    enrichment_term2gene = enrichment_term2gene,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    require_all_fdata_enrichment = require_all_fdata_enrichment
  )
  invisible(TRUE)
}

.report_loading_union_features <- function(fit, components, n_features = 50, dataset = NULL) {
  vmat <- .get_vmat(fit, dataset = dataset)
  vmat <- as.matrix(vmat)
  if (is.null(colnames(vmat))) colnames(vmat) <- .make_component_names(ncol(vmat))
  k <- vapply(components, function(cc) .resolve_component_index(cc, colnames(vmat)), integer(1))
  score <- apply(abs(vmat[, k, drop = FALSE]), 1, max, na.rm = TRUE)
  head(names(sort(score, decreasing = TRUE)), n_features)
}

.report_inference_available <- function(wf, dataset = NULL) {
  if (inherits(wf, "isvdxWorkflow")) {
    return(!is.null(wf$inf) && !is.null(wf$inf$dataset_results) && !is.null(wf$inf$dataset_results[[dataset]]))
  }
  !is.null(wf$inf)
}

.report_test_data_available <- function(wf, dataset = NULL) {
  if (inherits(wf, "isvdxWorkflow")) {
    return(
      !is.null(dataset) &&
        !is.null(wf$Ytest) &&
        !is.null(wf$data_test) &&
        !is.null(wf$Ytest[[dataset]])
    )
  }
  !is.null(wf$Ytest) && !is.null(wf$data_test)
}

.report_save_test_plot_safe <- function(wf, dataset = NULL, expr, file, ..., verbose = TRUE) {
  if (!.report_test_data_available(wf, dataset = dataset)) {
    ds_msg <- if (is.null(dataset)) "" else paste0(" for dataset '", dataset, "'")
    .report_message(
      "Skipping test-based plot", ds_msg, ": test data unavailable for ", basename(file), ".",
      verbose = verbose
    )
    return(NULL)
  }
  .report_save_plot_safe(expr, file, ..., verbose = verbose)
}

.report_profile_plot <- function(
    wf,
    file,
    selection_by = c("loading", "inference"),
    scope = c("union", "component", "global"),
    component = NULL,
    components = NULL,
    dataset = NULL,
    xaxis,
    color = "feature",
    facet = NULL,
    summarize = "mean",
    show_se = TRUE,
    se_width = 0.15,
    scale = "zscore",
    top_n = 5,
    verbose = TRUE
) {
  selection_by <- match.arg(selection_by)
  scope <- match.arg(scope)

  if (selection_by == "loading" && scope == "global") {
    return(NULL)
  }

  if (selection_by == "inference") {
    if (!.report_inference_available(wf, dataset = dataset)) {
      .report_message(
        "Skipping inference feature profile: inference results unavailable for ", basename(file), ".",
        verbose = verbose
      )
      return(NULL)
    }
    if (!.report_test_data_available(wf, dataset = dataset)) {
      .report_message(
        "Skipping inference feature profile: test data unavailable for ", basename(file), ".",
        verbose = verbose
      )
      return(NULL)
    }
  }

  .report_save_plot_safe(
    if (selection_by == "inference") {
      comp <- if (scope %in% c("union", "global")) NULL else component
      plotFeatureProfiles(
        wf,
        dataset = dataset,
        source = "test",
        inference_scope = scope,
        component = comp,
        select_by = "q",
        top_n = top_n,
        x = xaxis,
        color = color,
        facet = facet,
        summarize = summarize,
        show_se = show_se,
        se_width = se_width,
        scale = scale,
        line = TRUE,
        points = TRUE,
        fdr = wf$alpha
      )
    } else {
      feats <- if (scope == "union") {
        .report_loading_union_features(wf$fit, components = components, n_features = top_n, dataset = dataset)
      } else {
        idx <- .select_features_from_loadings(wf$fit, component = component, n_features = top_n, dataset = dataset)
        mat <- .get_vmat(wf$fit, dataset = dataset)
        rownames(mat)[idx]
      }
      plotFeatureProfiles(
        wf,
        dataset = dataset,
        features = feats,
        x = xaxis,
        color = color,
        facet = facet,
        summarize = summarize,
        show_se = show_se,
        se_width = se_width,
        scale = scale,
        line = TRUE,
        points = TRUE
      )
    },
    file = file,
    verbose = verbose
  )
}

#' Run an SSANOVA-SVD workflow and save a PDF report
#'
#' Runs [runSVDXWorkflow()], saves the workflow object and summary to disk,
#' and generates a first-pass set of PDF plots for fitted components,
#' loading heatmaps, inference results, and optional enrichment.
#'
#' @param Y Feature-by-sample matrix.
#' @param data Sample metadata data frame.
#' @param xformula Formula for metadata-driven component scores.
#' @param zformula Optional nuisance formula.
#' @param zdata Data frame for nuisance covariates. Defaults to `data`.
#' @param outdir Output directory.
#' @param overwrite Logical; if TRUE, remove any existing report directory before writing. If FALSE, stop when outdir already exists.
#' @param K Maximum number of components.
#' @param family Optional response family; see [runSVDXWorkflow()].
#' @param xaxis Metadata variable for one-dimensional U plots.
#' @param yaxis Optional second metadata variable for two-dimensional U plots.
#' @param group Optional grouping variable for U plots.
#' @param color Optional color variable for U embedding plots.
#' @param facet Optional faceting variable for U plots.
#' @param heatmap_order_by Optional metadata variables used to order heatmap columns.
#' @param heatmap_annotate Optional metadata variables used as heatmap annotations.
#'   If omitted and heatmap ordering is supplied, annotations inherit the heatmap
#'   ordering variables.
#' @param loading_outputs Logical; whether to generate loading-based report outputs.
#' @param inference_outputs Logical; whether to generate inference-based report outputs.
#' @param loading_heatmaps Logical; whether to generate loading-selected heatmaps.
#' @param loading_feature_profiles Logical; whether to generate loading-selected feature profile plots.
#' @param inference_scope Inference scopes to generate: `"all"`, `"component"`, `"union"`, or `"global"`. `"all"` generates component-, union-, and global-scope inference outputs.
#' @param inference_volcanoes Logical; whether to generate inference volcano plots.
#' @param inference_heatmaps Logical; whether to generate inference-selected heatmaps.
#' @param inference_feature_profiles Logical; whether to generate inference-selected feature profile plots.
#' @param inference_dotplots Logical; whether to generate inference dotplots.
#' @param inference_enrichment Logical; whether to generate inference enrichment outputs when enrichment identifiers are supplied.
#' @param alpha FDR threshold used for report-time inference, significance annotations, and thresholded enrichment. Defaults to 0.10.
#' @param z_adjustment Nuisance-adjustment mode for inference: `"fitted"` uses SVDX-estimated nuisance effects; `"reestimate"` re-estimates nuisance effects inside `inferV()`.
#' @param feature_profile_color Color variable for feature profile plots. Use
#'   `feature` to color by feature label.
#' @param feature_profile_facet Optional faceting variable for feature profile plots.
#' @param features_top_n Number of top features shown in feature-ranked plots such as loading bar plots, heatmaps, and inference feature dotplots.
#' @param feature_profiles_top_n Number of top features shown in each feature profile plot.
#' @param enrichment_top_n Number of enrichment categories shown in enrichment dotplots.
#' @param volcano_labels_top_n Number of top features labeled in volcano plots.
#' @param feature_profile_summarize Summary mode for feature profiles.
#' @param feature_profile_show_se Logical; show standard-error bars when `feature_profile_summarize = "mean"`.
#' @param feature_profile_se_width Width passed to ggplot2 error bars for summarized feature profiles.
#' @param feature_profile_scale Plotting scale for feature profiles.
#' @param fdata Optional `svdx_fdata` object created by [as_svdx_fdata()].
#'   Stored in the generated workflow and reused by plots, reports, and enrichment.
#' @param feature_sets Optional named list of feature sets for enrichment.
#' @param enrichment_backend Enrichment backend: auto, internal, clusterProfiler, or fgsea.
#' @param enrichment_id_map Two-column data frame used only for enrichment. The
#'   first column must contain SVDX feature identifiers and the second column
#'   must contain enrichment identifiers. Column names are ignored. Required
#'   when `inference_enrichment = TRUE` unless `fdata` contains an enrichment column.
#' @param msigdb_db_species MSigDB database species passed to `msigdbr`, e.g.
#'   `"HS"` for human MSigDB or `"MM"` for mouse MSigDB.
#' @param msigdb_species Species name passed to `msigdbr`, e.g.
#'   `"Homo sapiens"` or `"Mus musculus"`.
#' @param msigdb_collection MSigDB collection passed to `msigdbr`, e.g.
#'   `"H"`, `"C5"`, `"MH"`, or `"M5"`.
#' @param msigdb_subcollection Optional MSigDB subcollection, e.g. `"BP"`,
#'   `"MF"`, or `"CC"`.
#' @param enrichment_id_type Identifier column to use from MSigDB: `symbol`,
#'   `entrez`, or `ensembl`. The second column of `enrichment_id_map` should
#'   use the same identifier type.
#' @param enrichment_score Ranking score for fgsea/GSEA-style enrichment. The
#'   default `"auto"` uses -log10(union q-value) for union scope and signed
#'   -log10(component q-value) for component scope.
#' @param enrichment_term2gene Optional TERM2GENE table.
#' @param enrichment_extra_args Optional named list of additional arguments passed to backend enrichment functions; for integrated reports this may also be a named list by dataset.
#' @param min_feature_overlap_warn Minimum feature overlap below which to warn about likely identifier mismatch; set NULL to disable count-based warning.
#' @param min_feature_overlap_fraction Minimum fraction of inferred features overlapping enrichment sets below which to warn; set NULL to disable fraction-based warning.
#' @param components Components to plot. If `NULL`, all available components are plotted.
#' @param train_fraction Fraction of samples used for training. Values in (0, 1) create a train/test split and run conditional inference; `train_fraction = 1` fits on all samples and skips inference outputs.
#' @param seed Optional seed for train/test splitting.
#' @param numThreads Number of threads used by the fitter.
#' @param save_inputs Logical; whether to save the original input objects to disk.
#'   Defaults to `FALSE` because matrices may be large.
#' @param save_split Logical; whether to save train/test split metadata.
#' @param verbose Logical; whether to print progress messages.
#' @param modelType Optional shared model-space specification for numeric variables in `xformula`.
#'   When `Y` is an `svdx_data` object, this defaults to `Y$modelType`.
#' @param row_offset Optional feature-level offset vector. When `Y` is an
#'   `svdx_data` object, this defaults to `Y$row_offset`.
#' @param col_offset Optional sample-level offset vector. When `Y` is an
#'   `svdx_data` object, this defaults to `Y$col_offset`.
#' @param ... Additional arguments passed to [runSVDXWorkflow()].
#'
#' @return Invisibly returns the `"svdxWorkflow"` object.
#'
#' @examples
#' \dontrun{
#' wf <- runSVDXReport(
#'   Y = Y,
#'   data = meta,
#'   xformula = ~ group * t + severity * t,
#'   zformula = ~ batch,
#'   outdir = "svdx_report",
#'   K = 5,
#'   xaxis = "t",
#'   yaxis = "severity",
#'   group = "group",
#'   heatmap_order_by = c("group", "t"),
#'   heatmap_annotate = c("group", "batch")
#' )
#' }
#'
#' @export
runSVDXReport <- function(
    Y,
    data,
    xformula,
    zformula = NULL,
    zdata = data,
    outdir = "svdx_report",
    overwrite = FALSE,
    K = 5,
    family = NULL,
    z_adjustment = c("fitted", "reestimate"),
    modelType = NULL,
    row_offset = NULL,
    col_offset = NULL,
    xaxis,
    yaxis = NULL,
    group = NULL,
    color = group,
    facet = NULL,
    heatmap_order_by = NULL,
    heatmap_annotate = NULL,
    loading_outputs = TRUE,
    inference_outputs = TRUE,
    loading_heatmaps = TRUE,
    loading_feature_profiles = TRUE,
    inference_scope = c("all", "component", "union", "global"),
    inference_volcanoes = TRUE,
    inference_heatmaps = TRUE,
    inference_feature_profiles = TRUE,
    inference_dotplots = TRUE,
    inference_enrichment = TRUE,
    alpha = 0.1,
    feature_profile_color = "feature",
    feature_profile_facet = NULL,
    features_top_n = 50L,
    feature_profiles_top_n = 5L,
    enrichment_top_n = 20L,
    feature_profile_summarize = c("mean", "median", "none"),
    feature_profile_show_se = TRUE,
    feature_profile_se_width = 0.15,
    feature_profile_scale = c("zscore", "raw"),
    volcano_label_features = NULL,
    volcano_labels_top_n = 25L,
    volcano_label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    volcano_use_feature_labels = TRUE,
    fdata = NULL,
    feature_sets = NULL,
    enrichment_id_map = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    enrichment_term2gene = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    components = NULL,
    train_fraction = 0.70,
    seed = 1,
    numThreads = 4,
    save_inputs = FALSE,
    save_split = TRUE,
    verbose = TRUE,
    ...
) {
  if (is_svdx_data(Y)) {
    prepared <- Y
    if (isTRUE(prepared$integrated)) {
      stop("runSVDXReport() received integrated svdx_data; use runISVDXReport() instead.", call. = FALSE)
    }
    Y <- prepared$Y
    if (missing(data)) data <- prepared$xdata
    if (missing(xformula)) xformula <- prepared$xformula
    if (missing(zdata)) zdata <- prepared$zdata
    if (missing(zformula) || is.null(zformula)) zformula <- prepared$zformula
    if (missing(family)) family <- prepared$family
    if (missing(fdata) || is.null(fdata)) fdata <- prepared$fdata
    if (missing(row_offset)) row_offset <- prepared$row_offset
    if (missing(col_offset)) col_offset <- prepared$col_offset
    if (is.null(modelType)) modelType <- prepared$modelType
  }
  if (missing(xaxis)) {
    stop("`xaxis` must be supplied for report U plots and feature profile plots.", call. = FALSE)
  }
  enrichment_backend <- match.arg(enrichment_backend)
  report_inference_scope <- match.arg(inference_scope)
  inference_scopes <- .report_expand_choice(report_inference_scope, c("component", "union", "global", "all"))
  z_adjustment <- match.arg(z_adjustment)
  enrichment_id_type <- match.arg(enrichment_id_type)
  feature_profile_summarize <- match.arg(feature_profile_summarize)
  feature_profile_scale <- match.arg(feature_profile_scale)
  volcano_label_by <- match.arg(volcano_label_by)
  heatmap_annotate <- .report_heatmap_annotate(heatmap_order_by, heatmap_annotate)
  compute_enrichment <- isTRUE(inference_outputs) && isTRUE(inference_enrichment) && (!is.null(enrichment_id_map) || !is.null(fdata))

  .report_validate_common_args(
    data = data,
    zdata = zdata,
    xaxis = xaxis,
    yaxis = yaxis,
    group = group,
    color = color,
    facet = facet,
    heatmap_order_by = heatmap_order_by,
    heatmap_annotate = heatmap_annotate,
    train_fraction = train_fraction,
    inference_enrichment = inference_enrichment,
    fdata = fdata,
    enrichment_id_map = enrichment_id_map,
    feature_sets = feature_sets,
    enrichment_term2gene = enrichment_term2gene,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection
  )

  .report_validate_outdir_preflight(outdir, overwrite = overwrite)

  wf <- runSVDXWorkflow(
    Y = Y,
    data = data,
    zdata = zdata,
    xformula = xformula,
    zformula = zformula,
    K = K,
    family = family,
    z_adjustment = z_adjustment,
    modelType = modelType,
    row_offset = row_offset,
    col_offset = col_offset,
    train_fraction = train_fraction,
    seed = seed,
    fdata = fdata,
    feature_sets = feature_sets,
    inference_enrichment = inference_enrichment,
    alpha = alpha,
    enrichment_backend = enrichment_backend,
    inference_scope = report_inference_scope,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    enrichment_term2gene = enrichment_term2gene,
    enrichment_id_map = enrichment_id_map,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    numThreads = numThreads,
    return_design = TRUE,
    ...
  )
  .report_message("Workflow complete", verbose = verbose)

  outdir <- .report_prepare_outdir(outdir, overwrite = overwrite)
  .report_message("Output directory ready: ", outdir, verbose = verbose)
  
  plots_dir <- file.path(outdir, "plots")
  tables_dir <- file.path(outdir, "tables")
  u_dir <- file.path(plots_dir, "U")
  v_dir <- file.path(plots_dir, "V_loadings")
  inf_dir <- file.path(plots_dir, "inference")
  prof_dir <- file.path(plots_dir, "feature_profiles")
  
  .report_dir_create(plots_dir)
  .report_dir_create(tables_dir)
  .report_dir_create(u_dir)
  .report_dir_create(v_dir)
  .report_dir_create(inf_dir)
  .report_dir_create(prof_dir)
  
  plot_files <- list()
  
  report_enrichment_id_map <- enrichment_id_map
  if (is.null(report_enrichment_id_map) && !is.null(wf$fdata)) {
    report_enrichment_id_map <- .svdx_fdata_to_enrichment_id_map(wf$fdata, required = FALSE)
  }

  workflow_file <- file.path(outdir, "svdx_workflow.rds")
  summary_file <- file.path(outdir, "svdx_workflow_summary.rds")
  
  .report_save_rds(wf, workflow_file)
  .report_message("Workflow object saved", verbose = verbose)
  
  wf_summary <- tryCatch(
    summary(wf),
    error = function(e) {
      .report_message("Skipping workflow summary: ", conditionMessage(e), verbose = verbose)
      NULL
    }
  )
  if (!is.null(wf_summary)) {
    .report_save_rds(wf_summary, summary_file)
    .report_message("Workflow summary saved", verbose = verbose)
  }
  
  split_file <- NULL
  if (isTRUE(save_split)) {
    split_file <- file.path(outdir, "train_test_split.rds")
    .report_save_split(wf, split_file)
    .report_save_sample_split_tsv(wf, file.path(tables_dir, "sample_split.tsv"))
    .report_message("Train/test split saved", verbose = verbose)
  }
  
  inputs_file <- NULL
  if (isTRUE(save_inputs)) {
    inputs_file <- file.path(outdir, "report_inputs.rds")
    .report_save_rds(
      list(Y = Y, data = data, zdata = zdata),
      inputs_file
    )
    .report_message("Input objects saved", verbose = verbose)
  }
  
  comp_info <- .report_resolve_components(wf, components)
  components <- comp_info$components
  n_components <- comp_info$n_components
  
  .report_message("Detected ", n_components, " component(s)", verbose = verbose)
  inference_available <- .report_inference_available(wf)
  if (isTRUE(inference_available)) {
    .report_export_inference_tables(
      wf$inf,
      fdata = wf$fdata,
      tables_dir = tables_dir,
      components = components,
      alpha = wf$alpha
    )
  }
  enrichment_component_tables <- list()
  enrichment_global_table <- NULL
  if (isTRUE(inference_outputs) && !isTRUE(inference_available)) {
    .report_message(
      "Skipping inference and enrichment sections: inference results unavailable.",
      verbose = verbose
    )
  }
  inference_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_available)) inference_scopes else character(0)
  loading_heatmap_scopes <- if (isTRUE(loading_outputs) && isTRUE(loading_heatmaps)) c("union", "component") else character(0)
  inference_heatmap_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_heatmaps) && isTRUE(inference_available)) inference_scopes else character(0)
  volcano_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_volcanoes) && isTRUE(inference_available)) intersect(c("union", "global"), inference_scopes) else character(0)
  enrichment_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_enrichment) && isTRUE(inference_available)) inference_scopes else character(0)
  
  # ----------------------------------------------------------
  # U plots
  # ----------------------------------------------------------
  for (component in components) {
    lab <- .report_component_label(component)
    
    
    f <- file.path(u_dir, paste0("U_", lab, "_1D.pdf"))
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u",
        component = component,
        xaxis = xaxis,
        group = group,
        facet = facet,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
    
    f <- file.path(u_dir, paste0("U_terms_", lab, "_1D.pdf"))
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u_terms",
        component = component,
        xaxis = xaxis,
        group = group,
        facet = facet,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
    
    if (!is.null(yaxis)) {
      f <- file.path(u_dir, paste0("U_", lab, "_2D.pdf"))
      plot_files[[f]] <- .report_save_plot_safe(
        plotU(
          wf$fit,
          data = wf$data,
          component = component,
          x = xaxis,
          y = yaxis,
          group = group,
          facet = facet,
          fill = "typical"
        ),
        file = f,
        width = 9,
        height = 8,
        verbose = verbose
      )
    }
  }
  
  if (n_components >= 2) {
    
    f <- file.path(u_dir, "U_embedding_components_1_2.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u_embedding",
        components = c(1, 2),
        color = color,
        group = group,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
  }
  
  # ----------------------------------------------------------
  # V loading plots and loading heatmaps
  # ----------------------------------------------------------
  if (isTRUE(loading_outputs) && "union" %in% loading_heatmap_scopes) {
    f <- file.path(v_dir, "V_loading_heatmap_union.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "v_heatmap",
        features = .report_loading_union_features(wf$fit, components, n_features = features_top_n),
        order_by = heatmap_order_by,
        annotate = heatmap_annotate
      ),
      file = f,
      width = 10,
      height = 9,
      verbose = verbose
    )
  }

  for (component in components) {
    lab <- .report_component_label(component)
    
    
    f <- file.path(v_dir, paste0("V_loadings_", lab, ".pdf"))
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "v_loadings",
        component = component,
        n = features_top_n
      ),
      file = f,
      verbose = verbose
    )
    
    if (isTRUE(loading_outputs) && "component" %in% loading_heatmap_scopes) {
      f <- file.path(v_dir, paste0("V_loading_heatmap_", lab, ".pdf"))
      plot_files[[f]] <- .report_save_plot_safe(
        plot(
          wf,
          type = "v_heatmap",
          component = component,
          n_features = features_top_n,
          order_by = heatmap_order_by,
          annotate = heatmap_annotate
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }
  }
  
  # ----------------------------------------------------------
  # Inference union plots
  # ----------------------------------------------------------
  
  if ("union" %in% inference_heatmap_scopes) {
    f <- file.path(inf_dir, "inference_union_heatmap.pdf")
    plot_files[[f]] <- .report_save_test_plot_safe(
      wf,
      expr = plot(
        wf,
        type = "v_heatmap",
        source = "test",
        n_features = features_top_n,
        inference_scope = "union",
        select_by = "q",
        fdr = wf$alpha,
        order_by = heatmap_order_by,
        annotate = heatmap_annotate
      ),
      file = f,
      width = 10,
      height = 9,
      verbose = verbose
    )
  }
  if ("global" %in% inference_heatmap_scopes) {
    f <- file.path(inf_dir, "inference_global_heatmap.pdf")
    plot_files[[f]] <- .report_save_test_plot_safe(
      wf,
      expr = plot(
        wf,
        type = "v_heatmap",
        source = "test",
        n_features = features_top_n,
        inference_scope = "global",
        select_by = "q",
        fdr = wf$alpha,
        order_by = heatmap_order_by,
        annotate = heatmap_annotate
      ),
      file = f,
      width = 10,
      height = 9,
      verbose = verbose
    )
  }
  
  if (isTRUE(inference_outputs) && isTRUE(inference_dotplots) && "union" %in% inference_scopes) {
    f <- file.path(inf_dir, "inference_union_dotplot.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "v_inference_dotplot",
        n_features = features_top_n,
        inference_scope = "union",
        select_by = "q",
        fdr = wf$alpha
      ),
      file = f,
      width = 10,
      height = 9,
      verbose = verbose
    )
  }
  if (isTRUE(inference_outputs) && isTRUE(inference_dotplots) && "global" %in% inference_scopes) {
    f <- file.path(inf_dir, "inference_global_dotplot.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "v_inference_dotplot",
        n_features = features_top_n,
        inference_scope = "global",
        select_by = "q",
        fdr = wf$alpha
      ),
      file = f,
      width = 10,
      height = 9,
      verbose = verbose
    )
  }
  
  if (.report_inference_available(wf)) {
  if ("union" %in% volcano_scopes) {
    f <- file.path(inf_dir, "inference_union_volcano.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot_union_volcano(
        wf$inf,
        scope = "union",
        fdata = wf$fdata,
        label_features = volcano_label_features,
        label_top_n = volcano_labels_top_n,
        label_by = volcano_label_by,
        use_feature_labels = volcano_use_feature_labels
      ),
      file = f,
      verbose = verbose
    )
  }

  if ("global" %in% volcano_scopes) {
    f <- file.path(inf_dir, "inference_global_volcano.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot_union_volcano(
        wf$inf,
        scope = "global",
        fdata = wf$fdata,
        label_features = volcano_label_features,
        label_top_n = volcano_labels_top_n,
        label_by = volcano_label_by,
        use_feature_labels = volcano_use_feature_labels
      ),
      file = f,
      verbose = verbose
    )
  }
  
  if (isTRUE(compute_enrichment) && .report_enrichment_include_union(enrichment_scopes)) {
    f <- file.path(inf_dir, "inference_union_enrichment.pdf")
    enr_union <- .report_safe_enrichment(
      wf$inf,
      feature_sets = feature_sets,
      alpha = wf$alpha,
      scope = "union",
      enrichment_backend = enrichment_backend,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type,
      enrichment_score = enrichment_score,
      enrichment_term2gene = enrichment_term2gene,
      enrichment_id_map = report_enrichment_id_map,
      enrichment_extra_args = enrichment_extra_args,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction
    )
    
    if (!is.null(enr_union)) {
      .report_save_rds(enr_union, file.path(inf_dir, "inference_union_enrichment.rds"))
      plot_files[[f]] <- .report_save_plot_safe(
        plot_enrichment_dotplot(enr_union, top_n = enrichment_top_n),
        file = f,
        verbose = verbose
      )
    }
  }
  
  if (isTRUE(compute_enrichment) && .report_enrichment_include_global(enrichment_scopes)) {
    f <- file.path(inf_dir, "inference_global_enrichment.pdf")
    enr_global <- .report_safe_enrichment(
      wf$inf,
      feature_sets = feature_sets,
      alpha = wf$alpha,
      scope = "global",
      enrichment_backend = enrichment_backend,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type,
      enrichment_score = enrichment_score,
      enrichment_term2gene = enrichment_term2gene,
      enrichment_id_map = report_enrichment_id_map,
      enrichment_extra_args = enrichment_extra_args,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction
    )
    
    if (!is.null(enr_global)) {
      .report_save_rds(enr_global, file.path(inf_dir, "inference_global_enrichment.rds"))
      enrichment_global_table <- .report_flatten_enrichment(enr_global, fdata = wf$fdata)
      .report_save_tsv(enrichment_global_table, file.path(tables_dir, "enrichment_global.tsv"))
      plot_files[[f]] <- .report_save_plot_safe(
        plot_enrichment_dotplot(enr_global, top_n = enrichment_top_n),
        file = f,
        verbose = verbose
      )
    }
  }
  
  # ----------------------------------------------------------
  # Component-wise inference plots
  # ----------------------------------------------------------
  if ("component" %in% inference_scopes) {
    for (component in components) {
      lab <- .report_component_label(component)
      comp_dir <- file.path(inf_dir, lab)
      .report_dir_create(comp_dir)
      
      
      #comp_name <- colnames(wf$inf$coef)[component] %||% component
      comp_name <- tryCatch(
        .report_get_inference_component_name(wf$inf, component),
        error = function(e) {
          .report_message(
            "Skipping component-wise inference plots for ",
            lab,
            ": ",
            conditionMessage(e),
            verbose = verbose
          )
          NULL
        }
      )
      if (is.null(comp_name)) next
      
      if ("component" %in% inference_heatmap_scopes) {
        f <- file.path(comp_dir, paste0("inference_heatmap_", lab, ".pdf"))
        plot_files[[f]] <- .report_save_test_plot_safe(
          wf,
          expr = plot(
            wf,
            type = "v_heatmap",
            source = "test",
            component = comp_name,
            n_features = features_top_n,
            inference_scope = "component",
            select_by = "q",
            fdr = wf$alpha,
            order_by = heatmap_order_by,
            annotate = heatmap_annotate
          ),
          file = f,
          width = 10,
          height = 9,
          verbose = verbose
        )
      }
      
      if (isTRUE(inference_dotplots)) {
        f <- file.path(comp_dir, paste0("inference_dotplot_", lab, ".pdf"))
        plot_files[[f]] <- .report_save_plot_safe(
          plot(
            wf,
            type = "v_inference_dotplot",
            inference_scope = "component",
            component = comp_name,
            n_features = features_top_n,
            select_by = "q",
            fdr = wf$alpha
          ),
          file = f,
          width = 10,
          height = 9,
          verbose = verbose
        )
      }
      
      if (isTRUE(inference_volcanoes)) {
        f <- file.path(comp_dir, paste0("inference_volcano_", lab, ".pdf"))
        plot_files[[f]] <- .report_save_plot_safe(
          plot_component_volcano(
            wf$inf,
            component = comp_name,
            fdata = wf$fdata,
            label_features = volcano_label_features,
            label_top_n = volcano_labels_top_n,
            label_by = volcano_label_by,
            use_feature_labels = volcano_use_feature_labels
          ),
          file = f,
          verbose = verbose
        )
      }
      
      if (isTRUE(compute_enrichment) && component %in% .report_enrichment_components(enrichment_scopes, components)) {
        enr_comp <- .report_safe_enrichment(
          wf$inf,
          feature_sets = feature_sets,
          alpha = wf$alpha,
          scope = "component",
          component = comp_name,
          enrichment_backend = enrichment_backend,
          msigdb_db_species = msigdb_db_species,
          msigdb_species = msigdb_species,
          msigdb_collection = msigdb_collection,
          msigdb_subcollection = msigdb_subcollection,
          enrichment_id_type = enrichment_id_type,
          enrichment_score = enrichment_score,
          enrichment_term2gene = enrichment_term2gene,
          enrichment_id_map = report_enrichment_id_map,
          enrichment_extra_args = enrichment_extra_args,
          min_feature_overlap_warn = min_feature_overlap_warn,
          min_feature_overlap_fraction = min_feature_overlap_fraction
        )
        
        if (!is.null(enr_comp)) {
          enr_file <- file.path(comp_dir, paste0("inference_enrichment_", lab, ".rds"))
          .report_save_rds(enr_comp, enr_file)
          enr_tbl <- .report_flatten_enrichment(enr_comp, fdata = wf$fdata)
          enrichment_component_tables[[lab]] <- enr_tbl
          .report_save_tsv(enr_tbl, file.path(tables_dir, paste0("enrichment_", lab, ".tsv")))
          
          f <- file.path(comp_dir, paste0("inference_enrichment_", lab, ".pdf"))
          plot_files[[f]] <- .report_save_plot_safe(
            plot_enrichment_dotplot(enr_comp, top_n = enrichment_top_n),
            file = f,
            verbose = verbose
          )
        }
      }
    }
  }
  }

  if (length(enrichment_component_tables)) {
    enrichment_all <- do.call(rbind, enrichment_component_tables)
    rownames(enrichment_all) <- NULL
    .report_save_tsv(enrichment_all, file.path(tables_dir, "enrichment_allComponents.tsv"))
  }
  

  # ----------------------------------------------------------
  # Feature profile plots
  # ----------------------------------------------------------
  profile_by <- c(if (isTRUE(loading_outputs) && isTRUE(loading_feature_profiles)) "loading", if (isTRUE(inference_outputs) && isTRUE(inference_feature_profiles)) "inference")
  loading_profile_scopes <- if (isTRUE(loading_outputs) && isTRUE(loading_feature_profiles)) c("union", "component") else character(0)
  inference_profile_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_feature_profiles)) inference_scopes else character(0)

  for (sel in profile_by) {
    if ((sel == "loading" && "union" %in% loading_profile_scopes) || (sel == "inference" && "union" %in% inference_profile_scopes)) {
      f <- file.path(prof_dir, paste0("feature_profiles_", sel, "_union.pdf"))
      plot_files[[f]] <- .report_profile_plot(
        wf,
        file = f,
        selection_by = sel,
        scope = "union",
        components = components,
        xaxis = xaxis,
        color = feature_profile_color,
        facet = feature_profile_facet,
        summarize = feature_profile_summarize,
        show_se = feature_profile_show_se,
        se_width = feature_profile_se_width,
        scale = feature_profile_scale,
        top_n = feature_profiles_top_n,
        verbose = verbose
      )
    }
    if (sel == "inference" && "global" %in% inference_profile_scopes) {
      f <- file.path(prof_dir, paste0("feature_profiles_", sel, "_global.pdf"))
      plot_files[[f]] <- .report_profile_plot(
        wf,
        file = f,
        selection_by = sel,
        scope = "global",
        components = components,
        xaxis = xaxis,
        color = feature_profile_color,
        facet = feature_profile_facet,
        summarize = feature_profile_summarize,
        show_se = feature_profile_show_se,
        se_width = feature_profile_se_width,
        scale = feature_profile_scale,
        top_n = feature_profiles_top_n,
        verbose = verbose
      )
    }
    if ((sel == "loading" && "component" %in% loading_profile_scopes) || (sel == "inference" && "component" %in% inference_profile_scopes)) {
      for (component in components) {
        lab <- .report_component_label(component)
        f <- file.path(prof_dir, paste0("feature_profiles_", sel, "_", lab, ".pdf"))
        plot_files[[f]] <- .report_profile_plot(
          wf,
          file = f,
          selection_by = sel,
          scope = "component",
          component = component,
          components = components,
          xaxis = xaxis,
          color = feature_profile_color,
          facet = feature_profile_facet,
          summarize = feature_profile_summarize,
          show_se = feature_profile_show_se,
          se_width = feature_profile_se_width,
          scale = feature_profile_scale,
          top_n = feature_profiles_top_n,
          verbose = verbose
        )
      }
    }
  }

  manifest <- list(
    report_type = "svdx",
    workflow_file = workflow_file,
    summary_file = summary_file,
    split_file = split_file,
    inputs_file = inputs_file,
    outdir = outdir,
    plots_dir = plots_dir,
    plot_files = Filter(Negate(is.null), plot_files),
    n_components = n_components,
    components = components,
    call = match.call()
  )
  
  manifest_file <- file.path(outdir, "report_manifest.rds")
  .report_save_rds(manifest, manifest_file)
  .report_message("Report manifest saved", verbose = verbose)
  
  .report_message("Report complete: ", outdir, verbose = verbose)
  
  invisible(wf)
}


#' Run an integrated SSANOVA-SVD workflow and save a PDF report
#'
#' Runs [runISVDXWorkflow()], saves the workflow object and summary to disk,
#' and generates a first-pass set of PDF plots for shared U components,
#' dataset-specific loading plots, dataset-specific inference plots, and
#' optional enrichment.
#'
#' @param Y Named or unnamed list of feature-by-sample matrices.
#' @param data Shared sample metadata data frame.
#' @param zdata List of nuisance metadata data frames, one per dataset.
#' @param xformula Formula for shared metadata-driven component scores.
#' @param zformula List of nuisance formulas, one per dataset.
#' @param outdir Output directory.
#' @param overwrite Logical; if TRUE, remove any existing report directory before writing. If FALSE, stop when outdir already exists.
#' @param K Maximum number of components.
#' @param family Optional response family; see [runISVDXWorkflow()].
#' @param xaxis Metadata variable for one-dimensional U plots.
#' @param yaxis Optional second metadata variable for two-dimensional U plots.
#' @param group Optional grouping variable for U plots.
#' @param color Optional color variable for U embedding plots.
#' @param facet Optional faceting variable for U plots.
#' @param heatmap_order_by Optional metadata variables used to order heatmap columns.
#' @param heatmap_annotate Optional metadata variables used as heatmap annotations.
#'   If omitted and heatmap ordering is supplied, annotations inherit the heatmap
#'   ordering variables.
#' @param loading_outputs Logical; whether to generate loading-based report outputs.
#' @param inference_outputs Logical; whether to generate inference-based report outputs.
#' @param loading_heatmaps Logical; whether to generate loading-selected heatmaps.
#' @param loading_feature_profiles Logical; whether to generate loading-selected feature profile plots.
#' @param inference_scope Inference scopes to generate: `"all"`, `"component"`, `"union"`, or `"global"`. `"all"` generates component-, union-, and global-scope inference outputs.
#' @param inference_volcanoes Logical; whether to generate inference volcano plots.
#' @param inference_heatmaps Logical; whether to generate inference-selected heatmaps.
#' @param inference_feature_profiles Logical; whether to generate inference-selected feature profile plots.
#' @param inference_dotplots Logical; whether to generate inference dotplots.
#' @param inference_enrichment Logical; whether to generate inference enrichment outputs when enrichment identifiers are supplied.
#' @param alpha FDR threshold used for report-time inference, significance annotations, and thresholded enrichment. Defaults to 0.10.
#' @param z_adjustment Nuisance-adjustment mode for inference: `"fitted"` uses SVDX-estimated nuisance effects; `"reestimate"` re-estimates nuisance effects inside `inferV()`.
#' @param feature_profile_color Color variable for feature profile plots. Use
#'   `feature` to color by feature label.
#' @param feature_profile_facet Optional faceting variable for feature profile plots.
#' @param features_top_n Number of top features shown in feature-ranked plots such as loading bar plots, heatmaps, and inference feature dotplots.
#' @param feature_profiles_top_n Number of top features shown in each feature profile plot.
#' @param enrichment_top_n Number of enrichment categories shown in enrichment dotplots.
#' @param volcano_labels_top_n Number of top features labeled in volcano plots.
#' @param feature_profile_summarize Summary mode for feature profiles.
#' @param feature_profile_show_se Logical; show standard-error bars when `feature_profile_summarize = "mean"`.
#' @param feature_profile_se_width Width passed to ggplot2 error bars for summarized feature profiles.
#' @param feature_profile_scale Plotting scale for feature profiles.
#' @param fdata Optional named list of `svdx_fdata` objects, one per dataset.
#'   Stored in the generated workflow and reused by plots, reports, and enrichment.
#' @param feature_sets Optional named list of feature-set lists, one per dataset.
#' @param enrichment_backend Enrichment backend: auto, internal, clusterProfiler, or fgsea.
#' @param enrichment_id_map Named list of two-column data frames, one per
#'   dataset, used only for enrichment. In each data frame, the first column
#'   contains SVDX feature identifiers and the second column contains
#'   enrichment identifiers. Column names are ignored. Required when
#'   `inference_enrichment = TRUE` unless `fdata` contains an enrichment column.
#' @param msigdb_db_species MSigDB database species passed to `msigdbr`, e.g.
#'   `"HS"` for human MSigDB or `"MM"` for mouse MSigDB.
#' @param msigdb_species Species name passed to `msigdbr`, e.g.
#'   `"Homo sapiens"` or `"Mus musculus"`.
#' @param msigdb_collection MSigDB collection passed to `msigdbr`, e.g.
#'   `"H"`, `"C5"`, `"MH"`, or `"M5"`.
#' @param msigdb_subcollection Optional MSigDB subcollection, e.g. `"BP"`,
#'   `"MF"`, or `"CC"`.
#' @param enrichment_id_type Identifier column to use from MSigDB: `symbol`,
#'   `entrez`, or `ensembl`. The second column of each enrichment map should
#'   use the same identifier type.
#' @param enrichment_score Ranking score for fgsea/GSEA-style enrichment. The
#'   default `"auto"` uses -log10(union q-value) for union scope and signed
#'   -log10(component q-value) for component scope.
#' @param enrichment_term2gene Optional named list of TERM2GENE tables by dataset.
#' @param enrichment_extra_args Optional named list of additional arguments passed to backend enrichment functions; may be dataset-specific.
#' @param min_feature_overlap_warn Minimum feature overlap below which to warn about likely identifier mismatch; set NULL to disable count-based warning.
#' @param min_feature_overlap_fraction Minimum fraction of inferred features overlapping enrichment sets below which to warn; set NULL to disable fraction-based warning.
#' @param datasets Datasets to include. Defaults to all datasets in `Y`.
#' @param components Components to plot. If `NULL`, all available components are plotted.
#' @param train_fraction Fraction of samples used for training. Values in (0, 1) create a train/test split and run conditional inference; `train_fraction = 1` fits on all samples and skips inference outputs.
#' @param seed Optional seed for train/test splitting.
#' @param numThreads Number of threads used by the fitter.
#' @param save_inputs Logical; whether to save original input objects to disk.
#' @param save_split Logical; whether to save train/test split metadata.
#' @param verbose Logical; whether to print progress messages.
#' @param modelType Optional shared model-space specification for numeric variables in `xformula`.
#'   When `Y` is an `svdx_data` object, this defaults to `Y$modelType`.
#' @param ... Additional arguments passed to [runISVDXWorkflow()].
#'
#' @return Invisibly returns the `"isvdxWorkflow"` object.
#'
#' @examples
#' \dontrun{
#' iwf <- runISVDXReport(
#'   Y = Ylist,
#'   data = meta,
#'   zdata = zdata,
#'   xformula = ~ group * t + severity * t,
#'   zformula = zformula,
#'   outdir = "isvdx_report",
#'   K = 5,
#'   xaxis = "t",
#'   yaxis = "severity",
#'   group = "group",
#'   heatmap_order_by = c("group", "t"),
#'   heatmap_annotate = c("group")
#' )
#' }
#'
#' @export
runISVDXReport <- function(
    Y,
    data,
    zdata,
    xformula,
    zformula,
    outdir = "isvdx_report",
    overwrite = FALSE,
    K = 5,
    family = NULL,
    z_adjustment = c("fitted", "reestimate"),
    modelType = NULL,
    row_offset = NULL,
    col_offset = NULL,
    xaxis,
    yaxis = NULL,
    group = NULL,
    color = group,
    facet = NULL,
    heatmap_order_by = NULL,
    heatmap_annotate = NULL,
    loading_outputs = TRUE,
    inference_outputs = TRUE,
    loading_heatmaps = TRUE,
    loading_feature_profiles = TRUE,
    inference_scope = c("all", "component", "union", "global"),
    inference_volcanoes = TRUE,
    inference_heatmaps = TRUE,
    inference_feature_profiles = TRUE,
    inference_dotplots = TRUE,
    inference_enrichment = TRUE,
    alpha = 0.1,
    feature_profile_color = "feature",
    feature_profile_facet = NULL,
    features_top_n = 50L,
    feature_profiles_top_n = 5L,
    enrichment_top_n = 20L,
    feature_profile_summarize = c("mean", "median", "none"),
    feature_profile_show_se = TRUE,
    feature_profile_se_width = 0.15,
    feature_profile_scale = c("zscore", "raw"),
    volcano_label_features = NULL,
    volcano_labels_top_n = 25L,
    volcano_label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    volcano_use_feature_labels = TRUE,
    fdata = NULL,
    feature_sets = NULL,
    enrichment_id_map = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    enrichment_term2gene = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    datasets = NULL,
    components = NULL,
    train_fraction = 0.70,
    seed = 1,
    numThreads = 4,
    save_inputs = FALSE,
    save_split = TRUE,
    verbose = TRUE,
    ...
) {
  if (is_svdx_data(Y)) {
    prepared <- Y
    if (!isTRUE(prepared$integrated)) {
      stop("runISVDXReport() received single-dataset svdx_data; use runSVDXReport() instead.", call. = FALSE)
    }
    Y <- prepared$Y
    if (missing(data)) data <- prepared$xdata
    if (missing(zdata)) zdata <- prepared$zdata
    if (missing(xformula)) xformula <- prepared$xformula
    if (missing(zformula)) zformula <- prepared$zformula
    if (missing(family)) family <- prepared$family
    if (missing(fdata) || is.null(fdata)) fdata <- prepared$fdata
    if (missing(row_offset)) row_offset <- prepared$row_offset
    if (missing(col_offset)) col_offset <- prepared$col_offset
    if (is.null(modelType)) modelType <- prepared$modelType
  }
  if (missing(xaxis)) {
    stop("`xaxis` must be supplied for report U plots and feature profile plots.", call. = FALSE)
  }
  enrichment_backend <- match.arg(enrichment_backend)
  report_inference_scope <- match.arg(inference_scope)
  inference_scopes <- .report_expand_choice(report_inference_scope, c("component", "union", "global", "all"))
  z_adjustment <- match.arg(z_adjustment)
  enrichment_id_type <- match.arg(enrichment_id_type)
  feature_profile_summarize <- match.arg(feature_profile_summarize)
  feature_profile_scale <- match.arg(feature_profile_scale)
  volcano_label_by <- match.arg(volcano_label_by)
  heatmap_annotate <- .report_heatmap_annotate(heatmap_order_by, heatmap_annotate)
  compute_enrichment <- isTRUE(inference_outputs) && isTRUE(inference_enrichment) && (!is.null(enrichment_id_map) || !is.null(fdata))

  .report_validate_common_args(
    data = data,
    zdata = zdata,
    xaxis = xaxis,
    yaxis = yaxis,
    group = group,
    color = color,
    facet = facet,
    heatmap_order_by = heatmap_order_by,
    heatmap_annotate = heatmap_annotate,
    train_fraction = train_fraction,
    inference_enrichment = inference_enrichment,
    fdata = fdata,
    enrichment_id_map = enrichment_id_map,
    feature_sets = feature_sets,
    enrichment_term2gene = enrichment_term2gene,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    require_all_fdata_enrichment = FALSE
  )

  .report_validate_outdir_preflight(outdir, overwrite = overwrite)

  wf <- runISVDXWorkflow(
    Y = Y,
    data = data,
    zdata = zdata,
    xformula = xformula,
    zformula = zformula,
    K = K,
    family = family,
    z_adjustment = z_adjustment,
    modelType = modelType,
    row_offset = row_offset,
    col_offset = col_offset,
    train_fraction = train_fraction,
    seed = seed,
    fdata = fdata,
    feature_sets = feature_sets,
    inference_enrichment = inference_enrichment,
    alpha = alpha,
    enrichment_backend = enrichment_backend,
    inference_scope = report_inference_scope,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    enrichment_term2gene = enrichment_term2gene,
    enrichment_id_map = enrichment_id_map,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    numThreads = numThreads,
    return_design = TRUE,
    ...
  )
  .report_message("Workflow complete", verbose = verbose)

  outdir <- .report_prepare_outdir(outdir, overwrite = overwrite)
  .report_message("Output directory ready: ", outdir, verbose = verbose)
  
  plots_dir <- file.path(outdir, "plots")
  tables_dir <- file.path(outdir, "tables")
  u_dir <- file.path(plots_dir, "shared_U")
  dataset_root <- file.path(plots_dir, "datasets")
  
  .report_dir_create(plots_dir)
  .report_dir_create(tables_dir)
  .report_dir_create(u_dir)
  .report_dir_create(dataset_root)
  
  plot_files <- list()
  
  report_enrichment_id_map <- enrichment_id_map
  if (is.null(report_enrichment_id_map) && !is.null(wf$fdata)) {
    report_enrichment_id_map <- lapply(wf$fdata[wf$dataset_names %||% names(wf$Y)], .svdx_fdata_to_enrichment_id_map, required = FALSE)
  }

  workflow_file <- file.path(outdir, "isvdx_workflow.rds")
  summary_file <- file.path(outdir, "isvdx_workflow_summary.rds")
  
  .report_save_rds(wf, workflow_file)
  .report_message("Integrated workflow object saved", verbose = verbose)
  
  wf_summary <- tryCatch(
    summary(wf),
    error = function(e) {
      .report_message("Skipping integrated workflow summary: ", conditionMessage(e), verbose = verbose)
      NULL
    }
  )
  if (!is.null(wf_summary)) {
    .report_save_rds(wf_summary, summary_file)
    .report_message("Integrated workflow summary saved", verbose = verbose)
  }

  enrichment_notes_file <- NULL
  if (!is.null(wf$enrichment_notes) && length(wf$enrichment_notes) > 0L) {
    enrichment_notes_file <- file.path(outdir, "enrichment_notes.txt")
    notes <- unlist(wf$enrichment_notes, use.names = TRUE)
    notes <- paste0(names(notes), ": ", unname(notes))
    writeLines(notes, con = enrichment_notes_file)
    .report_message("Enrichment notes saved", verbose = verbose)
  }
  
  split_file <- NULL
  if (isTRUE(save_split)) {
    split_file <- file.path(outdir, "train_test_split.rds")
    .report_save_split(wf, split_file)
    .report_save_sample_split_tsv(wf, file.path(tables_dir, "sample_split.tsv"))
    .report_message("Train/test split saved", verbose = verbose)
  }
  
  inputs_file <- NULL
  if (isTRUE(save_inputs)) {
    inputs_file <- file.path(outdir, "report_inputs.rds")
    .report_save_rds(
      list(Y = Y, data = data, zdata = zdata),
      inputs_file
    )
    .report_message("Input objects saved", verbose = verbose)
  }
  
  if (is.null(datasets)) {
    datasets <- wf$dataset_names %||% names(wf$Y)
  }
  
  comp_info <- .report_resolve_components(wf, components)
  components <- comp_info$components
  n_components <- comp_info$n_components
  
  .report_message("Detected ", n_components, " shared component(s)", verbose = verbose)
  inference_available <- !is.null(wf$inf) && !is.null(wf$inf$dataset_results)
  if (isTRUE(inference_outputs) && !isTRUE(inference_available)) {
    .report_message(
      "Skipping inference and enrichment sections: inference results unavailable.",
      verbose = verbose
    )
  }
  inference_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_available)) inference_scopes else character(0)
  loading_heatmap_scopes <- if (isTRUE(loading_outputs) && isTRUE(loading_heatmaps)) c("union", "component") else character(0)
  inference_heatmap_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_heatmaps) && isTRUE(inference_available)) inference_scopes else character(0)
  volcano_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_volcanoes) && isTRUE(inference_available)) intersect(c("union", "global"), inference_scopes) else character(0)
  enrichment_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_enrichment) && isTRUE(inference_available)) inference_scopes else character(0)

  if (isTRUE(inference_available)) {
    for (ds in datasets) {
      if (!ds %in% names(wf$inf$dataset_results)) next
      ds_tables_dir <- file.path(tables_dir, "datasets", ds)
      fd_ds <- .svdx_resolve_fdata_for_dataset(wf$fdata, dataset = ds, required = FALSE)
      .report_export_inference_tables(
        wf$inf$dataset_results[[ds]],
        fdata = fd_ds,
        tables_dir = ds_tables_dir,
        components = components,
        alpha = wf$alpha
      )
    }
  }
  
  # ----------------------------------------------------------
  # Shared U plots
  # ----------------------------------------------------------
  for (component in components) {
    lab <- .report_component_label(component)
    
    
    f <- file.path(u_dir, paste0("shared_U_", lab, "_1D.pdf"))
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u",
        component = component,
        xaxis = xaxis,
        group = group,
        facet = facet,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
    
    f <- file.path(u_dir, paste0("shared_U_terms_", lab, "_1D.pdf"))
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u_terms",
        component = component,
        xaxis = xaxis,
        group = group,
        facet = facet,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
    
    if (!is.null(yaxis)) {
      f <- file.path(u_dir, paste0("shared_U_", lab, "_2D.pdf"))
      plot_files[[f]] <- .report_save_plot_safe(
        plotU(
          wf$fit,
          data = wf$data,
          component = component,
          x = xaxis,
          y = yaxis,
          group = group,
          facet = facet,
          fill = "typical"
        ),
        file = f,
        width = 9,
        height = 8,
        verbose = verbose
      )
    }
  }
  
  if (n_components >= 2) {
    
    f <- file.path(u_dir, "shared_U_embedding_components_1_2.pdf")
    plot_files[[f]] <- .report_save_plot_safe(
      plot(
        wf,
        type = "u_embedding",
        components = c(1, 2),
        color = color,
        group = group,
        fill = "typical"
      ),
      file = f,
      verbose = verbose
    )
  }
  
  # ----------------------------------------------------------
  # Dataset-specific V and inference plots
  # ----------------------------------------------------------
  for (ds in datasets) {
    
    ds_dir <- file.path(dataset_root, ds)
    ds_tables_dir <- file.path(tables_dir, "datasets", ds)
    .report_dir_create(ds_tables_dir)
    fd_ds <- .svdx_resolve_fdata_for_dataset(wf$fdata, dataset = ds, required = FALSE)
    enrichment_component_tables <- list()
    ds_v_dir <- file.path(ds_dir, "V_loadings")
    ds_inf_dir <- file.path(ds_dir, "inference")
    
    .report_dir_create(ds_dir)
    .report_dir_create(ds_v_dir)
    .report_dir_create(ds_inf_dir)
    
    # V loading plots and loading heatmaps for all components
    if (isTRUE(loading_outputs) && "union" %in% loading_heatmap_scopes) {
      f <- file.path(ds_v_dir, "V_loading_heatmap_union.pdf")
      plot_files[[f]] <- .report_save_plot_safe(
        plot(
          wf,
          type = "v_heatmap",
          dataset = ds,
          features = .report_loading_union_features(wf$fit, components, n_features = features_top_n, dataset = ds),
          order_by = heatmap_order_by,
          annotate = heatmap_annotate
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }

    for (component in components) {
      lab <- .report_component_label(component)
      
      
      f <- file.path(ds_v_dir, paste0("V_loadings_", lab, ".pdf"))
      plot_files[[f]] <- .report_save_plot_safe(
        plot(
          wf,
          type = "v_loadings",
          dataset = ds,
          component = component,
          n = 30
        ),
        file = f,
        verbose = verbose
      )
      
      if (isTRUE(loading_outputs) && "component" %in% loading_heatmap_scopes) {
        f <- file.path(ds_v_dir, paste0("V_loading_heatmap_", lab, ".pdf"))
        plot_files[[f]] <- .report_save_plot_safe(
          plot(
            wf,
            type = "v_heatmap",
            dataset = ds,
            component = component,
            n_features = features_top_n,
            order_by = heatmap_order_by,
            annotate = heatmap_annotate
          ),
          file = f,
          width = 10,
          height = 9,
          verbose = verbose
        )
      }
    }
    
    if (.report_inference_available(wf, dataset = ds) && isTRUE(inference_outputs) && isTRUE(inference_dotplots) && "union" %in% inference_scopes) {
      f <- file.path(ds_inf_dir, "inference_union_dotplot.pdf")
      plot_files[[f]] <- .report_save_plot_safe(
        plot(
          wf,
          type = "v_inference_dotplot",
          dataset = ds,
          n_features = features_top_n,
          inference_scope = "union",
          select_by = "q",
          fdr = wf$alpha
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }
    if (.report_inference_available(wf, dataset = ds) && isTRUE(inference_outputs) && isTRUE(inference_dotplots) && "global" %in% inference_scopes) {
      f <- file.path(ds_inf_dir, "inference_global_dotplot.pdf")
      plot_files[[f]] <- .report_save_plot_safe(
        plot(
          wf,
          type = "v_inference_dotplot",
          dataset = ds,
          n_features = features_top_n,
          inference_scope = "global",
          select_by = "q",
          fdr = wf$alpha
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }
    
    if (!.report_inference_available(wf, dataset = ds)) {
      .report_message(
        "Skipping inference plots for dataset '",
        ds,
        "': inference results unavailable.",
        verbose = verbose
      )
      next
    }
    inf_d <- wf$inf$dataset_results[[ds]]
    
    # Union inference plots
    
    if ("union" %in% inference_heatmap_scopes) {
      f <- file.path(ds_inf_dir, "inference_union_heatmap.pdf")
      plot_files[[f]] <- .report_save_test_plot_safe(
        wf,
        dataset = ds,
        expr = plot(
          wf,
          type = "v_heatmap",
          dataset = ds,
          source = "test",
          n_features = features_top_n,
          inference_scope = "union",
          select_by = "q",
          fdr = wf$alpha,
          order_by = heatmap_order_by,
          annotate = heatmap_annotate
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }
    if ("global" %in% inference_heatmap_scopes) {
      f <- file.path(ds_inf_dir, "inference_global_heatmap.pdf")
      plot_files[[f]] <- .report_save_test_plot_safe(
        wf,
        dataset = ds,
        expr = plot(
          wf,
          type = "v_heatmap",
          dataset = ds,
          source = "test",
          n_features = features_top_n,
          inference_scope = "global",
          select_by = "q",
          fdr = wf$alpha,
          order_by = heatmap_order_by,
          annotate = heatmap_annotate
        ),
        file = f,
        width = 10,
        height = 9,
        verbose = verbose
      )
    }
    
    if ("union" %in% volcano_scopes) {
      f <- file.path(ds_inf_dir, "inference_union_volcano.pdf")
      plot_files[[f]] <- .report_save_plot_safe(
        plot_union_volcano(
          inf_d,
          scope = "union",
          fdata = wf$fdata,
          dataset = ds,
          label_features = volcano_label_features,
          label_top_n = volcano_labels_top_n,
          label_by = volcano_label_by,
          use_feature_labels = volcano_use_feature_labels
        ),
        file = f,
        verbose = verbose
      )
    }

    if ("global" %in% volcano_scopes) {
      f <- file.path(ds_inf_dir, "inference_global_volcano.pdf")
      plot_files[[f]] <- .report_save_plot_safe(
        plot_union_volcano(
          inf_d,
          scope = "global",
          fdata = wf$fdata,
          dataset = ds,
          label_features = volcano_label_features,
          label_top_n = volcano_labels_top_n,
          label_by = volcano_label_by,
          use_feature_labels = volcano_use_feature_labels
        ),
        file = f,
        verbose = verbose
      )
    }
    
    ds_enrichment_scope <- if (isTRUE(inference_outputs) && isTRUE(inference_enrichment)) inference_scopes else character(0)
    if (.report_dataset_inference_enrichment(compute_enrichment, ds, wf$dataset_names %||% names(wf$Y)) && .report_enrichment_include_union(ds_enrichment_scope)) {
      enr_union <- .report_safe_integrated_enrichment(
        wf,
        dataset = ds,
        dataset_names = wf$dataset_names %||% names(wf$Y),
        feature_sets = feature_sets,
        alpha = wf$alpha,
        scope = "union",
        enrichment_backend = enrichment_backend,
        msigdb_db_species = msigdb_db_species,
        msigdb_species = msigdb_species,
        msigdb_collection = msigdb_collection,
        msigdb_subcollection = msigdb_subcollection,
        enrichment_id_type = enrichment_id_type,
        enrichment_score = enrichment_score,
        enrichment_term2gene = enrichment_term2gene,
        enrichment_id_map = report_enrichment_id_map,
        enrichment_extra_args = enrichment_extra_args,
        min_feature_overlap_warn = min_feature_overlap_warn,
        min_feature_overlap_fraction = min_feature_overlap_fraction
      )
      
      if (!is.null(enr_union)) {
        .report_save_rds(enr_union, file.path(ds_inf_dir, "inference_union_enrichment.rds"))
        
        f <- file.path(ds_inf_dir, "inference_union_enrichment.pdf")
        plot_files[[f]] <- .report_save_plot_safe(
          plot_enrichment_dotplot(enr_union, top_n = enrichment_top_n),
          file = f,
          verbose = verbose
        )
      }
    }
    
    if (.report_dataset_inference_enrichment(compute_enrichment, ds, wf$dataset_names %||% names(wf$Y)) && .report_enrichment_include_global(ds_enrichment_scope)) {
      enr_global <- .report_safe_integrated_enrichment(
        wf,
        dataset = ds,
        dataset_names = wf$dataset_names %||% names(wf$Y),
        feature_sets = feature_sets,
        alpha = wf$alpha,
        scope = "global",
        enrichment_backend = enrichment_backend,
        msigdb_db_species = msigdb_db_species,
        msigdb_species = msigdb_species,
        msigdb_collection = msigdb_collection,
        msigdb_subcollection = msigdb_subcollection,
        enrichment_id_type = enrichment_id_type,
        enrichment_score = enrichment_score,
        enrichment_term2gene = enrichment_term2gene,
        enrichment_id_map = report_enrichment_id_map,
        enrichment_extra_args = enrichment_extra_args,
        min_feature_overlap_warn = min_feature_overlap_warn,
        min_feature_overlap_fraction = min_feature_overlap_fraction
      )
      
      if (!is.null(enr_global)) {
        .report_save_rds(enr_global, file.path(ds_inf_dir, "inference_global_enrichment.rds"))
        enr_global_tbl <- .report_flatten_enrichment(enr_global, fdata = fd_ds)
        .report_save_tsv(enr_global_tbl, file.path(ds_tables_dir, "enrichment_global.tsv"))
        
        f <- file.path(ds_inf_dir, "inference_global_enrichment.pdf")
        plot_files[[f]] <- .report_save_plot_safe(
          plot_enrichment_dotplot(enr_global, top_n = enrichment_top_n),
          file = f,
          verbose = verbose
        )
      }
    }
    
    # Component-wise inference plots
    if ("component" %in% inference_scopes) {
      for (component in components) {
        lab <- .report_component_label(component)
        comp_dir <- file.path(ds_inf_dir, lab)
        .report_dir_create(comp_dir)
        
        #comp_name <- colnames(inf_d$coef)[component] %||% component
        comp_name <- tryCatch(
          .report_get_inference_component_name(inf_d, component),
          error = function(e) {
            .report_message(
              "Skipping component-wise inference plots for dataset ",
              ds,
              ", ",
              lab,
              ": ",
              conditionMessage(e),
              verbose = verbose
            )
            NULL
          }
        )
        if (is.null(comp_name)) next
        
        if ("component" %in% inference_heatmap_scopes) {
          f <- file.path(comp_dir, paste0("inference_heatmap_", lab, ".pdf"))
          plot_files[[f]] <- .report_save_test_plot_safe(
            wf,
            dataset = ds,
            expr = plot(
              wf,
              type = "v_heatmap",
              dataset = ds,
              source = "test",
              component = comp_name,
              n_features = features_top_n,
              inference_scope = "component",
              select_by = "q",
              fdr = wf$alpha,
              order_by = heatmap_order_by,
              annotate = heatmap_annotate
            ),
            file = f,
            width = 10,
            height = 9,
            verbose = verbose
          )
        }
        
        if (isTRUE(inference_dotplots)) {
          f <- file.path(comp_dir, paste0("inference_dotplot_", lab, ".pdf"))
          plot_files[[f]] <- .report_save_plot_safe(
            plot(
              wf,
              type = "v_inference_dotplot",
              dataset = ds,
              inference_scope = "component",
              component = comp_name,
              n_features = features_top_n,
              select_by = "q",
              fdr = wf$alpha
            ),
            file = f,
            width = 10,
            height = 9,
            verbose = verbose
          )
        }
        
        if (isTRUE(inference_volcanoes)) {
          f <- file.path(comp_dir, paste0("inference_volcano_", lab, ".pdf"))
          plot_files[[f]] <- .report_save_plot_safe(
            plot_component_volcano(
              inf_d,
              component = comp_name,
              fdata = wf$fdata,
              dataset = ds,
              label_features = volcano_label_features,
              label_top_n = volcano_labels_top_n,
              label_by = volcano_label_by,
              use_feature_labels = volcano_use_feature_labels
            ),
            file = f,
            verbose = verbose
          )
        }
        
        if (.report_dataset_inference_enrichment(compute_enrichment, ds, wf$dataset_names %||% names(wf$Y)) && component %in% .report_enrichment_components(ds_enrichment_scope, components)) {
          enr_comp <- .report_safe_integrated_enrichment(
            wf,
            dataset = ds,
            dataset_names = wf$dataset_names %||% names(wf$Y),
            feature_sets = feature_sets,
            alpha = wf$alpha,
            scope = "component",
            component = comp_name,
            enrichment_backend = enrichment_backend,
            msigdb_db_species = msigdb_db_species,
            msigdb_species = msigdb_species,
            msigdb_collection = msigdb_collection,
            msigdb_subcollection = msigdb_subcollection,
            enrichment_id_type = enrichment_id_type,
            enrichment_score = enrichment_score,
            enrichment_term2gene = enrichment_term2gene,
            enrichment_id_map = report_enrichment_id_map,
            enrichment_extra_args = enrichment_extra_args,
            min_feature_overlap_warn = min_feature_overlap_warn,
            min_feature_overlap_fraction = min_feature_overlap_fraction
          )
          
          if (!is.null(enr_comp)) {
            enr_file <- file.path(comp_dir, paste0("inference_enrichment_", lab, ".rds"))
            .report_save_rds(enr_comp, enr_file)
            enr_tbl <- .report_flatten_enrichment(enr_comp, fdata = fd_ds)
            enrichment_component_tables[[lab]] <- enr_tbl
            .report_save_tsv(enr_tbl, file.path(ds_tables_dir, paste0("enrichment_", lab, ".tsv")))
            
            f <- file.path(comp_dir, paste0("inference_enrichment_", lab, ".pdf"))
            plot_files[[f]] <- .report_save_plot_safe(
              plot_enrichment_dotplot(enr_comp, top_n = enrichment_top_n),
              file = f,
              verbose = verbose
            )
          }
        }
      }
    }

    if (length(enrichment_component_tables)) {
      enrichment_all <- do.call(rbind, enrichment_component_tables)
      rownames(enrichment_all) <- NULL
      .report_save_tsv(enrichment_all, file.path(ds_tables_dir, "enrichment_allComponents.tsv"))
    }
  }
  

  # ----------------------------------------------------------
  # Dataset-specific feature profile plots
  # ----------------------------------------------------------
  profile_by <- c(if (isTRUE(loading_outputs) && isTRUE(loading_feature_profiles)) "loading", if (isTRUE(inference_outputs) && isTRUE(inference_feature_profiles)) "inference")
  loading_profile_scopes <- if (isTRUE(loading_outputs) && isTRUE(loading_feature_profiles)) c("union", "component") else character(0)
  inference_profile_scopes <- if (isTRUE(inference_outputs) && isTRUE(inference_feature_profiles)) inference_scopes else character(0)

  for (ds in datasets) {
    ds_prof_dir <- file.path(dataset_root, ds, "feature_profiles")
    .report_dir_create(ds_prof_dir)
    for (sel in profile_by) {
      if ((sel == "loading" && "union" %in% loading_profile_scopes) || (sel == "inference" && "union" %in% inference_profile_scopes)) {
        f <- file.path(ds_prof_dir, paste0("feature_profiles_", sel, "_union.pdf"))
        plot_files[[f]] <- .report_profile_plot(
          wf,
          file = f,
          selection_by = sel,
          scope = "union",
          components = components,
          dataset = ds,
          xaxis = xaxis,
          color = feature_profile_color,
          facet = feature_profile_facet,
          summarize = feature_profile_summarize,
          show_se = feature_profile_show_se,
          se_width = feature_profile_se_width,
          scale = feature_profile_scale,
          top_n = feature_profiles_top_n,
          verbose = verbose
        )
      }
      if (sel == "inference" && "global" %in% inference_profile_scopes) {
        f <- file.path(ds_prof_dir, paste0("feature_profiles_", sel, "_global.pdf"))
        plot_files[[f]] <- .report_profile_plot(
          wf,
          file = f,
          selection_by = sel,
          scope = "global",
          components = components,
          dataset = ds,
          xaxis = xaxis,
          color = feature_profile_color,
          facet = feature_profile_facet,
          summarize = feature_profile_summarize,
          show_se = feature_profile_show_se,
          se_width = feature_profile_se_width,
          scale = feature_profile_scale,
          top_n = feature_profiles_top_n,
          verbose = verbose
        )
      }
      if ((sel == "loading" && "component" %in% loading_profile_scopes) || (sel == "inference" && "component" %in% inference_profile_scopes)) {
        for (component in components) {
          lab <- .report_component_label(component)
          f <- file.path(ds_prof_dir, paste0("feature_profiles_", sel, "_", lab, ".pdf"))
          plot_files[[f]] <- .report_profile_plot(
            wf,
            file = f,
            selection_by = sel,
            scope = "component",
            component = component,
            components = components,
            dataset = ds,
            xaxis = xaxis,
            color = feature_profile_color,
            facet = feature_profile_facet,
            summarize = feature_profile_summarize,
            scale = feature_profile_scale,
            top_n = feature_profiles_top_n,
            verbose = verbose
          )
        }
      }
    }
  }

  manifest <- list(
    report_type = "isvdx",
    workflow_file = workflow_file,
    summary_file = summary_file,
    enrichment_notes_file = enrichment_notes_file,
    split_file = split_file,
    inputs_file = inputs_file,
    outdir = outdir,
    plots_dir = plots_dir,
    plot_files = Filter(Negate(is.null), plot_files),
    datasets = datasets,
    n_components = n_components,
    components = components,
    call = match.call()
  )
  
  manifest_file <- file.path(outdir, "report_manifest.rds")
  .report_save_rds(manifest, manifest_file)
  .report_message("Report manifest saved", verbose = verbose)
  
  .report_message("Integrated report complete: ", outdir, verbose = verbose)
  
  invisible(wf)
}
