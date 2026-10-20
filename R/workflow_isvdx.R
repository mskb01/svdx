# ============================================================
# 6C. Real-world integrated workflow API
# ============================================================



.isvdx_dataset_enrichment_input <- function(x, dataset, dataset_names = NULL, arg = "input", list_kind = c("shared_list", "dataset_list")) {
  list_kind <- match.arg(list_kind)
  if (is.null(x)) return(NULL)

  # Data frames are ordinary shared objects, not dataset-parallel lists.
  if (!is.list(x) || is.data.frame(x)) return(x)

  nms <- names(x)
  if (!is.null(nms) && dataset %in% nms) {
    return(x[[dataset]])
  }

  # Some enrichment inputs are naturally lists (for example feature_sets is a
  # list of gene sets and enrichment_extra_args is a named list of backend
  # arguments).  For these objects, an unnamed list is treated as a shared
  # object, never guessed to be dataset-specific.  Dataset-specific versions
  # must be named by dataset.
  if (identical(list_kind, "shared_list")) {
    return(x)
  }

  # Other inputs, such as TERM2GENE, should only be lists in integrated runs
  # when they are explicitly dataset-specific.  Otherwise the intent is
  # ambiguous and we fail early with a useful message.
  if (is.null(nms) || any(!nzchar(nms))) {
    stop(
      "For integrated enrichment, `", arg, "` was supplied as an unnamed list. ",
      "Use a data frame/shared object, or provide a named list keyed by dataset.",
      call. = FALSE
    )
  }

  if (!is.null(dataset_names) && !all(dataset_names %in% nms)) {
    stop(
      "For integrated enrichment, named list `", arg, "` must contain all dataset names: ",
      paste(dataset_names, collapse = ", "), ".",
      call. = FALSE
    )
  }

  NULL
}

.isvdx_default_dataset_names <- function(n) paste0("d", seq_len(n))

.isvdx_dataset_names_from_y <- function(Y) {
  nms <- names(Y)
  if (is.null(nms) || length(nms) != length(Y) || any(!nzchar(nms))) {
    nms <- .isvdx_default_dataset_names(length(Y))
  }
  nms
}

.isvdx_recycle_dataset_option <- function(x, dataset_names, arg = "option", default = NULL) {
  if (is.null(x)) x <- default
  D <- length(dataset_names)

  if (is.list(x) && !is.data.frame(x)) {
    if (length(x) == 0L) {
      out <- replicate(D, x, simplify = FALSE)
      names(out) <- dataset_names
      return(out)
    }

    nms <- names(x)
    if (!is.null(nms) && all(dataset_names %in% nms)) {
      out <- x[dataset_names]
      names(out) <- dataset_names
      return(out)
    }

    if (length(x) == D) {
      names(x) <- dataset_names
      return(x)
    }
  }

  out <- rep(list(x), D)
  names(out) <- dataset_names
  out
}

.isvdx_recycle_shared_list_option <- function(x, dataset_names, default = list()) {
  if (is.null(x)) x <- default
  D <- length(dataset_names)

  if (is.list(x) && !is.data.frame(x)) {
    nms <- names(x)
    if (!is.null(nms) && all(dataset_names %in% nms)) {
      out <- x[dataset_names]
      names(out) <- dataset_names
      return(out)
    }
  }

  out <- rep(list(x), D)
  names(out) <- dataset_names
  out
}

.isvdx_resolve_enrichment_options <- function(
    dataset_names,
    inference_enrichment,
    enrichment_backend,
    inference_scope,
    msigdb_db_species,
    msigdb_species,
    msigdb_collection,
    msigdb_subcollection,
    enrichment_id_type,
    enrichment_score,
    enrichment_extra_args,
    min_feature_overlap_warn,
    min_feature_overlap_fraction
) {
  valid_backend <- c("auto", "internal", "clusterProfiler", "fgsea")
  valid_scope <- c("all", "union", "component", "global")
  valid_id <- c("symbol", "entrez", "ensembl")

  out <- list(
    inference_enrichment = .isvdx_recycle_dataset_option(inference_enrichment, dataset_names, "inference_enrichment", default = FALSE),
    enrichment_backend = .isvdx_recycle_dataset_option(enrichment_backend, dataset_names, "enrichment_backend"),
    inference_scope = .isvdx_recycle_dataset_option(inference_scope, dataset_names, "inference_scope"),
    msigdb_db_species = .isvdx_recycle_dataset_option(msigdb_db_species, dataset_names, "msigdb_db_species"),
    msigdb_species = .isvdx_recycle_dataset_option(msigdb_species, dataset_names, "msigdb_species"),
    msigdb_collection = .isvdx_recycle_dataset_option(msigdb_collection, dataset_names, "msigdb_collection"),
    msigdb_subcollection = .isvdx_recycle_dataset_option(msigdb_subcollection, dataset_names, "msigdb_subcollection"),
    enrichment_id_type = .isvdx_recycle_dataset_option(enrichment_id_type, dataset_names, "enrichment_id_type"),
    enrichment_score = .isvdx_recycle_dataset_option(enrichment_score, dataset_names, "enrichment_score"),
    enrichment_extra_args = .isvdx_recycle_shared_list_option(enrichment_extra_args, dataset_names, default = list()),
    min_feature_overlap_warn = .isvdx_recycle_dataset_option(min_feature_overlap_warn, dataset_names, "min_feature_overlap_warn"),
    min_feature_overlap_fraction = .isvdx_recycle_dataset_option(min_feature_overlap_fraction, dataset_names, "min_feature_overlap_fraction")
  )

  for (ds in dataset_names) {
    re <- out$inference_enrichment[[ds]]
    if (!is.logical(re) || length(re) != 1L || is.na(re)) {
      stop("`inference_enrichment` must be TRUE/FALSE or a list of TRUE/FALSE values parallel to Y.", call. = FALSE)
    }

    be <- out$enrichment_backend[[ds]]
    if (!is.character(be) || length(be) != 1L || !be %in% valid_backend) {
      stop("Invalid `enrichment_backend` for dataset '", ds, "'.", call. = FALSE)
    }

    sc <- out$inference_scope[[ds]]
    if (!is.character(sc) || length(sc) != 1L || !sc %in% valid_scope) {
      stop("Invalid `inference_scope` for dataset '", ds, "'.", call. = FALSE)
    }
    id <- out$enrichment_id_type[[ds]]
    if (!is.character(id) || length(id) != 1L || !id %in% valid_id) {
      stop("Invalid `enrichment_id_type` for dataset '", ds, "'.", call. = FALSE)
    }
  }

  out
}

#' Run a complete integrated SSANOVA-SVD workflow
#'
#' Fits an integrated SSANOVA-SVD model, summarizes the fitted training model,
#' performs dataset-specific conditional inference on held-out samples,
#' summarizes the inference results, and optionally runs dataset-specific
#' pathway enrichment.
#'
#' @param Y Named or unnamed list of feature-by-sample matrices. All datasets
#'   must share the same samples/columns.
#' @param data Shared sample metadata data frame.
#' @param zdata List of nuisance metadata data frames, one per dataset.
#' @param xformula Formula for shared metadata-driven component scores.
#' @param zformula List of nuisance formulas, one per dataset.
#' @param train_cols Optional training sample indices, names, or logical vector.
#' @param test_cols Optional test sample indices, names, or logical vector.
#' @param train_fraction Fraction of samples used for training when explicit
#'   train/test columns are not supplied. Values in (0, 1) create a train/test
#'   split and run conditional inference on held-out samples; `train_fraction = 1`
#'   fits on all samples and skips test-sample inference.
#' @param seed Optional random seed for train/test splitting.
#' @param K Maximum number of components to fit.
#' @param family Optional response family specification. Use `NULL` for the
#'   default Gaussian/non-family fitting path, a scalar family recycled to all
#'   datasets, an unnamed vector/list parallel to `Y`, or a named vector/list
#'   matched by dataset name. Entries may be `"gaussian"`, `"poisson"`, or
#'   `"quasipoisson"`; quasi-Poisson requests are fit with the Poisson objective
#'   and inferred with quasi-Poisson GLMs.
#' @param graphData Optional graph data used by graph-aware model components. Must 
#'   have dimensions graphFeatures x samples. This will be used with graphDistFn 
#'   to calculate graph based distance matrices. 
#' @param graphDistFn Optional function for graph distances. Must return a positive
#'   semi-definite / definite matrix. 
#' @param numThreads Number of threads for fitting.
#' @param row_offset Optional feature-level offsets used during inference.
#' @param col_offset Optional sample-level offsets, either a vector or list.
#' @param alpha Significance threshold used in inference summaries.
#' @param z_adjustment Nuisance-adjustment mode for inference: `"fitted"` uses SVDX-estimated nuisance effects; `"reestimate"` re-estimates nuisance effects inside `inferV()`.
#' @param feature_sets Optional feature sets for enrichment. For integrated
#'   workflows this may be a named list by dataset. If NULL and
#'   `inference_enrichment=TRUE`, feature sets are obtained from MSigDB for each
#'   dataset using the explicit MSigDB arguments.
#' @param fdata Optional named list of `svdx_fdata` objects, one per dataset.
#'   Stored in the workflow and reused by plots, reports, and enrichment. If
#'   enrichment is requested and `enrichment_id_map` is NULL, per-dataset
#'   enrichment maps are derived from `fdata`.
#' @param enrichment_id_map Required when `inference_enrichment=TRUE` unless `fdata`
#'   supplies enrichment columns. For ISVDX this
#'   must be a named list with one two-column data frame per dataset. In each
#'   data frame, the first column contains SVDX feature IDs matching
#'   `rownames(Y[[dataset]])`; the second column contains enrichment IDs used by
#'   the pathway database. Column names are ignored and are normalized
#'   internally.
#' @param inference_enrichment Logical; whether to run inference enrichment. Defaults to TRUE when enrichment identifiers are available.
#'   when `enrichment_id_map` is supplied.
#' @param enrichment_backend Enrichment backend: `"auto"`, `"internal"`,
#'   `"clusterProfiler"`, or `"fgsea"`. GSEA-style backends require unique
#'   mapped enrichment IDs within each dataset.
#' @param inference_scope Inference scope used for enrichment: `"all"`, `"union"`, `"component"`, or `"global"`. Defaults to `"all"`, which runs union-, global-,
#'   and component-level enrichment for each dataset. Component scope enriches
#'   all inferred components.
#' @param msigdb_db_species MSigDB database species passed to
#'   `msigdbr::msigdbr(db_species=...)`, for example `"HS"` or `"MM"`.
#' @param msigdb_species Species passed to `msigdbr::msigdbr(species=...)`, for
#'   example `"Homo sapiens"` or `"Mus musculus"`.
#' @param msigdb_collection MSigDB collection passed to
#'   `msigdbr::msigdbr(collection=...)`, for example `"H"`, `"C5"`, `"MH"`, or
#'   `"M5"`.
#' @param msigdb_subcollection Optional MSigDB subcollection passed to
#'   `msigdbr::msigdbr(subcollection=...)`.
#' @param enrichment_id_type Identifier column to extract from MSigDB when
#'   `feature_sets`/`enrichment_term2gene` are not supplied: `"symbol"`,
#'   `"entrez"`, or `"ensembl"`.
#' @param enrichment_score Ranking score for GSEA-style enrichment. The default
#'   `"auto"` uses union/component q-value based scores.
#' @param enrichment_term2gene Optional TERM2GENE table, or named list by dataset
#'   for integrated workflows. If NULL, it is built from MSigDB when needed.
#' @param enrichment_extra_args Optional named list of additional arguments
#'   passed to backend enrichment functions. For ISVDX this may also be a named
#'   list by dataset.
#' @param min_feature_overlap_warn Minimum feature-set overlap below which to
#'   warn about likely identifier mismatch; set NULL to disable count-based
#'   warning.
#' @param min_feature_overlap_fraction Minimum fraction of inferred enrichment
#'   IDs overlapping enrichment sets below which to warn; set NULL to disable
#'   fraction-based warning.
#' @param return_conf_distribution Logical; whether training summaries should
#'   include full x/z confounding cosine distributions.
#' @param return_design Logical; whether to retain design matrices and train/test
#'   data inside the workflow object.
#' @param verbose Logical; whether to print progress from feature-wise inference.
#' @param modelType Optional shared model-space specification for numeric variables in `xformula`.
#'   When `Y` is an `svdx_data` object, this defaults to `Y$modelType`.
#' @param ... Additional arguments passed to [isvdx()].
#'
#' @return An object of class `"isvdxWorkflow"` containing train/test split
#'   information, the fitted integrated model, training summary, integrated
#'   inference result, inference summary, optional enrichment results, and
#'   retained input objects. When enrichment is run, `iwf$enrichment` stores
#'   dataset-wise normalized results in `results[[dataset]]`, raw backend output
#'   in `raw[[dataset]]`, and resolved backends in
#'   `backend_resolved[[dataset]]`.
#'
#' @examples
#' \dontrun{
#' Y <- list(
#'   rna = Y_rna,
#'   protein = Y_protein
#' )
#'
#' zformula <- list(
#'   rna = ~ batch,
#'   protein = ~ plate
#' )
#'
#' zdata <- list(
#'   rna = rna_meta,
#'   protein = protein_meta
#' )
#'
#' iwf <- runISVDXWorkflow(
#'   Y = Y,
#'   data = sample_data,
#'   zdata = zdata,
#'   xformula = ~ group * time,
#'   zformula = zformula,
#'   K = 5,
#'   train_fraction = 0.7,
#'   seed = 1,
#'   inference_enrichment = TRUE,
#'   enrichment_backend = "internal",
#'   enrichment_id_map = list(
#'     rna = data.frame(feature = rownames(Y_rna), gene_symbol = rna_symbols),
#'     protein = data.frame(feature = rownames(Y_protein), gene_symbol = protein_symbols)
#'   ),
#'   enrichment_id_type = "symbol",
#'   msigdb_db_species = "HS",
#'   msigdb_species = "Homo sapiens",
#'   msigdb_collection = "H"
#' )
#'
#' summary(iwf)
#' plot(iwf, type = "u", component = 1, xaxis = "time", group = "group")
#' plot(iwf, type = "v_heatmap", dataset = "rna", component = 1)
#' topFeatures(iwf, dataset = "rna")
#' }
#'
#' @export
runISVDXWorkflow <- function(
    Y,
    data,
    zdata,
    xformula,
    zformula,
    train_cols = NULL,
    test_cols = NULL,
    train_fraction = 0.70,
    seed = NULL,
    K = 5,
    family = NULL,
    modelType = NULL,
    graphData = NULL,
    graphDistFn = NULL,
    numThreads = 1,
    row_offset = NULL,
    col_offset = NULL,
    alpha = 0.05,
    z_adjustment = c("fitted", "reestimate"),
    feature_sets = NULL,
    fdata = NULL,
    enrichment_id_map = NULL,
    inference_enrichment = !is.null(enrichment_id_map),
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    inference_scope = c("all", "union", "component", "global"),
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
    return_conf_distribution = FALSE,
    return_design = TRUE,
    verbose = FALSE,
    ...
) {
  if (is_svdx_data(Y)) {
    prepared <- Y
    if (!isTRUE(prepared$integrated)) {
      stop("runISVDXWorkflow() received single-dataset svdx_data; use runSVDXWorkflow() instead.", call. = FALSE)
    }
    Y <- prepared$Y
    if (missing(data)) data <- prepared$xdata
    if (missing(zdata)) zdata <- prepared$zdata
    if (missing(xformula)) xformula <- prepared$xformula
    if (missing(zformula)) zformula <- prepared$zformula
    if (missing(family)) family <- prepared$family
    prepared_fit_family <- prepared$fit_family
    prepared_inference_family <- prepared$inference_family
    prepared_fit_description <- prepared$fit_description
    if (missing(row_offset)) row_offset <- prepared$row_offset
    if (missing(col_offset)) col_offset <- prepared$col_offset
    if (missing(fdata) || is.null(fdata)) fdata <- prepared$fdata
    if (is.null(modelType)) modelType <- prepared$modelType
  }

  validated <- .validate_isvdx_workflow_inputs(
    Y = Y,
    data = data,
    zdata = zdata,
    zformula = zformula,
    family = family,
    graphData = graphData,
    row_offset = row_offset,
    col_offset = col_offset
  )
  
  Y <- validated$Y
  data <- validated$data
  zdata <- validated$zdata
  zformula <- validated$zformula
  family <- validated$family
  graphData <- validated$graphData
  row_offset <- validated$row_offset
  col_offset <- validated$col_offset
  dataset_names <- names(Y)
  z_adjustment <- match.arg(z_adjustment)

  enrichment_opts <- .isvdx_resolve_enrichment_options(
    dataset_names = dataset_names,
    inference_enrichment = inference_enrichment,
    enrichment_backend = enrichment_backend,
    inference_scope = inference_scope,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction
  )

  # Validate ambiguous list-valued enrichment inputs once, before fitting.
  # `feature_sets` is naturally a list of gene sets, so an unnamed list is a
  # shared gene-set collection. Dataset-specific feature_sets must be named.
  # `enrichment_term2gene` should be a shared data frame or a named list of
  # data frames; an unnamed list is ambiguous in integrated workflows.
  invisible(lapply(dataset_names, function(ds) {
    .isvdx_dataset_enrichment_input(feature_sets, ds, dataset_names, arg = "feature_sets", list_kind = "shared_list")
    .isvdx_dataset_enrichment_input(enrichment_term2gene, ds, dataset_names, arg = "enrichment_term2gene", list_kind = "dataset_list")
    .isvdx_dataset_enrichment_input(enrichment_extra_args, ds, dataset_names, arg = "enrichment_extra_args", list_kind = "shared_list")
  }))

  if (!exists("prepared_fit_family", inherits = FALSE)) prepared_fit_family <- NULL
  if (!exists("prepared_inference_family", inherits = FALSE)) prepared_inference_family <- NULL
  if (!exists("prepared_fit_description", inherits = FALSE)) prepared_fit_description <- NULL

  family_specs <- .svdx_family_spec_list(family, dataset_names)
  fit_family <- prepared_fit_family %||% {
    tmp <- setNames(lapply(family_specs, `[[`, "fit_family"), dataset_names)
    if (all(vapply(tmp, is.null, logical(1)))) NULL else tmp
  }
  inference_family <- prepared_inference_family %||% setNames(lapply(family_specs, `[[`, "inference_family"), dataset_names)
  fit_description <- prepared_fit_description %||% setNames(lapply(family_specs, `[[`, "fit_description"), dataset_names)

  .svdx_validate_nonnegative_for_family(Y, inference_family, context = "Y")

  if (!is.null(fdata)) {
    fdata <- .svdx_align_dataset_list(fdata, dataset_names, "fdata")
    for (ds in dataset_names) {
      .svdx_validate_fdata_matches_features(fdata[[ds]], rownames(Y[[ds]]), context = paste0("Y[[", ds, "]]"))
    }
  }

  enrichment_notes <- list()

  if (is.null(enrichment_id_map) && !is.null(fdata)) {
    enrichment_id_map <- vector("list", length(dataset_names))
    names(enrichment_id_map) <- dataset_names
    for (ds in dataset_names) {
      if (isTRUE(enrichment_opts$inference_enrichment[[ds]])) {
        enrichment_id_map[[ds]] <- tryCatch(
          .svdx_fdata_to_enrichment_id_map(fdata[[ds]], required = TRUE),
          error = function(e) {
            msg <- paste0("Enrichment skipped for dataset '", ds, "': could not derive enrichment identifiers from fdata (", conditionMessage(e), ").")
            message(msg)
            enrichment_notes[[ds]] <<- msg
            enrichment_opts$inference_enrichment[[ds]] <<- FALSE
            NULL
          }
        )
      }
    }
  } else if (!is.null(enrichment_id_map)) {
    enrichment_id_map <- .svdx_align_dataset_list(enrichment_id_map, dataset_names, "enrichment_id_map")
  }

  for (ds in dataset_names) {
    if (isTRUE(enrichment_opts$inference_enrichment[[ds]])) {
      if (is.null(enrichment_id_map) || is.null(enrichment_id_map[[ds]])) {
        msg <- paste0("Enrichment skipped for dataset '", ds, "': no enrichment_id_map or fdata enrichment column was available.")
        message(msg)
        enrichment_notes[[ds]] <- msg
        enrichment_opts$inference_enrichment[[ds]] <- FALSE
      } else {
        ok <- tryCatch({
          .svdx_validate_enrichment_id_map(
            enrichment_id_map = enrichment_id_map[[ds]],
            features = rownames(Y[[ds]]),
            context = paste0("Y[['", ds, "']]"),
            require_unique = enrichment_opts$enrichment_backend[[ds]] %in% c("fgsea", "clusterProfiler")
          )
          TRUE
        }, error = function(e) {
          msg <- paste0("Enrichment skipped for dataset '", ds, "': ", conditionMessage(e))
          message(msg)
          enrichment_notes[[ds]] <<- msg
          FALSE
        })
        if (!isTRUE(ok)) enrichment_opts$inference_enrichment[[ds]] <- FALSE
      }
    }
  }
  
  split <- .resolve_train_test_cols_integrated(
    Y = Y,
    data = data,
    train_cols = train_cols,
    test_cols = test_cols,
    train_fraction = train_fraction,
    seed = seed
  )
  if (isTRUE(verbose)) {
    if (length(split$test) > 0L) {
      message("Train/test split complete")
    } else {
      message("Using all samples for training; inference will be skipped")
    }
  }
  
  Y_work <- Y
  
  tr <- subset_isvdx_inputs(
    Y = Y_work,
    xdata = data,
    zdata = zdata,
    graphData = graphData,
    si = split$train
  )
  
  Ytrain <- tr$Y
  data_train <- tr$xdata
  zdata_train <- tr$zdata
  graphData_train <- tr$graphData
  
  has_test <- length(split$test) > 0L
  if (has_test) {
    te <- subset_isvdx_inputs(
      Y = Y_work,
      xdata = data,
      zdata = zdata,
      graphData = graphData,
      si = split$test
    )
    
    Ytest <- te$Y
    data_test <- te$xdata
    zdata_test <- te$zdata
  } else {
    Ytest <- NULL
    data_test <- NULL
    zdata_test <- NULL
  }
  
  row_offset_train <- row_offset
  row_offset_test <- if (has_test) row_offset else NULL
  col_offset_train <- .subset_i_col_offsets(col_offset, split$train)
  col_offset_test <- if (has_test) .subset_i_col_offsets(col_offset, split$test) else NULL
  
  fit_args <- list(
    Y = Ytrain,
    formula = xformula,
    data = data_train,
    zformula = zformula,
    zdata = zdata_train,
    K = K,
    family = fit_family,
    modelType = modelType,
    graphData = graphData_train,
    graphDistFn = graphDistFn,
    graphDistOnlyOnPoints = seq_len(ncol(Ytrain[[1]])),
    rowOffsets = row_offset_train,
    colOffsets = col_offset_train,
    numThreads = numThreads
  )
  
  fit_args <- c(fit_args, list(...))
  
  fit <- do.call(isvdx, fit_args)
  if (isTRUE(verbose)) message("Integrated SVDX fitting complete")
  
  fit_summary <- summary(
    fit,
    Y = Ytrain,
    return_conf_distribution = return_conf_distribution
  )
  
  if (has_test) {
    inf <- inferV(
      fit = fit,
      Ytest = Ytest,
      newdata = data_test,
      newzdata = zdata_test,
      zformula = zformula,
      modelType = modelType,
      family = inference_family,
      alpha = alpha,
      z_adjustment = z_adjustment,
      row_offset = row_offset_test,
      col_offset = col_offset_test,
      return_design = return_design,
      verbose = verbose
    )
    if (isTRUE(verbose)) message("Inference complete")
    
    inf_summary <- tryCatch(
      summary(inf, alpha = alpha),
      error = function(e) {
        if (isTRUE(verbose)) {
          message("Inference summary skipped: ", conditionMessage(e))
        }
        NULL
      }
    )
  } else {
    inf <- NULL
    inf_summary <- NULL
    if (isTRUE(verbose)) message("Inference skipped because train_fraction = 1")
  }
  
  enrichment <- NULL
  if (has_test && any(vapply(enrichment_opts$inference_enrichment, isTRUE, logical(1)))) {
    enrichment_results <- list()
    enrichment_raw <- list()
    enrichment_backend_resolved <- list()

    for (ds in dataset_names) {
      if (!isTRUE(enrichment_opts$inference_enrichment[[ds]])) next

      feature_sets_ds <- .isvdx_dataset_enrichment_input(feature_sets, ds, dataset_names, arg = "feature_sets", list_kind = "shared_list")
      term2gene_ds <- .isvdx_dataset_enrichment_input(enrichment_term2gene, ds, dataset_names, arg = "enrichment_term2gene", list_kind = "dataset_list")
      enrichment_id_map_ds <- enrichment_id_map[[ds]]

      enr_ds <- tryCatch(
        enrichPathways(
          inf = inf$dataset_results[[ds]],
          feature_sets = feature_sets_ds,
          alpha = alpha,
          scope = enrichment_opts$inference_scope[[ds]],
          component = NULL,
          enrichment_backend = enrichment_opts$enrichment_backend[[ds]],
          msigdb_db_species = enrichment_opts$msigdb_db_species[[ds]],
          msigdb_species = enrichment_opts$msigdb_species[[ds]],
          msigdb_collection = enrichment_opts$msigdb_collection[[ds]],
          msigdb_subcollection = enrichment_opts$msigdb_subcollection[[ds]],
          enrichment_id_type = enrichment_opts$enrichment_id_type[[ds]],
          enrichment_score = enrichment_opts$enrichment_score[[ds]],
          term2gene = term2gene_ds,
          enrichment_id_map = enrichment_id_map_ds,
          enrichment_extra_args = enrichment_opts$enrichment_extra_args[[ds]] %||% list(),
          min_feature_overlap_warn = enrichment_opts$min_feature_overlap_warn[[ds]],
          min_feature_overlap_fraction = enrichment_opts$min_feature_overlap_fraction[[ds]]
        ),
        error = function(e) {
          msg <- paste0("Enrichment skipped for dataset '", ds, "': ", conditionMessage(e))
          message(msg)
          enrichment_notes[[ds]] <<- msg
          NULL
        }
      )

      if (is.null(enr_ds)) next

      enrichment_results[[ds]] <- enr_ds$results
      enrichment_raw[[ds]] <- enr_ds$raw
      enrichment_backend_resolved[[ds]] <- enr_ds$backend_resolved %||% enr_ds$backend
    }

    if (!length(enrichment_results)) {
      enrichment <- NULL
      if (isTRUE(verbose)) {
        message("Enrichment skipped: no dataset produced enrichment results")
      }
    } else {
      enrichment <- list(
        backend_requested = enrichment_opts$enrichment_backend,
        backend_resolved = enrichment_backend_resolved,
        backend = enrichment_opts$enrichment_backend,
        params = list(
          backend_requested = enrichment_opts$enrichment_backend,
          backend_resolved = enrichment_backend_resolved,
          alpha = alpha,
          scope = enrichment_opts$inference_scope,
          component = NULL,
          msigdb_db_species = enrichment_opts$msigdb_db_species,
          msigdb_species = enrichment_opts$msigdb_species,
          msigdb_collection = enrichment_opts$msigdb_collection,
          msigdb_subcollection = enrichment_opts$msigdb_subcollection,
          id_type = enrichment_opts$enrichment_id_type,
          score = enrichment_opts$enrichment_score,
          enrichment_id_map = enrichment_id_map,
          extra_args = enrichment_opts$enrichment_extra_args,
          min_feature_overlap_warn = enrichment_opts$min_feature_overlap_warn,
          min_feature_overlap_fraction = enrichment_opts$min_feature_overlap_fraction,
          inference_enrichment = enrichment_opts$inference_enrichment
        ),
        results = enrichment_results,
        raw = enrichment_raw,
        notes = enrichment_notes
      )
      class(enrichment) <- c("isvdxEnrichment", "list")
      if (isTRUE(verbose)) message("Enrichment complete")
    }
  } else if (has_test && length(enrichment_notes) > 0L) {
    enrichment <- list(
      results = list(),
      raw = list(),
      notes = enrichment_notes
    )
    class(enrichment) <- c("isvdxEnrichment", "list")
  }
  
  # Keep optional workflow artifacts internally consistent.
  if (is.null(inf)) {
    inf_summary <- NULL
  }

  out <- list(
    split = split,
    Y = if (return_design) Y_work else NULL,
    data = if (return_design) data else NULL,
    zdata = if (return_design) zdata else NULL,
    Ytrain = if (return_design) Ytrain else NULL,
    Ytest = if (return_design) Ytest else NULL,
    data_train = if (return_design) data_train else NULL,
    data_test = if (return_design) data_test else NULL,
    zdata_train = if (return_design) zdata_train else NULL,
    zdata_test = if (return_design) zdata_test else NULL,
    fit = fit,
    fit_summary = fit_summary,
    inf = inf,
    inf_summary = inf_summary,
    enrichment = enrichment,
    enrichment_notes = enrichment_notes,
    fdata = fdata,
    xformula = xformula,
    zformula = zformula,
    modelType = modelType,
    family = family,
    fit_family = fit_family,
    inference_family = inference_family,
    fit_description = fit_description,
    alpha = alpha,
    dataset_names = dataset_names,
    call = match.call()
  )
  
  class(out) <- "isvdxWorkflow"
  if (isTRUE(verbose)) message("Workflow complete")
  out
}


summary.isvdxWorkflow <- function(object, n_top = 10, ...) {
  dataset_names <- object$dataset_names
  
  top_features <- NULL
  if (!is.null(object$inf) && !is.null(object$inf$dataset_results)) {
    available <- intersect(dataset_names, names(object$inf$dataset_results))
    top_list <- lapply(available, function(ds) {
      tf <- tryCatch(
        topFeatures(
          object$inf$dataset_results[[ds]],
          scope = "union",
          n = n_top
        ),
        error = function(e) NULL
      )
      if (is.null(tf) || nrow(tf) == 0L) return(NULL)
      dplyr::mutate(tf, dataset = ds, .before = 1)
    })
    top_list <- Filter(Negate(is.null), top_list)
    if (length(top_list)) {
      top_features <- dplyr::bind_rows(top_list)
      if (nrow(top_features) == 0L) top_features <- NULL
    }
  }
  
  fmt_family <- function(x, ds) {
    if (is.null(x)) return(rep("NULL", length(dataset_names)))
    vapply(dataset_names, function(d) {
      val <- x[[d]]
      if (is.null(val)) "NULL" else as.character(val)[1]
    }, character(1))
  }

  out <- list(
    model = tibble::tibble(
      dataset = dataset_names,
      requested_family = fmt_family(object$family, dataset_names),
      fit_family = fmt_family(object$fit_family, dataset_names),
      inference_family = fmt_family(object$inference_family, dataset_names),
      fit_description = fmt_family(object$fit_description, dataset_names)
    ),
    split = tibble::tibble(
      n_train = length(object$split$train),
      n_test = length(object$split$test),
      inference = if (length(object$split$test) > 0L) "held-out test samples" else "skipped (train_fraction = 1)"
    ),
    training_components = object$fit_summary$components,
    training_loadings = object$fit_summary$dataset_loadings,
    inference_union = if (!is.null(object$inf_summary)) object$inf_summary$union_summary else NULL,
    inference_components = if (!is.null(object$inf_summary)) object$inf_summary$component_summary else NULL,
    top_features = top_features,
    enrichment = if (!is.null(object$enrichment) && length(object$enrichment$results) > 0L) {
      dplyr::bind_rows(lapply(names(object$enrichment$results), function(ds) {
        object$enrichment$results[[ds]] |>
          dplyr::arrange(padj, pvalue) |>
          dplyr::slice_head(n = n_top) |>
          dplyr::mutate(dataset = ds, .before = 1)
      }))
    } else NULL,
    enrichment_notes = object$enrichment_notes %||% if (!is.null(object$enrichment)) object$enrichment$notes else NULL,
    call = object$call
  )
  
  class(out) <- "summary.isvdxWorkflow"
  out
}


print.summary.isvdxWorkflow <- function(x, ...) {
  cat("isvdx workflow summary\n")
  cat("======================\n\n")
  
  cat("Model:\n")
  print(x$model)
  cat("\nSplit:\n")
  print(x$split)
  
  cat("\nTraining components:\n")
  print(x$training_components)
  
  cat("\nDataset loading summaries:\n")
  print(x$training_loadings)
  
  cat("\nInference union summary:\n")
  print(x$inference_union)
  
  cat("\nInference component summary:\n")
  print(x$inference_components)
  
  cat("\nTop union features:\n")
  print(x$top_features)
  
  if (!is.null(x$enrichment)) {
    cat("\nTop enrichment results:\n")
    print(x$enrichment)
  }
  
  invisible(x)
}


plot.isvdxWorkflow <- function(
    x,
    type = c(
      "fit_pi",
      "fit_kappa",
      "fit_diagnostics",
      "xz_confounding",
      "union_volcano",
      "component_volcano",
      "enrichment",
      "u",
      "u_embedding",
      "u_terms",
      "v_loadings",
      "v_heatmap",
      "v_inference_heatmap",
      "v_inference_dotplot",
      "v_loading_inference_scatter",
      "feature_profiles"
    ),
    dataset = NULL,
    component = NULL,
    metric = NULL,
    show_values = FALSE,
    value_digits = 2,
    xaxis = NULL,
    source = c("auto", "train", "test", "all"),
    ...
) {
  type <- match.arg(type)
  source <- match.arg(source)
  dots <- list(...)
  
  if (type %in% c("u", "u_terms") && is.null(xaxis)) {
    stop(
      "xaxis must be supplied for type = 'u' and type = 'u_terms'.",
      call. = FALSE
    )
  }
  
  if (type %in% c(
    "union_volcano",
    "component_volcano",
    "v_inference_heatmap",
    "v_inference_dotplot",
    "v_loading_inference_scatter"
  )) {
    if (is.null(dataset)) {
      stop("dataset must be supplied for dataset-specific inference plots.", call. = FALSE)
    }
  }
  
  if (type %in% c("v_loadings", "v_heatmap")) {
    if (is.null(dataset)) {
      stop("dataset must be supplied for V/loading plots from isvdx workflows.", call. = FALSE)
    }
  }
  
  ds_inf <- if (!is.null(dataset)) {
    .get_isvdx_workflow_dataset_inf(x, dataset)
  } else {
    NULL
  }
  
  switch(
    type,
    
    fit_pi = plot_component_pi(
      x$fit_summary,
      show_values = show_values,
      value_digits = value_digits,
      ...
    ),
    
    fit_kappa = plot_component_kappa(
      x$fit_summary,
      show_values = show_values,
      value_digits = value_digits,
      ...
    ),
    
    fit_diagnostics = plot_component_diagnostics(
      x$fit_summary,
      metric = metric %||% "pi",
      show_values = show_values,
      value_digits = value_digits,
      ...
    ),
    
    xz_confounding = plot_xz_confounding(
      x$fit_summary,
      metric = metric %||% "q95_abs_cos",
      show_values = show_values,
      value_digits = value_digits,
      ...
    ),
    
    union_volcano = plot_union_volcano(ds_inf, fdata = x$fdata, dataset = dataset, ...),
    
    component_volcano = {
      if (is.null(component)) {
        stop("component must be supplied for component_volcano.", call. = FALSE)
      }
      plot_component_volcano(ds_inf, component = component, fdata = x$fdata, dataset = dataset, ...)
    },
    
    enrichment = {
      if (is.null(x$enrichment)) {
        stop("No enrichment results found in workflow object.", call. = FALSE)
      }
      if (is.null(dataset)) {
        stop("dataset must be supplied for integrated enrichment plots.", call. = FALSE)
      }
      if (is.null(x$enrichment$results[[dataset]])) {
        stop("No enrichment results found for dataset: ", dataset, call. = FALSE)
      }
      plot_enrichment_dotplot(x$enrichment$results[[dataset]])
    },
    
    u = {
      if (is.null(x$data)) {
        stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      }
      
      plotU(
        x$fit,
        data = x$data,
        component = component %||% 1,
        x = xaxis, 
        ...
      )
    },
    
    u_embedding = {
      if (is.null(x$data)) {
        stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      }
      
      plotUEmbedding(
        x$fit,
        data = x$data,
        ...
      )
    },
    
    u_terms = {
      if (is.null(x$data)) {
        stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      }
      
      plotUTerms(
        x$fit,
        data = x$data,
        component = component %||% 1,
        x=xaxis, 
        ...
      )
    },
    
    v_loadings = plotVLoadings(
      x$fit,
      dataset = dataset,
      component = component %||% 1,
      fdata = x$fdata,
      ...
    ),
    
    v_heatmap = {
      heatmap_source <- source
      if (identical(heatmap_source, "auto")) {
        select_by_arg <- dots$select_by %||% "loading"
        has_test <- !is.null(dataset) && !is.null(x$Ytest) && !is.null(x$data_test) &&
          !is.null(x$Ytest[[dataset]])
        heatmap_source <- if (!identical(select_by_arg, "loading") && has_test) "test" else "all"
      }

      if (identical(heatmap_source, "test")) {
        if (is.null(dataset) || is.null(x$Ytest) || is.null(x$data_test) || is.null(x$Ytest[[dataset]])) {
          stop("Workflow object does not contain Ytest for the requested dataset and shared data_test; rerun with return_design=TRUE.", call. = FALSE)
        }
        Y_use <- x$Ytest[[dataset]]
        data_use <- x$data_test
        zdata_use <- if (!is.null(x$zdata_test)) x$zdata_test[[dataset]] else NULL
      } else if (identical(heatmap_source, "train")) {
        if (is.null(dataset) || is.null(x$Ytrain) || is.null(x$data_train) || is.null(x$Ytrain[[dataset]])) {
          stop("Workflow object does not contain Ytrain for the requested dataset and shared data_train; rerun with return_design=TRUE.", call. = FALSE)
        }
        Y_use <- x$Ytrain[[dataset]]
        data_use <- x$data_train
        zdata_use <- if (!is.null(x$zdata_train)) x$zdata_train[[dataset]] else NULL
      } else {
        if (is.null(dataset) || is.null(x$Y) || is.null(x$data) || is.null(x$Y[[dataset]])) {
          stop("Workflow object does not contain Y for the requested dataset and shared data; rerun with return_design=TRUE.", call. = FALSE)
        }
        Y_use <- x$Y[[dataset]]
        data_use <- x$data
        zdata_use <- if (!is.null(x$zdata)) x$zdata[[dataset]] else NULL
      }
      
      plotVHeatmap(
        object = x,
        Y = Y_use,
        data = data_use,
        zdata = zdata_use,
        dataset = dataset,
        component = component,
        fdata = x$fdata,
        ...
      )
    },
    
    v_inference_heatmap = plotVInferenceHeatmap(
      ds_inf,
      fdata = x$fdata,
      dataset = dataset,
      ...
    ),
    
    v_inference_dotplot = plotVInferenceDotplot(
      ds_inf,
      fdata = x$fdata,
      dataset = dataset,
      ...
    ),
    
    v_loading_inference_scatter = plotVLoadingInferenceScatter(
      fit = x$fit,
      inf = ds_inf,
      dataset = dataset,
      component = component %||% 1,
      ...
    ),

    feature_profiles = plotFeatureProfiles(
      x,
      dataset = dataset,
      source = source,
      ...
    )
  )
}


.validate_isvdx_workflow_inputs <- function(
    Y,
    data,
    zdata,
    zformula,
    family = NULL,
    graphData = NULL,
    row_offset = NULL,
    col_offset = NULL
) {
  if (!is.list(Y)) {
    stop("Y must be a list of feature x sample matrices.", call. = FALSE)
  }
  
  D <- length(Y)
  
  dataset_names <- names(Y)
  if (is.null(dataset_names) || length(dataset_names) != D || any(!nzchar(dataset_names))) {
    dataset_names <- paste0("d", seq_len(D))
    names(Y) <- dataset_names
  } else {
    if (anyDuplicated(dataset_names)) {
      stop("Dataset names must be unique.", call. = FALSE)
    }
  }
  
  Y <- lapply(Y, as.matrix)
  names(Y) <- dataset_names
  
  n <- ncol(Y[[1]])
  bad_n <- vapply(Y, ncol, integer(1)) != n
  if (any(bad_n)) {
    stop("All datasets in Y must have the same number of samples/columns.", call. = FALSE)
  }
  
  if (nrow(data) != n) {
    stop("nrow(data) must equal ncol(Y[[d]]) for all datasets.", call. = FALSE)
  }
  
  for (ds in dataset_names) {
    if (is.null(colnames(Y[[ds]]))) {
      colnames(Y[[ds]]) <- paste0("sample_", seq_len(n))
    }
    
    if (is.null(rownames(Y[[ds]]))) {
      rownames(Y[[ds]]) <- paste0(ds, "_feature_", seq_len(nrow(Y[[ds]])))
    }
  }
  
  if (is.null(rownames(data))) {
    rownames(data) <- colnames(Y[[1]])
  }
  
  zdata <- .svdx_align_dataset_list(zdata, dataset_names, "zdata")
  for (ds in dataset_names) {
    .svdx_validate_sample_frame(zdata[[ds]], n, arg = paste0("zdata[['", ds, "']]"))
  }
  .svdx_validate_graph_data(graphData, n)
  
  if (is.null(zformula)) {
    zformula <- rep(list(stats::as.formula(~1)), D)
  } else if (inherits(zformula, "formula")) {
    zformula <- rep(list(zformula), D)
  } else if (!is.list(zformula) || length(zformula) != D) {
    stop("zformula must be NULL, a formula, or a list of length equal to length(Y).", call. = FALSE)
  } else {
    zformula <- .svdx_align_dataset_list(zformula, dataset_names, "zformula")
  }
  
  names(zformula) <- dataset_names
  
  family_specs <- .svdx_family_spec_list(family, dataset_names)
  family <- setNames(lapply(family_specs, `[[`, "family"), dataset_names)
  if (all(vapply(family, is.null, logical(1)))) {
    family <- NULL
  }

  if (!is.null(row_offset)) {
    if (is.list(row_offset)) {
      row_offset <- .svdx_align_dataset_list(row_offset, dataset_names, "row_offset")
      for (ds in dataset_names) {
        .svdx_validate_row_offset(row_offset[[ds]], nrow(Y[[ds]]), arg = paste0("row_offset[['", ds, "']]"))
      }
    } else if (D == 1L) {
      .svdx_validate_row_offset(row_offset, nrow(Y[[1]]), arg = "row_offset")
    } else {
      stop("For integrated workflows, `row_offset` must be a list with one entry per dataset.", call. = FALSE)
    }
  }

  if (!is.null(col_offset)) {
    if (is.list(col_offset)) {
      col_offset <- .svdx_align_dataset_list(col_offset, dataset_names, "col_offset")
      for (ds in dataset_names) {
        .svdx_validate_col_offset(col_offset[[ds]], n, arg = paste0("col_offset[['", ds, "']]"))
      }
    } else {
      .svdx_validate_col_offset(col_offset, n, arg = "col_offset")
    }
  }
  
  list(
    Y = Y,
    data = data,
    zdata = zdata,
    zformula = zformula,
    family = family,
    graphData = graphData,
    row_offset = row_offset,
    col_offset = col_offset
  )
}


.resolve_train_test_cols_integrated <- function(
    Y,
    data,
    train_cols = NULL,
    test_cols = NULL,
    train_fraction = 0.70,
    seed = NULL
) {
  n <- ncol(Y[[1]])
  all_cols <- seq_len(n)
  
  name_to_idx <- function(x, arg) {
    .svdx_resolve_sample_indices(x, colnames(Y[[1]]), arg = arg)
  }
  
  train_idx <- name_to_idx(train_cols, "train_cols")
  test_idx <- name_to_idx(test_cols, "test_cols")
  
  if (is.null(train_idx) && is.null(test_idx)) {
    .svdx_validate_train_fraction(train_fraction)
    if (identical(train_fraction, 1) || isTRUE(train_fraction == 1)) {
      train_idx <- all_cols
      test_idx <- integer(0)
    } else {
      if (!is.null(seed)) set.seed(seed)
      train_idx <- sort(sample(all_cols, size = round(train_fraction * n)))
      test_idx <- setdiff(all_cols, train_idx)
    }
  } else if (is.null(train_idx)) {
    test_idx <- sort(unique(test_idx))
    train_idx <- setdiff(all_cols, test_idx)
  } else if (is.null(test_idx)) {
    train_idx <- sort(unique(train_idx))
    test_idx <- setdiff(all_cols, train_idx)
  } else {
    train_idx <- sort(unique(train_idx))
    test_idx <- sort(unique(test_idx))
  }
  
  if (length(intersect(train_idx, test_idx)) > 0) {
    stop("train_cols and test_cols must not overlap.", call. = FALSE)
  }

  if (length(train_idx) == 0) {
    stop("Training split must contain at least one sample.", call. = FALSE)
  }
  
  if (length(test_idx) > 0 && (length(train_idx) < 3 || length(test_idx) < 3)) {
    warning("Very small train/test split; inference may be unstable.", call. = FALSE)
  }
  
  list(
    train = train_idx,
    test = test_idx,
    train_names = colnames(Y[[1]])[train_idx],
    test_names = colnames(Y[[1]])[test_idx]
  )
}


subset_isvdx_inputs <- function(
    Y,
    xdata,
    zdata,
    graphData = NULL,
    si
) {
  Y_sub <- lapply(Y, function(Yd) {
    Yd[, si, drop = FALSE]
  })
  
  xdata_sub <- xdata[si, , drop = FALSE]
  
  zdata_sub <- lapply(zdata, function(zd) {
    zd[si, , drop = FALSE]
  })
  
  graphData_sub <- if (is.null(graphData)) NULL else graphData[, si, drop = FALSE]
  
  list(
    Y = Y_sub,
    xdata = xdata_sub,
    zdata = zdata_sub,
    graphData = graphData_sub
  )
}

.standardize_Y_list_workflow <- function(Y, eps = 1e-8) {
  out <- lapply(Y, function(Yd) {
    mu <- rowMeans(Yd, na.rm = TRUE)
    sd <- apply(Yd, 1, stats::sd, na.rm = TRUE)
    sd[sd < eps | !is.finite(sd)] <- 1
    
    Ys <- sweep(sweep(Yd, 1, mu, "-"), 1, sd, "/")
    
    list(Y = Ys, center = mu, scale = sd)
  })
  
  list(
    Y = lapply(out, `[[`, "Y"),
    center = lapply(out, `[[`, "center"),
    scale = lapply(out, `[[`, "scale")
  )
}


.subset_i_col_offsets <- function(col_offset, idx) {
  if (is.null(col_offset)) return(NULL)
  
  if (is.list(col_offset)) {
    return(lapply(col_offset, function(x) as.numeric(x)[idx]))
  }
  
  as.numeric(col_offset)[idx]
}


.get_isvdx_workflow_dataset_inf <- function(x, dataset) {
  d <- if (is.numeric(dataset)) dataset else match(dataset, names(x$inf$dataset_results))
  
  if (is.na(d)) {
    stop(
      "Unknown dataset: ", dataset, ". Available datasets are: ",
      paste(names(x$inf$dataset_results), collapse = ", "),
      call. = FALSE
    )
  }
  
  x$inf$dataset_results[[d]]
}
