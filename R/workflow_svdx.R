# ============================================================
# 6B. Real-world single-dataset workflow API
# ============================================================
# First-class workflow object for applied use:
#   runSVDXWorkflow() -> class "svdxWorkflow"
#
# This wraps:
#   svdx() -> summary_svdx_fit() -> inferV() -> summary(inf)
# and optionally enrichPathways().
#
# The low-level functions remain available for power users.
# ============================================================

#' Run a complete single-dataset SSANOVA-SVD workflow
#'
#' Fits an SSANOVA-SVD model, summarizes the fitted training model, performs
#' conditional inference on held-out samples, summarizes the inference results,
#' and optionally runs pathway enrichment.
#'
#' @param Y Feature-by-sample matrix.
#' @param data Sample metadata data frame. Rows correspond to columns of `Y`.
#' @param zdata Data frame containing nuisance covariates. Defaults to `data`.
#' @param xformula Formula for metadata-driven component scores.
#' @param zformula Optional nuisance formula.
#' @param train_cols Optional training sample indices, names, or logical vector.
#' @param test_cols Optional test sample indices, names, or logical vector.
#' @param train_fraction Fraction of samples used for training when explicit
#'   train/test columns are not supplied. Values in (0, 1) create a train/test
#'   split and run conditional inference on held-out samples; `train_fraction = 1`
#'   fits on all samples and skips test-sample inference.
#' @param seed Optional random seed for train/test splitting.
#' @param K Maximum number of components to fit.
#' @param family Optional response family, currently `"gaussian"`, `"poisson"`, or `"quasipoisson"`. When `"quasipoisson"` is supplied, the SVDX fit uses the Poisson objective and inference uses quasi-Poisson GLMs.
#' @param graphData Optional graph data used by graph-aware model components. Must 
#'   have dimensions graphFeatures x samples. This will be used with graphDistFn 
#'   to calculate graph based distance matrices. 
#' @param graphDistFn Optional function for graph distances. Must return a positive
#'   semi-definite / definite matrix. 
#' @param numThreads Number of threads for fitting.
#' @param row_offset Optional feature-level offsets used during inference.
#' @param col_offset Optional sample-level offsets used during inference.
#' @param alpha Significance threshold used in inference summaries.
#' @param z_adjustment Nuisance-adjustment mode for inference: `"fitted"` uses SVDX-estimated nuisance effects; `"reestimate"` re-estimates nuisance effects inside `inferV()`.
#' @param feature_sets Optional named list of enrichment sets. If supplied, the
#'   list names are term names and each element is a character vector of
#'   enrichment identifiers. If NULL and `inference_enrichment=TRUE`, feature sets are
#'   obtained from MSigDB using `msigdb_db_species`, `msigdb_species`,
#'   `msigdb_collection`, and `msigdb_subcollection`.
#' @param fdata Optional `svdx_fdata` object created by [as_svdx_fdata()].
#'   Stored in the workflow object and reused by plots, reports, and enrichment.
#'   If enrichment is requested and `enrichment_id_map` is NULL, the enrichment
#'   mapping is derived from `fdata` when it defines an enrichment column.
#' @param enrichment_id_map Required when `inference_enrichment=TRUE` unless `fdata`
#'   supplies an enrichment column. A two-column
#'   data frame mapping SVDX feature IDs to enrichment IDs. The first column must
#'   contain SVDX feature names matching `rownames(Y)` / inference features. The
#'   second column must contain identifiers used by the enrichment database, such
#'   as gene symbols, NCBI/Entrez IDs, or Ensembl gene IDs. Column names are
#'   ignored and are normalized internally.
#' @param inference_enrichment Logical; whether to run inference enrichment. Defaults to TRUE when enrichment identifiers are available.
#'   when `enrichment_id_map` is supplied.
#' @param enrichment_backend Enrichment backend: `"auto"`, `"internal"`,
#'   `"clusterProfiler"`, or `"fgsea"`. The internal backend performs
#'   over-representation analysis. `fgsea` and `clusterProfiler` use
#'   GSEA-style ranked enrichment and require unique mapped enrichment IDs.
#' @param inference_scope Inference scope used for enrichment: `"all"`, `"union"`, `"component"`, or `"global"`. Defaults to `"all"`, which runs union-, global-,
#'   and component-level enrichment. Component scope enriches all inferred
#'   components.
#' @param msigdb_db_species MSigDB database species passed to
#'   `msigdbr::msigdbr(db_species=...)`, for example `"HS"` for human MSigDB or
#'   `"MM"` for mouse MSigDB. Required when `feature_sets` and
#'   `enrichment_term2gene` are NULL.
#' @param msigdb_species Species passed to `msigdbr::msigdbr(species=...)`, for
#'   example `"Homo sapiens"` or `"Mus musculus"`. Required when `feature_sets`
#'   and `enrichment_term2gene` are NULL.
#' @param msigdb_collection MSigDB collection passed to
#'   `msigdbr::msigdbr(collection=...)`, for example `"H"`, `"C5"`, `"MH"`, or
#'   `"M5"`. Use `msigdbr::msigdbr_collections()` to inspect valid values for
#'   your installed MSigDB database.
#' @param msigdb_subcollection Optional MSigDB subcollection passed to
#'   `msigdbr::msigdbr(subcollection=...)`, for example `"BP"`, `"MF"`, or
#'   `"CC"` for GO collections.
#' @param enrichment_id_type Identifier column to extract from MSigDB when
#'   `feature_sets`/`enrichment_term2gene` are not supplied: `"symbol"`,
#'   `"entrez"`, or `"ensembl"`. This must match the second column of
#'   `enrichment_id_map`.
#' @param enrichment_score Ranking score for GSEA-style enrichment. The default
#'   `"auto"` uses `-log10(union q-value)` for union scope and signed
#'   `-log10(component q-value)` for component scope.
#' @param enrichment_term2gene Optional TERM2GENE table used by
#'   `clusterProfiler::GSEA()`. If NULL and `inference_enrichment=TRUE`, it is built
#'   from MSigDB using the explicit MSigDB arguments above.
#' @param enrichment_extra_args Optional named list of additional arguments
#'   passed to backend enrichment functions, such as `minSize`/`maxSize` for
#'   `fgsea` or `pvalueCutoff`/`minGSSize`/`maxGSSize` for
#'   `clusterProfiler::GSEA()`.
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
#' @param ... Additional arguments passed to [svdx()].
#'
#' @return An object of class `"svdxWorkflow"` containing the split
#'   information, the fitted model, training summary, inference result,
#'   inference summary, optional enrichment results, and relevant input objects.
#'   When enrichment is run, `wf$enrichment` is an `svdxEnrichment` object with
#'   `backend_requested`, `backend_resolved`, `params`, normalized `results`,
#'   and raw backend output in `raw`.
#'
#' @examples
#' \dontrun{
#' wf <- runSVDXWorkflow(
#'   Y = Y,
#'   data = sample_data,
#'   xformula = ~ group * time,
#'   zformula = ~ batch,
#'   K = 5,
#'   train_fraction = 0.7,
#'   seed = 1,
#'   inference_enrichment = TRUE,
#'   enrichment_backend = "internal",
#'   enrichment_id_map = data.frame(
#'     feature = rownames(Y),
#'     gene_symbol = gene_symbols
#'   ),
#'   enrichment_id_type = "symbol",
#'   msigdb_db_species = "HS",
#'   msigdb_species = "Homo sapiens",
#'   msigdb_collection = "H"
#' )
#'
#' summary(wf)
#' plot(wf, type = "u", component = 1, x = "time", group = "group")
#' topFeatures(wf)
#' }
#'
#' @export
runSVDXWorkflow <- function(
    Y,
    data,
    zdata = data, 
    xformula,
    zformula = NULL,
    train_cols = NULL,
    test_cols = NULL,
    train_fraction = 0.70,
    seed = NULL,
    K = 5,
    family = NULL, #c("gaussian", "poisson", "quasipoisson"),
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
    if (isTRUE(prepared$integrated)) {
      stop("runSVDXWorkflow() received integrated svdx_data; use runISVDXWorkflow() instead.", call. = FALSE)
    }
    Y <- prepared$Y
    if (missing(data)) data <- prepared$xdata
    if (missing(xformula)) xformula <- prepared$xformula
    if (missing(zdata)) zdata <- prepared$zdata
    if (missing(zformula) || is.null(zformula)) zformula <- prepared$zformula
    if (missing(family)) family <- prepared$family
    prepared_fit_family <- prepared$fit_family
    prepared_inference_family <- prepared$inference_family
    prepared_fit_description <- prepared$fit_description
    if (missing(row_offset)) row_offset <- prepared$row_offset
    if (missing(col_offset)) col_offset <- prepared$col_offset
    if (missing(fdata) || is.null(fdata)) fdata <- prepared$fdata
    if (is.null(modelType)) modelType <- prepared$modelType
  }
  
  if (!exists("prepared_fit_family", inherits = FALSE)) prepared_fit_family <- NULL
  if (!exists("prepared_inference_family", inherits = FALSE)) prepared_inference_family <- NULL
  if (!exists("prepared_fit_description", inherits = FALSE)) prepared_fit_description <- NULL

  family_spec <- .svdx_family_spec(family)
  family <- family_spec$family
  fit_family <- prepared_fit_family %||% family_spec$fit_family
  inference_family <- prepared_inference_family %||% family_spec$inference_family
  fit_description <- prepared_fit_description %||% family_spec$fit_description
  
  zformula <- zformula %||% ~1
  enrichment_backend <- match.arg(enrichment_backend)
  inference_scope <- match.arg(inference_scope)
  z_adjustment <- match.arg(z_adjustment)
  enrichment_id_type <- match.arg(enrichment_id_type)
  
  validated <- .validate_svdx_workflow_inputs(
    Y = Y,
    data = data,
    zdata = zdata,
    graphData = graphData,
    row_offset = row_offset,
    col_offset = col_offset
  )
  Y <- validated$Y
  data <- validated$data
  zdata <- validated$zdata
  graphData <- validated$graphData

  .svdx_validate_nonnegative_for_family(Y, inference_family, context = "Y")

  if (!is.null(fdata)) {
    .svdx_validate_fdata_matches_features(fdata, rownames(Y), context = "Y")
  }

  if (isTRUE(inference_enrichment) && is.null(enrichment_id_map) && !is.null(fdata)) {
    enrichment_id_map <- .svdx_fdata_to_enrichment_id_map(fdata, required = TRUE)
  }

  if (isTRUE(inference_enrichment)) {
    .svdx_validate_enrichment_id_map(
      enrichment_id_map = enrichment_id_map,
      features = rownames(Y),
      context = "Y",
      require_unique = enrichment_backend %in% c("fgsea", "clusterProfiler")
    )
  }
  
  split <- .resolve_train_test_cols(
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
  
  tr <- subset_svdx_inputs( Y=Y_work, xdata=data, zdata=zdata, graphData=graphData, si = split$train )
  Ytrain <- tr$Y
  data_train <- tr$xdata
  zdata_train <- tr$zdata
  graphData_train <- tr$graphData
  
  has_test <- length(split$test) > 0L
  if (has_test) {
    te <- subset_svdx_inputs( Y_work, xdata=data, zdata=zdata, graphData=graphData, si = split$test )
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
  col_offset_train <- .subset_col_offsets(col_offset, split$train)
  col_offset_test <- if (has_test) .subset_col_offsets(col_offset, split$test) else NULL
  
  # ----------------------------------------------------------
  # Fit svdx
  # ----------------------------------------------------------
  fit_args <- list(
    Y = Ytrain,
    formula = xformula,
    zformula = zformula, 
    data = data_train, 
    zdata = zdata_train, 
    K = K,
    family = fit_family,
    modelType = modelType,
    graphData = graphData_train,
    graphDistFn = graphDistFn, 
    graphDistOnlyOnPoints = seq_len(ncol(Ytrain)), 
    rowOffsets = row_offset_train,
    colOffsets = col_offset_train,
    numThreads = numThreads
  )
  fit_args <- c(fit_args, list(...))
  
  fit <- do.call(svdx, fit_args)
  if (isTRUE(verbose)) message("SVDX fitting complete")
  
  # ----------------------------------------------------------
  # Training summary
  # ----------------------------------------------------------
  fit_summary <- summary_svdx_fit(
    fit = fit,
    Y = Ytrain,
    return_conf_distribution = return_conf_distribution
  )
  
  # ----------------------------------------------------------
  # Conditional inference on test samples
  # ----------------------------------------------------------
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
  
  # ----------------------------------------------------------
  # Optional built-in enrichment
  # ----------------------------------------------------------
  enrichment <- NULL
  if (has_test && isTRUE(inference_enrichment)) {
    enrichment <- tryCatch(
      enrichPathways(
        inf,
        feature_sets = feature_sets,
        alpha = alpha,
        scope = inference_scope,
        component = NULL,
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
        message("Enrichment skipped: ", conditionMessage(e))
        NULL
      }
    )
    if (isTRUE(verbose) && !is.null(enrichment)) message("Enrichment complete")
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
    fdata = fdata,
    xformula = xformula,
    zformula = zformula,
    modelType = modelType,
    family = family,
    fit_family = fit_family,
    inference_family = inference_family,
    fit_description = fit_description,
    alpha = alpha,
    call = match.call()
  )
  
  class(out) <- "svdxWorkflow"
  if (isTRUE(verbose)) message("Workflow complete")
  out
}


summary.svdxWorkflow <- function(object, n_top = 10, ...) {
  fit_components <- object$fit_summary$components
  inf_sum <- object$inf_summary
  
  out <- list(
    model = tibble::tibble(
      requested_family = if (is.null(object$family)) "NULL" else as.character(object$family),
      fit_family = if (is.null(object$fit_family)) "NULL" else as.character(object$fit_family),
      inference_family = if (is.null(object$inference_family)) NA_character_ else as.character(object$inference_family),
      fit_description = object$fit_description %||% NA_character_
    ),
    split = tibble::tibble(
      n_train = length(object$split$train),
      n_test = length(object$split$test),
      inference = if (length(object$split$test) > 0L) "held-out test samples" else "skipped (train_fraction = 1)"
    ),
    training_components = fit_components,
    inference_union = if (!is.null(inf_sum)) inf_sum$union_summary else NULL,
    inference_components = if (!is.null(inf_sum)) inf_sum$component_summary else NULL,
    top_features = if (!is.null(object$inf)) {
      tryCatch(
        topFeatures(object$inf, scope = "union", n = n_top),
        error = function(e) NULL
      )
    } else NULL,
    enrichment = if (!is.null(object$enrichment)) {
      .svdx_enrichment_results(object$enrichment) |>
        dplyr::arrange(padj, pvalue) |>
        dplyr::slice_head(n = n_top)
    } else NULL,
    call = object$call
  )
  
  class(out) <- "summary.svdxWorkflow"
  out
}


print.summary.svdxWorkflow <- function(x, ...) {
  cat("svdx workflow summary
")
  cat("=====================

")
  cat("Model:
")
  print(x$model)
  cat("
Split:
")
  print(x$split)
  cat("
Training components:
")
  print(x$training_components)
  cat("
Inference union summary:
")
  print(x$inference_union)
  cat("
Inference component summary:
")
  print(x$inference_components)
  cat("
Top union features:
")
  print(x$top_features)
  if (!is.null(x$enrichment)) {
    cat("
Top enrichment results:
")
    print(x$enrichment)
  }
  invisible(x)
}


plot.svdxWorkflow <- function(
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
    component = NULL,
    xaxis=NULL, 
    metric = NULL,
    show_values = FALSE,
    value_digits = 2,
    source = c("auto", "train", "test", "all"),
    ...
) {
  type <- match.arg(type)
  source <- match.arg(source)
  dots <- list(...)
  
  if (type %in% c("u", "u_terms") && is.null(xaxis)) {
    stop("xaxis must be supplied for type = 'u' and type = 'u_terms'.", call. = FALSE)
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
    
    union_volcano = plot_union_volcano(x$inf, fdata = x$fdata, ...),
    
    component_volcano = {
      if (is.null(component)) stop("component must be supplied for component_volcano.", call. = FALSE)
      plot_component_volcano(x$inf, component = component, fdata = x$fdata, ...)
    },
    
    enrichment = {
      if (is.null(x$enrichment)) stop("No enrichment results found in workflow object.", call. = FALSE)
      plot_enrichment_dotplot(x$enrichment$results)
    },
    
    u = {
      if (is.null(x$data)) stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      plotU(x$fit, data = x$data, component = component %||% 1, x=xaxis, ...)
    },
    
    u_embedding = {
      if (is.null(x$data)) stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      plotUEmbedding(x$fit, data = x$data, ...)
    },
    
    u_terms = {
      if (is.null(x$data)) stop("Workflow object does not contain data; rerun with return_design=TRUE.", call. = FALSE)
      plotUTerms(x$fit, data = x$data, component = component %||% 1, x=xaxis, ...)
    },
    
    v_loadings = plotVLoadings(
      x$fit,
      component = component %||% 1,
      fdata = x$fdata,
      ...
    ),
    
    v_heatmap = {
      heatmap_source <- source
      if (identical(heatmap_source, "auto")) {
        select_by_arg <- dots$select_by %||% "loading"
        heatmap_source <- if (!identical(select_by_arg, "loading") && !is.null(x$Ytest) && !is.null(x$data_test)) "test" else "all"
      }

      if (identical(heatmap_source, "test")) {
        if (is.null(x$Ytest) || is.null(x$data_test)) stop("Workflow object does not contain Ytest/data_test; rerun with return_design=TRUE.", call. = FALSE)
        Y_use <- x$Ytest
        data_use <- x$data_test
        zdata_use <- x$zdata_test
      } else if (identical(heatmap_source, "train")) {
        if (is.null(x$Ytrain) || is.null(x$data_train)) stop("Workflow object does not contain Ytrain/data_train; rerun with return_design=TRUE.", call. = FALSE)
        Y_use <- x$Ytrain
        data_use <- x$data_train
        zdata_use <- x$zdata_train
      } else {
        if (is.null(x$Y) || is.null(x$data)) stop("Workflow object does not contain Y/data; rerun with return_design=TRUE.", call. = FALSE)
        Y_use <- x$Y
        data_use <- x$data
        zdata_use <- x$zdata
      }

      plotVHeatmap(
        object = x,
        Y = Y_use,
        data = data_use,
        zdata = zdata_use,
        component = component,
        fdata = x$fdata,
        ...
      )
    },
    
    v_inference_heatmap = plotVInferenceHeatmap(
      x$inf,
      fdata = x$fdata,
      ...
    ),
    
    v_inference_dotplot = plotVInferenceDotplot(
      x$inf,
      fdata = x$fdata,
      ...
    ),
    
    v_loading_inference_scatter = plotVLoadingInferenceScatter(
      fit = x$fit,
      inf = x$inf,
      component = component %||% 1,
      ...
    ),

    feature_profiles = plotFeatureProfiles(
      x,
      source = source,
      ...
    )
  )
}



.validate_svdx_workflow_inputs <- function(
    Y,
    data,
    zdata = data,
    graphData = NULL,
    row_offset = NULL,
    col_offset = NULL
) {
  Y <- as.matrix(Y)
  
  if (ncol(Y) != nrow(data)) {
    stop(
      "ncol(Y) must equal nrow(data). Rows of data should correspond to columns/samples of Y.",
      call. = FALSE
    )
  }
  
  if (is.null(colnames(Y))) {
    colnames(Y) <- paste0("sample_", seq_len(ncol(Y)))
  }
  
  if (is.null(rownames(Y))) {
    rownames(Y) <- paste0("feature_", seq_len(nrow(Y)))
  }
  
  if (is.null(rownames(data))) {
    rownames(data) <- colnames(Y)
  }

  .svdx_validate_sample_frame(zdata, ncol(Y), arg = "zdata")
  .svdx_validate_graph_data(graphData, ncol(Y))
  .svdx_validate_row_offset(row_offset, nrow(Y), arg = "row_offset")
  .svdx_validate_col_offset(col_offset, ncol(Y), arg = "col_offset")
  
  list(Y = Y, data = data, zdata = zdata, graphData = graphData)
}

.resolve_train_test_cols <- function(
    Y,
    data,
    train_cols = NULL,
    test_cols = NULL,
    train_fraction = 0.70,
    seed = NULL
) {
  n <- ncol(Y)
  all_cols <- seq_len(n)
  
  name_to_idx <- function(x, arg) {
    .svdx_resolve_sample_indices(x, colnames(Y), arg = arg)
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
    train_names = colnames(Y)[train_idx],
    test_names = colnames(Y)[test_idx]
  )
}

.subset_col_offsets <- function(col_offset, idx) {
  if (is.null(col_offset)) return(NULL)
  as.numeric(col_offset)[idx]
}


# Vignette-style real-world usage:
# wf <- runSVDXWorkflow(
#   Y = Y,
#   data = meta,
#   xformula = ~ disease * time + severity * time,
#   zformula = ~ batch,
#   K = 8,
#   train_fraction = 0.7,
#   seed = 1,
#   feature_sets = hallmark_sets
# )
# summary(wf)
# plot(wf, type = "fit_pi")
# plot(wf, type = "fit_kappa")
# plot(wf, type = "xz_confounding")
# plot(wf, type = "union_volcano")
# topFeatures(wf)
# sig <- svdxSignificant(wf)
# universe <- svdxUniverse(wf)
# ranked <- asRankedList(wf)
