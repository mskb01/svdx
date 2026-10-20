#enrichment.R
# Move enrichment helpers here:
#   svdxSignificant()
#   svdxUniverse()
#   asRankedList()
#   enrichPathways()
#   runFGSEA()
#   runClusterProfilerORA()
#   runClusterProfilerGSEA()
#   getMSigDB()
#
# Prefer names:
#   svdxSignificant(), svdxUniverse()
# rather than foreground/background wording.

# ============================================================
# SVDXWorkflow objects
# ============================================================
topFeatures.svdxWorkflow <- function(x, scope = c("union", "component", "global"), component = NULL, n = 20) {
  scope <- match.arg(scope)
  topFeatures(x$inf, scope = scope, component = component, n = n)
}

svdxSignificant.svdxWorkflow <- function(x, alpha = x$alpha %||% 0.05, scope = c("union", "component", "global"), component = NULL) {
  scope <- match.arg(scope)
  svdxSignificant(x$inf, alpha = alpha, scope = scope, component = component)
}

svdxUniverse.svdxWorkflow <- function(x) {
  svdxUniverse(x$inf)
}

asRankedList.svdxWorkflow <- function(x, score = "auto", scope = "union", component = NULL) {
  asRankedList(x$inf, score = score, scope = scope, component = component)
}

# ============================================================
# SVDXINFERENCE objects
# ============================================================
topFeatures.svdxInference <- function(inf, scope = c("union", "component", "global"), component = NULL, n = 20) {
  scope <- match.arg(scope)
  
  if (scope == "union") {
    return(inf$union_results |>
             dplyr::arrange(union_q, min_p) |>
             dplyr::slice_head(n = n))
  }
  if (scope == "global") {
    if (!all(c("global_q", "global_p") %in% names(inf$union_results))) {
      message("No global inference statistics available; returning no top features.")
      return(inf$union_results[0, , drop = FALSE])
    }
    return(inf$union_results |>
             dplyr::arrange(global_q, global_p) |>
             dplyr::slice_head(n = n))
  }
  
  if (is.null(component)) stop("component must be supplied when scope='component'.", call. = FALSE)
  
  inf$feature_results |>
    dplyr::filter(component == !!component) |>
    dplyr::arrange(qval, pval) |>
    dplyr::slice_head(n = n)
}

svdxSignificant.svdxInference <- function(inf, alpha = inf$alpha %||% 0.05, scope = c("union", "component", "global"), component = NULL) {
  scope <- match.arg(scope)
  
  if (scope == "union") {
    return(inf$union_results$feature[!is.na(inf$union_results$union_q) & inf$union_results$union_q < alpha])
  }
  if (scope == "global") {
    return(inf$union_results$feature[!is.na(inf$union_results$global_q) & inf$union_results$global_q < alpha])
  }
  
  if (is.null(component)) stop("component must be supplied when scope='component'.", call. = FALSE)
  
  fr <- inf$feature_results
  fr$feature[fr$component == component & !is.na(fr$qval) & fr$qval < alpha]
}

svdxUniverse.svdxInference <- function(inf) {
  inf$union_results$feature
}

asRankedList.svdxInference <- function(
    inf,
    score = c("auto", "union_q", "neg_log10_union_q", "union_p", "neg_log10_union_p", "global_q", "neg_log10_global_q", "global_p", "neg_log10_global_p", "max_abs_stat", "best_stat", "signed_best_stat", "neg_log10_min_p", "qval", "neg_log10_qval", "signed_neg_log10_qval"),
    scope = c("union", "component", "global"),
    component = NULL
) {
  score <- match.arg(score)
  scope <- match.arg(scope)
  
  if (scope %in% c("union", "global")) {
    ur <- inf$union_results
    needed <- switch(score,
      auto = if (scope == "global") "global_q" else "union_q",
      union_q = "union_q", neg_log10_union_q = "union_q",
      union_p = "union_p", neg_log10_union_p = "union_p",
      global_q = "global_q", neg_log10_global_q = "global_q",
      global_p = "global_p", neg_log10_global_p = "global_p",
      max_abs_stat = "max_abs_stat", best_stat = "best_stat", signed_best_stat = "best_stat",
      neg_log10_min_p = "min_p", qval = "union_q", neg_log10_qval = "union_q",
      signed_neg_log10_qval = "union_q"
    )
    if (!needed %in% names(ur)) {
      message("Ranked list skipped: statistic '", needed, "' is not available.")
      return(stats::setNames(numeric(), character()))
    }

    vals <- switch(
      score,
      auto = if (scope == "global") -log10(pmax(ur$global_q, .Machine$double.xmin)) else -log10(pmax(ur$union_q, .Machine$double.xmin)),
      union_q = -log10(pmax(ur$union_q, .Machine$double.xmin)),
      neg_log10_union_q = -log10(pmax(ur$union_q, .Machine$double.xmin)),
      union_p = -log10(pmax(ur$union_p, .Machine$double.xmin)),
      neg_log10_union_p = -log10(pmax(ur$union_p, .Machine$double.xmin)),
      global_q = -log10(pmax(ur$global_q, .Machine$double.xmin)),
      neg_log10_global_q = -log10(pmax(ur$global_q, .Machine$double.xmin)),
      global_p = -log10(pmax(ur$global_p, .Machine$double.xmin)),
      neg_log10_global_p = -log10(pmax(ur$global_p, .Machine$double.xmin)),
      max_abs_stat = ur$max_abs_stat,
      best_stat = ur$best_stat,
      signed_best_stat = ur$best_stat,
      neg_log10_min_p = -log10(pmax(ur$min_p, .Machine$double.xmin)),
      qval = -log10(pmax(ur$union_q, .Machine$double.xmin)),
      neg_log10_qval = -log10(pmax(ur$union_q, .Machine$double.xmin)),
      signed_neg_log10_qval = -log10(pmax(ur$union_q, .Machine$double.xmin))
    )

    names(vals) <- ur$feature
    vals <- vals[is.finite(vals)]
    return(sort(vals, decreasing = TRUE))
  }
  
  if (is.null(component)) stop("component must be supplied when scope='component'.", call. = FALSE)
  
  fr <- inf$feature_results |>
    dplyr::filter(component == !!component)

  if (nrow(fr) == 0L) {
    message("Ranked list skipped: no inference results found for component '", component, "'.")
    return(stats::setNames(numeric(), character()))
  }

  if (score %in% c("union_p", "neg_log10_union_p", "global_q", "neg_log10_global_q", "global_p", "neg_log10_global_p")) {
    stop(score, " is only available for union-level ranked lists.", call. = FALSE)
  }
  
  vals <- switch(
    score,
    auto = sign(fr$stat) * -log10(pmax(fr$qval, .Machine$double.xmin)),
    union_q = -log10(pmax(fr$qval, .Machine$double.xmin)),
    neg_log10_union_q = -log10(pmax(fr$qval, .Machine$double.xmin)),
    max_abs_stat = abs(fr$stat),
    best_stat = fr$stat,
    signed_best_stat = fr$stat,
    neg_log10_min_p = -log10(pmax(fr$pval, .Machine$double.xmin)),
    qval = -log10(pmax(fr$qval, .Machine$double.xmin)),
    neg_log10_qval = -log10(pmax(fr$qval, .Machine$double.xmin)),
    signed_neg_log10_qval = sign(fr$stat) * -log10(pmax(fr$qval, .Machine$double.xmin))
  )
  
  names(vals) <- fr$feature
  vals <- vals[is.finite(vals)]
  sort(vals, decreasing = TRUE)
}


# ============================================================
# Enrichment
# ============================================================

.svdx_term2gene_to_feature_sets <- function(term2gene) {
  if (is.null(term2gene) || nrow(term2gene) == 0) {
    stop("term2gene must contain at least one term-feature mapping.", call. = FALSE)
  }

  term_col <- if ("term" %in% names(term2gene)) "term" else if ("gs_name" %in% names(term2gene)) "gs_name" else names(term2gene)[1]
  gene_col <- if ("gene" %in% names(term2gene)) "gene" else if ("gene_symbol" %in% names(term2gene)) "gene_symbol" else names(term2gene)[2]

  ids <- as.character(term2gene[[gene_col]])
  terms <- as.character(term2gene[[term_col]])
  ok <- !is.na(ids) & nzchar(ids) & !is.na(terms) & nzchar(terms)

  split(ids[ok], terms[ok]) |>
    lapply(unique)
}



.svdx_validate_enrichment_id_map <- function(
    enrichment_id_map,
    features,
    context = "enrichment",
    require_unique = FALSE
) {
  if (is.null(enrichment_id_map)) {
    stop(
      "enrichment_id_map must be supplied when enrichment analysis is requested. ",
      "It must be a two-column data frame. The first column must contain SVDX feature names ",
      "matching rownames(Y) / inf$union_results$feature, and the second column must contain ",
      "the identifiers used by the selected enrichment backend.",
      call. = FALSE
    )
  }

  if (!is.data.frame(enrichment_id_map)) {
    stop("enrichment_id_map must be a data frame.", call. = FALSE)
  }

  if (ncol(enrichment_id_map) < 2) {
    stop(
      "enrichment_id_map must be a two-column data frame: first column = SVDX feature, second column = enrichment identifier.",
      call. = FALSE
    )
  }
  required <- c("feature", "enrichment_id")
  enrichment_id_map <- enrichment_id_map[, seq_len(2), drop = FALSE]
  names(enrichment_id_map) <- required

  if (is.null(features) || !length(features)) {
    stop("Cannot validate enrichment_id_map because no feature names were supplied.", call. = FALSE)
  }
  features <- as.character(features)
  if (any(is.na(features)) || any(!nzchar(features))) {
    stop("All SVDX features used for enrichment must be non-missing, non-empty strings.", call. = FALSE)
  }
  if (anyDuplicated(features)) {
    dups <- unique(features[duplicated(features)])
    stop(
      "SVDX feature names used for enrichment must be unique. ",
      "Feature names are taken from rownames(Y) / inf$union_results$feature. ",
      "Duplicated examples include: ", paste(utils::head(dups, 5), collapse = ", "),
      call. = FALSE
    )
  }
  map <- enrichment_id_map[, required, drop = FALSE]
  map$feature <- as.character(map$feature)
  map$enrichment_id <- as.character(map$enrichment_id)

  bad_feature <- is.na(map$feature) | !nzchar(map$feature)
  if (any(bad_feature)) {
    stop(
      "enrichment_id_map contains missing or empty feature values.",
      call. = FALSE
    )
  }

  # A modeled feature need not have an identifier in the enrichment namespace.
  # Preserve the feature in the map and represent unmapped enrichment IDs as NA;
  # enrichment backends exclude these features from their mapped universe.
  map$enrichment_id[
    is.na(map$enrichment_id) | !nzchar(map$enrichment_id)
  ] <- NA_character_

  if (anyDuplicated(map$feature)) {
    dups <- unique(map$feature[duplicated(map$feature)])
    stop(
      "enrichment_id_map must contain at most one enrichment_id per SVDX feature. ",
      "Duplicated feature values include: ",
      paste(utils::head(dups, 5), collapse = ", "),
      call. = FALSE
    )
  }

  missing <- setdiff(features, map$feature)
  if (length(missing)) {
    stop(
      "enrichment_id_map does not contain mappings for all features in ", context, ". ",
      "Missing examples: ", paste(utils::head(missing, 5), collapse = ", "),
      call. = FALSE
    )
  }

  map <- map[match(features, map$feature), , drop = FALSE]

  mapped_ids <- map$enrichment_id[
    !is.na(map$enrichment_id) & nzchar(map$enrichment_id)
  ]
  if (isTRUE(require_unique) && anyDuplicated(mapped_ids)) {
    dups <- unique(mapped_ids[duplicated(mapped_ids)])
    stop(
      "GSEA-style enrichment backends require unique enrichment identifiers after applying enrichment_id_map. ",
      "Multiple SVDX features map to the same enrichment_id; duplicated examples include: ",
      paste(utils::head(dups, 5), collapse = ", "), ". ",
      "SVDX does not collapse or aggregate features for FGSEA. ",
      "Please provide a one-to-one enrichment_id_map for GSEA-style enrichment, or consider ORA-based enrichment options ",
      "such as enrichment_backend='internal' or enrichment_backend='clusterProfiler', which do not require unique enrichment IDs.",
      call. = FALSE
    )
  }

  map
}

.svdx_validate_fgsea_ids_before_fit <- function(
    features,
    enrichment_id_map,
    context = "Y"
) {
  .svdx_validate_enrichment_id_map(
    enrichment_id_map = enrichment_id_map,
    features = features,
    context = context,
    require_unique = TRUE
  )
  invisible(TRUE)
}

.svdx_map_feature_ids <- function(features, enrichment_id_map) {
  map <- .svdx_validate_enrichment_id_map(
    enrichment_id_map = enrichment_id_map,
    features = features,
    context = "inference object",
    require_unique = FALSE
  )
  ids <- map$enrichment_id
  names(ids) <- map$feature
  ids
}

.svdx_map_feature_vector <- function(x, enrichment_id_map) {
  features <- as.character(x)
  if (!length(features)) return(character(0))
  ids <- .svdx_map_feature_ids(features, enrichment_id_map)
  unname(ids[features])
}

.svdx_enrichment_feature_universe <- function(inf, universe = NULL, enrichment_id_map = NULL) {
  if (!is.null(universe)) return(as.character(universe))
  features <- svdxUniverse(inf)
  if (is.null(features) || !length(features)) {
    stop(
      "Could not determine enrichment feature universe from the inference object. ",
      "Features are expected in inf$union_results$feature.",
      call. = FALSE
    )
  }
  if (!is.null(enrichment_id_map)) {
    ids <- .svdx_map_feature_vector(features, enrichment_id_map)
    ids <- ids[!is.na(ids) & nzchar(ids)]
    return(unique(ids))
  }
  as.character(features)
}

.svdx_feature_set_features <- function(feature_sets = NULL, term2gene = NULL) {
  if (!is.null(feature_sets)) {
    return(unique(as.character(unlist(feature_sets, use.names = FALSE))))
  }
  if (!is.null(term2gene)) {
    if (!is.data.frame(term2gene) || ncol(term2gene) < 2) {
      stop("term2gene must be a data frame with at least two columns.", call. = FALSE)
    }
    gene_col <- if ("gene" %in% names(term2gene)) "gene" else if ("gene_symbol" %in% names(term2gene)) "gene_symbol" else names(term2gene)[2]
    return(unique(as.character(term2gene[[gene_col]])))
  }
  character()
}

.svdx_warn_low_feature_overlap <- function(
    inf,
    feature_sets = NULL,
    term2gene = NULL,
    universe = NULL,
    enrichment_id_map = NULL,
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    backend = NULL
) {
  if (is.null(min_feature_overlap_warn) && is.null(min_feature_overlap_fraction)) {
    return(invisible(NULL))
  }

  universe <- .svdx_enrichment_feature_universe(inf, universe = universe, enrichment_id_map = enrichment_id_map)
  set_features <- .svdx_feature_set_features(feature_sets = feature_sets, term2gene = term2gene)
  set_features <- set_features[!is.na(set_features) & nzchar(set_features)]
  if (!length(universe) || !length(set_features)) return(invisible(NULL))

  n_overlap <- length(intersect(universe, set_features))
  overlap_fraction <- n_overlap / length(unique(universe))
  warn_count <- !is.null(min_feature_overlap_warn) && n_overlap < min_feature_overlap_warn
  warn_fraction <- !is.null(min_feature_overlap_fraction) && overlap_fraction < min_feature_overlap_fraction

  if (warn_count || warn_fraction) {
    warning(
      "Only ", n_overlap, " inferred features overlap the enrichment feature sets",
      if (!is.null(backend)) paste0(" for backend='", backend, "'") else "",
      ". Feature names in inf$union_results$feature must use the same identifier system ",
      "as feature_sets, TERM2GENE, or the selected enrichment database.",
      call. = FALSE
    )
  }

  invisible(list(n_overlap = n_overlap, overlap_fraction = overlap_fraction))
}

.svdx_resolve_term2gene <- function(
    term2gene = NULL,
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl")
) {
  if (!is.null(term2gene)) return(term2gene)

  enrichment_id_type <- match.arg(enrichment_id_type)
  getMSigDB(
    db_species = msigdb_db_species,
    species = msigdb_species,
    collection = msigdb_collection,
    subcollection = msigdb_subcollection,
    id_type = enrichment_id_type,
    as = "term2gene"
  )
}

.svdx_resolve_feature_sets <- function(
    feature_sets = NULL,
    term2gene = NULL,
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl")
) {
  if (!is.null(feature_sets)) return(feature_sets)

  term2gene <- .svdx_resolve_term2gene(
    term2gene = term2gene,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = match.arg(enrichment_id_type)
  )

  .svdx_term2gene_to_feature_sets(term2gene)
}

.resolve_enrichment_backend <- function(
    backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    feature_sets = NULL,
    term2gene = NULL
) {
  backend <- match.arg(backend)
  if (backend != "auto") return(backend)

  # Keep automatic resolution conservative and reproducible.
  # The internal ORA backend can operate with user-supplied feature sets
  # or package-resolved MSigDB sets and therefore remains the safest
  # default when users request automatic enrichment.
  "internal"
}

.svdx_empty_enrichment_table <- function() {
  tibble::tibble(
    backend = character(), method = character(), scope = character(),
    component = character(), direction = character(), term_name = character(),
    term_id = character(), collection = character(), subcollection = character(),
    n_query = integer(), n_term = integer(), n_overlap = integer(),
    overlap_features = character(), score = numeric(), statistic = numeric(),
    pvalue = numeric(), padj = numeric(), qvalue = numeric(), rank_metric = character()
  )
}

.svdx_enrichment_result <- function(
    results,
    raw,
    backend_requested,
    backend_resolved,
    params = list()
) {
  out <- list(
    backend_requested = backend_requested,
    backend_resolved = backend_resolved,
    backend = backend_resolved,
    params = params,
    results = results,
    raw = raw
  )
  class(out) <- c("svdxEnrichment", "list")
  out
}

.svdx_enrichment_results <- function(x) {
  if (inherits(x, "svdxEnrichment")) return(x$results)
  if (is.list(x) && !is.null(x$results)) return(x$results)
  x
}

.svdx_collapse_features <- function(x) {
  if (is.null(x)) return(NA_character_)
  if (is.list(x)) {
    return(vapply(x, function(z) paste(as.character(z), collapse = ";"), character(1)))
  }
  as.character(x)
}

.svdx_parse_ratio_num <- function(x, part = c("num", "den")) {
  part <- match.arg(part)
  if (is.null(x)) return(NA_integer_)
  x <- as.character(x)
  pieces <- strsplit(x, "/", fixed = TRUE)
  vapply(pieces, function(z) {
    if (length(z) < 2) return(NA_integer_)
    val <- suppressWarnings(as.integer(z[if (part == "num") 1 else 2]))
    ifelse(is.na(val), NA_integer_, val)
  }, integer(1))
}

.safe_fisher_enrichment <- function(discovered, in_set) {
  discovered <- as.logical(discovered)
  in_set <- as.logical(in_set)
  
  tab <- matrix(
    c(
      sum(discovered & in_set, na.rm = TRUE),
      sum(discovered & !in_set, na.rm = TRUE),
      sum(!discovered & in_set, na.rm = TRUE),
      sum(!discovered & !in_set, na.rm = TRUE)
    ),
    nrow = 2,
    byrow = TRUE
  )
  
  dimnames(tab) <- list(
    significant = c("yes", "no"),
    in_set = c("yes", "no")
  )
  
  if (sum(tab[1, ]) == 0 || sum(tab[, 1]) == 0) {
    return(list(odds_ratio = NA_real_, p_value = NA_real_))
  }
  
  ft <- fisher.test(tab, alternative = "greater")
  list(odds_ratio = unname(ft$estimate), p_value = ft$p.value)
}

.svdx_internal_ora <- function(
    inf,
    feature_sets,
    alpha = inf$alpha %||% 0.05,
    scope = c("union", "component", "global"),
    component = NULL,
    universe = NULL,
    enrichment_id_map = NULL
) {
  scope <- match.arg(scope)

  if (is.null(universe)) universe <- svdxUniverse(inf)
  significant <- svdxSignificant(inf, alpha = alpha, scope = scope, component = component)
  if (!is.null(enrichment_id_map)) {
    universe <- unique(.svdx_map_feature_vector(universe, enrichment_id_map))
    significant <- unique(.svdx_map_feature_vector(significant, enrichment_id_map))
    universe <- universe[!is.na(universe) & nzchar(universe)]
    significant <- significant[!is.na(significant) & nzchar(significant)]
  }
  component_value <- if (is.null(component)) NA_character_ else as.character(component)

  if (is.null(feature_sets) || length(feature_sets) == 0L) {
    message("Enrichment skipped: no gene sets were resolved.")
    return(tibble::tibble(
      pathway = character(), scope = character(), component = character(),
      set_size = integer(), n_significant = integer(), overlap = integer(),
      overlap_features = character(), odds_ratio = numeric(), p_value = numeric(), q_value = numeric()
    ))
  }

  # Restrict the enrichment universe to identifiers represented in
  # the supplied feature-set collection.
  evaluable_ids <- unique(as.character(unlist(feature_sets, use.names = FALSE)))
  evaluable_ids <- evaluable_ids[
    !is.na(evaluable_ids) & nzchar(evaluable_ids)
  ]

  universe <- intersect(universe, evaluable_ids)
  significant <- intersect(significant, universe)

  if (!length(universe)) {
    message("Enrichment skipped: no valid enrichment IDs overlap the inference universe.")
    return(tibble::tibble(
      pathway = character(), scope = character(), component = character(),
      set_size = integer(), n_significant = integer(), overlap = integer(),
      overlap_features = character(), odds_ratio = numeric(), p_value = numeric(), q_value = numeric()
    ))
  }

  if (!length(significant)) {
    message("Enrichment skipped: no significant features were available for scope '", scope, "'.")
    return(tibble::tibble(
      pathway = character(), scope = character(), component = character(),
      set_size = integer(), n_significant = integer(), overlap = integer(),
      overlap_features = character(), odds_ratio = numeric(), p_value = numeric(), q_value = numeric()
    ))
  }

  out <- lapply(names(feature_sets), function(set_name) {
    set_features <- intersect(feature_sets[[set_name]], universe)
    in_set <- universe %in% set_features
    discovered <- universe %in% significant
    overlap_features <- universe[discovered & in_set]
    sf <- .safe_fisher_enrichment(discovered, in_set)
    
    tibble::tibble(
      pathway = set_name,
      scope = scope,
      component = component_value,
      set_size = length(set_features),
      n_significant = length(significant),
      overlap = length(overlap_features),
      overlap_features = paste(overlap_features, collapse = ";"),
      odds_ratio = sf$odds_ratio,
      p_value = sf$p_value
    )
  })
  
  out <- dplyr::bind_rows(out)
  if (nrow(out) == 0L || !any(out$overlap > 0L, na.rm = TRUE)) {
    message("Enrichment skipped: resolved gene sets had no overlap with the inference universe/significant features.")
    return(out |> dplyr::mutate(q_value = .safe_bh(p_value)))
  }
  out |>
    dplyr::mutate(q_value = .safe_bh(p_value)) |>
    dplyr::arrange(q_value, p_value)
}

.normalize_enrichment_result <- function(
    x,
    backend = c("internal", "clusterProfiler", "fgsea"),
    scope = "union",
    component = NULL,
    direction = "any",
    collection = NA_character_,
    subcollection = NA_character_,
    rank_metric = NA_character_
) {
  backend <- match.arg(backend)
  component_value <- if (is.null(component)) NA_character_ else as.character(component)

  if (backend == "internal") {
    df <- as.data.frame(x)
    if (nrow(df) == 0L) return(.svdx_empty_enrichment_table())
    return(tibble::tibble(
      backend = "internal",
      method = "ORA",
      scope = as.character(df$scope %||% scope),
      component = as.character(df$component %||% component_value),
      direction = direction,
      term_name = as.character(df$pathway),
      term_id = NA_character_,
      collection = collection,
      subcollection = subcollection,
      n_query = as.integer(df$n_significant),
      n_term = as.integer(df$set_size),
      n_overlap = as.integer(df$overlap),
      overlap_features = as.character(df$overlap_features %||% NA_character_),
      score = as.numeric(df$odds_ratio),
      statistic = as.numeric(df$odds_ratio),
      pvalue = as.numeric(df$p_value),
      padj = as.numeric(df$q_value),
      qvalue = as.numeric(df$q_value),
      rank_metric = NA_character_
    ))
  }

  if (backend == "clusterProfiler") {
    df <- as.data.frame(x)
    if (nrow(df) == 0) {
      return(.svdx_empty_enrichment_table())
    }
    gene_ratio <- df$GeneRatio %||% NA_character_
    bg_ratio <- df$BgRatio %||% NA_character_
    return(tibble::tibble(
      backend = "clusterProfiler",
      method = if ("NES" %in% names(df) || "enrichmentScore" %in% names(df)) "GSEA" else "ORA",
      scope = scope,
      component = component_value,
      direction = direction,
      term_name = as.character(df$Description %||% df$ID),
      term_id = as.character(df$ID %||% NA_character_),
      collection = collection,
      subcollection = subcollection,
      n_query = .svdx_parse_ratio_num(gene_ratio, "den"),
      n_term = as.integer(df$setSize %||% .svdx_parse_ratio_num(bg_ratio, "num")),
      n_overlap = if ("core_enrichment" %in% names(df)) lengths(strsplit(as.character(df$core_enrichment), "/", fixed = TRUE)) else as.integer(df$Count %||% .svdx_parse_ratio_num(gene_ratio, "num")),
      overlap_features = as.character(df$core_enrichment %||% df$geneID %||% NA_character_),
      score = as.numeric(df$NES %||% df$Count %||% NA_real_),
      statistic = as.numeric(df$enrichmentScore %||% df$Count %||% NA_real_),
      pvalue = as.numeric(df$pvalue %||% NA_real_),
      padj = as.numeric(df$p.adjust %||% NA_real_),
      qvalue = as.numeric(df$qvalues %||% df$qvalue %||% df$p.adjust %||% NA_real_),
      rank_metric = rank_metric
    ))
  }

  df <- as.data.frame(x)
  if (nrow(df) == 0) return(.svdx_empty_enrichment_table())

  tibble::tibble(
    backend = "fgsea",
    method = "GSEA",
    scope = scope,
    component = component_value,
    direction = direction,
    term_name = as.character(df$pathway),
    term_id = NA_character_,
    collection = collection,
    subcollection = subcollection,
    n_query = NA_integer_,
    n_term = as.integer(df$size %||% NA_integer_),
    n_overlap = if ("leadingEdge" %in% names(df)) lengths(df$leadingEdge) else NA_integer_,
    overlap_features = if ("leadingEdge" %in% names(df)) .svdx_collapse_features(df$leadingEdge) else NA_character_,
    score = as.numeric(df$NES %||% NA_real_),
    statistic = as.numeric(df$ES %||% NA_real_),
    pvalue = as.numeric(df$pval %||% NA_real_),
    padj = as.numeric(df$padj %||% NA_real_),
    qvalue = as.numeric(df$padj %||% NA_real_),
    rank_metric = rank_metric
  )
}


.svdx_enrichment_components <- function(inf) {
  comps <- sort(unique(inf$feature_results$component))
  comps[!is.na(comps)]
}

.svdx_combine_enrichment_objects <- function(objects, backend_requested, params = list()) {
  objects <- objects[!vapply(objects, is.null, logical(1))]
  objects <- objects[vapply(objects, function(x) !is.null(x$results), logical(1))]
  if (!length(objects)) return(NULL)
  out <- list(
    backend_requested = backend_requested,
    backend_resolved = vapply(objects, function(x) as.character(x$backend_resolved %||% x$backend), character(1)),
    backend = unique(vapply(objects, function(x) as.character(x$backend_resolved %||% x$backend), character(1))),
    params = params,
    results = dplyr::bind_rows(lapply(objects, `[[`, "results")),
    raw = lapply(objects, `[[`, "raw")
  )
  names(out$raw) <- paste0(
    vapply(objects, function(x) unique(as.character(x$results$scope))[1] %||% "scope", character(1)),
    "_",
    vapply(objects, function(x) unique(as.character(x$results$component))[1] %||% "NA", character(1))
  )
  class(out) <- c("svdxEnrichment", "list")
  out
}
enrichPathways.svdxInference <- function(
    inf,
    feature_sets = NULL,
    alpha = inf$alpha %||% 0.05,
    scope = c("all", "union", "component", "global"),
    component = NULL,
    universe = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    ...
) {
  scope <- match.arg(scope)
  backend_requested <- match.arg(enrichment_backend)

  if (scope == "all" || (scope == "component" && is.null(component))) {
    component_values <- if (is.null(component)) .svdx_enrichment_components(inf) else component
    calls <- list()
    if (scope == "all") {
      calls[["union"]] <- enrichPathways(
        inf = inf, feature_sets = feature_sets, alpha = alpha, scope = "union", component = NULL,
        universe = universe, enrichment_backend = enrichment_backend, msigdb_db_species = msigdb_db_species,
        msigdb_species = msigdb_species, msigdb_collection = msigdb_collection,
        msigdb_subcollection = msigdb_subcollection, enrichment_id_type = enrichment_id_type,
        enrichment_score = enrichment_score, term2gene = term2gene, enrichment_id_map = enrichment_id_map,
        enrichment_extra_args = enrichment_extra_args, min_feature_overlap_warn = min_feature_overlap_warn,
        min_feature_overlap_fraction = min_feature_overlap_fraction, ...
      )
      calls[["global"]] <- enrichPathways(
        inf = inf, feature_sets = feature_sets, alpha = alpha, scope = "global", component = NULL,
        universe = universe, enrichment_backend = enrichment_backend, msigdb_db_species = msigdb_db_species,
        msigdb_species = msigdb_species, msigdb_collection = msigdb_collection,
        msigdb_subcollection = msigdb_subcollection, enrichment_id_type = enrichment_id_type,
        enrichment_score = if (identical(enrichment_score, "auto")) "global_q" else enrichment_score, term2gene = term2gene, enrichment_id_map = enrichment_id_map,
        enrichment_extra_args = enrichment_extra_args, min_feature_overlap_warn = min_feature_overlap_warn,
        min_feature_overlap_fraction = min_feature_overlap_fraction, ...
      )
    }
    for (cc in component_values) {
      calls[[paste0("component_", cc)]] <- enrichPathways(
        inf = inf, feature_sets = feature_sets, alpha = alpha, scope = "component", component = cc,
        universe = universe, enrichment_backend = enrichment_backend, msigdb_db_species = msigdb_db_species,
        msigdb_species = msigdb_species, msigdb_collection = msigdb_collection,
        msigdb_subcollection = msigdb_subcollection, enrichment_id_type = enrichment_id_type,
        enrichment_score = enrichment_score, term2gene = term2gene, enrichment_id_map = enrichment_id_map,
        enrichment_extra_args = enrichment_extra_args, min_feature_overlap_warn = min_feature_overlap_warn,
        min_feature_overlap_fraction = min_feature_overlap_fraction, ...
      )
    }
    return(.svdx_combine_enrichment_objects(
      calls,
      backend_requested = backend_requested,
      params = list(scope = scope, component = component, enrichment_id_map = enrichment_id_map)
    ))
  }
  enrichment_id_type <- match.arg(enrichment_id_type)
  backend_resolved <- .resolve_enrichment_backend(
    backend = backend_requested,
    feature_sets = feature_sets,
    term2gene = term2gene
  )
  features_for_enrichment <- svdxUniverse(inf)
  if (is.null(enrichment_id_map)) {
    stop(
      "enrichment_id_map must be supplied when enrichment analysis is requested. ",
      "It must be a two-column data frame: first column = SVDX feature, second column = enrichment identifier.",
      call. = FALSE
    )
  }
  enrichment_id_map <- .svdx_validate_enrichment_id_map(
    enrichment_id_map = enrichment_id_map,
    features = features_for_enrichment,
    context = "inference object",
    require_unique = backend_resolved %in% c("fgsea", "clusterProfiler")
  )

  backend_args <- c(list(...), enrichment_extra_args)

  params <- list(
    backend_requested = backend_requested,
    backend_resolved = backend_resolved,
    alpha = alpha,
    scope = scope,
    component = component,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    id_type = enrichment_id_type,
    score = enrichment_score,
    enrichment_id_map = enrichment_id_map,
    extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction
  )

  if (backend_resolved == "internal") {
    feature_sets <- .svdx_resolve_feature_sets(
      feature_sets = feature_sets,
      term2gene = term2gene,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type
    )
    .svdx_warn_low_feature_overlap(
      inf = inf,
      feature_sets = feature_sets,
      universe = universe,
      enrichment_id_map = enrichment_id_map,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction,
      backend = "internal"
    )
    raw <- .svdx_internal_ora(
      inf = inf,
      feature_sets = feature_sets,
      alpha = alpha,
      scope = scope,
      component = component,
      universe = universe,
      enrichment_id_map = enrichment_id_map
    )
    results <- .normalize_enrichment_result(
      raw,
      backend = "internal",
      scope = scope,
      component = component,
      collection = msigdb_collection,
      subcollection = msigdb_subcollection,
      rank_metric = enrichment_score
    )
    return(.svdx_enrichment_result(
      results = results,
      raw = raw,
      backend_requested = backend_requested,
      backend_resolved = "internal",
      params = params
    ))
  }

  if (backend_resolved == "clusterProfiler") {
    term2gene <- .svdx_resolve_term2gene(
      term2gene = term2gene,
      msigdb_db_species = msigdb_db_species,
      msigdb_species = msigdb_species,
      msigdb_collection = msigdb_collection,
      msigdb_subcollection = msigdb_subcollection,
      enrichment_id_type = enrichment_id_type
    )
    .svdx_warn_low_feature_overlap(
      inf = inf,
      term2gene = term2gene,
      universe = universe,
      enrichment_id_map = enrichment_id_map,
      min_feature_overlap_warn = min_feature_overlap_warn,
      min_feature_overlap_fraction = min_feature_overlap_fraction,
      backend = "clusterProfiler"
    )
    raw <- do.call(runClusterProfilerGSEA, c(list(
      inf = inf,
      term2gene = term2gene,
      score = enrichment_score,
      scope = scope,
      component = component,
      enrichment_id_map = enrichment_id_map
    ), backend_args))
    results <- .normalize_enrichment_result(
      raw,
      backend = "clusterProfiler",
      scope = scope,
      component = component,
      collection = msigdb_collection,
      subcollection = msigdb_subcollection,
      rank_metric = enrichment_score
    )
    return(.svdx_enrichment_result(
      results = results,
      raw = raw,
      backend_requested = backend_requested,
      backend_resolved = "clusterProfiler",
      params = params
    ))
  }

  pathways <- .svdx_resolve_feature_sets(
    feature_sets = feature_sets,
    term2gene = term2gene,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type
  )
  .svdx_warn_low_feature_overlap(
    inf = inf,
    feature_sets = pathways,
    universe = universe,
    enrichment_id_map = enrichment_id_map,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    backend = "fgsea"
  )
  raw <- do.call(runFGSEA, c(list(
    inf = inf,
    pathways = pathways,
    score = enrichment_score,
    scope = scope,
    component = component,
    enrichment_id_map = enrichment_id_map
  ), backend_args))
  results <- .normalize_enrichment_result(
    raw,
    backend = "fgsea",
    scope = scope,
    component = component,
    collection = msigdb_collection,
    subcollection = msigdb_subcollection,
    rank_metric = enrichment_score
  )
  .svdx_enrichment_result(
    results = results,
    raw = raw,
    backend_requested = backend_requested,
    backend_resolved = "fgsea",
    params = params
  )
}

enrichPathways.svdxWorkflow <- function(
    inf,
    feature_sets = NULL,
    alpha = inf$alpha %||% 0.05,
    scope = c("all", "union", "component", "global"),
    component = NULL,
    universe = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    ...
) {
  enrichPathways(
    inf = inf$inf,
    feature_sets = feature_sets,
    alpha = alpha,
    scope = scope,
    component = component,
    universe = universe,
    enrichment_backend = enrichment_backend,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    term2gene = term2gene,
    enrichment_id_map = enrichment_id_map,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    ...
  )
}

#' Run fgsea from SVDX inference
#'
#' Builds a ranked vector from SVDX inference, maps SVDX feature IDs to
#' enrichment IDs using `enrichment_id_map`, and calls `fgsea::fgsea()`.
#'
#' @param inf An `svdxInference` object.
#' @param pathways Named list of gene sets, with members expressed in the same
#'   identifier system as the second column of `enrichment_id_map`.
#' @param score Ranking score. The default `"auto"` uses q-value based scores.
#' @param scope `"union"` or `"component"`.
#' @param component Component number for component-level enrichment.
#' @param enrichment_id_map Two-column feature-to-enrichment-ID map.
#' @param ... Additional arguments passed to `fgsea::fgsea()`.
#'
#' @return Raw `fgsea` result table.
#'
runFGSEA <- function(
    inf,
    pathways = NULL,
    score = "auto",
    scope = "union",
    component = NULL,
    enrichment_id_map,
    ...
) {
  if (is.null(pathways)) stop("pathways must be supplied to runFGSEA().", call. = FALSE)
  if (missing(enrichment_id_map) || is.null(enrichment_id_map)) {
    stop("enrichment_id_map must be supplied for FGSEA.", call. = FALSE)
  }
  stats <- asRankedList(inf, score = score, scope = scope, component = component)
  mapped_ids <- .svdx_map_feature_vector(names(stats), enrichment_id_map)
  keep <- !is.na(mapped_ids) & nzchar(mapped_ids)
  stats <- stats[keep]
  names(stats) <- mapped_ids[keep]

  if (!length(stats)) {
    message("FGSEA skipped: no valid ranked enrichment IDs were available after mapping.")
    return(data.frame())
  }

  if (is.null(pathways) || length(pathways) == 0L) {
    message("FGSEA skipped: no gene sets were resolved.")
    return(data.frame())
  }

  if (anyDuplicated(names(stats))) {
    dups <- unique(names(stats)[duplicated(names(stats))])
    stop(
      "GSEA-style enrichment backends require unique enrichment identifiers after applying enrichment_id_map. ",
      "Duplicated examples include: ", paste(utils::head(dups, 5), collapse = ", "), ". ",
      "SVDX does not collapse or aggregate features for FGSEA. ",
      "Please provide a one-to-one enrichment_id_map for GSEA-style enrichment, or consider ORA-based enrichment options ",
      "such as enrichment_backend='internal' or enrichment_backend='clusterProfiler', which do not require unique enrichment IDs.",
      call. = FALSE
    )
  }

  .require_namespace("fgsea")
  fgsea::fgsea(pathways = pathways, stats = stats, ...)
}

runClusterProfilerORA <- function(
    inf,
    term2gene,
    alpha = inf$alpha %||% 0.05,
    scope = "union",
    component = NULL,
    universe = NULL,
    enrichment_id_map = NULL,
    ...
) {
  .require_namespace("clusterProfiler")
  
  sig <- svdxSignificant(inf, alpha = alpha, scope = scope, component = component)
  if (is.null(universe)) universe <- svdxUniverse(inf)
  if (!is.null(enrichment_id_map)) {
    sig <- unique(.svdx_map_feature_vector(sig, enrichment_id_map))
    universe <- unique(.svdx_map_feature_vector(universe, enrichment_id_map))
    sig <- sig[!is.na(sig) & nzchar(sig)]
    universe <- universe[!is.na(universe) & nzchar(universe)]
  }
  
  clusterProfiler::enricher(
    gene = sig,
    universe = universe,
    TERM2GENE = term2gene,
    ...
  )
}

#' Run clusterProfiler GSEA from SVDX inference
#'
#' Builds a ranked vector from SVDX inference, maps SVDX feature IDs to
#' enrichment IDs using `enrichment_id_map`, and calls
#' `clusterProfiler::GSEA()` with `TERM2GENE`.
#'
#' @param inf An `svdxInference` object.
#' @param term2gene TERM2GENE table whose gene column uses the same identifier
#'   system as the second column of `enrichment_id_map`.
#' @param score Ranking score. The default `"auto"` uses q-value based scores.
#' @param scope `"union"` or `"component"`.
#' @param component Component number for component-level enrichment.
#' @param enrichment_id_map Two-column feature-to-enrichment-ID map.
#' @param ... Additional arguments passed to `clusterProfiler::GSEA()`.
#'
#' @return Raw `clusterProfiler::GSEA()` result object.
#'
runClusterProfilerGSEA <- function(
    inf,
    term2gene,
    score = "auto",
    scope = "union",
    component = NULL,
    enrichment_id_map = NULL,
    ...
) {
  geneList <- asRankedList(inf, score = score, scope = scope, component = component)
  if (!is.null(enrichment_id_map)) {
    mapped_ids <- .svdx_map_feature_vector(names(geneList), enrichment_id_map)
    keep <- !is.na(mapped_ids) & nzchar(mapped_ids)
    geneList <- geneList[keep]
    names(geneList) <- mapped_ids[keep]
  }
  if (!length(geneList)) {
    message("clusterProfiler GSEA skipped: no valid ranked enrichment IDs were available after mapping.")
    return(data.frame())
  }
  if (anyDuplicated(names(geneList))) {
    dups <- unique(names(geneList)[duplicated(names(geneList))])
    stop(
      "clusterProfiler::GSEA requires unique enrichment identifiers after applying enrichment_id_map. ",
      "Duplicated examples include: ", paste(utils::head(dups, 5), collapse = ", "), ". ",
      "SVDX does not collapse or aggregate features for GSEA-style enrichment. ",
      "Please provide a one-to-one enrichment_id_map, or use enrichment_backend='internal' for ORA-based enrichment.",
      call. = FALSE
    )
  }
  geneList <- sort(geneList, decreasing = TRUE)
  .require_namespace("clusterProfiler")
  clusterProfiler::GSEA(geneList = geneList, TERM2GENE = term2gene, ...)
}

#' Get MSigDB gene sets for SVDX enrichment
#'
#' Retrieves MSigDB annotations with `msigdbr::msigdbr()` and converts them to
#' either a TERM2GENE table for `clusterProfiler::GSEA()` or a named list of
#' feature sets for the internal and `fgsea` backends.
#'
#' @param db_species MSigDB database species passed directly to
#'   `msigdbr::msigdbr(db_species=...)`, for example `"HS"` or `"MM"`.
#' @param species Species passed directly to `msigdbr::msigdbr(species=...)`,
#'   for example `"Homo sapiens"` or `"Mus musculus"`.
#' @param collection Collection passed directly to
#'   `msigdbr::msigdbr(collection=...)`. Use
#'   `msigdbr::msigdbr_collections()` to inspect valid values.
#' @param subcollection Optional subcollection passed directly to
#'   `msigdbr::msigdbr(subcollection=...)`.
#' @param id_type Identifier column to extract: `"symbol"`, `"entrez"`, or
#'   `"ensembl"`.
#' @param as Output format: `"term2gene"` or `"feature_sets"`.
#'
#' @return A two-column TERM2GENE data frame or a named list of feature sets.
#'
getMSigDB <- function(
    db_species,
    species,
    collection,
    subcollection = NULL,
    id_type = c("symbol", "entrez", "ensembl"),
    as = c("term2gene", "feature_sets")
) {
  .require_namespace("msigdbr")
  .require_namespace("dplyr")

  id_type <- match.arg(id_type)
  as <- match.arg(as)

  if (missing(db_species) || is.null(db_species) || !nzchar(as.character(db_species))) {
    stop("msigdb_db_species must be supplied when feature_sets/term2gene are not supplied.", call. = FALSE)
  }
  if (missing(species) || is.null(species) || !nzchar(as.character(species))) {
    stop("msigdb_species must be supplied when feature_sets/term2gene are not supplied.", call. = FALSE)
  }
  if (missing(collection) || is.null(collection) || !nzchar(as.character(collection))) {
    stop("msigdb_collection must be supplied when feature_sets/term2gene are not supplied. Use msigdbr::msigdbr_collections() to see valid options.", call. = FALSE)
  }

  args <- list(
    db_species = as.character(db_species),
    species = as.character(species),
    collection = as.character(collection)
  )
  if (!is.null(subcollection)) args$subcollection <- as.character(subcollection)

  msig <- do.call(msigdbr::msigdbr, args)

  id_col <- switch(
    id_type,
    symbol = "gene_symbol",
    entrez = "ncbi_gene",
    ensembl = "ensembl_gene"
  )

  if (!id_col %in% names(msig)) {
    stop(
      "MSigDB result does not contain the requested identifier column '", id_col, "'. ",
      "Try another enrichment_id_type or supply feature_sets/term2gene directly.",
      call. = FALSE
    )
  }

  term2gene <- msig |>
    dplyr::transmute(
      term = .data$gs_name,
      gene = as.character(.data[[id_col]])
    ) |>
    dplyr::filter(!is.na(.data$term), nzchar(.data$term), !is.na(.data$gene), nzchar(.data$gene)) |>
    dplyr::distinct()

  if (identical(as, "feature_sets")) {
    return(.svdx_term2gene_to_feature_sets(term2gene))
  }

  term2gene
}

# ============================================================
# isvdxInference objects
# ============================================================

.get_isvdx_dataset_inf <- function(x, dataset) {
  if (is.null(dataset)) {
    stop("dataset must be supplied for isvdxInference objects.", call. = FALSE)
  }
  
  d <- if (is.numeric(dataset)) dataset else match(dataset, names(x$dataset_results))
  
  if (is.na(d)) {
    stop(
      "Unknown dataset: ", dataset, ". Available datasets are: ",
      paste(names(x$dataset_results), collapse = ", "),
      call. = FALSE
    )
  }
  
  x$dataset_results[[d]]
}

topFeatures.isvdxInference <- function(x, dataset, scope = c("union", "component", "global"), component = NULL, n = 20) {
  topFeatures(
    .get_isvdx_dataset_inf(x, dataset),
    scope = scope,
    component = component,
    n = n
  )
}

svdxSignificant.isvdxInference <- function(x, dataset, alpha = x$alpha %||% 0.05, scope = c("union", "component", "global"), component = NULL) {
  svdxSignificant(
    .get_isvdx_dataset_inf(x, dataset),
    alpha = alpha,
    scope = scope,
    component = component
  )
}

svdxUniverse.isvdxInference <- function(x, dataset) {
  svdxUniverse(.get_isvdx_dataset_inf(x, dataset))
}

asRankedList.isvdxInference <- function(x, dataset, score = "auto", scope = "union", component = NULL) {
  asRankedList(
    .get_isvdx_dataset_inf(x, dataset),
    score = score,
    scope = scope,
    component = component
  )
}

enrichPathways.isvdxInference <- function(
    inf,
    dataset,
    feature_sets = NULL,
    alpha = inf$alpha %||% 0.05,
    scope = c("all", "union", "component", "global"),
    component = NULL,
    universe = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    ...
) {
  enrichPathways(
    inf = .get_isvdx_dataset_inf(inf, dataset),
    feature_sets = feature_sets,
    alpha = alpha,
    scope = scope,
    component = component,
    universe = universe,
    enrichment_backend = enrichment_backend,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    term2gene = term2gene,
    enrichment_id_map = enrichment_id_map,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    ...
  )
}

# ============================================================
# isvdxWorkflow objects
# ============================================================

topFeatures.isvdxWorkflow <- function(x, dataset, scope = c("union", "component", "global"), component = NULL, n = 20) {
  topFeatures(
    x$inf,
    dataset = dataset,
    scope = scope,
    component = component,
    n = n
  )
}

svdxSignificant.isvdxWorkflow <- function(x, dataset, alpha = x$alpha %||% 0.05, scope = c("union", "component", "global"), component = NULL) {
  svdxSignificant(
    x$inf,
    dataset = dataset,
    alpha = alpha,
    scope = scope,
    component = component
  )
}

svdxUniverse.isvdxWorkflow <- function(x, dataset) {
  svdxUniverse(x$inf, dataset = dataset)
}

asRankedList.isvdxWorkflow <- function(x, dataset, score = "auto", scope = "union", component = NULL) {
  asRankedList(
    x$inf,
    dataset = dataset,
    score = score,
    scope = scope,
    component = component
  )
}

enrichPathways.isvdxWorkflow <- function(
    inf,
    dataset,
    feature_sets = NULL,
    alpha = inf$alpha %||% 0.05,
    scope = c("all", "union", "component", "global"),
    component = NULL,
    universe = NULL,
    enrichment_backend = c("auto", "internal", "clusterProfiler", "fgsea"),
    msigdb_db_species = NULL,
    msigdb_species = NULL,
    msigdb_collection = NULL,
    msigdb_subcollection = NULL,
    enrichment_id_type = c("symbol", "entrez", "ensembl"),
    enrichment_score = "auto",
    term2gene = NULL,
    enrichment_id_map = NULL,
    enrichment_extra_args = list(),
    min_feature_overlap_warn = 5L,
    min_feature_overlap_fraction = 0.01,
    ...
) {
  enrichPathways(
    inf = inf$inf,
    dataset = dataset,
    feature_sets = feature_sets,
    alpha = alpha,
    scope = scope,
    component = component,
    universe = universe,
    enrichment_backend = enrichment_backend,
    msigdb_db_species = msigdb_db_species,
    msigdb_species = msigdb_species,
    msigdb_collection = msigdb_collection,
    msigdb_subcollection = msigdb_subcollection,
    enrichment_id_type = enrichment_id_type,
    enrichment_score = enrichment_score,
    term2gene = term2gene,
    enrichment_id_map = enrichment_id_map,
    enrichment_extra_args = enrichment_extra_args,
    min_feature_overlap_warn = min_feature_overlap_warn,
    min_feature_overlap_fraction = min_feature_overlap_fraction,
    ...
  )
}
