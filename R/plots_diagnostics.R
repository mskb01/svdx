# ============================================================
# 5. Plot hooks
# ============================================================

.format_heatmap_label <- function(x, digits = 2) {
  ifelse(
    is.na(x),
    "",
    formatC(x, digits = digits, format = "f")
  )
}

plot_component_diagnostics <- function(
    summary_fit,
    metric = c("pi", "kappa", "cos_y", "cos_e", "norm"),
    show_values = FALSE,
    value_digits = 2,
    title = NULL
) {
  metric <- match.arg(metric)
  
  if (is.null(summary_fit$term_effects)) {
    stop("summary_fit must contain a $term_effects table.", call. = FALSE)
  }
  
  if (!metric %in% colnames(summary_fit$term_effects)) {
    stop("Metric '", metric, "' not found in summary_fit$term_effects.", call. = FALSE)
  }
  
  default_title <- switch(
    metric,
    pi = "Signed term decomposition of fitted U",
    kappa = "Functional identifiability of U terms",
    cos_y = "Term alignment with component response",
    cos_e = "Term alignment with residuals",
    norm = "Magnitude of fitted term contribution"
  )
  
  df <- summary_fit$term_effects |>
    dplyr::mutate(.value = .data[[metric]])
  
  p <- ggplot2::ggplot(
    df,
    ggplot2::aes(
      x = term,
      y = component,
      fill = .value
    )
  ) +
    ggplot2::geom_tile() +
    ggplot2::theme_bw() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    ) +
    ggplot2::labs(
      x = "xformula term",
      y = "Component",
      fill = metric,
      title = title %||% default_title
    )
  
  if (show_values) {
    p <- p +
      ggplot2::geom_text(
        ggplot2::aes(label = .format_heatmap_label(.value, digits = value_digits)),
        size = 3
      )
  }
  
  p
}

plot_component_pi <- function(summary_fit, show_values = FALSE, value_digits = 2, ...) {
  plot_component_diagnostics(
    summary_fit,
    metric = "pi",
    show_values = show_values,
    value_digits = value_digits,
    ...
  )
}

plot_component_kappa <- function(summary_fit, show_values = FALSE, value_digits = 2, ...) {
  plot_component_diagnostics(
    summary_fit,
    metric = "kappa",
    show_values = show_values,
    value_digits = value_digits,
    ...
  )
}

plot_component_cos_y <- function(summary_fit, show_values = FALSE, value_digits = 2, ...) {
  plot_component_diagnostics(
    summary_fit,
    metric = "cos_y",
    show_values = show_values, value_digits = value_digits,
    ...
  )
}

plot_component_cos_e <- function(summary_fit, show_values = FALSE, value_digits = 2, ...) {
  plot_component_diagnostics(
    summary_fit,
    metric = "cos_e",
    show_values = show_values, 
    value_digits=value_digits,
    ...
  )
}

plot_component_norm <- function(summary_fit, show_values = FALSE, value_digits = 2,...) {
  plot_component_diagnostics(
    summary_fit,
    metric = "norm",
    show_values = show_values, value_digits = value_digits,
    ...
  )
}

plot_xz_confounding <- function(
    summary_fit,
    metric = c("q95_abs_cos", "q99_abs_cos", "median_abs_cos", "mean_abs_cos", "max_abs_cos"),
    components = NULL,
    layout = c("stacked", "facet", "list"),
    show_values = FALSE,
    value_digits = 2,
    title = NULL
) {
  metric <- match.arg(metric)
  layout <- match.arg(layout)
  
  df <- summary_fit$xz_confounding
  if (is.null(df)) stop("No x/z confounding diagnostics found.", call. = FALSE)
  
  if (!metric %in% colnames(df)) {
    stop("Metric '", metric, "' not found in summary_fit$xz_confounding.", call. = FALSE)
  }
  
  if (!is.null(components)) {
    df <- df |>
      dplyr::filter(component %in% components)
  }
  
  if (nrow(df) == 0) {
    stop("No x/z confounding rows remain after filtering components.", call. = FALSE)
  }
  
  df <- df |>
    dplyr::mutate(
      .value = .data[[metric]],
      component_u_term = interaction(component, u_term, sep = ":", lex.order = TRUE)
    )
  
  make_one <- function(d, subtitle = NULL) {
    p <- ggplot2::ggplot(
      d,
      ggplot2::aes(
        x = z_term,
        y = if (layout == "stacked") component_u_term else u_term,
        fill = .value
      )
    ) +
      ggplot2::geom_tile() +
      ggplot2::theme_bw() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
      ) +
      ggplot2::labs(
        x = "zformula term",
        y = if (layout == "stacked") "component:xformula term" else "xformula term",
        fill = metric,
        title = title %||% "Feature-wise x/z fitted-effect overlap",
        subtitle = subtitle
      )
    
    if (show_values) {
      p <- p +
        ggplot2::geom_text(
          ggplot2::aes(label = .format_heatmap_label(.value, digits = value_digits)),
          size = 3
        )
    }
    
    p
  }
  
  if (layout == "stacked") {
    return(make_one(df))
  }
  
  if (layout == "facet") {
    p <- make_one(df) + ggplot2::facet_wrap(~ component)
    return(p)
  }
  
  # layout == "list": return one ggplot per component.
  split_df <- split(df, df$component)
  out <- lapply(names(split_df), function(cc) make_one(split_df[[cc]], subtitle = cc))
  names(out) <- names(split_df)
  out
}


.svdx_volcano_label_map <- function(fdata = NULL, dataset = NULL) {
  fdata <- .svdx_resolve_fdata_for_dataset(fdata, dataset = dataset, required = FALSE)
  if (is.null(fdata)) return(NULL)
  labels <- .svdx_fdata_to_row_labels(fdata)
  if (is.null(labels) || !nrow(labels)) return(NULL)
  stats::setNames(as.character(labels$label), as.character(labels$feature))
}

.svdx_volcano_resolve_labels <- function(
    df,
    label_features = NULL,
    label_top_n = 0,
    label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    use_feature_labels = TRUE,
    fdata = NULL,
    dataset = NULL,
    ranking = c(q_value = NA_character_, abs_statistic = NA_character_, abs_effect = NA_character_, abs_loading = NA_character_)
) {
  if (!"feature" %in% names(df)) return(df[0, , drop = FALSE])
  label_by <- match.arg(label_by)
  label_top_n <- as.integer(label_top_n %||% 0L)
  if (is.na(label_top_n) || label_top_n < 0L) {
    stop("label_top_n must be a non-negative integer.", call. = FALSE)
  }
  features <- as.character(df$feature)
  label_map <- .svdx_volcano_label_map(fdata = fdata, dataset = dataset)
  display <- features
  if (isTRUE(use_feature_labels) && !is.null(label_map)) {
    mapped <- unname(label_map[features])
    display <- ifelse(!is.na(mapped) & nzchar(mapped), mapped, features)
  }

  selected <- character(0)
  if (!is.null(label_features) && length(label_features)) {
    requested <- unique(as.character(label_features))
    by_id <- features[features %in% requested]
    by_label <- character(0)
    if (isTRUE(use_feature_labels) && !is.null(label_map)) {
      by_label <- features[display %in% requested]
    }
    selected <- unique(c(by_id, by_label))
  }

  if (label_top_n > 0L) {
    rank_col <- unname(ranking[[label_by]])
    if (!is.na(rank_col) && rank_col %in% names(df)) {
      vals <- df[[rank_col]]
      ok <- !is.na(vals) & is.finite(vals)
      if (any(ok)) {
        ord <- if (label_by == "q_value") {
          order(vals[ok], decreasing = FALSE, na.last = NA)
        } else {
          order(abs(vals[ok]), decreasing = TRUE, na.last = NA)
        }
        top_features <- features[ok][ord][seq_len(min(label_top_n, length(ord)))]
        selected <- unique(c(selected, top_features))
      }
    } else {
      message("Volcano auto-labeling skipped: label_by = '", label_by, "' is unavailable for this plot.")
    }
  }

  out <- df[features %in% selected, , drop = FALSE]
  if (!nrow(out)) return(out)
  out$.svdx_label <- display[features %in% selected]
  out
}

.svdx_add_volcano_labels <- function(p, label_df) {
  if (is.null(label_df) || !nrow(label_df)) return(p)
  p <- p + ggplot2::geom_point(
    data = label_df,
    shape = 21,
    size = 2.4,
    stroke = 0.8,
    alpha = 1
  )
  if (requireNamespace("ggrepel", quietly = TRUE)) {
    p <- p + ggrepel::geom_text_repel(
      data = label_df,
      ggplot2::aes(label = .data$.svdx_label),
      size = 3,
      colour = "#1F4E79",
      segment.color = "grey60",
      max.overlaps = Inf,
      min.segment.length = 0,
      seed = 1
    )
  } else {
    p <- p + ggplot2::geom_text(
      data = label_df,
      ggplot2::aes(label = .data$.svdx_label),
      size = 3,
      colour = "#1F4E79",
      vjust = -0.6,
      check_overlap = TRUE
    )
  }
  p
}

plot_union_volcano <- function(
    inf,
    scope = c("union", "global"),
    label_features = NULL,
    label_top_n = 0,
    label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    use_feature_labels = TRUE,
    fdata = NULL,
    dataset = NULL
) {
  scope <- match.arg(scope)
  label_by <- match.arg(label_by)

  make_union <- function() {
    df <- inf$union_results

    if (is.null(df) || nrow(df) == 0L) {
      message("Union volcano skipped: union results unavailable.")
      return(NULL)
    }
    if (!all(c("best_coef", "union_q") %in% names(df))) {
      message("Union volcano skipped: required union columns are unavailable.")
      return(NULL)
    }
    df <- df |>
      dplyr::mutate(
        neg_log10_q = -log10(pmax(.data$union_q, .Machine$double.xmin))
      )

    xmin <- min(df$best_coef, na.rm = TRUE)
    xmax <- max(df$best_coef, na.rm = TRUE)
    if (!is.finite(xmin)) xmin <- 0
    if (!is.finite(xmax)) xmax <- 0

    label_df <- .svdx_volcano_resolve_labels(
      df,
      label_features = label_features,
      label_top_n = label_top_n,
      label_by = label_by,
      use_feature_labels = use_feature_labels,
      fdata = fdata,
      dataset = dataset,
      ranking = c(q_value = "union_q", abs_statistic = "max_abs_stat", abs_effect = "best_coef", abs_loading = NA_character_)
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$best_coef, y = .data$neg_log10_q)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(
        yintercept = -log10(0.10),
        color = "orange",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      ggplot2::geom_hline(
        yintercept = -log10(0.05),
        color = "red",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      ggplot2::annotate(
        "text",
        x = xmin,
        y = -log10(0.10)-.05,
        label = "10% FDR",
        hjust = -0.1,
        color = "orange",
        size = 3
      ) +
      ggplot2::annotate(
        "text",
        x = xmax,
        y = -log10(0.05)+.05,
        label = "5% FDR",
        hjust = 1.1,
        color = "red",
        size = 3
      ) +
      ggplot2::coord_cartesian(clip = "off") +
      ggplot2::theme_bw() +
      ggplot2::theme(
        panel.grid.minor = ggplot2::element_blank(),
        plot.margin = ggplot2::margin(
          t = 5.5,
          r = 20,
          b = 5.5,
          l = 20
        )
      ) +
      ggplot2::labs(
        x = "Effect size of each feature's top-scoring component",
        y = "-log10(union q-value)",
        title = "Union feature inference across components"
      )
    .svdx_add_volcano_labels(p, label_df)
  }

  make_global <- function() {
    df <- inf$union_results

    if (is.null(df) || nrow(df) == 0L) {
      message("Global volcano skipped: union/global results unavailable.")
      return(NULL)
    }
    if (!all(c("best_coef", "global_q") %in% names(df))) {
      message("Global volcano skipped: required global columns are unavailable.")
      return(NULL)
    }
    df <- df |>
      dplyr::mutate(
        neg_log10_q = -log10(pmax(.data$global_q, .Machine$double.xmin))
      )
    keep <- is.finite(df$best_coef) & !is.na(df$global_q)
    if (!any(keep)) {
      message("Global volcano skipped: no finite coefficients/global q-values available.")
      return(NULL)
    }
    df <- df[keep, , drop = FALSE]

    xmin <- min(df$best_coef, na.rm = TRUE)
    xmax <- max(df$best_coef, na.rm = TRUE)
    if (!is.finite(xmin)) xmin <- 0
    if (!is.finite(xmax)) xmax <- 0

    label_df <- .svdx_volcano_resolve_labels(
      df,
      label_features = label_features,
      label_top_n = label_top_n,
      label_by = label_by,
      use_feature_labels = use_feature_labels,
      fdata = fdata,
      dataset = dataset,
      ranking = c(q_value = "global_q", abs_statistic = "max_abs_stat", abs_effect = "best_coef", abs_loading = NA_character_)
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$best_coef, y = .data$neg_log10_q)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(
        yintercept = -log10(0.10),
        color = "orange",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      ggplot2::geom_hline(
        yintercept = -log10(0.05),
        color = "red",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      ggplot2::annotate(
        "text",
        x = xmin,
        y = -log10(0.10)-.05,
        label = "10% FDR",
        hjust = -0.1,
        color = "orange",
        size = 3
      ) +
      ggplot2::annotate(
        "text",
        x = xmax,
        y = -log10(0.05)+.05,
        label = "5% FDR",
        hjust = 1.1,
        color = "red",
        size = 3
      ) +
      ggplot2::coord_cartesian(clip = "off") +
      ggplot2::theme_bw() +
      ggplot2::theme(
        panel.grid.minor = ggplot2::element_blank(),
        plot.margin = ggplot2::margin(
          t = 5.5,
          r = 20,
          b = 5.5,
          l = 20
        )
      ) +
      ggplot2::labs(
        x = "Effect size of each feature's top-scoring component",
        y = "-log10(global q-value)",
        title = "Global feature inference across components"
      )
    .svdx_add_volcano_labels(p, label_df)
  }

  if (identical(scope, "union")) return(make_union())
  if (identical(scope, "global")) return(make_global())
}

plot_component_volcano <- function(
    inf,
    component = 1,
    label_features = NULL,
    label_top_n = 0,
    label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    use_feature_labels = TRUE,
    fdata = NULL,
    dataset = NULL
) {
  comps <- unique(inf$feature_results$component)
  k <- .resolve_component_index(component, comps)
  component_id <- comps[k]
  
  df <- inf$feature_results |>
    dplyr::filter(.data$component == component_id) |>
    dplyr::mutate(
      neg_log10_q = -log10(pmax(.data$qval, .Machine$double.xmin))
    )
  
  if (nrow(df) == 0) {
    stop(
      "No rows found for component = ", component_id,
      ". Available components are: ",
      paste(comps, collapse = ", "),
      call. = FALSE
    )
  }
  
  xmin <- min(df$coef, na.rm = TRUE)
  xmax <- max(df$coef, na.rm = TRUE)
  label_df <- .svdx_volcano_resolve_labels(
    df,
    label_features = label_features,
    label_top_n = label_top_n,
    label_by = label_by,
    use_feature_labels = use_feature_labels,
    fdata = fdata,
    dataset = dataset,
    ranking = c(q_value = "qval", abs_statistic = "stat", abs_effect = "coef", abs_loading = NA_character_)
  )
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$coef, y = .data$neg_log10_q)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_hline(
      yintercept = -log10(0.10),
      color = "orange",
      linetype = "dashed",
      linewidth = 0.6
    ) +
    ggplot2::geom_hline(
      yintercept = -log10(0.05),
      color = "red",
      linetype = "dashed",
      linewidth = 0.6
    ) +
    ggplot2::annotate(
      "text",
      x = xmin,
      y = -log10(0.10)-.05,
      label = "10% FDR",
      hjust = -0.1,
      color = "orange",
      size = 3
    ) +
    ggplot2::annotate(
      "text",
      x = xmax,
      y = -log10(0.05)+.05,
      label = "5% FDR",
      hjust = 1.1,
      color = "red",
      size = 3
    ) +
    ggplot2::coord_cartesian(clip = "off") +
    ggplot2::theme_bw() +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(
        t = 5.5,
        r = 20,
        b = 5.5,
        l = 20
      )
    ) +
    ggplot2::labs(
      x = "Estimated effect size for component",
      y = "-log10(component q-value)",
      title = paste("Feature inference:", component_id)
    )
  .svdx_add_volcano_labels(p, label_df)
}

plot_enrichment_dotplot <- function(
    enrich_tbl,
    n = 20,
    top_n = n,
    rank_by = c("padj", "pvalue", "score"),
    padj_max = NULL,
    label_wrap_width = 40
) {
  rank_by <- match.arg(rank_by)
  if (!is.null(label_wrap_width)) {
    if (length(label_wrap_width) != 1L || !is.numeric(label_wrap_width) ||
        is.na(label_wrap_width) || label_wrap_width <= 0) {
      stop("label_wrap_width must be NULL or a single positive number.", call. = FALSE)
    }
  }

  # if (inherits(enrich_tbl, "svdxEnrichment") ||
  #     (is.list(enrich_tbl) && !is.null(enrich_tbl$results))) {
  #   enrich_tbl <- enrich_tbl$results
  # }
  if (inherits(enrich_tbl, "svdxEnrichment") ||
      (is.list(enrich_tbl) &&
       !is.data.frame(enrich_tbl) &&
       !is.null(enrich_tbl[["results"]]))) {
    enrich_tbl <- enrich_tbl[["results"]]
  }

  df <- enrich_tbl

  if (!is.null(padj_max) && "padj" %in% names(df)) {
    df <- df |> dplyr::filter(.data$padj <= !!padj_max)
  }

  if (rank_by == "score") {
    df <- df |> dplyr::arrange(dplyr::desc(abs(.data$score)), .data$padj, .data$pvalue)
  } else {
    df <- df |> dplyr::arrange(.data[[rank_by]], .data$pvalue)
  }

  df <- df |>
    dplyr::slice_head(n = top_n) |>
    dplyr::mutate(
      term_name = stats::reorder(.data$term_name, -log10(pmax(.data$padj, .Machine$double.xmin))),
      plot_score = dplyr::if_else(is.finite(.data$score), .data$score, NA_real_)
    )

  ggplot2::ggplot(
    df,
    ggplot2::aes(
      x = -log10(pmax(.data$padj, .Machine$double.xmin)),
      y = .data$term_name,
      size = .data$n_overlap,
      color = .data$plot_score
    )
  ) +
    ggplot2::geom_point() +
    ggplot2::scale_y_discrete(
      labels = if (is.null(label_wrap_width) || is.infinite(label_wrap_width)) {
        ggplot2::waiver()
      } else {
        function(x) vapply(
          as.character(x),
          function(label) {
            if (is.na(label) || !nzchar(label)) return(label)
            starts <- seq.int(1L, nchar(label), by = as.integer(label_wrap_width))
            ends <- pmin(starts + as.integer(label_wrap_width) - 1L, nchar(label))
            paste(substring(label, starts, ends), collapse = "\n")
          },
          character(1)
        )
      }
    ) +
    ggplot2::theme_bw() +
    ggplot2::labs(
      x = "-log10(adjusted p-value)",
      y = NULL,
      size = "Overlap",
      color = "Score",
      title = "Pathway/module enrichment"
    )
}


plot_volcano <- function(
    inf,
    scope = c("all", "component", "union", "global"),
    component = NULL,
    label_features = NULL,
    label_top_n = 0,
    label_by = c("q_value", "abs_statistic", "abs_effect", "abs_loading"),
    use_feature_labels = TRUE,
    fdata = NULL,
    dataset = NULL
) {
  scope <- match.arg(scope)
  label_by <- match.arg(label_by)

  scopes <- if (identical(scope, "all")) c("component", "union", "global") else scope
  out <- list()

  if ("component" %in% scopes) {
    comps <- sort(unique(inf$feature_results$component))
    if (!is.null(component)) {
      comps <- comps[comps %in% component]
    }
    component_plots <- lapply(comps, function(cc) {
      plot_component_volcano(
        inf = inf,
        component = cc,
        label_features = label_features,
        label_top_n = label_top_n,
        label_by = label_by,
        use_feature_labels = use_feature_labels,
        fdata = fdata,
        dataset = dataset
      )
    })
    names(component_plots) <- paste0("component_", comps)
    out <- c(out, component_plots)
  }

  if ("union" %in% scopes) {
    out$union <- plot_union_volcano(
      inf = inf,
      scope = "union",
      label_features = label_features,
      label_top_n = label_top_n,
      label_by = label_by,
      use_feature_labels = use_feature_labels,
      fdata = fdata,
      dataset = dataset
    )
  }

  if ("global" %in% scopes) {
    out$global <- plot_union_volcano(
      inf = inf,
      scope = "global",
      label_features = label_features,
      label_top_n = label_top_n,
      label_by = label_by,
      use_feature_labels = use_feature_labels,
      fdata = fdata,
      dataset = dataset
    )
  }

  out <- Filter(Negate(is.null), out)
  if (length(out) == 1L) return(out[[1L]])
  if (!length(out)) return(NULL)
  out
}
