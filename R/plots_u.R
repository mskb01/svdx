#plots_u.R
# Move U plotting utilities here:
#   plotU()
#   plotUEmbedding()
#   plotUTerms()
#   .make_prediction_grid()
#   .predict_U_df()
#
# Avoid grep("^U[0-9]+$") assumptions if component names may differ.


.resolve_component_index <- function(
    component,
    component_names,
    context = "component"
) {
  n_components <- length(component_names)
  
  if (n_components == 0) {
    stop("No components are available.", call. = FALSE)
  }
  
  if (is.numeric(component)) {
    if (length(component) != 1) {
      stop(context, " must be a single component index or name.", call. = FALSE)
    }
    
    idx <- as.integer(component)
    
    if (is.na(idx) || idx < 1 || idx > n_components) {
      stop(
        "Invalid ", context, ": ", component, ".\n",
        "Available components are 1 through ", n_components,
        if (!is.null(component_names)) {
          paste0(" (", paste(component_names, collapse = ", "), ").")
        } else {
          "."
        },
        call. = FALSE
      )
    }
    
    return(idx)
  }
  
  idx <- match(component, component_names)
  
  if (length(idx) != 1 || is.na(idx)) {
    stop(
      "Invalid ", context, ": ", component, ".\n",
      "Available components are: ",
      paste(component_names, collapse = ", "),
      call. = FALSE
    )
  }
  
  idx
}
.typical_value <- function(x) {
  if (is.factor(x)) {
    tab <- table(x)
    return(factor(names(tab)[which.max(tab)], levels = levels(x)))
  }
  
  if (is.character(x)) {
    tab <- table(x)
    return(names(tab)[which.max(tab)])
  }
  
  if (is.logical(x)) return(stats::median(as.numeric(x), na.rm = TRUE) > 0.5)
  
  stats::median(x, na.rm = TRUE)
}

.get_x_formula_from_fit <- function(fit) {
  fit$fitAtOptCV$trainModel$mod$formula %||%
    fit$fitAtOptCV$trainModel$mod$call$formula
}

.get_formula_variable_names <- function(fit) {
  unique(all.vars(.get_x_formula_from_fit(fit)))
}
# .get_formula_variable_names <- function(fit) {
#   vars <- all.vars(fit$fitAtOptCV$trainModel$mod$formula %||% fit$fitAtOptCV$trainModel$mod$call$formula)
#   unique(vars)
# }



.make_prediction_grid <- function(
    data,
    x = NULL,
    y = NULL,
    group = NULL,
    facet = NULL,
    fixed = list(),
    fill = c("error", "typical"),
    n_grid = 100,
    vars_needed = NULL
) {
  fill <- match.arg(fill)
  
  vary_vars <- unique(stats::na.omit(c(x, y, group, facet)))
  supplied_vars <- unique(c(vary_vars, names(fixed)))
  
  if (is.null(vars_needed)) {
    vars_needed <- colnames(data)
  }
  
  missing_vars <- setdiff(vars_needed, supplied_vars)
  
  if (length(missing_vars) > 0 && fill == "error") {
    stop(
      "Variables not plotted, grouped, faceted, or fixed: ",
      paste(missing_vars, collapse = ", "),
      ". Supply fixed = list(...) or use fill = 'typical'.",
      call. = FALSE
    )
  }
  
  grid_list <- list()
  
  for (vv in vary_vars) {
    z <- data[[vv]]
    if (is.factor(z)) {
      grid_list[[vv]] <- levels(z)
    } else if (is.character(z)) {
      grid_list[[vv]] <- sort(unique(z))
    } else if (is.logical(z)) {
      grid_list[[vv]] <- c(FALSE, TRUE)
    } else {
      grid_list[[vv]] <- seq(min(z, na.rm = TRUE), max(z, na.rm = TRUE), length.out = n_grid)
    }
  }
  
  for (vv in names(fixed)) {
    grid_list[[vv]] <- fixed[[vv]]
  }
  
  if (fill == "typical") {
    for (vv in missing_vars) {
      grid_list[[vv]] <- .typical_value(data[[vv]])
    }
  }
  
  if (length(grid_list) == 0) {
    newdata <- data
  } else {
    newdata <- expand.grid(grid_list, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  }
  
  # restore factor levels/classes where possible
  for (vv in intersect(colnames(data), colnames(newdata))) {
    if (is.factor(data[[vv]])) {
      newdata[[vv]] <- factor(newdata[[vv]], levels = levels(data[[vv]]))
    }
  }
  
  # preserve columns not in formula but needed by predict internals, if any
  newdata
}

.predict_U_df <- function(fit, newdata, include = NULL) {
  U <- if (is.null(include)) {
    predict(fit, newdata)
  } else {
    predict(fit, newdata, include = include)
  }
  
  U <- as.matrix(U)
  colnames(U) <- colnames(U) %||% .make_component_names(ncol(U))
  
  out <- cbind(newdata, as.data.frame(U, check.names = FALSE))
  attr(out, "component_names") <- colnames(U)
  out
}


# ------------------------------------------------------------
# U plots
# ------------------------------------------------------------

#' Plot predicted U component or term contribution over metadata space
#' @export
plotU.SSANOVASVD <- function(
    fit,
    data,
    component = 1,
    include = NULL,
    x,
    y = NULL,
    group = NULL,
    facet = NULL,
    fixed = list(),
    fill = c("error", "typical"),
    n_grid = 100,
    observed = FALSE,
    point_alpha = 0.6
) {
  fill <- match.arg(fill)
  vars_needed <- all.vars(fit$fitAtOptCV$trainModel$mod$formula)

  # Plotting variables are display coordinates. They need not belong to the
  # fitted SSANOVA-SVD model, but they must be present in the supplied data.
  # If every display coordinate is a model variable, construct the usual
  # prediction grid. If any display coordinate is outside the model, preserve
  # the supplied design rows so relationships between the display coordinate
  # and fitted model variables are not broken by Cartesian expansion.
  plot_vars <- unique(stats::na.omit(c(x, y, group, facet)))

  missing_plot_vars <- setdiff(plot_vars, colnames(data))
  if (length(missing_plot_vars) > 0L) {
    stop(
      "The following plotting variable(s) were not found in data: ",
      paste(missing_plot_vars, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  missing_model_vars <- setdiff(vars_needed, colnames(data))
  if (length(missing_model_vars) > 0L) {
    stop(
      "The supplied data are missing variable(s) required by the fitted SSANOVA-SVD model: ",
      paste(missing_model_vars, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  non_model_vars <- setdiff(plot_vars, vars_needed)
  supplied_design <- length(non_model_vars) > 0L

  if (supplied_design && !observed) {
    message(
      "Plotting variable(s) not in the fitted SSANOVA-SVD model: ",
      paste(non_model_vars, collapse = ", "),
      ". Predictions are being evaluated at the supplied design points rather ",
      "than on a constructed prediction grid. To evaluate a custom grid, ",
      "construct newdata explicitly and use predict()."
    )
  }

  if (observed || supplied_design) {
    newdata <- data

    # Retain the established meaning of fixed = list(...) when predictions are
    # evaluated at supplied design points by overriding those columns directly.
    if (length(fixed) > 0L) {
      bad_fixed <- setdiff(names(fixed), colnames(newdata))
      if (length(bad_fixed) > 0L) {
        stop(
          "The following fixed variable(s) were not found in data: ",
          paste(bad_fixed, collapse = ", "),
          ".",
          call. = FALSE
        )
      }
      for (vv in names(fixed)) {
        value <- fixed[[vv]]
        if (!(length(value) %in% c(1L, nrow(newdata)))) {
          stop(
            "fixed[['", vv, "']] must have length 1 or nrow(data) when ",
            "predictions are evaluated at supplied design points.",
            call. = FALSE
          )
        }
        newdata[[vv]] <- value
      }
    }
  } else {
    newdata <- .make_prediction_grid(
      data = data,
      x = x,
      y = y,
      group = group,
      facet = facet,
      fixed = fixed,
      fill = fill,
      n_grid = n_grid,
      vars_needed = vars_needed
    )
  }
  
  pred <- .predict_U_df(fit, newdata, include = include)
  comp_names <- attr(pred, "component_names")
  k <- .resolve_component_index(component, comp_names)
  comp <- comp_names[k]

  # The prediction grid already varies every variable supplied in `group`.
  # When more than one grouping variable is supplied, create a temporary
  # interaction factor solely for ggplot grouping/coloring.
  group_aes <- group
  group_label <- group
  if (!is.null(group) && length(group) > 1L) {
    group_aes <- ".svdx_plot_group"
    while (group_aes %in% names(pred)) {
      group_aes <- paste0(group_aes, "_")
    }
    pred[[group_aes]] <- do.call(
      interaction,
      c(pred[group], list(drop = TRUE, sep = " - "))
    )
    group_label <- paste(group, collapse = " x ")
  }
  
  if (is.null(y)) {
    aes_args <- ggplot2::aes(x = .data[[x]], y = .data[[comp]])
    if (!is.null(group_aes)) {
      aes_args$colour <- rlang::expr(.data[[!!group_aes]])
      aes_args$group <- rlang::expr(.data[[!!group_aes]])
    }
    
    p <- ggplot2::ggplot(pred, aes_args) +
      ggplot2::theme_bw() +
      ggplot2::labs(
        x = x,
        y = comp,
        color = group_label,
        title = paste0(comp, if (!is.null(include)) paste0(" contribution: ", paste(include, collapse = " + ")) else "")
      )
    
    x_is_continuous <-
      .is_numeric_like(pred[[x]]) ||
      inherits(pred[[x]], "Date") ||
      inherits(pred[[x]], "POSIXt")
    x_is_ordered <- .is_ordered_factor_like(pred[[x]])
    x_is_factor <- .is_factor_like(pred[[x]])
    
    if (observed) {
      if (x_is_factor && !is.null(group)) {
        p <- p + ggplot2::geom_point(
          alpha = point_alpha,
          position = ggplot2::position_dodge(width = 0.4)
        )
      } else {
        p <- p + ggplot2::geom_point(alpha = point_alpha)
      }
    } else if (x_is_continuous) {
      if (is.null(group)) {
        p <- p + ggplot2::geom_line(
          ggplot2::aes(group = 1),
          linewidth = 1
        )
      } else {
        p <- p + ggplot2::geom_line(linewidth = 1)
      }
    } else if (x_is_ordered) {
      if (is.null(group)) {
        p <- p + ggplot2::geom_point(size = 2) +
          ggplot2::geom_line(ggplot2::aes(group = 1), linewidth = 0.6)
      } else {
        p <- p + ggplot2::geom_point(size = 2) +
          ggplot2::geom_line(linewidth = 0.6)
      }
    } else if (x_is_factor) {
      if (!is.null(group)) {
        p <- p + ggplot2::geom_point(
          size = 2,
          position = ggplot2::position_dodge(width = 0.4)
        )
      } else {
        p <- p + ggplot2::geom_point(size = 2)
      }
    } else {
      p <- p + ggplot2::geom_point(alpha = point_alpha)
    }
  } else {
    p <- ggplot2::ggplot(
      pred,
      ggplot2::aes(x = .data[[x]], y = .data[[y]], fill = .data[[comp]])
    ) +
      ggplot2::geom_tile() +
      ggplot2::theme_bw() +
      ggplot2::labs(
        x = x,
        y = y,
        fill = comp,
        title = paste0(comp, if (!is.null(include)) paste0(" contribution: ", paste(include, collapse = " + ")) else "")
      )
  }
  
  if (!is.null(facet)) {
    p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", facet)))
  }

  if (.is_factor_like(pred[[x]])) {
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

#' Plot two predicted U components/contributions against each other
#' @export
plotUEmbedding.SSANOVASVD <- function(
    fit,
    data,
    components = c(1, 2),
    include = NULL,
    vary = NULL,
    color = NULL,
    group = NULL,
    facet = NULL,
    fixed = list(),
    fill = c("error", "typical"),
    n_grid = 100,
    observed = is.null(vary),
    point_alpha = 0.7
) {
  fill <- match.arg(fill)
  vars_needed <- all.vars(fit$fitAtOptCV$trainModel$mod$formula)
  
  if (observed) {
    newdata <- data
  } else {
    xvar <- vary[1] %||% NULL
    yvar <- if (length(vary) >= 2) vary[2] else NULL
    newdata <- .make_prediction_grid(
      data = data,
      x = xvar,
      y = yvar,
      group = group,
      facet = facet,
      fixed = fixed,
      fill = fill,
      n_grid = n_grid,
      vars_needed = vars_needed
    )
  }
  
  pred <- .predict_U_df(fit, newdata, include = include)
  comp_names <- attr(pred, "component_names")
  if (length(comp_names) < 2) {
    stop(
      "plotUEmbedding() requires at least two components.\n",
      "Only ", length(comp_names), " component is available.\n",
      "Consider using plotU(), or plotUTerms() to visualize the fitted decomposition.",
      call. = FALSE
    )
    
    if (length(components) < 2) {
      stop("components must contain two component indices or names.", call. = FALSE)
    }
  }
  
  k1 <- .resolve_component_index(components[1], comp_names)
  k2 <- .resolve_component_index(components[2], comp_names)
  c1 <- comp_names[k1]
  c2 <- comp_names[k2]
  
  aes_args <- ggplot2::aes(x = .data[[c1]], y = .data[[c2]])
  if (!is.null(color)) aes_args$colour <- rlang::expr(.data[[!!color]])
  if (!is.null(group)) aes_args$group <- rlang::expr(.data[[!!group]])
  
  p <- ggplot2::ggplot(pred, aes_args) +
    ggplot2::theme_bw() +
    ggplot2::labs(
      x = c1,
      y = c2,
      color = color,
      title = paste0(c1, " vs ", c2, if (!is.null(include)) paste0(" contribution: ", paste(include, collapse = " + ")) else "")
    )
  
  if (!observed && length(vary) == 1) {
    p <- p + ggplot2::geom_path(linewidth = 1)
  } else {
    p <- p + ggplot2::geom_point(alpha = point_alpha)
  }
  
  if (!is.null(facet)) {
    p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", facet)))
  }
  
  p
}


.is_ordered_factor_like <- function(x) {
  is.ordered(x)
}

.is_factor_like <- function(x) {
  is.factor(x) || is.character(x) || is.logical(x)
}

.is_numeric_like <- function(x) {
  is.numeric(x) || is.integer(x)
}

.get_term_vars <- function(term) {
  vars <- tryCatch(
    all.vars(stats::as.formula(paste("~", term))),
    error = function(e) character(0)
  )
  if (!length(vars)) {
    vars <- strsplit(term, ":", fixed = TRUE)[[1]]
    vars <- trimws(vars)
    vars <- vars[nzchar(vars)]
  }
  unique(vars)
}

.classify_term_vars <- function(data, vars, term) {
  missing_vars <- setdiff(vars, colnames(data))
  if (length(missing_vars)) {
    message(
      "Skipping term '", term, "': variable(s) not found in data: ",
      paste(missing_vars, collapse = ", "), "."
    )
    return(NULL)
  }
  kinds <- vapply(vars, function(vv) {
    x <- data[[vv]]
    if (.is_numeric_like(x)) return("numeric")
    if (.is_factor_like(x)) return("factor")
    "unsupported"
  }, character(1))
  if (any(kinds == "unsupported")) {
    message(
      "Skipping term '", term, "': unsupported variable type(s): ",
      paste(vars[kinds == "unsupported"], collapse = ", "), "."
    )
    return(NULL)
  }
  kinds
}

.make_term_prediction_grid <- function(
    data,
    vars,
    kinds,
    facet = NULL,
    fixed = list(),
    fill = c("error", "typical"),
    n_grid = 100,
    vars_needed = NULL
) {
  fill <- match.arg(fill)
  n_vars <- length(vars)
  if (n_vars == 1) {
    .make_prediction_grid(
      data = data,
      x = vars[1],
      facet = facet,
      fixed = fixed,
      fill = fill,
      n_grid = n_grid,
      vars_needed = vars_needed
    )
  } else if (n_vars == 2) {
    if (all(kinds == "numeric")) {
      .make_prediction_grid(
        data = data,
        x = vars[1],
        y = vars[2],
        facet = facet,
        fixed = fixed,
        fill = fill,
        n_grid = n_grid,
        vars_needed = vars_needed
      )
    } else if (sum(kinds == "numeric") == 1 && sum(kinds == "factor") == 1) {
      num_var <- vars[kinds == "numeric"][1]
      fac_var <- vars[kinds == "factor"][1]
      .make_prediction_grid(
        data = data,
        x = num_var,
        group = fac_var,
        facet = facet,
        fixed = fixed,
        fill = fill,
        n_grid = n_grid,
        vars_needed = vars_needed
      )
    } else {
      .make_prediction_grid(
        data = data,
        x = vars[1],
        y = vars[2],
        facet = facet,
        fixed = fixed,
        fill = fill,
        n_grid = n_grid,
        vars_needed = vars_needed
      )
    }
  } else {
    NULL
  }
}

.u_term_diverging_fill_scale <- function(x) {
  vals <- x[is.finite(x)]

  if (!length(vals)) {
    return(
      ggplot2::scale_fill_distiller(
        type = "div",
        palette = "PuOr",
        direction = -1
      )
    )
  }

  lim <- max(abs(vals))

  if (!is.finite(lim) || lim == 0) {
    return(
      ggplot2::scale_fill_distiller(
        type = "div",
        palette = "PuOr",
        direction = -1
      )
    )
  }

  ggplot2::scale_fill_distiller(
    type = "div",
    palette = "PuOr",
    direction = -1,
    limits = c(-lim, lim)
  )
}

.plot_u_term_df <- function(df, term, vars, kinds, data, facet = NULL) {
  n_vars <- length(vars)
  base <- ggplot2::theme_bw()
  title <- paste0("Fitted effect function: ", term)
  
  if (n_vars == 1) {
    vv <- vars[1]
    if (kinds[1] == "numeric") {
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[vv]], y = contribution)) +
        ggplot2::geom_line(linewidth = 1) +
        base +
        ggplot2::labs(x = vv, y = "Fitted effect function", title = title)
    } else {
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[vv]], y = contribution)) +
        ggplot2::geom_col() +
        base +
        ggplot2::labs(x = vv, y = "Fitted effect function", title = title) +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(
            angle = 90,
            hjust = 1,
            vjust = 0.5
          )
        )
    }
  } else if (n_vars == 2) {
    if (all(kinds == "numeric")) {
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[vars[1]]], y = .data[[vars[2]]], fill = contribution)) +
        ggplot2::geom_tile() +
        .u_term_diverging_fill_scale(df$contribution) +
        base +
        ggplot2::labs(x = vars[1], y = vars[2], fill = "Fitted effect function", title = title)
    } else if (sum(kinds == "numeric") == 1 && sum(kinds == "factor") == 1) {
      num_var <- vars[kinds == "numeric"][1]
      fac_var <- vars[kinds == "factor"][1]
      p <- ggplot2::ggplot(
        df,
        ggplot2::aes(x = .data[[num_var]], y = contribution, color = .data[[fac_var]], group = .data[[fac_var]])
      ) +
        ggplot2::geom_line(linewidth = 1) +
        base +
        ggplot2::labs(x = num_var, y = "Fitted effect function", color = fac_var, title = title)
    } else {
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[vars[1]]], y = .data[[vars[2]]], fill = contribution)) +
        ggplot2::geom_tile() +
        .u_term_diverging_fill_scale(df$contribution) +
        base +
        ggplot2::labs(x = vars[1], y = vars[2], fill = "Fitted effect function", title = title) +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(
            angle = 90,
            hjust = 1,
            vjust = 0.5
          )
        )
    }
  } else {
    return(NULL)
  }
  
  if (!is.null(facet) && facet %in% colnames(df)) {
    p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", facet)))
  }
  
  p
}

#' Plot term-wise decompositions of one U component
#' @export
plotUTerms.SSANOVASVD <- function(
    fit,
    data,
    component = 1,
    terms = NULL,
    x,
    y = NULL,
    group = NULL,
    facet = NULL,
    fixed = list(),
    fill = c("error", "typical"),
    include_intercept = FALSE,
    n_grid = 100
) {
  fill <- match.arg(fill)
  
  if (missing(x)) x <- NULL
  if (!is.null(x) || !is.null(y) || !is.null(group)) {
    message(
      "plotUTerms() now chooses term-specific plotting variables from each term. ",
      "Arguments x, y, and group are retained for API compatibility but ignored for term-wise plotting."
    )
  }
  
  if (is.null(terms)) {
    terms <- .get_x_terms(fit)
    terms <- terms[!terms %in% c("1", "(Intercept)", "offset")]
  }
  
  vars_needed <- all.vars(fit$fitAtOptCV$trainModel$mod$formula)
  plots <- list()
  
  for (tt in terms) {
    term_vars <- .get_term_vars(tt)
    if (length(term_vars) == 0) {
      message("Skipping term '", tt, "': no plottable variables were found.")
      next
    }
    if (length(term_vars) > 2) {
      message("Skipping term '", tt, "': interactions with more than two variables are not plotted automatically.")
      next
    }
    
    kinds <- .classify_term_vars(data, term_vars, tt)
    if (is.null(kinds)) next
    
    newdata <- .make_term_prediction_grid(
      data = data,
      vars = term_vars,
      kinds = kinds,
      facet = facet,
      fixed = fixed,
      fill = fill,
      n_grid = n_grid,
      vars_needed = vars_needed
    )
    if (is.null(newdata) || nrow(newdata) == 0) {
      message("Skipping term '", tt, "': prediction grid is empty.")
      next
    }
    
    include <- if (include_intercept) c("1", tt) else tt
    pred <- .predict_U_df(fit, newdata, include = include)
    comp_names <- attr(pred, "component_names")
    k <- .resolve_component_index(component, comp_names)
    comp <- comp_names[k]
    pred$term <- tt
    pred$contribution <- pred[[comp]]
    
    p <- .plot_u_term_df(
      df = pred,
      term = tt,
      vars = term_vars,
      kinds = kinds,
      data = data,
      facet = facet
    )
    if (!is.null(p)) plots[[tt]] <- p
  }
  
  if (!length(plots)) {
    message("plotUTerms() skipped all requested terms; no plots were produced.")
    return(NULL)
  }
  
  if (length(plots) == 1) {
    return(plots[[1]])
  }
  
  message("plotUTerms() returning one plot per term because terms require different visualization types.")
  plots
}

#' @method plotU iSSANOVASVD
#' @export
plotU.iSSANOVASVD <- plotU.SSANOVASVD

#' @method plotUTerms iSSANOVASVD
#' @export
plotUTerms.iSSANOVASVD <- plotUTerms.SSANOVASVD

#' @method plotUEmbedding iSSANOVASVD
#' @export
plotUEmbedding.iSSANOVASVD <- plotUEmbedding.SSANOVASVD
