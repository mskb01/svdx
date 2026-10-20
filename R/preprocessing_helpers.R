# R/preprocessing_helpers.R

is_sparse_matrix <- function(x) {
  inherits(x, "sparseMatrix")
}

as_numeric_matrix_or_sparse <- function(Y) {
  if (is_sparse_matrix(Y)) {
    if (!requireNamespace("Matrix", quietly = TRUE)) {
      stop("Package 'Matrix' is required for sparse matrix support.")
    }
    return(Y)
  }
  
  Y <- as.matrix(Y)
  
  if (!is.numeric(Y)) {
    stop("Y_raw must be numeric.")
  }
  
  Y
}

any_negative <- function(Y) {
  if (is_sparse_matrix(Y)) {
    any(Y@x < 0)
  } else {
    any(Y < 0)
  }
}

.svdx_validate_nonnegative_for_family <- function(Y, family, context = "Y") {
  if (is.null(family)) return(invisible(TRUE))

  is_count_family <- function(fam) {
    !is.null(fam) && length(fam) == 1L && as.character(fam) %in% c("poisson", "quasipoisson")
  }

  if (is.list(family) && !is.data.frame(family) && !inherits(family, "formula")) {
    if (!is.list(Y) || is.data.frame(Y)) {
      stop("Internal error: list-valued family requires list-valued Y.", call. = FALSE)
    }
    dataset_names <- names(Y) %||% paste0("dataset_", seq_along(Y))
    names(Y) <- dataset_names
    if (is.null(names(family)) || !all(dataset_names %in% names(family))) {
      names(family) <- dataset_names
    }
    for (ds in dataset_names) {
      fam <- family[[ds]]
      if (is_count_family(fam) && any_negative(Y[[ds]])) {
        stop(context, "[['", ds, "']] must be nonnegative for family = 'poisson' or 'quasipoisson'.", call. = FALSE)
      }
    }
    return(invisible(TRUE))
  }

  if (is_count_family(family) && any_negative(Y)) {
    stop(context, " must be nonnegative for family = 'poisson' or 'quasipoisson'.", call. = FALSE)
  }

  invisible(TRUE)
}

any_nonfinite_matrix <- function(Y) {
  if (is_sparse_matrix(Y)) {
    any(!is.finite(Y@x))
  } else {
    any(!is.finite(Y))
  }
}

matrix_col_sums <- function(Y) {
  if (is_sparse_matrix(Y)) {
    Matrix::colSums(Y)
  } else {
    colSums(Y)
  }
}

matrix_row_means <- function(Y) {
  if (is_sparse_matrix(Y)) {
    Matrix::rowMeans(Y)
  } else {
    rowMeans(Y)
  }
}

matrix_row_sums_sq <- function(Y) {
  if (is_sparse_matrix(Y)) {
    Matrix::rowSums(Y^2)
  } else {
    rowSums(Y^2)
  }
}

matrix_row_vars <- function(Y) {
  n <- ncol(Y)
  
  if (n <= 1) {
    stop("Need at least two samples to compute row variances.")
  }
  
  mu <- matrix_row_means(Y)
  ss <- matrix_row_sums_sq(Y)
  
  vars <- (ss - n * mu^2) / (n - 1)
  vars[vars < 0 & vars > -1e-12] <- 0
  
  vars
}


validate_logical_scalar <- function(x, arg_name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(arg_name, " must be TRUE or FALSE.", call. = FALSE)
  }
  invisible(TRUE)
}

validate_numeric_vector <- function(x, expected_length, arg_name, allow_null = TRUE) {
  if (is.null(x)) {
    if (isTRUE(allow_null)) return(invisible(TRUE))
    stop(arg_name, " cannot be NULL.", call. = FALSE)
  }
  if (!is.numeric(x) || length(x) != expected_length) {
    stop(arg_name, " must be a numeric vector of length ", expected_length, ".", call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop(arg_name, " contains non-finite values.", call. = FALSE)
  }
  invisible(TRUE)
}

compute_size_factors <- function(t_j) {
  if (is.null(t_j)) {
    stop("t_j cannot be NULL.")
  }
  
  if (any(!is.finite(t_j))) {
    stop("t_j contains non-finite values.")
  }
  
  if (any(t_j <= 0)) {
    stop("All entries of t_j must be positive.")
  }
  
  t_j / exp(mean(log(t_j)))
}

formula_variables <- function(formula) {
  if (is.null(formula)) {
    return(character(0))
  }
  
  all.vars(formula)
}

is_allowed_model_column <- function(x) {
  is.numeric(x) || is.integer(x) || is.factor(x) || is.ordered(x) ||
    is.logical(x) || is.character(x) || is.matrix(x)
}

.svdx_date_time_message <- function(data_name, v, formula_name) {
  paste0(
    data_name, "$", v, " is a Date/POSIXct/POSIXlt variable, which is not directly supported in ", formula_name, ".\n",
    "Please convert date/time variables into analysis-specific covariates before calling prepare_svdx_data(), such as day, month, year, day_of_year, hours_since_start, or circadian_time.\n",
    "Place derived variables in xformula when they represent structured/relevant variation to model and interpret, in zformula when they represent nuisance or adjustment variation, or in both only when different derived date/time aspects have different modeling roles."
  )
}

prepare_formula_data <- function(data, formula, data_name, formula_name) {
  vars <- formula_variables(formula)
  term_info <- list()

  if (length(vars) == 0) {
    return(list(data = data, term_info = term_info))
  }

  missing_vars <- setdiff(vars, colnames(data))

  if (length(missing_vars) > 0) {
    stop(
      formula_name,
      " contains variables not found in ",
      data_name,
      ": ",
      paste(missing_vars, collapse = ", "),
      call. = FALSE
    )
  }

  for (v in vars) {
    x <- data[[v]]
    original_class <- class(x)

    if (inherits(x, "Date") || inherits(x, "POSIXct") || inherits(x, "POSIXlt") || inherits(x, "POSIXt")) {
      stop(.svdx_date_time_message(data_name, v, formula_name), call. = FALSE)
    }

    if (is.logical(x)) {
      data[[v]] <- factor(x)
      x <- data[[v]]
      message(data_name, "$", v, " was logical and has been converted to a factor for SVDX modeling.")
    } else if (is.character(x)) {
      data[[v]] <- factor(x)
      x <- data[[v]]
      message(data_name, "$", v, " was character and has been converted to a factor for SVDX modeling.")
    } else if (is.integer(x)) {
      data[[v]] <- as.numeric(x)
      x <- data[[v]]
      message(data_name, "$", v, " was integer and has been converted to numeric for SVDX modeling.")
    }

    if (!is_allowed_model_column(x)) {
      stop(
        data_name,
        "$",
        v,
        " has unsupported type: ",
        paste(original_class, collapse = "/"),
        ". Variables in ",
        formula_name,
        " must be factor, ordered factor, numeric vector, or numeric matrix. Integer variables are converted to numeric; logical/character variables are converted to factor.",
        call. = FALSE
      )
    }

    if (is.matrix(x)) {
      if (!is.numeric(x)) {
        stop(data_name, "$", v, " is a matrix but is not numeric. Numerical matrix terms are allowed; non-numeric matrices are not.", call. = FALSE)
      }
      if (anyNA(x)) {
        stop(data_name, "$", v, " contains missing values.", call. = FALSE)
      }
      if (any(!is.finite(x))) {
        stop(data_name, "$", v, " contains non-finite values.", call. = FALSE)
      }
      term_info[[v]] <- list(
        original_class = paste(original_class, collapse = "/"),
        svdx_type = "numeric_matrix",
        ncol = ncol(x)
      )
      next
    }

    if (any(is.na(x))) {
      stop(data_name, "$", v, " contains missing values.", call. = FALSE)
    }

    if (is.factor(x) || is.ordered(x)) {
      if (nlevels(x) < 2) {
        stop(data_name, "$", v, " must have at least two levels.", call. = FALSE)
      }
      term_info[[v]] <- list(
        original_class = paste(original_class, collapse = "/"),
        svdx_type = if (is.ordered(x)) "ordered_factor" else "factor",
        levels = levels(x)
      )
      next
    }

    if (is.numeric(x)) {
      if (!is.finite(stats::var(x)) || stats::var(x) == 0) {
        stop(data_name, "$", v, " has zero or non-finite variance.", call. = FALSE)
      }
      term_info[[v]] <- list(
        original_class = paste(original_class, collapse = "/"),
        svdx_type = "numeric",
        range = range(x, na.rm = TRUE)
      )
      next
    }
  }

  list(data = data, term_info = term_info)
}

validate_formula_data <- function(data, formula, data_name, formula_name) {
  prepare_formula_data(data, formula, data_name, formula_name)
  invisible(TRUE)
}

validate_sample_alignment <- function(Y, xdata, zdata) {
  if (nrow(xdata) != ncol(Y)) {
    stop("nrow(xdata) must equal ncol(Y).")
  }
  
  if (nrow(zdata) != ncol(Y)) {
    stop("nrow(zdata) must equal ncol(Y).")
  }
  
  if (!is.null(rownames(xdata)) && !is.null(colnames(Y))) {
    if (!all(rownames(xdata) == colnames(Y))) {
      stop("rownames(xdata) must exactly match colnames(Y).")
    }
  }
  
  if (!is.null(rownames(zdata)) && !is.null(colnames(Y))) {
    if (!all(rownames(zdata) == colnames(Y))) {
      stop("rownames(zdata) must exactly match colnames(Y).")
    }
  }
  
  invisible(TRUE)
}

sparse_safe_col_normalize <- function(Y, t_tilde_j) {
  if (is_sparse_matrix(Y)) {
    Y %*% Matrix::Diagonal(x = 1 / t_tilde_j)
  } else {
    sweep(Y, 2, t_tilde_j, "/")
  }
}

sparse_safe_log1p <- function(Y) {
  if (is_sparse_matrix(Y)) {
    Y@x <- log1p(Y@x)
    Y
  } else {
    log1p(Y)
  }
}

select_top_hvg <- function(Y, top_ftrs) {
  if (is.null(top_ftrs)) {
    return(seq_len(nrow(Y)))
  }
  
  top_ftrs <- as.integer(top_ftrs)
  
  if (length(top_ftrs) != 1 || top_ftrs <= 0) {
    stop("top_ftrs must be a single positive integer or NULL.")
  }
  
  n_select <- min(top_ftrs, nrow(Y))
  
  vars <- matrix_row_vars(Y)
  
  if (any(!is.finite(vars))) {
    stop("Some feature variances are non-finite.")
  }
  
  order(vars, decreasing = TRUE)[seq_len(n_select)]
}

row_center_scale <- function(
    Y,
    center_rows = TRUE,
    scale_rows = TRUE,
    allow_sparse_densification = FALSE
) {
  row_center <- rep(0, nrow(Y))
  row_scale <- rep(1, nrow(Y))
  
  names(row_center) <- rownames(Y)
  names(row_scale) <- rownames(Y)
  
  if (is_sparse_matrix(Y) && center_rows && !allow_sparse_densification) {
    stop(
      "center_rows = TRUE would densify sparse Y. ",
      "Set center_rows = FALSE, or explicitly set ",
      "allow_sparse_densification = TRUE."
    )
  }
  
  if (is_sparse_matrix(Y) && center_rows && allow_sparse_densification) {
    warning(
      "Converting sparse Y to dense matrix because center_rows = TRUE ",
      "and allow_sparse_densification = TRUE."
    )
    Y <- as.matrix(Y)
  }
  
  if (center_rows) {
    row_center <- rowMeans(Y)
    Y <- sweep(Y, 1, row_center, "-")
  }
  
  if (scale_rows) {
    if (is_sparse_matrix(Y)) {
      vars <- matrix_row_vars(Y)
      row_scale <- sqrt(vars)
    } else {
      row_scale <- apply(Y, 1, stats::sd)
    }
    
    bad_scale <- !is.finite(row_scale) | row_scale == 0
    
    if (any(bad_scale)) {
      warning(
        sum(bad_scale),
        " rows had zero or non-finite SD; setting scale to 1."
      )
      row_scale[bad_scale] <- 1
    }
    
    if (is_sparse_matrix(Y)) {
      Y <- Matrix::Diagonal(x = 1 / row_scale) %*% Y
    } else {
      Y <- sweep(Y, 1, row_scale, "/")
    }
  }
  
  list(
    Y = Y,
    row_center = row_center,
    row_scale = row_scale
  )
}



resolve_modelType_xdata <- function(modelType, xdata, xformula) {
  xvars <- formula_variables(xformula)

  if (is.null(modelType)) {
    modelType <- list()
  }

  if (!is.list(modelType) || inherits(modelType, "formula") || is.data.frame(modelType)) {
    stop("modelType must be a named list, or NULL.", call. = FALSE)
  }

  if (length(modelType) && (is.null(names(modelType)) || any(!nzchar(names(modelType))))) {
    stop("modelType must be a named list when supplied.", call. = FALSE)
  }

  extra <- setdiff(names(modelType), xvars)
  if (length(extra)) {
    stop(
      "modelType contains variables not present in xformula: ",
      paste(extra, collapse = ", "),
      call. = FALSE
    )
  }

  for (v in xvars) {
    if (!v %in% colnames(xdata)) next
    x <- xdata[[v]]

    is_numeric_vector <- is.numeric(x) && is.null(dim(x))
    is_numeric_matrix <- is.matrix(x) && is.numeric(x)

    if (is_numeric_vector && is.null(modelType[[v]])) {
      rng <- range(x, na.rm = TRUE)
      if (length(rng) != 2L || any(!is.finite(rng)) || rng[1L] >= rng[2L]) {
        stop("Cannot construct default modelType for numeric xdata$", v, ": range is not finite and increasing.", call. = FALSE)
      }
      modelType[[v]] <- list("cubic", rng)
    } else if (is_numeric_matrix && is.null(modelType[[v]])) {
      modelType[[v]] <- "tp"
    }
  }

  for (v in names(modelType)) {
    if (!v %in% colnames(xdata)) {
      stop("modelType variable '", v, "' was not found in xdata.", call. = FALSE)
    }

    x <- xdata[[v]]
    is_numeric_vector <- is.numeric(x) && is.null(dim(x))
    is_numeric_matrix <- is.matrix(x) && is.numeric(x)

    if (!is_numeric_vector && !is_numeric_matrix) {
      stop("modelType was supplied for non-numeric xdata variable '", v, "'. Factor and ordered-factor variables do not need modelType entries.", call. = FALSE)
    }

    spec <- modelType[[v]]
    if (is_numeric_vector && is.list(spec) && length(spec) >= 2L) {
      rng <- spec[[2L]]
      if (is.numeric(rng) && length(rng) == 2L) {
        if (any(!is.finite(rng)) || rng[1L] >= rng[2L]) {
          stop("modelType[['", v, "']][[2]] must be a finite increasing numeric range of length 2.", call. = FALSE)
        }
        observed <- range(x, na.rm = TRUE)
        if (observed[1L] < rng[1L] || observed[2L] > rng[2L]) {
          stop(
            "xdata$", v, " contains values outside the modelType range [",
            paste(rng, collapse = ", "), "]. Observed range is [",
            paste(observed, collapse = ", "), "].",
            call. = FALSE
          )
        }
      }
    }
  }

  modelType
}

validate_modelType_xdata <- function(modelType, xdata, xformula) {
  resolve_modelType_xdata(modelType, xdata, xformula)
  invisible(TRUE)
}

standardize_family <- function(family) {
  if (is.null(family)) {
    return(NULL)
  }

  if (!is.character(family) || length(family) != 1) {
    stop("family must be NULL, 'gaussian', 'poisson', or 'quasipoisson'.", call. = FALSE)
  }

  match.arg(family, choices = c("gaussian", "poisson", "quasipoisson"))
}

.svdx_family_spec <- function(family) {
  family <- standardize_family(family)

  if (is.null(family)) {
    return(list(
      family = NULL,
      fit_family = NULL,
      inference_family = "gaussian",
      fit_description = "PCA-like Gaussian/nonparametric objective; no explicit probabilistic family used for fitting"
    ))
  }

  if (identical(family, "gaussian")) {
    return(list(
      family = "gaussian",
      fit_family = "gaussian",
      inference_family = "gaussian",
      fit_description = "Gaussian exponential-family SVDX objective"
    ))
  }

  if (identical(family, "poisson")) {
    return(list(
      family = "poisson",
      fit_family = "poisson",
      inference_family = "poisson",
      fit_description = "Poisson exponential-family SVDX objective"
    ))
  }

  if (identical(family, "quasipoisson")) {
    return(list(
      family = "quasipoisson",
      fit_family = "poisson",
      inference_family = "quasipoisson",
      fit_description = "Poisson exponential-family SVDX objective; quasi-Poisson GLMs are used for inference"
    ))
  }

  stop("Unsupported family.", call. = FALSE)
}

.svdx_family_spec_list <- function(family, dataset_names) {
  D <- length(dataset_names)

  if (is.null(family)) {
    fam <- vector("list", D)
    names(fam) <- dataset_names
  } else if (is.character(family) && length(family) == 1L) {
    fam <- rep(list(family), D)
    names(fam) <- dataset_names
  } else if (is.character(family) && length(family) == D) {
    nms <- names(family)
    if (!is.null(nms) && any(nzchar(nms))) {
      if (is.null(nms) || any(!nzchar(nms))) {
        stop("family must be either fully named or fully unnamed.", call. = FALSE)
      }
      missing <- setdiff(dataset_names, nms)
      extra <- setdiff(nms, dataset_names)
      if (length(missing) > 0L || length(extra) > 0L) {
        stop(
          "family names must exactly match dataset names. Missing: ",
          paste(missing, collapse = ", "),
          "; extra: ",
          paste(extra, collapse = ", "),
          ".",
          call. = FALSE
        )
      }
      fam <- as.list(family[dataset_names])
    } else if (!is.null(nms) && any(nzchar(nms))) {
      stop("family must be either fully named or fully unnamed.", call. = FALSE)
    } else {
      fam <- as.list(family)
      names(fam) <- dataset_names
    }
  } else {
    if (!is.list(family) || inherits(family, "formula") || is.data.frame(family)) {
      stop(
        "For integrated SVDX data, family must be NULL, a scalar family recycled to all datasets, ",
        "or a vector/list of length ", D, " parallel to Y.",
        call. = FALSE
      )
    }

    if (length(family) != D) {
      stop(
        "For integrated SVDX data, family must be NULL, a scalar family recycled to all datasets, ",
        "or a vector/list of length ", D, " parallel to Y.",
        call. = FALSE
      )
    }

    nms <- names(family)
    if (!is.null(nms) && any(nzchar(nms))) {
      if (any(!nzchar(nms))) {
        stop("family must be either fully named or fully unnamed.", call. = FALSE)
      }
      missing <- setdiff(dataset_names, nms)
      extra <- setdiff(nms, dataset_names)
      if (length(missing) > 0L || length(extra) > 0L) {
        stop(
          "family names must exactly match dataset names. Missing: ",
          paste(missing, collapse = ", "),
          "; extra: ",
          paste(extra, collapse = ", "),
          ".",
          call. = FALSE
        )
      }
      fam <- family[dataset_names]
    } else if (!is.null(nms) && any(nzchar(nms))) {
      stop("family must be either fully named or fully unnamed.", call. = FALSE)
    } else {
      fam <- family
      names(fam) <- dataset_names
    }

    has_null <- vapply(fam, is.null, logical(1))
    if (any(has_null)) {
      stop(
        "family list cannot contain NULL entries; use family = NULL for the default family for all datasets.",
        call. = FALSE
      )
    }
  }

  specs <- lapply(fam, .svdx_family_spec)
  names(specs) <- dataset_names
  specs
}


.svdx_arg_for_dataset <- function(x, dataset, default = NULL) {
  if (is.null(x)) return(default)
  if (is.list(x) && !inherits(x, "formula") && !is.data.frame(x)) {
    if (!is.null(names(x)) && dataset %in% names(x)) return(x[[dataset]])
    return(x[[1L]])
  }
  x
}

.svdx_validate_integrated_scalar_or_list <- function(x, D, dataset_names, arg_name, validator) {
  if (!is.list(x) || inherits(x, "formula") || is.data.frame(x)) {
    validator(x, arg_name)
    return(invisible(TRUE))
  }
  if (length(x) != D) {
    stop("For integrated SVDX data, ", arg_name, " must be a scalar value or a list of length ", D, " parallel to Y.", call. = FALSE)
  }
  names(x) <- dataset_names
  for (ds in dataset_names) {
    validator(x[[ds]], paste0(arg_name, "[[", ds, "]]"))
  }
  invisible(TRUE)
}

.svdx_validate_family_for_prepare <- function(x, arg_name = "family") {
  .svdx_family_spec(x)
  invisible(TRUE)
}

.validate_svdx_fdata_for_Y <- function(fdata, Y, context = "Y") {
  if (is.null(fdata) || !is_svdx_fdata(fdata)) {
    stop("fdata must be supplied as an svdx_fdata object created by as_svdx_fdata().", call. = FALSE)
  }
  .svdx_validate_fdata_matches_features(fdata, rownames(Y), context = context)
  invisible(TRUE)
}


.svdx_integrated_dataset_names <- function(Y, dataset_name = NULL) {
  if (!is.list(Y) || is.data.frame(Y) || is_sparse_matrix(Y)) {
    stop("Integrated SVDX data require Y to be a list of matrices.", call. = FALSE)
  }
  D <- length(Y)
  if (D < 1L) stop("Integrated SVDX data require Y to contain at least one dataset.", call. = FALSE)

  yn <- names(Y)
  if (!is.null(yn) && length(yn) == D && all(nzchar(yn))) {
    if (anyDuplicated(yn)) stop("Dataset names in Y must be unique.", call. = FALSE)
    return(yn)
  }

  if (!is.null(dataset_name)) {
    dn <- unlist(dataset_name, use.names = FALSE)
    if (length(dn) == D && all(!is.na(dn)) && all(nzchar(as.character(dn)))) {
      dn <- as.character(dn)
      if (anyDuplicated(dn)) stop("dataset_name entries must be unique when supplied for integrated SVDX data.", call. = FALSE)
      return(dn)
    }
    if (length(dn) != D) {
      stop("For integrated SVDX data, dataset_name must be NULL or have one entry per dataset.", call. = FALSE)
    }
  }

  paste0("d", seq_len(D))
}

.svdx_parallel_list <- function(x, D, dataset_names, arg_name, required = FALSE, default = NULL) {
  if (is.null(x)) {
    if (isTRUE(required)) stop("For integrated SVDX data, ", arg_name, " must be a list parallel to Y.", call. = FALSE)
    if (is.null(default)) return(NULL)
    out <- rep(list(default), D)
    names(out) <- dataset_names
    return(out)
  }

  if (!is.list(x) || inherits(x, "formula") || is.data.frame(x)) {
    stop("For integrated SVDX data, ", arg_name, " must be a list parallel to Y.", call. = FALSE)
  }
  if (length(x) != D) {
    stop("For integrated SVDX data, ", arg_name, " must have length ", D, " to match Y.", call. = FALSE)
  }

  names(x) <- dataset_names
  x
}

.svdx_parallel_formula_list <- function(x, D, dataset_names, arg_name = "zformula") {
  if (is.null(x)) {
    out <- rep(list(stats::as.formula(~1)), D)
    names(out) <- dataset_names
    return(out)
  }
  if (inherits(x, "formula")) {
    out <- rep(list(x), D)
    names(out) <- dataset_names
    return(out)
  }
  out <- .svdx_parallel_list(x, D, dataset_names, arg_name, required = TRUE)
  bad <- !vapply(out, function(z) inherits(z, "formula") || is.null(z), logical(1))
  if (any(bad)) {
    stop("For integrated SVDX data, ", arg_name, " entries must be formulas or NULL.", call. = FALSE)
  }
  out[vapply(out, is.null, logical(1))] <- list(stats::as.formula(~1))
  out
}

.svdx_parallel_optional_dataset_vector <- function(x, D, dataset_names, arg_name) {
  if (is.null(x)) return(NULL)
  .svdx_parallel_list(x, D, dataset_names, arg_name, required = TRUE)
}

.normalize_integrated_inputs <- function(
    Y,
    zdata,
    zformula,
    fdata,
    row_offset = NULL,
    col_offset = NULL,
    t_j = NULL,
    dataset_name = NULL
) {
  D <- length(Y)
  dataset_names <- .svdx_integrated_dataset_names(Y, dataset_name = dataset_name)
  names(Y) <- dataset_names

  zdata <- .svdx_parallel_list(zdata, D, dataset_names, "zdata", required = TRUE)
  zformula <- .svdx_parallel_formula_list(zformula, D, dataset_names, "zformula")
  fdata <- .svdx_parallel_list(fdata, D, dataset_names, "fdata", required = TRUE)
  row_offset <- .svdx_parallel_optional_dataset_vector(row_offset, D, dataset_names, "row_offset")
  col_offset <- .svdx_parallel_optional_dataset_vector(col_offset, D, dataset_names, "col_offset")
  t_j <- .svdx_parallel_optional_dataset_vector(t_j, D, dataset_names, "t_j")

  list(
    Y = Y,
    zdata = zdata,
    zformula = zformula,
    fdata = fdata,
    row_offset = row_offset,
    col_offset = col_offset,
    t_j = t_j,
    dataset_names = dataset_names
  )
}

#' Prepare and validate SVDX input data
#'
#' Constructs an `svdx_data` object for high-level workflow and report wrappers.
#' The function validates sample alignment, formula variables, feature metadata,
#' response family compatibility, offsets, optional normalization, and optional
#' row-wise transformations. Low-level `svdx()` and `isvdx()` are not called.
#'
#' @param Y Feature-by-sample matrix, sparse matrix, or named list of matrices
#'   for integrated analyses.
#' @param xdata Shared sample metadata. Rows must match columns of `Y`.
#' @param xformula Formula for metadata-driven component scores.
#' @param zdata Nuisance metadata data frame for single-dataset analyses, or
#'   named list of data frames parallel to `Y` for integrated analyses.
#' @param zformula Nuisance formula, or named list of formulas for integrated
#'   analyses.
#' @param modelType Optional model-space specification passed to `svdx()`/`isvdx()`
#'   by high-level wrappers. Named entries should correspond to numeric variables
#'   in `xformula`; entries with explicit two-value ranges are validated against
#'   `xdata`.
#' @param fdata `svdx_fdata` object, or named list of `svdx_fdata` objects for
#'   integrated analyses. This argument is required.
#' @param family NULL/`"gaussian"`, `"poisson"`, or `"quasipoisson"`.
#'   Quasi-Poisson data are prepared like Poisson/count data; high-level
#'   workflows fit a Poisson objective and use quasi-Poisson inference.
#' @param row_offset Optional feature-level offsets.
#' @param col_offset Optional sample-level offsets.
#' @param t_j Optional normalization totals. If omitted and `normalize=TRUE`,
#'   column sums are used.
#' @param top_ftrs Optional number of high-variance features to retain.
#' @param normalize Logical; whether to apply column-sum size normalization. For
#'   count-family data this is applied through `col_offset`.
#' @param log1p_transform Logical; apply log1p transform for Gaussian data.
#' @param center_rows,scale_rows Logical row centering/scaling controls for
#'   Gaussian data.
#' @param allow_sparse_densification Logical; whether row-centering may densify
#'   sparse matrices.
#' @param dataset_name Optional label for single-dataset objects.
#' @return An object of class `svdx_data`.
#' @export
prepare_svdx_data <- function(
    Y,
    xdata,
    xformula,
    zdata = NULL,
    zformula = ~1,
    modelType = NULL,
    fdata,
    family = NULL,
    row_offset = NULL,
    col_offset = NULL,
    t_j = NULL,
    top_ftrs = NULL,
    normalize = FALSE,
    log1p_transform = TRUE,
    center_rows = TRUE,
    scale_rows = TRUE,
    allow_sparse_densification = FALSE,
    dataset_name = NULL
) {
  if (missing(xformula)) {
    stop("xformula must be provided. There is no SVDX analysis without xformula.", call. = FALSE)
  }
  if (missing(fdata) || is.null(fdata)) {
    stop("fdata must be provided. Use as_svdx_fdata() to construct it.", call. = FALSE)
  }

  integrated <- is.list(Y) && !is.data.frame(Y) && !is_sparse_matrix(Y)

  # Validate family early. This preserves family = NULL as the fast
  # PCA-like/nonparametric fitting path, while still rejecting unsupported
  # family specifications before expensive workflow/report code runs.
  family_spec_input <- if (isTRUE(integrated)) NULL else .svdx_family_spec(family)

  # For single-dataset preparation, preprocessing controls must be scalar
  # logical values. For integrated preparation, these may be supplied either
  # as scalar logical values shared across datasets or as lists parallel to Y;
  # dataset-specific validation happens in the recursive single-dataset calls.
  if (!isTRUE(integrated)) {
    validate_logical_scalar(normalize, "normalize")
    validate_logical_scalar(log1p_transform, "log1p_transform")
    validate_logical_scalar(center_rows, "center_rows")
    validate_logical_scalar(scale_rows, "scale_rows")
  } else {
    for (arg_nm in c("normalize", "log1p_transform", "center_rows", "scale_rows")) {
      arg_val <- get(arg_nm, inherits = FALSE)
      if (!is.list(arg_val)) validate_logical_scalar(arg_val, arg_nm)
    }
  }
  validate_logical_scalar(allow_sparse_densification, "allow_sparse_densification")

  if (isTRUE(integrated)) {
    normalized <- .normalize_integrated_inputs(
      Y = Y,
      zdata = zdata,
      zformula = zformula,
      fdata = fdata,
      row_offset = row_offset,
      col_offset = col_offset,
      t_j = t_j,
      dataset_name = dataset_name
    )

    Y <- normalized$Y
    zdata <- normalized$zdata
    zformula <- normalized$zformula
    fdata <- normalized$fdata
    row_offset <- normalized$row_offset
    col_offset <- normalized$col_offset
    t_j <- normalized$t_j
    dataset_names <- normalized$dataset_names

    .svdx_family_spec_list(family, dataset_names)
    .svdx_validate_integrated_scalar_or_list(normalize, length(dataset_names), dataset_names, "normalize", validate_logical_scalar)
    .svdx_validate_integrated_scalar_or_list(log1p_transform, length(dataset_names), dataset_names, "log1p_transform", validate_logical_scalar)
    .svdx_validate_integrated_scalar_or_list(center_rows, length(dataset_names), dataset_names, "center_rows", validate_logical_scalar)
    .svdx_validate_integrated_scalar_or_list(scale_rows, length(dataset_names), dataset_names, "scale_rows", validate_logical_scalar)

    prepared <- lapply(dataset_names, function(ds) {
      prepare_svdx_data(
        Y = Y[[ds]],
        xdata = xdata,
        xformula = xformula,
        zdata = zdata[[ds]],
        zformula = zformula[[ds]],
        modelType = modelType,
        fdata = fdata[[ds]],
        family = .svdx_arg_for_dataset(family, ds, default = NULL),
        row_offset = if (is.null(row_offset)) NULL else row_offset[[ds]],
        col_offset = if (is.null(col_offset)) NULL else col_offset[[ds]],
        t_j = if (is.null(t_j)) NULL else t_j[[ds]],
        top_ftrs = .svdx_arg_for_dataset(top_ftrs, ds, default = NULL),
        normalize = .svdx_arg_for_dataset(normalize, ds, default = FALSE),
        log1p_transform = .svdx_arg_for_dataset(log1p_transform, ds, default = TRUE),
        center_rows = .svdx_arg_for_dataset(center_rows, ds, default = TRUE),
        scale_rows = .svdx_arg_for_dataset(scale_rows, ds, default = TRUE),
        allow_sparse_densification = allow_sparse_densification,
        dataset_name = ds
      )
    })
    names(prepared) <- dataset_names

    family_list <- setNames(lapply(prepared, `[[`, "family"), dataset_names)
    fit_family_list <- setNames(lapply(prepared, `[[`, "fit_family"), dataset_names)
    inference_family_list <- setNames(lapply(prepared, `[[`, "inference_family"), dataset_names)
    fit_description_list <- setNames(lapply(prepared, `[[`, "fit_description"), dataset_names)
    family_out <- if (all(vapply(family_list, is.null, logical(1)))) NULL else family_list
    fit_family_out <- if (all(vapply(fit_family_list, is.null, logical(1)))) NULL else fit_family_list

    out <- list(
      integrated = TRUE,
      dataset_name = dataset_name,
      dataset_names = dataset_names,
      family = family_out,
      fit_family = fit_family_out,
      inference_family = inference_family_list,
      fit_description = fit_description_list,
      Y = setNames(lapply(prepared, `[[`, "Y"), dataset_names),
      xdata = prepared[[1L]]$xdata,
      zdata = setNames(lapply(prepared, `[[`, "zdata"), dataset_names),
      xformula = xformula,
      zformula = setNames(lapply(prepared, `[[`, "zformula"), dataset_names),
      modelType = prepared[[1L]]$modelType,
      x_term_info = prepared[[1L]]$x_term_info,
      z_term_info = setNames(lapply(prepared, `[[`, "z_term_info"), dataset_names),
      fdata = setNames(lapply(prepared, `[[`, "fdata"), dataset_names),
      row_offset = setNames(lapply(prepared, `[[`, "row_offset"), dataset_names),
      col_offset = setNames(lapply(prepared, `[[`, "col_offset"), dataset_names),
      row_offsets = setNames(lapply(prepared, `[[`, "row_offset"), dataset_names),
      col_offsets = setNames(lapply(prepared, `[[`, "col_offset"), dataset_names),
      normalization_factors = setNames(lapply(prepared, `[[`, "normalization_factors"), dataset_names),
      row_center = setNames(lapply(prepared, `[[`, "row_center"), dataset_names),
      row_scale = setNames(lapply(prepared, `[[`, "row_scale"), dataset_names),
      sparse_input = setNames(vapply(prepared, `[[`, logical(1), "sparse_input"), dataset_names),
      sparse_output = setNames(vapply(prepared, `[[`, logical(1), "sparse_output"), dataset_names)
    )
    class(out) <- "svdx_data"
    return(out)
  }

  family_spec <- family_spec_input
  family_type <- family_spec$family
  count_family <- family_spec$inference_family %in% c("poisson", "quasipoisson")

  Y_raw <- as_numeric_matrix_or_sparse(Y)

  if (any_nonfinite_matrix(Y_raw)) {
    stop("Y contains non-finite values.", call. = FALSE)
  }

  if (is.null(rownames(Y_raw))) {
    stop("Y must have rownames identifying features. These must match fdata.", call. = FALSE)
  }

  if (is.null(colnames(Y_raw))) {
    stop("Y must have colnames identifying samples. These must match rownames(xdata) and rownames(zdata).", call. = FALSE)
  }

  xdata <- as.data.frame(xdata)
  if (is.null(zdata)) {
    zdata <- xdata
  } else {
    zdata <- as.data.frame(zdata)
  }

  validate_sample_alignment(Y_raw, xdata, zdata)

  xprep <- prepare_formula_data(xdata, xformula, "xdata", "xformula")
  xdata <- xprep$data
  x_term_info <- xprep$term_info
  modelType <- resolve_modelType_xdata(modelType, xdata = xdata, xformula = xformula)

  zprep <- prepare_formula_data(zdata, zformula, "zdata", "zformula")
  zdata <- zprep$data
  z_term_info <- zprep$term_info

  .validate_svdx_fdata_for_Y(fdata, Y_raw, context = dataset_name %||% "Y")

  Y_has_negative <- any_negative(Y_raw)

  if (count_family) {
    if (Y_has_negative) {
      stop("family = 'poisson' or 'quasipoisson' requires nonnegative count data.", call. = FALSE)
    }
    if (isTRUE(log1p_transform)) {
      stop("log1p_transform = TRUE is not allowed for count-family SVDX data.", call. = FALSE)
    }
    if (isTRUE(center_rows)) {
      stop("center_rows = TRUE is not allowed for count-family SVDX data.", call. = FALSE)
    }
    if (isTRUE(scale_rows)) {
      stop("scale_rows = TRUE is not allowed for count-family SVDX data.", call. = FALSE)
    }
  }

  Y_work <- Y_raw
  normalize_requested <- isTRUE(normalize)
  normalize_used_on_Y <- FALSE
  normalization_applied_via_col_offset <- FALSE

  if (is.null(col_offset)) {
    col_offset <- rep(0, ncol(Y_raw))
  } else {
    validate_numeric_vector(col_offset, ncol(Y_raw), "col_offset", allow_null = FALSE)
  }
  names(col_offset) <- colnames(Y_raw)

  if (normalize_requested) {
    if (is.null(t_j)) {
      if (Y_has_negative) {
        stop("Default column-sum normalization requires nonnegative data.", call. = FALSE)
      }
      t_j <- matrix_col_sums(Y_raw)
    } else {
      validate_numeric_vector(t_j, ncol(Y_raw), "t_j", allow_null = FALSE)
    }
    names(t_j) <- colnames(Y_raw)
    t_tilde_j <- compute_size_factors(t_j)

    if (count_family) {
      col_offset <- col_offset + log(t_tilde_j)
      normalization_applied_via_col_offset <- TRUE
    } else {
      Y_work <- sparse_safe_col_normalize(Y_work, t_tilde_j)
      normalize_used_on_Y <- TRUE
    }
  } else {
    t_j <- rep(NA_real_, ncol(Y_raw))
    t_tilde_j <- rep(1, ncol(Y_raw))
    names(t_j) <- names(t_tilde_j) <- colnames(Y_raw)
  }

  if (!count_family && isTRUE(log1p_transform)) {
    if (any_negative(Y_work)) {
      stop("Cannot apply log1p_transform because Y contains negative values.", call. = FALSE)
    }
    Y_work <- sparse_safe_log1p(Y_work)
  }

  selected_idx <- select_top_hvg(Y_work, top_ftrs = top_ftrs)
  Y_work <- Y_work[selected_idx, , drop = FALSE]
  selected_features <- rownames(Y_work)

  if (is.null(row_offset)) {
    row_offset <- rep(0, nrow(Y_work))
  } else if (length(row_offset) == nrow(Y_raw)) {
    validate_numeric_vector(row_offset, nrow(Y_raw), "row_offset", allow_null = FALSE)
    row_offset <- row_offset[selected_idx]
  } else if (length(row_offset) == nrow(Y_work)) {
    validate_numeric_vector(row_offset, nrow(Y_work), "row_offset", allow_null = FALSE)
  } else {
    stop("row_offset must have length nrow(Y), length nrow(Y after feature selection), or be NULL.", call. = FALSE)
  }
  names(row_offset) <- rownames(Y_work)

  if (!count_family) {
    cs <- row_center_scale(
      Y_work,
      center_rows = center_rows,
      scale_rows = scale_rows,
      allow_sparse_densification = allow_sparse_densification
    )
    Y_processed <- cs$Y
    row_center <- cs$row_center
    row_scale <- cs$row_scale
  } else {
    Y_processed <- Y_work
    row_center <- rep(0, nrow(Y_processed))
    row_scale <- rep(1, nrow(Y_processed))
    names(row_center) <- names(row_scale) <- rownames(Y_processed)
  }

  fdata_selected <- .svdx_subset_fdata(fdata, selected_features)

  stopifnot(nrow(Y_processed) == length(row_offset))
  stopifnot(ncol(Y_processed) == length(col_offset))
  stopifnot(nrow(xdata) == ncol(Y_processed))
  stopifnot(nrow(zdata) == ncol(Y_processed))

  out <- list(
    integrated = FALSE,
    dataset_name = dataset_name,
    family = family_type,
    fit_family = family_spec$fit_family,
    inference_family = family_spec$inference_family,
    fit_description = family_spec$fit_description,
    Y = Y_processed,
    xdata = xdata,
    zdata = zdata,
    xformula = xformula,
    zformula = zformula,
    modelType = modelType,
    x_term_info = x_term_info,
    z_term_info = z_term_info,
    fdata = fdata_selected,
    row_offset = row_offset,
    col_offset = col_offset,
    row_offsets = row_offset,
    col_offsets = col_offset,
    normalization_factors = t_tilde_j,
    row_center = row_center,
    row_scale = row_scale,
    selected_idx = selected_idx,
    selected_features = selected_features,
    sparse_input = is_sparse_matrix(Y_raw),
    sparse_output = is_sparse_matrix(Y_processed),
    normalize_requested = normalize_requested,
    normalize_used_on_Y = normalize_used_on_Y,
    normalization_applied_via_col_offset = normalization_applied_via_col_offset,
    log1p_transform = isTRUE(log1p_transform),
    center_rows = isTRUE(center_rows),
    scale_rows = isTRUE(scale_rows)
  )
  class(out) <- "svdx_data"
  out
}

preprocess_svdx_data <- function(...) {
  prepare_svdx_data(...)
}
