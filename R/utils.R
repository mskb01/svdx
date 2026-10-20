`%||%` <- function(x, y) if (!is.null(x)) x else y

.require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required but not installed.", call. = FALSE)
  }
}

.center_vec <- function(x, w = NULL) {
  x <- as.numeric(x)
  if (is.null(w)) {
    x - mean(x, na.rm = TRUE)
  } else {
    x - sum(w * x, na.rm = TRUE) / sum(w, na.rm = TRUE)
  }
}

.cosine <- function(x, y, w = NULL, center = TRUE, eps = 1e-12) {
  x <- as.numeric(x)
  y <- as.numeric(y)
  
  if (center) {
    x <- .center_vec(x, w)
    y <- .center_vec(y, w)
  }
  
  if (!is.null(w)) {
    x <- sqrt(w) * x
    y <- sqrt(w) * y
  }
  
  nx <- sqrt(sum(x^2, na.rm = TRUE))
  ny <- sqrt(sum(y^2, na.rm = TRUE))
  
  if (!is.finite(nx) || !is.finite(ny) || nx < eps || ny < eps) {
    return(NA_real_)
  }
  
  sum(x * y, na.rm = TRUE) / (nx * ny)
}

.safe_bh <- function(p) {
  out <- rep(NA_real_, length(p))
  ok <- is.finite(p) & !is.na(p)
  out[ok] <- stats::p.adjust(p[ok], method = "BH")
  out
}

.safe_auc_score <- function(score, truth) {
  truth <- as.logical(truth)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(truth)
  if (sum(ok) == 0 || length(unique(truth[ok])) < 2) return(NA_real_)

  if (requireNamespace("pROC", quietly = TRUE)) {
    roc_obj <- pROC::roc(response = truth[ok], predictor = score[ok], quiet = TRUE, direction = "<")
    return(as.numeric(pROC::auc(roc_obj)))
  }

  r <- rank(score[ok], ties.method = "average")
  y <- truth[ok]
  n1 <- sum(y)
  n0 <- sum(!y)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

.safe_cor_vec <- function(x, y, method = "pearson") {
  x <- as.numeric(x); y <- as.numeric(y)
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  stats::cor(x[ok], y[ok], method = method)
}

.safe_rel_frob <- function(A, B, eps = 1e-8) {
  num <- sqrt(sum((A - B)^2, na.rm = TRUE))
  den <- sqrt(sum(B^2, na.rm = TRUE))
  if (!is.finite(den) || den < eps) return(NA_real_)
  num / den
}

standardize_features <- function(Y, eps = 1e-8) {
  mu <- rowMeans(Y, na.rm = TRUE)
  sdv <- apply(Y, 1, stats::sd, na.rm = TRUE)
  sdv[sdv < eps | !is.finite(sdv)] <- 1
  Y_std <- sweep(Y, 1, mu, "-")
  Y_std <- sweep(Y_std, 1, sdv, "/")
  list(Y = Y_std, center = mu, scale = sdv)
}

make_design <- function(meta, formula) stats::model.matrix(formula, data = meta)

adjust_Y_by_formula <- function(Y, meta, zformula) {
  .require_namespace("limma")
  Z <- make_design(meta, zformula)
  ft <- limma::lmFit(Y, Z)
  Y - fitted(ft)
}

.make_feature_names <- function(Y, p = NULL) {
  rn <- rownames(Y)
  if (!is.null(rn)) return(rn)
  if (is.null(p)) p <- nrow(Y)
  paste0("feature_", seq_len(p))
}

.make_component_names <- function(K) paste0("U", seq_len(K))

subset_svdx_inputs <- function(Y, xdata, zdata = xdata, graphData = NULL, si) {
  list(
    Y = Y[, si, drop = FALSE],
    xdata = xdata[si, , drop = FALSE],
    zdata = zdata[si, , drop = FALSE],
    graphData = if (is.null(graphData)) NULL else graphData[, si, drop = FALSE]
  )
}

.svdx_validate_train_fraction <- function(train_fraction) {
  if (!is.numeric(train_fraction) || length(train_fraction) != 1L || !is.finite(train_fraction) ||
      train_fraction <= 0 || train_fraction > 1) {
    stop("`train_fraction` must be a single finite number in (0, 1].", call. = FALSE)
  }
  invisible(TRUE)
}

.svdx_resolve_sample_indices <- function(x, sample_names, arg = "sample indices") {
  if (is.null(x)) return(NULL)

  n <- length(sample_names)

  if (is.character(x)) {
    idx <- match(x, sample_names)
    if (any(is.na(idx))) {
      stop("Some ", arg, " names were not found in the sample names.", call. = FALSE)
    }
    return(idx)
  }

  if (is.logical(x)) {
    if (length(x) != n) {
      stop("Logical ", arg, " must have length equal to the number of samples.", call. = FALSE)
    }
    if (any(is.na(x))) {
      stop("Logical ", arg, " cannot contain NA values.", call. = FALSE)
    }
    return(which(x))
  }

  if (!is.numeric(x) || any(is.na(x))) {
    stop(arg, " must be sample names, logical values, or numeric indices.", call. = FALSE)
  }

  idx <- as.integer(x)
  if (any(idx != x) || any(idx < 1L) || any(idx > n)) {
    stop(arg, " must be integer sample indices between 1 and the number of samples.", call. = FALSE)
  }

  idx
}

.svdx_validate_sample_frame <- function(x, n, arg = "zdata") {
  if (is.null(x)) return(invisible(TRUE))
  if (is.null(nrow(x)) || nrow(x) != n) {
    stop("`", arg, "` must have one row per sample.", call. = FALSE)
  }
  invisible(TRUE)
}

.svdx_validate_graph_data <- function(graphData, n) {
  if (is.null(graphData)) return(invisible(TRUE))
  if (is.null(dim(graphData)) || ncol(graphData) != n) {
    stop("`graphData` must have one column per sample.", call. = FALSE)
  }
  invisible(TRUE)
}

.svdx_validate_row_offset <- function(row_offset, p, arg = "row_offset") {
  if (is.null(row_offset)) return(invisible(TRUE))
  if (!is.numeric(row_offset) || length(row_offset) != p || any(!is.finite(row_offset))) {
    stop("`", arg, "` must be a finite numeric vector with one value per feature.", call. = FALSE)
  }
  invisible(TRUE)
}

.svdx_validate_col_offset <- function(col_offset, n, arg = "col_offset") {
  if (is.null(col_offset)) return(invisible(TRUE))
  if (!is.numeric(col_offset) || length(col_offset) != n || any(!is.finite(col_offset))) {
    stop("`", arg, "` must be a finite numeric vector with one value per sample.", call. = FALSE)
  }
  invisible(TRUE)
}

.svdx_align_dataset_list <- function(x, dataset_names, arg, allow_null = FALSE) {
  if (is.null(x)) {
    if (isTRUE(allow_null)) return(NULL)
    stop("`", arg, "` must not be NULL.", call. = FALSE)
  }

  if (!is.list(x) || is.data.frame(x)) {
    stop("`", arg, "` must be a list with one entry per dataset.", call. = FALSE)
  }

  if (length(x) != length(dataset_names)) {
    stop("`", arg, "` must have one entry per dataset.", call. = FALSE)
  }

  nm <- names(x)
  if (!is.null(nm) && all(nzchar(nm))) {
    missing <- setdiff(dataset_names, nm)
    extra <- setdiff(nm, dataset_names)
    if (length(missing) > 0 || length(extra) > 0) {
      stop(
        "`", arg, "` names must exactly match dataset names. Missing: ",
        paste(missing, collapse = ", "),
        "; extra: ",
        paste(extra, collapse = ", "),
        ".",
        call. = FALSE
      )
    }
    x <- x[dataset_names]
  } else if (!is.null(nm) && any(nzchar(nm))) {
    stop("`", arg, "` must be either fully named or fully unnamed.", call. = FALSE)
  }

  names(x) <- dataset_names
  x
}

# .get_vmat <- function(fit) {
#   vmat <- fit$fitAtOptCV$vmat %||% fit$vmat
#   if (is.null(vmat)) stop("Could not find vmat in fit object.", call. = FALSE)
#   as.matrix(vmat)
# }
.get_vmat <- function(fit, dataset = NULL) {
  if (inherits(fit, "SSANOVASVD")) {
    return(fit$fitAtOptCV$vmat %||% fit$vmat)
  }
  
  if (inherits(fit, "iSSANOVASVD")) {
    vlist <- fit$fitAtOptCV$vmat %||% fit$vmat
    if (!is.list(vlist)) stop("iSSANOVASVD vmat must be list-valued.", call. = FALSE)
    
    if (is.null(dataset)) {
      stop("dataset must be supplied for iSSANOVASVD V/loadings plots.", call. = FALSE)
    }
    
    d <- if (is.numeric(dataset)) dataset else match(dataset, names(vlist))
    if (is.na(d)) stop("Unknown dataset.", call. = FALSE)
    
    return(vlist[[d]])
  }
  
  stop("Unsupported fit class.", call. = FALSE)
}

.get_umat <- function(fit) {
  umat <- fit$fitAtOptCV$umat %||% fit$umat
  if (is.null(umat)) stop("Could not find umat in fit object.", call. = FALSE)
  as.matrix(umat)
}
