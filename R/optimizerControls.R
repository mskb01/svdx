# Internal optimizer-control helpers.
#
# Public entry points (svdx/isvdx) accept compact control bundles. These
# helpers normalize those bundles once, preserve legacy scalar arguments,
# and return a single resolved control object. Downstream code should receive
# resolved scalar values rather than the lists themselves.

.default_fitter_control <- function() {
  list(
    ftol_abs = 0.001,
    xtol_abs = 0.01,   # reserved for fitter parameter-change convergence
    maxIter = 200L,
    inner_maxIter = 25L,
    inner_ftol_abs = 1e-4
  )
}

.default_optCV_control <- function() {
  list(
    ftol_abs = 0.001,
    xtol_abs = 0.01,
    maxIter = 20L
  )
}

.default_nlopt_control <- function() {
  list(
    ftol_abs = 0.001,
    xtol_abs = 0.01,
    maxeval = 0L       # NLopt convention: 0 means no evaluation limit
  )
}

.assert_control_list <- function(x, name) {
  if (is.null(x)) {
    return(list())
  }
  if (!is.list(x)) {
    stop(sprintf("%s must be a list.", name), call. = FALSE)
  }
  x
}

.check_control_names <- function(x, allowed, name) {
  if (length(x) == 0L) {
    return(invisible(NULL))
  }
  nms <- names(x)
  if (is.null(nms) || any(!nzchar(nms))) {
    stop(sprintf("All entries in %s must be named.", name), call. = FALSE)
  }
  unknown <- setdiff(nms, allowed)
  if (length(unknown)) {
    stop(
      sprintf(
        "Unknown %s entr%s: %s.",
        name,
        if (length(unknown) == 1L) "y" else "ies",
        paste(unknown, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(NULL)
}

.validate_nonnegative_scalar <- function(x, label) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < 0) {
    stop(sprintf("%s must be one finite non-negative number.", label), call. = FALSE)
  }
  as.numeric(x)
}

.validate_positive_integer <- function(x, label) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x <= 0 || x != floor(x)) {
    stop(sprintf("%s must be one positive integer.", label), call. = FALSE)
  }
  as.integer(x)
}

.validate_nonnegative_integer <- function(x, label) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x < 0 || x != floor(x)) {
    stop(sprintf("%s must be one non-negative integer.", label), call. = FALSE)
  }
  as.integer(x)
}

.normalize_fitter_control <- function(x, defaults = .default_fitter_control()) {
  x <- .assert_control_list(x, "fitter_control")
  .check_control_names(x, c("ftol_abs", "xtol_abs", "maxIter", "inner_maxIter", "inner_ftol_abs"), "fitter_control")

  out <- defaults
  if (!is.null(x$ftol_abs)) {
    out$ftol_abs <- .validate_nonnegative_scalar(x$ftol_abs, "fitter_control$ftol_abs")
  }
  if (!is.null(x$xtol_abs)) {
    out$xtol_abs <- .validate_nonnegative_scalar(x$xtol_abs, "fitter_control$xtol_abs")
  }
  if (!is.null(x$maxIter)) {
    out$maxIter <- .validate_positive_integer(x$maxIter, "fitter_control$maxIter")
  }
  if (!is.null(x$inner_maxIter)) {
    out$inner_maxIter <- .validate_positive_integer(x$inner_maxIter, "fitter_control$inner_maxIter")
  }
  if (!is.null(x$inner_ftol_abs)) {
    out$inner_ftol_abs <- .validate_nonnegative_scalar(x$inner_ftol_abs, "fitter_control$inner_ftol_abs")
  }
  out
}

.normalize_optCV_control <- function(x, defaults = .default_optCV_control()) {
  x <- .assert_control_list(x, "optCV_control")
  .check_control_names(x, c("ftol_abs", "xtol_abs", "maxIter"), "optCV_control")

  out <- defaults
  if (!is.null(x$ftol_abs)) {
    out$ftol_abs <- .validate_nonnegative_scalar(x$ftol_abs, "optCV_control$ftol_abs")
  }
  if (!is.null(x$xtol_abs)) {
    out$xtol_abs <- .validate_nonnegative_scalar(x$xtol_abs, "optCV_control$xtol_abs")
  }
  if (!is.null(x$maxIter)) {
    out$maxIter <- .validate_positive_integer(x$maxIter, "optCV_control$maxIter")
  }
  out
}

.normalize_nlopt_control <- function(x, defaults = .default_nlopt_control()) {
  x <- .assert_control_list(x, "nlopt_control")
  .check_control_names(x, c("ftol_abs", "xtol_abs", "maxeval"), "nlopt_control")

  out <- defaults
  if (!is.null(x$ftol_abs)) {
    out$ftol_abs <- .validate_nonnegative_scalar(x$ftol_abs, "nlopt_control$ftol_abs")
  }
  if (!is.null(x$xtol_abs)) {
    out$xtol_abs <- .validate_nonnegative_scalar(x$xtol_abs, "nlopt_control$xtol_abs")
  }
  if (!is.null(x$maxeval)) {
    out$maxeval <- .validate_nonnegative_integer(x$maxeval, "nlopt_control$maxeval")
  }
  out
}

.resolve_control_value <- function(bundle_value,
                                   legacy_value,
                                   legacy_supplied,
                                   default_value,
                                   bundle_label,
                                   legacy_label,
                                   validator) {
  if (!is.null(bundle_value)) {
    if (isTRUE(legacy_supplied)) {
      message(sprintf(
        "Both %s and %s were supplied; using %s.",
        bundle_label, legacy_label, bundle_label
      ))
    }
    return(validator(bundle_value, bundle_label))
  }

  if (isTRUE(legacy_supplied)) {
    return(validator(legacy_value, legacy_label))
  }

  default_value
}

.normalize_init_control <- function(init_control, fitter, optCV, nlopt) {
  init_control <- .assert_control_list(init_control, "init_control")
  .check_control_names(init_control, c("fitter", "optCV", "nlopt"), "init_control")

  list(
    fitter = .normalize_fitter_control(init_control$fitter, defaults = fitter),
    optCV = .normalize_optCV_control(init_control$optCV, defaults = optCV),
    nlopt = .normalize_nlopt_control(init_control$nlopt, defaults = nlopt)
  )
}

.resolve_optimizer_controls <- function(
    fitter_control = list(),
    optCV_control = list(),
    nlopt_control = list(),
    init_control = list(),
    fitter_maxIter,
    fitter_eps,
    optCV_maxIter,
    optCV_eps,
    xtol_abs,
    ftol_abs,
    legacy_supplied = list()
) {
  fitter_control <- .assert_control_list(fitter_control, "fitter_control")
  optCV_control <- .assert_control_list(optCV_control, "optCV_control")
  nlopt_control <- .assert_control_list(nlopt_control, "nlopt_control")

  .check_control_names(fitter_control, c("ftol_abs", "xtol_abs", "maxIter", "inner_maxIter", "inner_ftol_abs"), "fitter_control")
  .check_control_names(optCV_control, c("ftol_abs", "xtol_abs", "maxIter"), "optCV_control")
  .check_control_names(nlopt_control, c("ftol_abs", "xtol_abs", "maxeval"), "nlopt_control")

  fitter_defaults <- .default_fitter_control()
  optCV_defaults <- .default_optCV_control()
  nlopt_defaults <- .default_nlopt_control()

  supplied <- function(name) isTRUE(legacy_supplied[[name]])

  fitter <- list(
    ftol_abs = .resolve_control_value(
      fitter_control$ftol_abs, fitter_eps, supplied("fitter_eps"),
      fitter_defaults$ftol_abs,
      "fitter_control$ftol_abs", "fitter_eps", .validate_nonnegative_scalar
    ),
    xtol_abs = if (!is.null(fitter_control$xtol_abs))
      .validate_nonnegative_scalar(fitter_control$xtol_abs, "fitter_control$xtol_abs")
    else fitter_defaults$xtol_abs,
    maxIter = .resolve_control_value(
      fitter_control$maxIter, fitter_maxIter, supplied("fitter_maxIter"),
      fitter_defaults$maxIter,
      "fitter_control$maxIter", "fitter_maxIter", .validate_positive_integer
    ),
    inner_maxIter = if (!is.null(fitter_control$inner_maxIter))
      .validate_positive_integer(fitter_control$inner_maxIter, "fitter_control$inner_maxIter")
    else fitter_defaults$inner_maxIter,
    inner_ftol_abs = if (!is.null(fitter_control$inner_ftol_abs))
      .validate_nonnegative_scalar(fitter_control$inner_ftol_abs, "fitter_control$inner_ftol_abs")
    else fitter_defaults$inner_ftol_abs
  )

  optCV <- list(
    # Legacy ftol_abs historically controlled both NLopt ftol and the
    # coordinate-descent objective-change stopping rule.
    ftol_abs = .resolve_control_value(
      optCV_control$ftol_abs, ftol_abs, supplied("ftol_abs"),
      optCV_defaults$ftol_abs,
      "optCV_control$ftol_abs", "ftol_abs", .validate_nonnegative_scalar
    ),
    # Legacy optCV_eps is the RMS log10-parameter-change tolerance.
    xtol_abs = .resolve_control_value(
      optCV_control$xtol_abs, optCV_eps, supplied("optCV_eps"),
      optCV_defaults$xtol_abs,
      "optCV_control$xtol_abs", "optCV_eps", .validate_nonnegative_scalar
    ),
    maxIter = .resolve_control_value(
      optCV_control$maxIter, optCV_maxIter, supplied("optCV_maxIter"),
      optCV_defaults$maxIter,
      "optCV_control$maxIter", "optCV_maxIter", .validate_positive_integer
    )
  )

  nlopt <- list(
    ftol_abs = .resolve_control_value(
      nlopt_control$ftol_abs, ftol_abs, supplied("ftol_abs"),
      nlopt_defaults$ftol_abs,
      "nlopt_control$ftol_abs", "ftol_abs", .validate_nonnegative_scalar
    ),
    xtol_abs = .resolve_control_value(
      nlopt_control$xtol_abs, xtol_abs, supplied("xtol_abs"),
      nlopt_defaults$xtol_abs,
      "nlopt_control$xtol_abs", "xtol_abs", .validate_nonnegative_scalar
    ),
    maxeval = if (!is.null(nlopt_control$maxeval))
      .validate_nonnegative_integer(nlopt_control$maxeval, "nlopt_control$maxeval")
    else nlopt_defaults$maxeval
  )

  list(
    fitter = fitter,
    optCV = optCV,
    nlopt = nlopt,
    init = .normalize_init_control(init_control, fitter, optCV, nlopt)
  )
}
