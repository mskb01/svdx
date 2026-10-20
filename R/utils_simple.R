# `%||%` <- function(x, y) {
#   if (!is.null(x)) x else y
# }
# 
# 
# 
# .safe_bh <- function(p) {
#   out <- rep(NA_real_, length(p))
#   ok <- is.finite(p) & !is.na(p)
#   out[ok] <- p.adjust(p[ok], method = "BH")
#   out
# }
# 
# .make_feature_names <- function(Y, p = NULL) {
#   rn <- rownames(Y)
#   if (!is.null(rn)) return(rn)
#   if (is.null(p)) p <- nrow(Y)
#   paste0("feature_", seq_len(p))
# }
# 
# .make_component_names <- function(K) {
#   paste0("U", seq_len(K))
# }
# 
# .require_namespace <- function(pkg) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     stop("Package '", pkg, "' is required but not installed.", call. = FALSE)
#   }
# }
