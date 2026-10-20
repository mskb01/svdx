//Custom Rcpp wrappers for wrapping std::vector of matrices and vectors.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]



namespace Rcpp {

// wrap a std::vector of matrices
template <>
SEXP wrap(const std::vector<arma::mat>& x);

// wrap a std::vector of matrices
template <>
SEXP wrap(const std::vector<arma::vec>& x);

}
