//Custom Rcpp wrappers for wrapping std::vector of matrices and vectors.

#include "customRcppWrappers.h"
// [[Rcpp::depends(RcppArmadillo)]]


// wrap a std::vector of matrices
namespace Rcpp {
template <>
SEXP wrap(const std::vector<arma::mat>& x) {
  int n = x.size();
  Rcpp::List list(n);
  for (int i = 0; i < n; ++i) {
    list[i] = Rcpp::wrap(x[i]);
  }
  return list;
}
}

// wrap a std::vector of vectors
namespace Rcpp {
template <>
SEXP wrap(const std::vector<arma::vec>& x) {
  int n = x.size();
  Rcpp::List list(n);
  for (int i = 0; i < n; ++i) {
    list[i] = Rcpp::wrap(x[i]);
  }
  return list;
}
}
