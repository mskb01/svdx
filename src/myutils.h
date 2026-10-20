// myutils.h

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppParallel;
//using namespace arma;

#include<string>
#include<cmath>


//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>



// [[Rcpp::export]]
arma::mat invpsd( const arma::mat& X, double thresh, bool force);

double mse_vec(const arma::vec& vec1, const arma::vec& vec2);

double mse_mats(const arma::mat& mat1, const arma::mat& mat2);

double mse_VectorOfVecs(const std::vector<arma::vec>& vec1, const std::vector<arma::vec>& vec2);
double mse_VectorOfMats(const std::vector<arma::mat>& vecMat1, const std::vector<arma::mat>& vecMat2);

std::vector<arma::mat> transposeVecOfMats(const std::vector<arma::mat>& vecOfMats);
