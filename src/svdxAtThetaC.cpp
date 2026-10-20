

/*
 * wrapper for calls to C++ svdx from R side
 */


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <tbb/global_control.h>
#include <Rinternals.h>


#include<unordered_map>
#include<list>
#include<vector>
#include<array>
#include<random>
#include<string>
#include<cmath>
#include<stdexcept>

#include "svdxAtTheta.h"
#include "svdx.h"
#include "smr.h"
#include "smttfr.h"


using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// [[Rcpp::export]]
Rcpp::List svdxAtThetaC(
    const SEXP& Y,
    const double f2,
    const Rcpp::Nullable<double> f3,
    const arma::vec theta,
    const int& K,
    const Rcpp::List& allDataMod,
    const int& fitter_maxIter=300,
    const double& fitter_ftol_abs=.001,
    const double& fitter_xtol_abs=.01,
    const int& fitter_inner_maxIter=25,
    const double& fitter_inner_ftol_abs=1e-4,
    const double f1=1,  //const Rcpp::Nullable<arma::uvec> trainColumnsR=R_NilValue, //train columns in R index, not C index
    const bool estK = true
){


  double f3v = 1;//just a placeholder on stack
  double* f3p=nullptr; //critical to not assign &f3v here unless f3 is defined in the input.
  if( f3.isNotNull() ){
    f3v = Rcpp::as<double>(f3);
    f3p = &f3v;  //critical to note take this statement outside!
  }

  //Read from R passed train test folds, and all data mods
  smr s( allDataMod  );

  Rcpp::List fitAtOptCV;
  if( Rf_isS4( Y   )  ){ //sparse?
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat = Rcpp::as<arma::sp_mat>(Y);
      svdxAtTheta<arma::sp_mat> svx( Ymat, s, f1, K, f2, theta, f3p, fitter_maxIter, fitter_ftol_abs, fitter_xtol_abs, fitter_inner_maxIter, fitter_inner_ftol_abs );
      svx.estimate();
      fitAtOptCV = svx.getFitAtOptCV();
    } else {
      throw std::invalid_argument( "Unknown S4 matrix type" );
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    svdxAtTheta<arma::mat> svx( Ymat, s, f1, K, f2, theta, f3p, fitter_maxIter, fitter_ftol_abs, fitter_xtol_abs, fitter_inner_maxIter, fitter_inner_ftol_abs );
    svx.estimate();
    fitAtOptCV = svx.getFitAtOptCV();
  }

  return fitAtOptCV;
}
