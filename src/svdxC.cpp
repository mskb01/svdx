

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
Rcpp::List svdxC(
    const SEXP& Y,
    const int& KMAX,
    const Rcpp::List& allDataMod,
    const Rcpp::List& trainTestFoldMods,
    const int& optCV_maxIter=20,
    const double& optCV_xtol_abs=.01,
    const double& optCV_ftol_abs=.001,
    const int& fitter_maxIter=300,
    const double& fitter_ftol_abs=.001,
    const double& fitter_xtol_abs=.01,
    const int& fitter_inner_maxIter=25,
    const double& fitter_inner_ftol_abs=1e-4,
    const double& nlopt_xtol_abs=.01,
    const double& nlopt_ftol_abs=.001,
    const int& nlopt_maxeval=0,
    const double f1=1,  //const Rcpp::Nullable<arma::uvec> trainColumnsR=R_NilValue, //train columns in R index, not C index
    const bool estK = true, 
    const int numThreads = 2

){

  ////const int numThreads=2
  //tbb::global_control gcp( tbb::global_control::max_allowed_parallelism, numThreads );


  //Read from R passed train test folds, and all data mods
  smr s( allDataMod  );
  smttfr tt( trainTestFoldMods );

  bool optForK =estK;
  bool optForf2=true;
  bool optForf3=(tt.f3Init_ptr != nullptr);
  bool optFortheta=true;

  Rcpp::List optCV, fitAtOptCV;
  
  //create strict arena cage
  tbb::task_arena arena(numThreads);
  arena.execute([&]() {
    if( Rf_isS4( Y   )  ){ //sparse?
      if(Rf_inherits(Y , "dgCMatrix")) {
        arma::sp_mat Ymat = Rcpp::as<arma::sp_mat>(Y);
        svdx<arma::sp_mat> svx( Ymat,tt, s );
        svx.estimate(f1,
                     KMAX,
                     optForK,
                     optForf2,
                     optForf3,
                     optFortheta,
                     optCV_maxIter,
                     optCV_xtol_abs,
                     optCV_ftol_abs,
                     nlopt_xtol_abs,
                     nlopt_ftol_abs,
                     nlopt_maxeval,
                     fitter_maxIter,
                     fitter_ftol_abs,
                     fitter_xtol_abs,
                     fitter_inner_maxIter,
                     fitter_inner_ftol_abs);
        optCV = svx.getOptCV();
        fitAtOptCV = svx.getFitAtOptCV();
      } else {
        throw std::invalid_argument( "Unknown S4 matrix type" );
      }
    } else {
      arma::mat Ymat = Rcpp::as<arma::mat>(Y);
      
      svdx<arma::mat> svx( Ymat, tt, s );
      svx.estimate(f1,
                   KMAX,
                   optForK,
                   optForf2,
                   optForf3,
                   optFortheta,
                   optCV_maxIter,
                   optCV_xtol_abs,
                   optCV_ftol_abs,
                   nlopt_xtol_abs,
                   nlopt_ftol_abs,
                   nlopt_maxeval,
                   fitter_maxIter,
                   fitter_ftol_abs,
                   fitter_xtol_abs,
                   fitter_inner_maxIter,
                   fitter_inner_ftol_abs);
      optCV = svx.getOptCV();
      fitAtOptCV = svx.getFitAtOptCV();
    }
  });

  Rcpp::List result;
  result["optCV"]=optCV;
  result["fitAtOptCV"]=fitAtOptCV;
  return result;
}
