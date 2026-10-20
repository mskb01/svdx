#ifndef ISVDX_H
#define ISVDX_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>
#include <memory>
#include "nlopt.hpp"

#include "IMLFCOnColsListTT.h"
#include "IMLFCOptWorker.h"
#include "isvdxAtTheta.h"


using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

///////////////////////////////////////////////////////////////////////
/*
 * Trains on object, and evaluates MSE on folds.
 * Same as above but only on columns
 * the "mses" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
class isvdx{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data

private:
  const dsListC& Y;
  const imttfr& ttf;
  const imr& s;
  std::unique_ptr< IMLFCOpt > opt;
  std::unique_ptr< isvdxAtTheta > isvxAtopt;

public:

  explicit isvdx(
    const dsListC& Y,
    const imttfr& ttf,
    const imr& s
  );

  ~isvdx()=default;

  void optcv(
      double f1,
      unsigned int KInit,//initial conditions
      double f2Init,
      const arma::vec& thetaInit,
      double* f3Init=nullptr,

      bool optForK=false, //optional opts to switch on / off
      bool optForf2 = true,
      bool optForf3 = false,
      bool optFortheta = true,

      int optCV_maxIter = 20,
      double optCV_xtol_abs = .01,
      double optCV_ftol_abs = .001,
      double nlopt_xtol_abs = .01,
      double nlopt_ftol_abs = .001,
      int nlopt_maxeval = 0,
      int fitter_maxIter=300,
      double fitter_ftol_abs=.001,
      double fitter_xtol_abs=.01,
      int fitter_inner_maxIter=25,
      double fitter_inner_ftol_abs=1e-4
  );

  void estimWithTheta(double& f1,
                      unsigned int& K,
                      double& f2, double& f3,
                      const arma::vec& theta,
                      int fitter_maxIter=300,
                      double fitter_ftol_abs=.001,
                      double fitter_xtol_abs=.01,
                      int fitter_inner_maxIter=25,
                      double fitter_inner_ftol_abs=1e-4 );
  void estimate(
      double f1,
      unsigned int K,
      bool optForK=false,
      bool optForf2 = true,
      bool optForf3 = false,
      bool optFortheta = true,
      int optCV_maxIter = 20,
      double optCV_xtol_abs = .01,
      double optCV_ftol_abs = .001,
      double nlopt_xtol_abs = .01,
      double nlopt_ftol_abs = .001,
      int nlopt_maxeval = 0,
      int fitter_maxIter=300,
      double fitter_ftol_abs=.001,
      double fitter_xtol_abs=.01,
      int fitter_inner_maxIter=25,
      double fitter_inner_ftol_abs=1e-4);

  /*
   * Get
   */
  Rcpp::List getOptCV();
  Rcpp::List getFitAtOptCV();
};




#endif
