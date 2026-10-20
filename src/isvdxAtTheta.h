#ifndef ISVDXATTHETA_H
#define ISVDXATTHETA_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]


#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>
#include <memory>
#include <stdexcept>
#include "nlopt.hpp"
#include<vector>
#include<cmath>
#include<string>

#include "IMLFCOptWorker.h"
#include "imr.h"
#include "myutils.h"
#include "customRcppWrappers.h"

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
class isvdxAtTheta { //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data


public:
  //All lists below mst be of the same length.
  //mses should have the same length.

  const dsListC& Y;
  const arma::uvec& columns1;
  const std::vector<arma::mat>& Z;       //Z may or may not be present
  std::vector<arma::mat> Zeta;
  arma::mat hmat; //initial conditions
  arma::mat umat;//initial conditions
  std::vector<arma::mat> vmat;//initial conditions
  const arma::uvec& qids;
  const int q;   // saves number of qids for each fold
  const arma::mat& S;
  const arma::mat& SStq;
  const arma::mat& RsCbound;
  const std::vector<arma::vec>& rowOffsets;
  const std::vector<arma::vec>& colOffsets;

  const int& r;
  const int& maxIter;
  const double& ftol_abs;
  const double& xtol_abs;
  const int& inner_maxIter;
  const double& inner_ftol_abs;

  // PARAMETERS, and dependents
  double f1;
  unsigned int K;
  double f2;

  arma::vec theta;
  unsigned int nTheta;
  const arma::vec& wd; 

  arma::mat RTheta; // working values qxq penalty
  arma::mat MTheta; // qxq penalty
  arma::mat RtildeTheta; // qxq penalty
  arma::mat BTheta;      //qxq full kernel inverse

  //graph specific additions : first two do not alter through the optimization
  const arma::sp_mat* graphDistMat = nullptr;
  const arma::uvec* graphDistOnlyOnPoints = nullptr;
  double f3=0.0;
  bool f3set=false;

  //exponential family
  const std::vector<std::string>* family = nullptr;
  std::vector<arma::vec> aphi; //not a pointer this time


  int dispatch; //0 if single-np, 1 if single-np-graph, 2 if single-generalized, 3, if single-generalized-graph
  arma::vec ones_w;
  double objective;
  arma::mat debugTrace;
  arma::mat hDebugTrace;
  arma::mat hNewtonDebugTrace;
  arma::mat hProjectionDebugTrace;

  void setTheta( const arma::vec& theta );

  // After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
  explicit isvdxAtTheta( const dsListC& Y,
               const imr& s,
               const double& f1,
               const unsigned int& K,
               const double& f2,
               const arma::vec& theta_init,
               const double* f3p,
               const int& maxIter,
               const double& ftol_abs,
               const double& xtol_abs,
               const int& inner_maxIter,
               const double& inner_ftol_abs

  );
  void estimate();
  Rcpp::List getFitAtOptCV();
};

#endif
