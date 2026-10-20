#ifndef SVDX_H
#define SVDX_H


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>
#include <memory>
#include "nlopt.hpp"

#include "SMLFCOnColsListTT.h"
#include "SMLFCOptWorker.h"
#include "svdxAtTheta.h"


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
template<typename T> //sp_mat or mat
class svdx{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data

private:
  const T& Y;
  const smttfr& ttf;
  const smr& s;
  std::unique_ptr< SMLFCOpt<T> > opt;
  std::unique_ptr< svdxAtTheta<T> > svxAtopt;

public:
  svdx(
    const T& Y,
    const smttfr& ttf,
    const smr& s
  ): Y(Y), ttf(ttf), s(s), opt(nullptr), svxAtopt(nullptr) {};

  ~svdx()=default;

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
  ){
    opt = std::make_unique<SMLFCOpt<T>>( Y,ttf,
                                         f1,
                                         optCV_maxIter,optCV_xtol_abs,optCV_ftol_abs,
                                         fitter_maxIter,fitter_ftol_abs,fitter_xtol_abs,fitter_inner_maxIter,fitter_inner_ftol_abs
                             );
    opt->optimize( KInit,
                   f2Init,
                   thetaInit,
                   f3Init,
                   optForK,
                   optForf2,
                   optForf3,
                   optFortheta,
                   nlopt_xtol_abs,
                   nlopt_ftol_abs,
                   nlopt_maxeval
                   );
  }

  void estimWithTheta(double& f1, unsigned int& K, double& f2, double& f3, const arma::vec& theta,
                      int fitter_maxIter=300,
                      double fitter_ftol_abs=.001,
                      double fitter_xtol_abs=.01,
                      int fitter_inner_maxIter=25,
                      double fitter_inner_ftol_abs=1e-4 ){

    //Rcpp::Rcout << "Now calling svxAtopt ... " << std::endl;
    svxAtopt=std::make_unique<svdxAtTheta<T>>( Y,
                                               s,
                                               f1,
                                               K,
                                               f2,
                                               theta,
                                               &f3,
                                               fitter_maxIter,
                                               fitter_ftol_abs,
                                               fitter_xtol_abs,
                                               fitter_inner_maxIter,
                                               fitter_inner_ftol_abs
    );
    //Rcpp::Rcout << "Getting to estimate ... " << std::endl;
    svxAtopt->estimate();

  }

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
      double fitter_inner_ftol_abs=1e-4){

    Rcpp::Rcout << " + +> Optimizing hyperparameters ... " << std::endl;
    this->optcv( f1,
                 K,
                 this->ttf.f2Init,
                 this->ttf.thetaInit,
                 this->ttf.f3Init_ptr,
                 optForK,
                 optForf2,
                 optForf3,
                 optFortheta ,
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
                 fitter_inner_ftol_abs
    );
    Rcpp::Rcout << " + +> Estimation at theta ... " << std::endl;
    this->estimWithTheta( opt->f1_w, opt->K_w, opt->f2_w, opt->f3_w, opt->theta_w, fitter_maxIter, fitter_ftol_abs, fitter_xtol_abs, fitter_inner_maxIter, fitter_inner_ftol_abs );
  }


  /*
   * Get
   */
  Rcpp::List getOptCV(){

    if (!svxAtopt) {
      throw std::runtime_error("Estimation at theta has not been executed yet.");
    }

    Rcpp::List L= Rcpp::List::create(
      Rcpp::Named("f1") = svxAtopt->f1 , //we store this for look up purposes. We don't optimize over it.
      Rcpp::Named("f2") = svxAtopt->f2 ,
      Rcpp::Named("theta") = svxAtopt->theta,
      Rcpp::Named("K") = svxAtopt->K
    );
    if( svxAtopt->f3set ){
      L.push_back( svxAtopt->f3, "f3" );
    }
    if( opt ){
      L.push_back(opt->getCVObjectiveTrace(), "cvObjectiveTrace");
    }
    return L;
  }
  Rcpp::List getFitAtOptCV(){
    return svxAtopt->getFitAtOptCV();
  }



};

template class svdx<arma::mat>;
template class svdx<arma::sp_mat>;




#endif
