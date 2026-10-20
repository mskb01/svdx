
//KEY FUNCTION USED FOR CV OF NPFITTER!



// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include "isvdx.h"

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

isvdx::isvdx(
  const dsListC& Y,
  const imttfr& ttf,
  const imr& s
): Y(Y), ttf(ttf), s(s), opt(nullptr), isvxAtopt(nullptr) {};

void isvdx::optcv(
    double f1,
    unsigned int KInit,//initial conditions
    double f2Init,
    const arma::vec& thetaInit,
    double* f3Init,

    bool optForK, //optional opts to switch on / off
    bool optForf2,
    bool optForf3,
    bool optFortheta,

    int optCV_maxIter,
    double optCV_xtol_abs,
    double optCV_ftol_abs,
    double nlopt_xtol_abs,
    double nlopt_ftol_abs,
    int nlopt_maxeval,
    int fitter_maxIter,
    double fitter_ftol_abs,
    double fitter_xtol_abs,
    int fitter_inner_maxIter,
    double fitter_inner_ftol_abs
){
  opt = std::make_unique<IMLFCOpt>( Y,ttf,
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

void isvdx::estimWithTheta(double& f1, unsigned int& K, double& f2, double& f3, const arma::vec& theta,
                           int fitter_maxIter,
                           double fitter_ftol_abs,
                           double fitter_xtol_abs,
                           int fitter_inner_maxIter,
                           double fitter_inner_ftol_abs){

  //Rcpp::Rcout << "Now calling svxAtopt ... " << std::endl;
  isvxAtopt=std::make_unique<isvdxAtTheta>( Y,
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
  isvxAtopt->estimate();

}

void isvdx::estimate(
    double f1,
    unsigned int K,
    bool optForK,
    bool optForf2,
    bool optForf3,
    bool optFortheta,
    int optCV_maxIter,
    double optCV_xtol_abs,
    double optCV_ftol_abs,
    double nlopt_xtol_abs,
    double nlopt_ftol_abs,
    int nlopt_maxeval,
    int fitter_maxIter,
    double fitter_ftol_abs,
    double fitter_xtol_abs,
    int fitter_inner_maxIter,
    double fitter_inner_ftol_abs){

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
Rcpp::List isvdx::getOptCV(){

  if (!isvxAtopt) {
    throw std::runtime_error("Estimation at theta has not been executed yet.");
  }

  Rcpp::List L= Rcpp::List::create(
    Rcpp::Named("f1") = isvxAtopt->f1 , //we store this for look up purposes. We don't optimize over it.
    Rcpp::Named("f2") = isvxAtopt->f2 ,
    Rcpp::Named("theta") = isvxAtopt->theta,
    Rcpp::Named("K") = isvxAtopt->K
  );
  if( isvxAtopt->f3set ){
    L.push_back( isvxAtopt->f3, "f3" );
  }
  if( opt ){
    L.push_back(opt->getCVObjectiveTrace(), "cvObjectiveTrace");
  }
  return L;
}
Rcpp::List isvdx::getFitAtOptCV(){
  return isvxAtopt->getFitAtOptCV();
}









