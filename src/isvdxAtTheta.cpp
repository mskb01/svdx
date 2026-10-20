
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

#include "isvdxAtTheta.h"

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
isvdxAtTheta::isvdxAtTheta( const dsListC& Y,
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

): Y(Y), columns1(s.columns1c), Z(s.Zc), Zeta(s.Zetac), S(s.Sc), SStq(s.SStqc), RsCbound(s.RsCboundc),
hmat(s.hmatc), umat(s.umatc),vmat(s.vmatc),
qids(s.qidsc), q(s.qc),
rowOffsets(s.rowOffsetsc), colOffsets(s.colOffsetsc),
graphDistMat(s.graphDistMatc), graphDistOnlyOnPoints(s.graphDistOnlyOnPointsc),
family( s.family ),
r(s.r),K(K),f1(f1), f2(f2), wd(s.wd),  theta(theta_init), maxIter(maxIter), ftol_abs(ftol_abs), xtol_abs(xtol_abs), inner_maxIter(inner_maxIter), inner_ftol_abs(inner_ftol_abs)
{

  /*
   * set K and their dependents, f1, f2, theta and their dependents.
   */

  //set K and its dependents
  if( K > this->umat.n_cols ){
    throw std::invalid_argument( "K to set is higher than the number of columns in initializer umat" );
  }
  this->K=K;
  this->umat=umat.cols(0,K-1);
  this->hmat=hmat.cols(0,K-1);
  for( int d=0; d < Y.size(); d++ ){
    this->vmat[d] = vmat[d].cols(0,K-1);
  }

  //set theta and its dependents
  this->nTheta = theta_init.n_elem;
  setTheta( theta );

  if( s.aphi != nullptr ){
    this->aphi = (*s.aphi);
  }
  if( graphDistMat != nullptr ){ //graph
    if( family != nullptr ){     // family
      this->dispatch = 3;
    } else { //no family
      this->dispatch = 1;
    }
  } else { //no graph
    if( family !=nullptr ){ //family
      this->dispatch=2;
    } else { //no family
      this->dispatch=0;
    }
  }

  if( f3p != nullptr ){
    f3 = *f3p;
    f3set = true;
  }

  if( this->family != nullptr ){
    Rcpp::Rcout << "AtTheta: Family option set. Dispatch mode: " << dispatch << std::endl;
  } else {
    Rcpp::Rcout << "AtTheta: No family set. Dispatch mode: " << dispatch << std::endl;
  }


};

void isvdxAtTheta::setTheta( const arma::vec& theta ){
  this->theta = theta;
  if( theta.n_elem != nTheta ){
    throw std::invalid_argument("theta vector does not have the expected length");
  }

  //prepare Mtheta=[S, Rtheta], RtildeTheta=Rtheta[,qids] for each fold
  int st=0, en= st+q-1;
  RTheta = theta[0]*RsCbound.cols( st, en );
  st=en+1;
  for( int t=1; t < nTheta ; t++ ){ //loop over rest of the terms
    en =st+q-1;
    RTheta += theta[ t ] * RsCbound.cols( st, en );
    st=en+1;
  }
  RtildeTheta = RTheta.rows( qids );
  BTheta=invpsd(SStq + RtildeTheta, 1e-7, false);  //the orth kernel

  //generate M
  MTheta = arma::join_horiz( S, RTheta );

}

void isvdxAtTheta::estimate() {

  //Copying for any future thread safety.
  std::vector<arma::mat> Zeta_w = Zeta;
  arma::mat hmat_w = hmat;
  arma::mat umat_w = umat;
  std::vector<arma::mat> vmat_w = vmat;
  arma::mat MTheta_w = MTheta;
  arma::mat RtildeTheta_w = RtildeTheta;
  arma::mat BTheta_w = BTheta;

  if( dispatch == 0 ){
    inpFitterOnColumns D( Y, columns1,
                          Z, Zeta_w,
                          MTheta_w, hmat_w, umat_w, vmat_w,
                          RtildeTheta_w,
                          qids,
                          BTheta_w,
                          rowOffsets,
                          colOffsets,
                          K,
                          r, 
                          wd//,
                          //Q
    );
    D.estimate(f1, f2, maxIter, ftol_abs, xtol_abs);
    this->objective = D.objective(f1, f2);
    this->hmat = D.hmat;
    this->umat = D.umat;
    this->vmat = D.vmat;
    this->Zeta = D.Zeta;
  } else if( dispatch == 1  ) {
    if (!f3set) {
      throw std::runtime_error("Graph regularization flag active but f3 parameter is not set.");
    }
    inpFitterOnColumnsGraph D( Y, columns1,
                               Z, Zeta_w,
                               MTheta_w, hmat_w, umat_w, vmat_w,
                               RtildeTheta_w,
                               *graphDistMat,
                               *graphDistOnlyOnPoints,
                               qids,
                               BTheta_w,
                               rowOffsets,
                               colOffsets,
                               K,
                               r, 
                               wd//,
                               //Q
    );
    D.estimate(f1, f2, f3, maxIter, ftol_abs, xtol_abs);
    this->objective = D.objective(f1, f2, f3);
    this->hmat = D.hmat;
    this->umat = D.umat;
    this->vmat = D.vmat;
    this->Zeta = D.Zeta;
  } else if( dispatch == 2 ){
    std::vector<arma::vec> aphi_w = this->aphi;
    igFitterOnColumns D( Y, columns1,
                         (*family) ,
                         Z, Zeta_w,
                         MTheta_w, hmat_w, umat_w, vmat_w,
                         aphi_w,
                         RtildeTheta_w,
                         qids,
                         BTheta_w,
                         rowOffsets,
                         colOffsets,
                         K,
                         r, 
                         wd//,
                         //Q
    );
    D.estimate(f1, f2, maxIter, ftol_abs, xtol_abs, inner_maxIter, inner_ftol_abs);
    this->objective = D.objective(true, true, f1, f2);
    this->hmat = D.hmat;
    this->umat = D.umat;
    this->vmat = D.vmat;
    this->Zeta = D.Zeta;
    this->aphi = D.aphi;
    this->debugTrace = D.getDebugTrace();
    this->hDebugTrace = D.getHDebugTrace();
    this->hNewtonDebugTrace = D.getHNewtonDebugTrace();
    this->hProjectionDebugTrace = D.getHProjectionDebugTrace();
  } else if( dispatch == 3 ) {
    std::vector<arma::vec> aphi_w = this->aphi;
    igFitterOnColumnsGraph D( Y, columns1,
                              (*family) ,
                              Z, Zeta_w,
                              MTheta_w, hmat_w, umat_w, vmat_w,
                              aphi_w,
                              RtildeTheta_w,
                              *graphDistMat,
                              *graphDistOnlyOnPoints,
                              qids,
                              BTheta_w,
                              rowOffsets,
                              colOffsets,
                              K,
                              r, 
                              wd//,
                              //Q
    );
    D.estimate(f1, f2, f3, maxIter, ftol_abs, xtol_abs, inner_maxIter, inner_ftol_abs);
    this->objective = D.objective(true, true, f1, f2, f3);
    this->hmat = D.hmat;
    this->umat = D.umat;
    this->vmat = D.vmat;
    this->Zeta = D.Zeta;
    this->aphi = D.aphi;
    this->debugTrace = D.getDebugTrace();
    this->hDebugTrace = D.getHDebugTrace();
    this->hNewtonDebugTrace = D.getHNewtonDebugTrace();
    this->hProjectionDebugTrace = D.getHProjectionDebugTrace();

  } else {
    throw std::invalid_argument("undefined dispatch specification. ");
  }

}

Rcpp::List isvdxAtTheta::getFitAtOptCV(){
  Rcpp::List result=Rcpp::List::create(
    Rcpp::Named("objective") = objective,
    Rcpp::Named("Z") = Rcpp::wrap(Z),
    Rcpp::Named("Zeta") = Rcpp::wrap(transposeVecOfMats(Zeta)), //change to the usual format
    Rcpp::Named("hmat") = hmat,
    Rcpp::Named("umat") = umat,
    Rcpp::Named("vmat") = Rcpp::wrap(vmat),
    Rcpp::Named("aphi") = NumericVector::get_na(),
    Rcpp::Named("family") = NumericVector::get_na(),
    Rcpp::Named("debugTrace") = debugTrace,
    Rcpp::Named("hDebugTrace") = hDebugTrace,
    Rcpp::Named("hNewtonDebugTrace") = hNewtonDebugTrace,
    Rcpp::Named("hProjectionDebugTrace") = hProjectionDebugTrace
  //Rcpp::Named("gmat") = svxAtopt.gmat,  //u = f(=Sd) + g(=Rc)
  );
  if( family != nullptr ){
    result[ "aphi" ] = Rcpp::wrap(this->aphi);
    result[ "family" ] = Rcpp::wrap((*family));
  } else {
    result[ "aphi" ] = NumericVector::get_na();
    result[ "family" ] = NumericVector::get_na();
  }
  return result;
}



