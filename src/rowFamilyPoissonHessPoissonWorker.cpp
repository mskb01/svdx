
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include "rowFamilyPoissonHessPoissonWorker.h"

using namespace arma;
using namespace Rcpp;
using namespace RcppParallel;


/*
 * HESS POISSON FOR COLUMN J
 */
// join my value with that of another Sum
void hessPoissonForColj::join(const hessPoissonForColj& rhs) {
  value += rhs.value;
}

void hessPoissonForColj::operator()(std::size_t begin, std::size_t end){
  for( std::size_t i=begin; i!=end; i++  ){ //going over features
    value += v_k2(i)*exp_eta_j(i)*mj;
  }
}

/*
 * HESS POISSON FOR COLUMN J
 */
// join my value with that of another Sum
void hessPoissonForColjK1::join(const hessPoissonForColjK1& rhs) {
  value += rhs.value;
}

void hessPoissonForColjK1::operator()(std::size_t begin, std::size_t end){
  for( std::size_t i=begin; i!=end; i++  ){ //going over features
    value += v_k2(i)*exp_eta_j(i)*mj;
  }
}


/*
 * HESS POISSON ALL COLS
 * i.e., all cols represented in M.
 * the model construction of M appropriately reflects the design
 * of the samples used for analysis.
 */
// join my value with that of another Sum
void hessPoissonAllCols::join(const hessPoissonAllCols& rhs) {
  value += rhs.value;
}

void hessPoissonAllCols::operator()(std::size_t begin, std::size_t end){
  for( std::size_t j=begin; j!=end; j++  ){ //going over columns/samples
    arma::rowvec mjr = M.row(j);
    hessPoissonForColj hpcj( M,
                             Z,
                             Zeta,
                             umatOthers, //columns other than the k^th
                             vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
                             aphi,
                             rowOffsets,
                             colOffsets,
                             h_k,      //the h that will determine u_k = M h_k
                             v_k,      // the corresponding feature loadings vector already fixed
                             j );

    if(this->innerHessWorkerSerial){
      hpcj( 0, nftrs);
    } else {
      parallelReduce( 0, nftrs, hpcj  );
    }
    
    value += (hpcj.value * mjr);
  }
}

//for K1
void hessPoissonAllColsK1::join(const hessPoissonAllColsK1& rhs) {
  value += rhs.value;
}

void hessPoissonAllColsK1::operator()(std::size_t begin, std::size_t end){
  for( std::size_t j=begin; j!=end; j++  ){ //going over columns/samples
    arma::rowvec mjr = M.row(j);
    hessPoissonForColjK1 hpcj( M,
                             Z,
                             Zeta,
                             aphi,
                             rowOffsets,
                             colOffsets,
                             h_k,      //the h that will determine u_k = M h_k
                             v_k,      // the corresponding feature loadings vector already fixed
                             j );
    
    if(this->innerHessWorkerSerial){
      hpcj( 0, nftrs);
    } else {
      parallelReduce( 0, nftrs, hpcj  );
    }
    
    value += (hpcj.value * mjr);
  }
}


