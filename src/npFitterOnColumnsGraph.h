#ifndef NPFITTERONCOLUMNSGRAPH_H
#define NPFITTERONCOLUMNSGRAPH_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

#include<unordered_map>
#include<list>
#include<vector>
#include<array>
#include<random>
#include<string>
#include<cmath>

#include "myutils.h"
#include "npFitterOnColumns.h"
#include "parallelWokersTemplates.h"

template<typename T>
class npFitterOnColumnsGraph final: public npFitterOnColumns<T> { //same as npFitter but operates on columns specified, w/o a copy. NOTE: ALl other input variables (e.g., qids) must have been subsetted. Only Y[,cols] is passed as Y, cols

  public:

    const arma::sp_mat& graphDistMat; //qxq graph dist matrix, must be pos semi def/pos def
    const arma::uvec& graphDistOnlyOnPoints;    //the graph distances are defined for these points, in the order

    using npFitterOnColumns<T>::Y;
    using npFitterOnColumns<T>::columns; //these are the only columns that will be used for fitting the data. Useful for CV, instead of making copies of Y!
    using npFitterOnColumns<T>::Z;       //Z may or may not be present; can be a column vector of 1 simply
    using npFitterOnColumns<T>::M;       //M=[S; R] , S is nxr R is nxq entering in Sd + Rc = Mh
    using npFitterOnColumns<T>::qids;    //let's not use a refernece here, we will be subbin 1 too, don't want to the R part to reflect that!!!!
    using npFitterOnColumns<T>::Rtilde; //qxq penalty = Rtheta[qids,]

    using npFitterOnColumns<T>::B;      //qxq full kernel inverse
    using npFitterOnColumns<T>::Q;       //any space to be orthogonal to; this is not a reference as declaring it so will require initialization in the initialiezr list, and we have situations where we don't have a Q (we will be overloading the constructor for this one).
    using npFitterOnColumns<T>::rowOffsets;
    using npFitterOnColumns<T>::colOffsets;
    using npFitterOnColumns<T>::colOffsets_for_columns; //stores colOffests corresponding to desired columns

    using npFitterOnColumns<T>::K;
    using npFitterOnColumns<T>::r;
    using npFitterOnColumns<T>::q;
    using npFitterOnColumns<T>::n;
    using npFitterOnColumns<T>::p;


    //these two get updated based on
    //  whatever last f1, f2 pair the
    //  estimate function was called with.
    // This is useful to know because the umat, vmat, hmat etc.,
    // only make sense with these hyperparameter settings.
    using npFitterOnColumns<T>::last_f1_estimatedWith;
    using npFitterOnColumns<T>::last_f2_estimatedWith;
    double last_f3_estimatedWith;


    using npFitterOnColumns<T>::Zeta;
    using npFitterOnColumns<T>::hmat;    //umat = M hmat
    using npFitterOnColumns<T>::umat;
    using npFitterOnColumns<T>::vmat;

    using npFitterOnColumns<T>::A; //holds M'M
    arma::mat G; //holds Mq'graphDistMat Mq
    using npFitterOnColumns<T>::Aex; //expanded form of A matrix that incorporates +f2 c'Rc in the right indices, and the orthogonal constraints.
    using npFitterOnColumns<T>::Aex_pinv;
    using npFitterOnColumns<T>::Aex_pinv_rPlusq_submat;
    using npFitterOnColumns<T>::Zt;
    using npFitterOnColumns<T>::Mt;
    using npFitterOnColumns<T>::ZtZ;
    using npFitterOnColumns<T>::Zls;
    using npFitterOnColumns<T>::YZlst;
    using npFitterOnColumns<T>::w_hat;
    using npFitterOnColumns<T>::seqOfComponents;
    using npFitterOnColumns<T>::qindx;
    using npFitterOnColumns<T>::rqindx;

    //whether to fit zeta
    // conditioning useful for struc discover
    using npFitterOnColumns<T>::fitZeta;

    //lf if needed
    using npFitterOnColumns<T>::lf_swise;

    //some ones for offsets
    using npFitterOnColumns<T>::ones_p;
    using npFitterOnColumns<T>::ones_n; //ones of length roffset, coloffsets respectively

    //usefulf or estimation
    using npFitterOnColumns<T>::move_Zeta;             //this using brings both overloaded move_Zeta for k=0, and k>0 components
    using npFitterOnColumns<T>::move_v_k;
    using npFitterOnColumns<T>::populate_w_hat_for_k; //for k = 0
    using npFitterOnColumns<T>::append_to_Aex_and_update_Aexpinv;
    using npFitterOnColumns<T>::estimate_k0; //for k=0, the first component
    using npFitterOnColumns<T>::estimate_k; //for k>0
    using npFitterOnColumns<T>::move_h_k;

    //Constructors
    // note, we make copies of Zeta, hmat, umat, vmat
    // when offsets get implemented, they need to be too.
    // these copies, instead of references, allows us to not interfere with multithreaded wrappers built
    //. on top of this.
    explicit npFitterOnColumnsGraph(const T& Y,   //without Q, orthogonal space to avoid
                      const arma::uvec& columns1,
                      const arma::mat& Z, arma::mat Zeta,
                      const arma::mat& M,arma::mat hmat,
                      arma::mat umat, arma::mat vmat, //it is very important these umat, vmat, hmat, Zeta etc., are all copies, as we will be writing to these! Otherwise what is passed from R gets messed up, and very hard to track why we get different results when rerunning code!
                      const arma::mat& Rtilde,
                      const arma::sp_mat& graphDistMat,
                      const arma::uvec& graphDistOnlyOnPoints,
                      const arma::uvec& qids1,
                      const arma::mat& B,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const int K,
                      const int r,
                      const bool fitZeta=true
    );

    explicit npFitterOnColumnsGraph(const T& Y, //with Q,
                      const arma::uvec& columns1,
                      const arma::mat& Z,arma::mat Zeta,
                      const arma::mat& M,arma::mat hmat,
                      arma::mat umat, arma::mat vmat,
                      const arma::mat& Rtilde,
                      const arma::sp_mat& graphDistMat,
                      const arma::uvec& graphDistOnlyOnPoints,
                      const arma::uvec& qids1,
                      const arma::mat& B,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const int K,
                      const int r,
                      const arma::mat& Q,
                      const bool fitZeta=true
               );

    //Estimation utilities
    using npFitterOnColumns<T>::estimateHelper;
    void estimate(double f1, double f2, double f3, double maxIter=25, double eps=1e-4, double xtol_abs=.01);
    void estimate(const estimatePar& ep );

    //objectives
    double getPenalty();
    double getPenaltyForGraph();
    double objective(double f1, double f2, double f3);
    using npFitterOnColumns<T>::mseForTest;
    double objectiveForTest(const arma::uvec& columns1Test,
                      const arma::mat& ZTest,
                      const arma::mat& MTest,
                      const arma::vec& rowOffsetsTest,
                      const arma::vec& colOffsetsTest
    );

    //outputs, althoguh everyhing is public.
    using npFitterOnColumns<T>::get_Zeta;
    using npFitterOnColumns<T>::get_hmat;
    using npFitterOnColumns<T>::get_umat;
    using npFitterOnColumns<T>::get_vmat;
    using npFitterOnColumns<T>::setAexToAandLock;

    ~npFitterOnColumnsGraph()=default;

};

template class npFitterOnColumnsGraph<arma::mat>;
template class npFitterOnColumnsGraph<arma::sp_mat>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * CONSTRUCTORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
 * With Q
 */
template<typename T>
npFitterOnColumnsGraph<T>::npFitterOnColumnsGraph(
  const T& Y,
  const arma::uvec& columns1, //NOTE!! we assume columns -> columns-1 has already been made before. Similarly, for qids below!
  const arma::mat& Z,       //Z may or may not be present
  arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  const arma::mat& Rtilde, // qxq penalty
  const arma::sp_mat& graphDistMat,
  const arma::uvec& graphDistOnlyOnPoints,
  const arma::uvec& qids1,//NOTE!! we assume qids -> qids-1 has already been made before. Similarly, for columns1 above.
  const arma::mat& B,      //qxq full kernel inverse
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const int K,
  const int r,
  const arma::mat& Q,       //nxq
  const bool fitZeta
): graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
npFitterOnColumns<T>( Y,
                      columns1, //NOTE!! we assume columns -> columns-1 has already been made before. Similarly, for qids below!
                      Z,       //Z may or may not be present
                      Zeta,
                      M,
                      hmat,umat, vmat,
                      Rtilde, // qxq penalty
                      qids1,//NOTE!! we assume qids -> qids-1 has already been made before. Similarly, for columns1 above.
                      B,      //qxq full kernel inverse
                      rowOffsets,
                      colOffsets,
                      K,
                      r,
                      Q,       //nxq
                      fitZeta )
{

  this->G = Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
  
}


/*
 * Without Q
 */
template<typename T>
npFitterOnColumnsGraph<T>::npFitterOnColumnsGraph(
  const T& Y,
  const arma::uvec& columns1,
  const arma::mat& Z,       //Z may or may not be present
  arma::mat Zeta,
  const arma::mat& M,arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  const arma::mat& Rtilde, // qxq penalty
  const arma::sp_mat& graphDistMat,
  const arma::uvec& graphDistOnlyOnPoints,
  const arma::uvec& qids1,
  const arma::mat& B,      //qxq full kernel inverse
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const int K,
  const int r,
  const bool fitZeta
):  graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
npFitterOnColumns<T>( Y,
                      columns1, //NOTE!! we assume columns -> columns-1 has already been made before. Similarly, for qids below!
                      Z,       //Z may or may not be present
                      Zeta,
                      M,
                      hmat,umat, vmat,
                      Rtilde, // qxq penalty
                      qids1,//NOTE!! we assume qids -> qids-1 has already been made before. Similarly, for columns1 above.
                      B,      //qxq full kernel inverse
                      rowOffsets,
                      colOffsets,
                      K,
                      r,
                      fitZeta ) {



  this->G = Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
  
}

template<typename T>
void npFitterOnColumnsGraph<T>::estimate(double f1, double f2, double f3, double maxIter, double eps, double xtol_abs){
  (void)xtol_abs; // reserved for future parameter-change convergence

  if( Y.n_rows==0 ){
    throw std::runtime_error("Y has not been assigned. Has the right constructor been called? ");
  }

  this->setAexToAandLock(); ////ressets Aex = A , and locks a derived class inaccessible private boolean in base class, so that we are able to call base class's estimateHelper.
  Aex += 2*(f3/f1)*n*G; //this is the only addition compared to base npFitterOnColumns<T> estimate()
  this->estimateHelper(f1, f2, maxIter,eps); //now we are ready to call estimateHelper.
  last_f3_estimatedWith = f3;

}
template<typename T>
void npFitterOnColumnsGraph<T>::estimate(const estimatePar& ep){
  this->estimate( ep.f1, ep.f2, ep.f3, ep.maxIter, ep.eps, .01 );
}

// USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
template<typename T>
double npFitterOnColumnsGraph<T>::getPenalty(){

  double penalty = 0;
  arma::vec hk(hmat.n_rows);
  arma::vec hk_q(r+q-1);
  for( int k=0; k<K; k++ ){
    hk = hmat.col(k);
    hk_q=hk.elem( qindx );
    penalty += arma::dot( hk_q, Rtilde * hk_q  );
  }
  return penalty;
}


template<typename T>
double npFitterOnColumnsGraph<T>::getPenaltyForGraph(){

  double penalty = 0;
  arma::vec hk(hmat.n_rows);
  for( int k=0; k<K; k++ ){
    hk = hmat.col(k);
    penalty += arma::dot( hk, G * hk  );
  }
  return penalty;
}


// USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
template<typename T>
double npFitterOnColumnsGraph<T>::objective(double f1, double f2, double f3){
  npSMLFCOnColumns<T> LFC( Y, columns, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  double lf = arma::mean( lf_swise );
  double penalty = this->getPenalty();
  double gpenalty = this->getPenaltyForGraph();
  return (f1*lf)+(f2*penalty)+(f3*gpenalty);
}

template<typename T>
double npFitterOnColumnsGraph<T>::objectiveForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  double lf = this->mseForTest( columns1Test, ZTest, MTest,
                                rowOffsetsTest, colOffsetsTest); //mse
  double penalty = this->getPenalty();
  double gpenalty = this->getPenaltyForGraph();
  double obj= last_f1_estimatedWith* lf + last_f2_estimatedWith* penalty + last_f3_estimatedWith*gpenalty;
  return obj;
}

#endif
