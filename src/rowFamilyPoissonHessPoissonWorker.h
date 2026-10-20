#ifndef ROWFAMILYPOISSONHESSPOISSONWORKER_H
#define ROWFAMILYPOISSONHESSPOISSONWORKER_H


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <tbb/tbb.h>

using namespace arma;
using namespace Rcpp;
using namespace RcppParallel;


/*
 * This computes for each column j, sum_i v_ki^2 e^{eta_ij} m_j,
 * where m_j is the j^th row of M.
 * eta_ij the link
 * v_ki loading for ith feature in kth component
 *
 * The result is a vector.
 */
class hessPoissonForColj : public Worker {

public:
  //inputs
  static constexpr double clampUpperThresh=1e15;
  static constexpr double clampLowerThresh=-1e15;
  static constexpr double eta_clampUpperThresh=20;
  static constexpr double eta_clampLowerThresh=-20;
  const arma::mat& M;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umatOthers;
  const arma::mat& vmatOthers;
  const arma::vec& aphi;
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
  const arma::vec& h_k;
  const arma::vec& v_k;

  // const arma::mat& M, Z, Zeta, umatOthers, vmatOthers;
  // const arma::vec& aphi, rowOffsets, colOffsets, h_k, v_k;
  const int j;
  arma::vec mj;
  arma::vec otherEtaContrib_j; //px1 vec corresp to col j
  arma::vec eta_j; //adding the k^th components contribution
  arma::vec exp_eta_j;
  arma::vec v_k2;

  //output
  arma::vec value;

  //constructurors
  hessPoissonForColj( const arma::mat& M,
                      const arma::mat& Z,
                      const arma::mat& Zeta,
                      const arma::mat& umatOthers, //columns other than the k^th
                      const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
                      const arma::vec& aphi,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const arma::vec& h_k,      //the h that will determine u_k = M h_k
                      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
                      const int& j ): M(M), Z(Z), Zeta(Zeta),
                      umatOthers(umatOthers), vmatOthers(vmatOthers),
                      aphi(aphi), rowOffsets(rowOffsets), colOffsets(colOffsets),
                      h_k(h_k), v_k(v_k), j(j){
    mj = M.row(j).t();
    otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*umatOthers.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
    eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
    exp_eta_j = arma::exp( arma::clamp(eta_j, eta_clampLowerThresh, eta_clampUpperThresh) );
    v_k2 = arma::pow(v_k,2);
    this->value = arma::vec( M.n_cols, arma::fill::zeros ) ;
  };

  hessPoissonForColj(const hessPoissonForColj& hpcj, Split):M(hpcj.M), Z(hpcj.Z), Zeta(hpcj.Zeta),
  umatOthers(hpcj.umatOthers), vmatOthers(hpcj.vmatOthers),aphi(hpcj.aphi),
  rowOffsets(hpcj.rowOffsets), colOffsets(hpcj.colOffsets),h_k(hpcj.h_k), v_k(hpcj.v_k), j(hpcj.j){
    this->value=arma::vec( M.n_cols, arma::fill::zeros );
    mj = hpcj.mj;
    otherEtaContrib_j=hpcj.otherEtaContrib_j;
    eta_j=hpcj.eta_j;
    exp_eta_j = hpcj.exp_eta_j;
    v_k2 = hpcj.v_k2;
  };

  void join(const hessPoissonForColj& rhs);// join my value with that of another Sum
  void operator()(std::size_t begin, std::size_t end);


};


/*
 * This adds the output of the above hessPoissonForCol_j
 * for all samples j, after post multiplying by M.t()
 * The result is a matrix.
 */
class hessPoissonAllCols : public Worker {

public:
  //inputs
  const arma::mat& M;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umatOthers;
  const arma::mat& vmatOthers;
  const arma::vec& aphi;
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
  const arma::vec& h_k;
  const arma::vec& v_k;
  int nftrs;
  const bool innerHessWorkerSerial; 


  //output
  arma::mat value;

  //constructurors
  hessPoissonAllCols( const arma::mat& M,
                      const arma::mat& Z,
                      const arma::mat& Zeta,
                      const arma::mat& umatOthers, //columns other than the k^th
                      const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
                      const arma::vec& aphi,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const arma::vec& h_k,      //the h that will determine u_k = M h_k
                      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
                      const bool& innerHessWorkerSerial
                      ): M(M), Z(Z), Zeta(Zeta),
                      umatOthers(umatOthers), vmatOthers(vmatOthers),
                      aphi(aphi), rowOffsets(rowOffsets), colOffsets(colOffsets),
                      h_k(h_k), v_k(v_k), innerHessWorkerSerial(innerHessWorkerSerial){
    nftrs = v_k.size();
    this->value = arma::mat( M.n_cols, M.n_cols, arma::fill::zeros ) ;
  };

  hessPoissonAllCols(const hessPoissonAllCols& hpcac, Split):M(hpcac.M), Z(hpcac.Z), Zeta(hpcac.Zeta),
  umatOthers(hpcac.umatOthers), vmatOthers(hpcac.vmatOthers),aphi(hpcac.aphi),
  rowOffsets(hpcac.rowOffsets), colOffsets(hpcac.colOffsets),h_k(hpcac.h_k), v_k(hpcac.v_k), innerHessWorkerSerial(hpcac.innerHessWorkerSerial){
    nftrs = v_k.size();
    this->value=arma::mat( M.n_cols, M.n_cols, arma::fill::zeros );
  };

  // join my value with that of another Sum
  void join(const hessPoissonAllCols& rhs);

  void operator()(std::size_t begin, std::size_t end);

};


/*
 *  We do the same but as above without
 *  umatOthers and vmatOthers, for the purpose of
 *  K = 1
 *
 */
class hessPoissonForColjK1 : public Worker {

public:
  //inputs
  static constexpr double clampUpperThresh=1e15;
  static constexpr double clampLowerThresh=-1e15;
  static constexpr double eta_clampUpperThresh=20;
  static constexpr double eta_clampLowerThresh=-20;
  const arma::mat& M;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::vec& aphi;
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
  const arma::vec& h_k;
  const arma::vec& v_k;

  const int j;
  arma::vec mj;
  arma::vec otherEtaContrib_j; //px1 vec corresp to col j
  arma::vec eta_j; //adding the k^th components contribution
  arma::vec exp_eta_j;
  arma::vec v_k2;

  //output
  arma::vec value;

  //constructurors
  hessPoissonForColjK1( const arma::mat& M,
                      const arma::mat& Z,
                      const arma::mat& Zeta,
                      const arma::vec& aphi,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const arma::vec& h_k,      //the h that will determine u_k = M h_k
                      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
                      const int& j ): M(M), Z(Z), Zeta(Zeta),
                      aphi(aphi), rowOffsets(rowOffsets), colOffsets(colOffsets),
                      h_k(h_k), v_k(v_k), j(j){
    mj = M.row(j).t();
    otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) +  (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
    eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
    exp_eta_j = arma::exp( arma::clamp(eta_j, eta_clampLowerThresh, eta_clampUpperThresh) );
    v_k2 = arma::pow(v_k,2);
    this->value = arma::vec( M.n_cols, arma::fill::zeros ) ;
  };

  hessPoissonForColjK1(const hessPoissonForColjK1& hpcj, Split):M(hpcj.M), Z(hpcj.Z), Zeta(hpcj.Zeta),
  aphi(hpcj.aphi),
  rowOffsets(hpcj.rowOffsets), colOffsets(hpcj.colOffsets),h_k(hpcj.h_k), v_k(hpcj.v_k), j(hpcj.j){
    this->value=arma::vec( M.n_cols, arma::fill::zeros );
    mj = hpcj.mj;
    otherEtaContrib_j=hpcj.otherEtaContrib_j;
    eta_j=hpcj.eta_j;
    exp_eta_j = hpcj.exp_eta_j;
    v_k2 = hpcj.v_k2;
  };

  void join(const hessPoissonForColjK1& rhs);// join my value with that of another Sum
  void operator()(std::size_t begin, std::size_t end);


};


/*
 * This adds the output of the above hessPoissonForCol_j
 * for all samples j, after post multiplying by M.t()
 * The result is a matrix.
 */
class hessPoissonAllColsK1 : public Worker {

public:
  //inputs
  const arma::mat& M;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::vec& aphi;
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
  const arma::vec& h_k;
  const arma::vec& v_k;
  int nftrs;
  const bool innerHessWorkerSerial;


  //output
  arma::mat value;

  //constructurors
  hessPoissonAllColsK1( const arma::mat& M,
                      const arma::mat& Z,
                      const arma::mat& Zeta,
                      const arma::vec& aphi,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const arma::vec& h_k,      //the h that will determine u_k = M h_k
                      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
                      const bool& innerHessWorkerSerial
  ): M(M), Z(Z), Zeta(Zeta),
  aphi(aphi), rowOffsets(rowOffsets), colOffsets(colOffsets),
  h_k(h_k), v_k(v_k),innerHessWorkerSerial(innerHessWorkerSerial){
    nftrs = v_k.size();
    this->value = arma::mat( M.n_cols, M.n_cols, arma::fill::zeros ) ;
  };

  hessPoissonAllColsK1(const hessPoissonAllColsK1& hpcac, Split):M(hpcac.M), Z(hpcac.Z), Zeta(hpcac.Zeta),
  aphi(hpcac.aphi),
  rowOffsets(hpcac.rowOffsets), colOffsets(hpcac.colOffsets),h_k(hpcac.h_k), v_k(hpcac.v_k),
  innerHessWorkerSerial(hpcac.innerHessWorkerSerial){
    nftrs = v_k.size();
    this->value=arma::mat( M.n_cols, M.n_cols, arma::fill::zeros );
  };

  // join my value with that of another Sum
  void join(const hessPoissonAllColsK1& rhs);

  void operator()(std::size_t begin, std::size_t end);

};




#endif
