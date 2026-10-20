#ifndef ROWFAMILY_H
#define ROWFAMILY_H


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>

#include "rowFamilyPoissonHessPoissonWorker.h"

using namespace Rcpp;
using namespace RcppParallel;
//using namespace arma;


#include<string>
#include<cmath>

/**
 * Defining a row family for a feature x n matrix.
 * The implementation allows a single family for all rws in a matrix.
 * Can be straightforwardly extended to incorporate multiple families for rows of a data matrix.
 *
 *
 * Exp family: g(y_ij | eta_ij) = g_ij = exp{ (y_ij eta_ij - b(eta_ij))/a(phi_i)  + c(y_ij, phi_i) }
 * The log likelihood for the observation: log g_ij = (y_ij eta_ij - b(eta_ij))/a(phi_i)  + c(y_ij, phi_i) )
 * In the "unweighted" situations, we consider the full data log likelihood: L=sum_ij log g_ij
 * In all the "weighted" situations, we consider the weighted log likelihood : Lw= sum_ij w_ij log g_ij *
 * For implementation, we consider w_ij = roweights_i * colweights_j
 *
 */
template <typename T>
class rowFamily { // Models a row
  /*
   * Acronyms
   *
   * ULL : unnorm. log likelihood
   * Pot : Potential
   * Rem : Removal of an item
   * Add : Addition of an item
   * TS  : Thread safe
   * indx : a unique sequence's id.
   */


public:

  virtual ~rowFamily()=default;



  /*
   * Log liklihood calculations. For dense and sparse matrices.
   * For sparse matrices, multi-threading is done to accumulate
   * log-likelihood computations over columns (as column traversal
   * is more efficient than row-wise traversal in armadillo). Row-
   * wise log-likelihood computations are however needed for Zeta_i
   * estimation. We therefore write functions for calculations of
   * log-likelihood over rows, and over columns, for sparse and dense
   * matrices.
   *
   * For our use cases, we are only considering row and column offsets,
   * and similarly, row and column weights, and never full offsetMatrix and
   * full weights matrix. We will construct them if needed later.
   *
   * Calculation of full data loglikelihood is useful for Daarem / fpiter
   * speedups.
   *
   */


  ////////////////////////////////////
  /*
   *  LOGLIKELIHOODS
   */
  ////////////////////////////////////
  virtual double calcNegLogLL_i( const arma::vec& y,
                         const arma::vec& eta,
                         const double& aphi
  )=0;
  virtual double calcNegLogLL_j( const arma::vec& y,
                                 const arma::vec& eta,
                                 const arma::vec& aphi
  )=0;
  virtual double calcNegLogLLForCols( const T& Y,
                               const arma::uvec columns1,
                               const arma::mat& Zmat,
                               const arma::mat& Zeta,
                               const arma::mat& umat,
                               const arma::mat& vmat,
                               const arma::vec& aphi,
                               const arma::vec& rowOffsets,
                               const arma::vec& colOffsets
  )=0;

  ////////////////////////////////////
  /*
   *  ZETA:
   *  Zeta_i matrix in T= Z Zeta_i
   */
  ////////////////////////////////////
  virtual arma::vec gradNegLogLL_Zeta_i( const arma::vec& y_i, //nx1
                                         const arma::vec& eta_i, //nx1
                                         const arma::mat& Z, // n x B
                                         const arma::vec& zeta_i,  // B x p
                                         const double& aphi_i)=0;

  //hessian
  virtual arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                         const arma::vec& eta_i,
                                         const arma::mat& Z,
                                         const arma::vec& zeta_i,
                                         const double aphi_i)=0;
  virtual arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                         const arma::vec& eta_i,
                                         const arma::mat& Z,
                                         const arma::vec& zeta_i,
                                         const double aphi,
                                         const arma::mat& ZtZ
  )=0;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  v_k: in vmat=[v1, ... vK].
   *  v_ik i.e., for the ith index of v in v=[v0,v1, ... v(p-1) ] of p features
   */
  ////////////////////////////////////////////////////////////////////////
  virtual double gradNegLogLL_v_ik(
      const arma::vec& y_i,
      const arma::vec& eta_i,
      const arma::vec& u_k, // for a single vector u, entering as uv'
      const double v_ik,  // for a single vector v entering as uv'
      const double& aphi_i
  ) =0 ;

  virtual double hessNegLogLL_v_ik( const arma::vec& y_i,
                                    const arma::vec& eta_i,
                                    const arma::vec& u_k, // for a single component
                                    const double v_ik,  // for the i^th feature in k^th component
                                    const double& aphi_i
  ) = 0;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  h_k: in hmat=[h1, ... hK].
   *  U=Mh
   */
  ////////////////////////////////////////////////////////////////////////

  // this is when K=1, when there are no umatOthers and vmatOthers to adjust for
  virtual double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k, // the feature loading vector corresponding to the k^th component
      const int j
  ) = 0;

  // this is when K>1, with umatOthers and vmatOthers to adjust for
  virtual double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k, // the feature loading vector corresponding to the k^th component
      const int j
  ) = 0;

  // this is for K = 1, w/o umatOthers and vmatOthers
  virtual double cs_j_gradNegLogLL_h_k( //same as above, but only on columns of Y with Z, M, umatOthers, vmatOthers already correctly reflcting the columns
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k, // the feature loading vector corresponding to the k^th component
      const int j
  ) = 0;

  virtual double cs_j_gradNegLogLL_h_k( //same as above, but only on columns of Y with Z, M, umatOthers, vmatOthers already correctly reflcting the columns
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k, // the feature loading vector corresponding to the k^th component
      const int j
  ) = 0;

  // for K=1, w/o umatOthers and vmatOthers
  virtual void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) = 0;

  virtual void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) = 0;

  // for K = 1, withou umatOthers and vmatOthers
  virtual void gradNegLogLL_h_k_onColsOf( //same as above but only based on columns1 of Y
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) = 0;
  //for K >1
  virtual void gradNegLogLL_h_k_onColsOf( //same as above but only based on columns1 of Y
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) = 0;

  //for K = 1
  virtual void hessNegLogLL_h_k( const T& Y,
                                 const arma::mat& M,
                                 const arma::mat& Z,
                                 const arma::mat& Zeta,
                                 const arma::vec& aphi,
                                 const arma::vec& rowOffsets,
                                 const arma::vec& colOffsets,
                                 const arma::mat& MtM,
                                 const arma::vec& h_k,
                                 const arma::vec& v_k,
                                 arma::mat& hess_h_k

  )=0;
  virtual void hessNegLogLL_h_k( const T& Y,
                                 const arma::mat& M,
                                 const arma::mat& Z,
                                 const arma::mat& Zeta,
                                 const arma::mat& umatOthers,
                                 const arma::mat& vmatOthers,
                                 const arma::vec& aphi,
                                 const arma::vec& rowOffsets,
                                 const arma::vec& colOffsets,
                                 const arma::mat& MtM,
                                 const arma::vec& h_k,
                                 const arma::vec& v_k,
                                 arma::mat& hess_h_k

  )=0;
  //for K = 1
  virtual void hessNegLogLL_h_k_onColsOf( const T& Y, //same as above but only based on columns1 of Y
                                          const arma::uvec& columns1,
                                          const arma::mat& M,
                                          const arma::mat& Z,
                                          const arma::mat& Zeta,
                                          const arma::vec& aphi,
                                          const arma::vec& rowOffsets,
                                          const arma::vec& colOffsets,
                                          const arma::mat& MtM,
                                          const arma::vec& h_k,
                                          const arma::vec& v_k,
                                          arma::mat& hess_h_k

  )=0;
  virtual void hessNegLogLL_h_k_onColsOf( const T& Y, //same as above but only based on columns1 of Y
                                          const arma::uvec& columns1,
                                          const arma::mat& M,
                                          const arma::mat& Z,
                                          const arma::mat& Zeta,
                                          const arma::mat& umatOthers,
                                          const arma::mat& vmatOthers,
                                          const arma::vec& aphi,
                                          const arma::vec& rowOffsets,
                                          const arma::vec& colOffsets,
                                          const arma::mat& MtM,
                                          const arma::vec& h_k,
                                          const arma::vec& v_k,
                                          arma::mat& hess_h_k

  )=0;

  //////////////////////////////////////////////////////////////
  /*
   * Aphi updates
   */
  //////////////////////////////////////////////////////////////

  virtual double calc_aphi(const arma::vec& y_i, const arma::vec& eta_i)=0;



  //////////////////////////////////////////////////////////////
  /*
   * Deviance and MSE
   *
   * Both for test data
   *
   * MSE is calculated as the mean of (y-eta)^2,
   *   where eta is the appropriate link.
   *
   */
  //////////////////////////////////////////////////////////////
  virtual double deviance(const arma::vec& y_i,
                                    const arma::vec& eta_i,
                                    const double& aphi
  )=0;


  virtual double deviance(const arma::vec& y_j,
                                    const arma::vec& eta_j,
                                    const arma::vec& aphi
                                    )=0; //overloaded form

  virtual double mse(const arma::vec& y_i,
                     const arma::vec& eta_i,
                     const double& aphi
  )=0;

  virtual double mse(const arma::vec& y_j,
                     const arma::vec& eta_j,
                     const arma::vec& aphi
                       )=0; //overloaded form



};


/**
 * Gaussian family
 */

template<typename T>
class rowFamilyGaussian final: public rowFamily<T> { // Models a row
public:

  static constexpr double clampThreshUpper=1e15;
  static constexpr double clampThreshLower=-1e15;
  
  ////////////////////////////////////
  /*
   *  LOGLIKELIHOODS
   */
  ////////////////////////////////////
  double calcNegLogLL_i( const arma::vec& y,
                         const arma::vec& eta,
                         const double& aphi
  ) override;

  double calcNegLogLL_j( const arma::vec& y,
                         const arma::vec& eta,
                         const arma::vec& aphi //a vector now
  ) override;

  double calcNegLogLLForCols( const T& Y,
                              const arma::uvec columns1,
                              const arma::mat& Zmat,
                              const arma::mat& Zeta,
                              const arma::mat& umat,
                              const arma::mat& vmat,
                              const arma::vec& aphi,
                              const arma::vec& rowOffsets,
                              const arma::vec& colOffsets
  ) override;

  ////////////////////////////////////
  /*
   *  ZETA:
   *  Zeta_i matrix in T= Z Zeta_i
   */
  ////////////////////////////////////
  arma::vec gradNegLogLL_Zeta_i( const arma::vec& y_i, //nx1
                                 const arma::vec& eta_i, //nx1
                                 const arma::mat& Z, // n x B
                                 const arma::vec& zeta_i,  // B x p
                                 const double& aphi_i) override;

  //hessian
  arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::mat& Z,
                                 const arma::vec& zeta_i,
                                 const double aphi_i) override;
  arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::mat& Z,
                                 const arma::vec& zeta_i,
                                 const double aphi,
                                 const arma::mat& ZtZ
  ) override;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  v_k: in vmat=[v1, ... vK].
   *  v_ik i.e., for the ith index of v in v=[v0,v1, ... v(p-1) ] of p features
   */
  ////////////////////////////////////////////////////////////////////////
  double gradNegLogLL_v_ik(
      const arma::vec& y_i,
      const arma::vec& eta_i,
      const arma::vec& u_k, // for a single vector u, entering as uv'
      const double v_ik,  // for a single vector v entering as uv'
      const double& aphi_i
  ) override ;

  double hessNegLogLL_v_ik( const arma::vec& y_i,
                            const arma::vec& eta_i,
                            const arma::vec& u_k, // for a single component
                            const double v_ik,  // for the i^th feature in k^th component
                            const double& aphi_i
  ) override;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  h_k: in hmat=[h1, ... hK].
   *  U=Mh
   */
  ////////////////////////////////////////////////////////////////////////
  //for K=1 without umatOthers and vmatOthers
  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  //for K=1
  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  //for K = 1
  void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  //for K =1
  void gradNegLogLL_h_k_onColsOf(
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  void gradNegLogLL_h_k_onColsOf(
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  //for K = 1
  void hessNegLogLL_h_k( const T& Y,
                         const arma::mat& M,
                         const arma::mat& Z,
                         const arma::mat& Zeta,
                         const arma::vec& aphi,
                         const arma::vec& rowOffsets,
                         const arma::vec& colOffsets,
                         const arma::mat& MtM,
                         const arma::vec& h_k,
                         const arma::vec& v_k,
                         arma::mat& hess_h_k
  ) override;

  void hessNegLogLL_h_k( const T& Y,
                         const arma::mat& M,
                         const arma::mat& Z,
                         const arma::mat& Zeta,
                         const arma::mat& umatOthers,
                         const arma::mat& vmatOthers,
                         const arma::vec& aphi,
                         const arma::vec& rowOffsets,
                         const arma::vec& colOffsets,
                         const arma::mat& MtM,
                         const arma::vec& h_k,
                         const arma::vec& v_k,
                         arma::mat& hess_h_k
  ) override;

  //for K = 1
  void hessNegLogLL_h_k_onColsOf( const T& Y,
                                  const arma::uvec& columns1,
                                  const arma::mat& M,
                                  const arma::mat& Z,
                                  const arma::mat& Zeta,
                                  const arma::vec& aphi,
                                  const arma::vec& rowOffsets,
                                  const arma::vec& colOffsets,
                                  const arma::mat& MtM,
                                  const arma::vec& h_k,
                                  const arma::vec& v_k,
                                  arma::mat& hess_h_k

  ) override;

  void hessNegLogLL_h_k_onColsOf( const T& Y,
                                  const arma::uvec& columns1,
                                  const arma::mat& M,
                                  const arma::mat& Z,
                                  const arma::mat& Zeta,
                                  const arma::mat& umatOthers,
                                  const arma::mat& vmatOthers,
                                  const arma::vec& aphi,
                                  const arma::vec& rowOffsets,
                                  const arma::vec& colOffsets,
                                  const arma::mat& MtM,
                                  const arma::vec& h_k,
                                  const arma::vec& v_k,
                         arma::mat& hess_h_k

  ) override;

  //////////////////////////////////////////////////////////////
  /*
   * Aphi updates
   */
  //////////////////////////////////////////////////////////////

  double calc_aphi(const arma::vec& y_i, const arma::vec& eta_i) override;


  //Deviance and MSE

  double deviance(const arma::vec& y_i,
                                    const arma::vec& eta_i,
                                    const double& aphi
  ) override;


  double deviance(const arma::vec& y_j,
                                    const arma::vec& eta_j,
                                    const arma::vec& aphi
  ) override; //overloaded form

  double mse(const arma::vec& y_i,
                     const arma::vec& eta_i,
                     const double& aphi
  ) override;

  double mse(const arma::vec& y_j,
                     const arma::vec& eta_j,
                     const arma::vec& aphi
  ) override; //overloaded form


};


template class rowFamilyGaussian<arma::mat>;
template class rowFamilyGaussian<arma::sp_mat>;



/**
 * Poisson family
 */
template<typename T>
class rowFamilyPoisson final: public rowFamily<T> { // Models a row
public:

  
  static constexpr double ETA_MAX = 700.0;
  
  static constexpr double clampThreshUpper=1e15;
  static constexpr double clampThreshLower=-1e15;
  static constexpr double eta_clampThreshUpper=740;
  static constexpr double eta_clampThreshLower=-740;
  static constexpr bool poissonHessSerial = true; //to avoid oversubscription in nested tbb workers, 
                                                  // we make the inner hessian calculation for poisson serial. 
                                                  // For gausian, we don't use such calculators
                                                  // the 
  
  
  ////////////////////////////////////
  /*
   *  Descent moves
   */
  ////////////////////////////////////
  inline bool validEta(const arma::vec& eta) const {
    return eta.is_finite() && eta.max() <= ETA_MAX;
  }
  
  
  ////////////////////////////////////
  /*
   *  LOGLIKELIHOODS
   */
  ////////////////////////////////////
  double calcNegLogLL_i( const arma::vec& y,
                         const arma::vec& eta,
                         const double& aphi
  ) override;

  double calcNegLogLL_j( const arma::vec& y,
                         const arma::vec& eta,
                         const arma::vec& aphi //a vector now
  ) override;

  double calcNegLogLLForCols( const T& Y,
                              const arma::uvec columns1,
                              const arma::mat& Zmat,
                              const arma::mat& Zeta,
                              const arma::mat& umat,
                              const arma::mat& vmat,
                              const arma::vec& aphi,
                              const arma::vec& rowOffsets,
                              const arma::vec& colOffsets
  ) override;




  ////////////////////////////////////
  /*
   *  ZETA:
   *  Zeta_i matrix in T= Z Zeta_i
   */
  ////////////////////////////////////
  arma::vec gradNegLogLL_Zeta_i( const arma::vec& y_i, //nx1
                                 const arma::vec& eta_i, //nx1
                                 const arma::mat& Z, // n x B
                                 const arma::vec& zeta_i,  // B x p
                                 const double& aphi_i) override;

  //hessian
  arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::mat& Z,
                                 const arma::vec& zeta_i,
                                 const double aphi_i) override;
  arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::mat& Z,
                                 const arma::vec& zeta_i,
                                 const double aphi,
                                 const arma::mat& ZtZ
  ) override;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  v_k: in vmat=[v1, ... vK].
   *  v_ik i.e., for the ith index of v in v=[v0,v1, ... v(p-1) ] of p features
   */
  ////////////////////////////////////////////////////////////////////////
  double gradNegLogLL_v_ik(
      const arma::vec& y_i,
      const arma::vec& eta_i,
      const arma::vec& u_k, // for a single vector u, entering as uv'
      const double v_ik,  // for a single vector v entering as uv'
      const double& aphi_i
  ) override ;

  double hessNegLogLL_v_ik( const arma::vec& y_i,
                            const arma::vec& eta_i,
                            const arma::vec& u_k, // for a single component
                            const double v_ik,  // for the i^th feature in k^th component
                            const double& aphi_i
  ) override;

  ////////////////////////////////////////////////////////////////////////
  /*
   *  h_k: in hmat=[h1, ... hK].
   *  U=Mh
   */
  ////////////////////////////////////////////////////////////////////////
    // for K = 1
  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  //for K = 1
  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  double cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k, // the point at which the gradient is needed. h_k enters via u_k = M h_k. This is the rowFamily model assumed.
      const arma::vec& v_k,
      const int j //sample id
  ) override;

  //for K=1
  void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  void gradNegLogLL_h_k(
      const T& Y,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  //for K =1
  void gradNegLogLL_h_k_onColsOf(
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;
  void gradNegLogLL_h_k_onColsOf(
      const T& Y,
      const arma::uvec& columns1,
      const arma::mat& M,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& aphi,
      const arma::vec& rowOffsets,
      const arma::vec& colOffsets,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::vec& grad_h_k //reference to which output is written
  ) override;

  //for K = 1
  void hessNegLogLL_h_k( const T& Y,
                         const arma::mat& M,
                         const arma::mat& Z,
                         const arma::mat& Zeta,
                         const arma::vec& aphi,
                         const arma::vec& rowOffsets,
                         const arma::vec& colOffsets,
                         const arma::mat& MtM,
                         const arma::vec& h_k,
                         const arma::vec& v_k,
                         arma::mat& hess_h_k
  ) override;
  void hessNegLogLL_h_k( const T& Y,
                         const arma::mat& M,
                         const arma::mat& Z,
                         const arma::mat& Zeta,
                         const arma::mat& umatOthers,
                         const arma::mat& vmatOthers,
                         const arma::vec& aphi,
                         const arma::vec& rowOffsets,
                         const arma::vec& colOffsets,
                         const arma::mat& MtM,
                         const arma::vec& h_k,
                         const arma::vec& v_k,
                         arma::mat& hess_h_k
  ) override;

  //for K=1
  void hessNegLogLL_h_k_onColsOf( const T& Y,
                                  const arma::uvec& columns1,
                                  const arma::mat& M,
                                  const arma::mat& Z,
                                  const arma::mat& Zeta,
                                  const arma::vec& aphi,
                                  const arma::vec& rowOffsets,
                                  const arma::vec& colOffsets,
                                  const arma::mat& MtM,
                                  const arma::vec& h_k,
                                  const arma::vec& v_k,
                                  arma::mat& hess_h_k

  ) override;
  void hessNegLogLL_h_k_onColsOf( const T& Y,
                                  const arma::uvec& columns1,
                                  const arma::mat& M,
                                  const arma::mat& Z,
                                  const arma::mat& Zeta,
                                  const arma::mat& umatOthers,
                                  const arma::mat& vmatOthers,
                                  const arma::vec& aphi,
                                  const arma::vec& rowOffsets,
                                  const arma::vec& colOffsets,
                                  const arma::mat& MtM,
                                  const arma::vec& h_k,
                                  const arma::vec& v_k,
                         arma::mat& hess_h_k

  ) override;

  //////////////////////////////////////////////////////////////
  /*
   * Aphi updates
   */
  //////////////////////////////////////////////////////////////

  double calc_aphi(const arma::vec& y_i, const arma::vec& eta_i) override;


  //////////////////////////////////////////////////////////////
  /*
   * Deviance and MSE
   *
   * Both for test data
   */
  //////////////////////////////////////////////////////////////

  double deviance(const arma::vec& y_i,
                                    const arma::vec& eta_i,
                                    const double& aphi
  ) override;


  double deviance(const arma::vec& y_j,
                                    const arma::vec& eta_j,
                                    const arma::vec& aphi
  ) override; //overloaded form

  double mse(const arma::vec& y_i,
                     const arma::vec& eta_i,
                     const double& aphi
  ) override;

  double mse(const arma::vec& y_j,
                     const arma::vec& eta_j,
                     const arma::vec& aphi
  ) override; //overloaded form




};


template class rowFamilyPoisson<arma::mat>;
template class rowFamilyPoisson<arma::sp_mat>;



///////////////////////////////////////////////////////////////////////
//////////////////////////////GAUSSIAN//////////////////////////////////
///////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
/*
 * Gradient and hessians for technical effects operating on rows i.
 * Zeta_i matrix in T= Z Zeta_i
 * ll.ik <- sum((y.i - eta.ik)^2)/s2i #negative log like, when s2i is fixed constant
 * gradll.ik <- f1*c(crossprod( Z,eta.ik-y.i ))/s2i
 * hessZeta.ik <- f1*ZtZ/s2i, with weights this is f1* Z^TWZ/s2i
 *  NOTE: IT IS BETTER TO CALCULATE HESSIAN ONCE AND STORE! WITH OR WITHOUT WEIGHTS,
 *        IT IS ThE SAME EXPRESSION  that only gets divided by s2i fresh.
 */
// Overloaded forms appear depending on the argument, example constant weight.
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK

//////////////////////////////////////////////////////////////////////////////////

// gradient, with NO weights
template<typename T>
arma::vec rowFamilyGaussian<T>::gradNegLogLL_Zeta_i( const arma::vec& y_i, //nx1
                                                     const arma::vec& eta_i, //nx1
                                                     const arma::mat& Z, // n x B
                                                     const arma::vec& zeta_i,  // B x p
                                                     const double& aphi_i)  {

  // std::cout << "Calling Zeta_i gradient in rowfam gaussian" << std::endl;

  return( ( Z.t()  * (eta_i-y_i) )/aphi_i );
}

//hessian
template<typename T>
arma::mat rowFamilyGaussian<T>::hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                                     const arma::vec& eta_i,
                                                     const arma::mat& Z,
                                                     const arma::vec& zeta_i,
                                                     const double aphi_i
)  {
  return( ( Z.t() * Z )/aphi_i );
}

template<typename T>
arma::mat rowFamilyGaussian<T>::hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                                     const arma::vec& eta_i,
                                                     const arma::mat& Z,
                                                     const arma::vec& zeta_i,
                                                     const double aphi,
                                                     const arma::mat& ZtZ
) {
  return( ZtZ/aphi );
}




//////////////////////////////////////////////////////////////////////////////////
/*
 * For the ith index of v (really , this is ith index of the k^th component v)
 * for the ith index of v in v=[v0,v1, ... v(p-1) ] of p features
 *
 * ll.ki <- (1/s2.i)*sum( (Y.i - eta.ki)^2 ), with weights, (1/s2.i)*(Y.i - eta.ki)'W(Y.i - eta.ki)
 * gradll.ki <- (f1/s2.i)*sum( (eta.ki-Y.i)*u ), with weights, (f1/s2i)*(eta.ki-Y.i)'W u
 * hessv.ki <- f1*usqnorm/s2.i, with weights f1*u'Wu/s2.i                   //really just the second derivative
 *
 */
//////////////////////////////////////////////////////////////////////////////////
//gradient and hessians for simple row vectors passed
template<typename T>
double rowFamilyGaussian<T>::gradNegLogLL_v_ik(
    const arma::vec& y_i,
    const arma::vec& eta_i,
    const arma::vec& u_k, // for a single vector u, entering as uv'
    const double v_ik,  // for a single vector v entering as uv'
    const double& aphi_i
)  {

  // std::cout << "Calling v_ik gradient in rowfam gaussian" << std::endl;

  return( arma::dot( (eta_i-y_i), u_k )/aphi_i);

}


template <typename T>
double rowFamilyGaussian<T>::hessNegLogLL_v_ik( const arma::vec& y_i,
                                                const arma::vec& eta_i,
                                                const arma::vec& u_k, // for a single component
                                                const double v_ik,  // for the i^th feature in k^th component
                                                const double& aphi_i
)  {

  // std::cout << "Calling v_ik hessian in rowfam gaussian" << std::endl;

  return( arma::dot(u_k, u_k)/aphi_i );
}


//////////////////////////////////////////////////////////////////////////////////
/* For h in u=Mh
 *
 * Here Y is a full blown matrix. Eta is too. More efficient implementations need to
 *  be made for large, sparse matrices.
 *
 * ll.k <- sum((1/s2)*((Y - eta.k)^2))
 * gradll.k <- c( f1*crossprod( M, c(colSums( (eta.k-Y)*(v/s2)))) ); gradll.k[ .r+(1:.q) ] <- gradll.k[ .r+(1:.q) ] + 2*f2*RtilTheta %*% h.k[.r+(1:.q)]
 * hessh.k <-  f1*sum((v^2)/s2)*MtM ; hessh.k[ .r+(1:.q), .r+(1:.q) ] <- hessh.k[ .r+(1:.q), .r+(1:.q) ] + 2*f2*RtilTheta
 *
 */
//////////////////////////////////////////////////////////////////////////////////

/*
 * For K = 1, Case: DENSE MATRIX, Without weights
 */
template <typename T>
double rowFamilyGaussian<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec yj;
  yj=Y.col(j);
  return( arma::sum( (eta_j - yj) % (v_k/aphi) ) );
}
/*
 * For K > 1, Case: DENSE MATRIX, Without weights
 */
template <typename T>
double rowFamilyGaussian<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers, //columns other than the k^th, must not be empty, no check made
    const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*umatOthers.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec yj;
  yj=Y.col(j);
  return( arma::sum( (eta_j - yj) % (v_k/aphi) ) );
}


//for K =1
template <typename T>
double rowFamilyGaussian<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) +  (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec yj;
  yj=Y.col(columns1(j));
  return( arma::sum( (eta_j - yj) % (v_k/aphi) ) );
}
template <typename T>
double rowFamilyGaussian<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers, //columns other than the k^th
    const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*umatOthers.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec yj;
  yj=Y.col(columns1(j));
  return( arma::sum( (eta_j - yj) % (v_k/aphi) ) );
}


//a direct tbb version without weights
// for K =1
template <typename T>
void rowFamilyGaussian<T>::gradNegLogLL_h_k(
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  arma::vec cs(Y.n_cols);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y, M, Z, Zeta,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}
template <typename T>
void rowFamilyGaussian<T>::gradNegLogLL_h_k(
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  arma::vec cs(Y.n_cols);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y, M, Z, Zeta,
                           umatOthers, vmatOthers,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}


//a direct tbb version without weights
// for K = 1
template<typename T>
void rowFamilyGaussian<T>::gradNegLogLL_h_k_onColsOf(
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  const int n = columns1.size();
  arma::vec cs(n);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y,
                           columns1, //this is the key change as we need to do onColsOf
                           M, Z, Zeta,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}
template<typename T>
void rowFamilyGaussian<T>::gradNegLogLL_h_k_onColsOf(
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  const int n = columns1.size();
  arma::vec cs(n);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y,
                           columns1, //this is the key change as we need to do onColsOf
                           M, Z, Zeta,
                           umatOthers, vmatOthers,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}

// HESSIAN, FOR THE CASE WITHOUT WEIGHTS
// sBut notice the same function can be used
//  when there are only colweights, and no rowWeights, as MtM can be set to Mtdiag(colWeights)M
// for K =1
template <typename T>
void rowFamilyGaussian<T>::hessNegLogLL_h_k( const T& Y,
                                             const arma::mat& M,
                                             const arma::mat& Z,
                                             const arma::mat& Zeta,
                                             const arma::vec& aphi,
                                             const arma::vec& rowOffsets,
                                             const arma::vec& colOffsets,
                                             const arma::mat& MtM,
                                             const arma::vec& h_k,
                                             const arma::vec& v_k,
                                             arma::mat& hess_h_k

)  {
  hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
}

template <typename T>
void rowFamilyGaussian<T>::hessNegLogLL_h_k( const T& Y,
                                             const arma::mat& M,
                                             const arma::mat& Z,
                                             const arma::mat& Zeta,
                                             const arma::mat& umatOthers,
                                             const arma::mat& vmatOthers,
                                             const arma::vec& aphi,
                                             const arma::vec& rowOffsets,
                                             const arma::vec& colOffsets,
                                             const arma::mat& MtM,
                                             const arma::vec& h_k,
                                             const arma::vec& v_k,
                                             arma::mat& hess_h_k

)  {
  hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
}

//for K =1
template <typename T>
void rowFamilyGaussian<T>::hessNegLogLL_h_k_onColsOf( const T& Y,
                                                      const arma::uvec& columns1,
                                                      const arma::mat& M,
                                                      const arma::mat& Z,
                                                      const arma::mat& Zeta,
                                                      const arma::vec& aphi,
                                                      const arma::vec& rowOffsets,
                                                      const arma::vec& colOffsets,
                                                      const arma::mat& MtM,
                                                      const arma::vec& h_k,
                                                      const arma::vec& v_k,
                                                      arma::mat& hess_h_k

)  {
  hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
}

template <typename T>
void rowFamilyGaussian<T>::hessNegLogLL_h_k_onColsOf( const T& Y,
                                                      const arma::uvec& columns1,
                                                      const arma::mat& M,
                                                      const arma::mat& Z,
                                                      const arma::mat& Zeta,
                                                      const arma::mat& umatOthers,
                                                      const arma::mat& vmatOthers,
                                                      const arma::vec& aphi,
                                                      const arma::vec& rowOffsets,
                                                      const arma::vec& colOffsets,
                                                      const arma::mat& MtM,
                                                      const arma::vec& h_k,
                                                      const arma::vec& v_k,
                                                      arma::mat& hess_h_k

)  {
  hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
}


//////////////////////////////////////////////////////////////////////////////////
/*
 * Calculation of negative log likeslihoods for ROW TYPE OBJECTS.
 * These functions are overloaded for row vectors, matrices
 *  and sparse matrices. Thank you C++,Armadillo developers for
 *  making all of this easy for the rest of us.
 */
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK
//////////////////////////////////////////////////////////////////////////////////

//For a row vector of y, its means, and offsets, with no weights.
template<typename T>
double rowFamilyGaussian<T>::calcNegLogLL_i( const arma::vec& y,
                                             const arma::vec& eta,
                                             const double& aphi
)  {
  double nll = (.5/aphi)*arma::sum( pow((y-eta),2) ) + (y.n_elem)*log(std::sqrt(aphi));
  return( nll );
}

template <typename T>
double rowFamilyGaussian<T>::calcNegLogLL_j( const arma::vec& y,
                                             const arma::vec& eta,
                                             const arma::vec& aphi //overloaded form with vector
)  {
  double nll = (.5)*arma::sum( pow((y-eta),2)/aphi ) + arma::sum(log(arma::sqrt(aphi)));
  return( nll );
}


template <typename T>
double rowFamilyGaussian<T>::calcNegLogLLForCols( const T& Y,
                                                  const arma::uvec columns1,
                                                  const arma::mat& Z,
                                                  const arma::mat& Zeta,
                                                  const arma::mat& umat,
                                                  const arma::mat& vmat,
                                                  const arma::vec& aphi,
                                                  const arma::vec& rowOffsets,
                                                  const arma::vec& colOffsets
                                                    )  {

  arma::vec nll(Y.n_cols, arma::fill::zeros);
  
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::vec eta_j = (Zeta.t() * Z.row(j).t()) + (vmat*umat.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
                         arma::vec y_j;
                         y_j=Y.col(columns1(j));
                         nll(j)=calcNegLogLL_j(y_j, eta_j, aphi);
                       }
                     }
                     );

  return arma::sum(nll);


}




//////////////////////////////////////////////////////////////////////////////////
/*
 * DEVIANCE AND MSES
 */
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK
//////////////////////////////////////////////////////////////////////////////////
template<typename T>
double rowFamilyGaussian<T>::deviance(const arma::vec& y_i,
                                      const arma::vec& eta_i,
                                      const double& aphi
){

  arma::vec dev = arma::clamp( arma::pow((y_i - eta_i), 2)/aphi, 0.0, clampThreshUpper ); //deviance is positive,so lower obound is just 0
  return arma::mean( dev ); //just averaging instead of sum, fr matrix data; equivalent upto a constant
}

template<typename T>
double rowFamilyGaussian<T>::deviance(const arma::vec& y_j,
                                      const arma::vec& eta_j,
                                      const arma::vec& aphi
){

  arma::vec dev = arma::clamp( arma::pow((y_j - eta_j), 2)/aphi, 0.0, clampThreshUpper );
  return arma::mean( dev );
}

template <typename T>
double rowFamilyGaussian<T>::mse(const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const double& aphi
){

  arma::vec resid=arma::clamp( arma::pow(y_i-eta_i,2), 0.0, clampThreshUpper);
  return arma::mean(resid);

}

template<typename T>
double rowFamilyGaussian<T>::mse(const arma::vec& y_j,
                                 const arma::vec& eta_j,
                                 const arma::vec& aphi
){
  arma::vec resid=arma::clamp( arma::pow(y_j-eta_j, 2), 0.0, clampThreshUpper);
  return arma::mean(resid);

}


//////////////////////////////////////////////////////////////////////////////////
/*
 * aphi = s2 calculation
 */
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK
//////////////////////////////////////////////////////////////////////////////////
template<typename T>
double rowFamilyGaussian<T>::calc_aphi(const arma::vec& y_i, const arma::vec& eta_i)  {
  return( arma::sum( pow( y_i -eta_i, 2  )  )/y_i.size()  );

}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
/*
 * POISSON POISSON
 */
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////
/*
 * Gradient and hessians for technical effects operating on rows i.
 * Zeta_i matrix in T= Z Zeta_i
 * ll.ik <- sum((y.i - eta.ik)^2)/s2i #negative log like, when s2i is fixed constant
 * gradll.ik <- f1*c(crossprod( Z,eta.ik-y.i ))/s2i
 * hessZeta.ik <- f1*ZtZ/s2i, with weights this is f1* Z^TWZ/s2i
 */
// Overloaded forms appear depending on the argument, example constant weight.
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK

//////////////////////////////////////////////////////////////////////////////////

// gradient, with NO weights
template <typename T>
arma::vec rowFamilyPoisson<T>::gradNegLogLL_Zeta_i( const arma::vec& y_i, //nx1
                                                    const arma::vec& eta_i, //nx1
                                                    const arma::mat& Z, // n x B
                                                    const arma::vec& zeta_i,  // B x p
                                                    const double& aphi_i)  {

  // std::cout << "Calling Zeta_i gradient in rowfam gaussian" << std::endl;
  //arma::vec exp_eta_i = arma::clamp( arma::exp(eta_i), clampThreshLower, clampThreshUpper);
  
  
  
  
  arma::vec exp_eta_i = arma::exp( arma::clamp( eta_i, eta_clampThreshLower, eta_clampThreshUpper ) );
  return( ( Z.t()  * (exp_eta_i-y_i) )/aphi_i ); //aphi_i is one for poisson
}

//hessian
template <typename T>
arma::mat rowFamilyPoisson<T>::hessNegLogLL_Zeta_i( const  arma::vec& y_i,
                                                    const arma::vec& eta_i,
                                                    const arma::mat& Z,
                                                    const arma::vec& zeta_i,
                                                    const double aphi_i
)  {

  
  arma::vec exp_eta_i = arma::exp( arma::clamp( eta_i, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::mat Ztilde_i = Z;
  for( int j=0; j< Z.n_cols; j++){
    Ztilde_i.col(j) %= exp_eta_i;
  }
  return( ( Ztilde_i.t() * Z )/aphi_i );
}

template <typename T>
arma::mat rowFamilyPoisson<T>::hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                                    const arma::vec& eta_i,
                                                    const arma::mat& Z,
                                                    const arma::vec& zeta_i,
                                                    const double aphi_i,
                                                    const arma::mat& ZtZ
) {

  return hessNegLogLL_Zeta_i( y_i,eta_i,Z,zeta_i,aphi_i);
}


//////////////////////////////////////////////////////////////////////////////////
/*
 * For the ith index of v (really , this is ith index of the k^th component v)
 * for the ith index of v in v=[v0,v1, ... v(p-1) ] of p features
 *
 * ll.ki <- (1/s2.i)*sum( (Y.i - eta.ki)^2 ), with weights, (1/s2.i)*(Y.i - eta.ki)'W(Y.i - eta.ki)
 * gradll.ki <- (f1/s2.i)*sum( (eta.ki-Y.i)*u ), with weights, (f1/s2i)*(eta.ki-Y.i)'W u
 * hessv.ki <- f1*usqnorm/s2.i, with weights f1*u'Wu/s2.i                   //really just the second derivative
 *
 */
//////////////////////////////////////////////////////////////////////////////////
//gradient and hessians for simple row vectors passed
template <typename T>
double rowFamilyPoisson<T>::gradNegLogLL_v_ik(
    const arma::vec& y_i,
    const arma::vec& eta_i,
    const arma::vec& u_k, // for a single vector u, entering as uv'
    const double v_ik,  // for a single vector v entering as uv'
    const double& aphi_i
)  {

  arma::vec exp_eta_i = arma::exp( arma::clamp( eta_i, eta_clampThreshLower, eta_clampThreshUpper ) );

  return( arma::dot( (exp_eta_i-y_i), u_k)/aphi_i);

}


template <typename T>
double rowFamilyPoisson<T>::hessNegLogLL_v_ik( const arma::vec& y_i,
                                               const arma::vec& eta_i,
                                               const arma::vec& u_k, // for a single component
                                               const double v_ik,  // for the i^th feature in k^th component
                                               const double& aphi_i
)  {

  arma::vec exp_eta_i = arma::exp( arma::clamp( eta_i, eta_clampThreshLower, eta_clampThreshUpper ) );
  return( arma::dot(u_k % exp_eta_i, u_k)/aphi_i );
}


//////////////////////////////////////////////////////////////////////////////////
/* For h in u=Mh
 *
 * Here Y is a full blown matrix. Eta is too. More efficient implementations need to
 *  be made for large, sparse matrices.
 *
 * ll.k <- sum((1/s2)*((Y - eta.k)^2))
 * gradll.k <- c( f1*crossprod( M, c(colSums( (eta.k-Y)*(v/s2)))) ); gradll.k[ .r+(1:.q) ] <- gradll.k[ .r+(1:.q) ] + 2*f2*RtilTheta %*% h.k[.r+(1:.q)]
 * hessh.k <-  f1*sum((v^2)/s2)*MtM ; hessh.k[ .r+(1:.q), .r+(1:.q) ] <- hessh.k[ .r+(1:.q), .r+(1:.q) ] + 2*f2*RtilTheta
 *
 */
//////////////////////////////////////////////////////////////////////////////////

/*
 * For K =1, w/o umatOthers, vmatOthers. Case: DENSE MATRIX, Without weights
 */
template <typename T>
double rowFamilyPoisson<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) +  (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec exp_eta_j = arma::exp( arma::clamp( eta_j, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec yj;
  yj = Y.col(j);
  return( arma::sum( (exp_eta_j - yj) % (v_k/aphi) ) );
}

/*
 * For K >1 Case: DENSE MATRIX, Without weights
 */
template <typename T>
double rowFamilyPoisson<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers, //columns other than the k^th
    const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*umatOthers.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec exp_eta_j = arma::exp( arma::clamp( eta_j, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec yj;
  yj = Y.col(j);
  return( arma::sum( (exp_eta_j - yj) % (v_k/aphi) ) );
}

//For K=1
template <typename T>
double rowFamilyPoisson<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) +  (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec exp_eta_j = arma::exp( arma::clamp( eta_j, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec yj;
  yj = Y.col(columns1(j));
  return( arma::sum( (exp_eta_j - yj) % (v_k/aphi) ) );
}
template <typename T>
double rowFamilyPoisson<T>::cs_j_gradNegLogLL_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers, //columns other than the k^th
    const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*umatOthers.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec exp_eta_j = arma::exp( arma::clamp( eta_j, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec yj;
  yj = Y.col(columns1(j));
  return( arma::sum( (exp_eta_j - yj) % (v_k/aphi) ) );
}

//a direct tbb version without weights
// for K = 1
template <typename T>
void rowFamilyPoisson<T>::gradNegLogLL_h_k(
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  arma::vec cs(Y.n_cols);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y, M, Z, Zeta,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}
template <typename T>
void rowFamilyPoisson<T>::gradNegLogLL_h_k(
    const T& Y,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  arma::vec cs(Y.n_cols);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y, M, Z, Zeta,
                           umatOthers, vmatOthers,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}

template <typename T>
void rowFamilyPoisson<T>::hessNegLogLL_h_k( const T& Y,
                                            const arma::mat& M,
                                            const arma::mat& Z,
                                            const arma::mat& Zeta,
                                            const arma::vec& aphi,
                                            const arma::vec& rowOffsets,
                                            const arma::vec& colOffsets,
                                            const arma::mat& MtM,
                                            const arma::vec& h_k,
                                            const arma::vec& v_k,
                                            arma::mat& hess_h_k

)  {

  const bool phs = this->poissonHessSerial; //this additional step is needed to avoid poissonHessSerial being ODR-used , as we don't have a separate cpp file with an out of class definition. 
  hessPoissonAllColsK1 hpcac( M,
                            Z,
                            Zeta,
                            aphi,
                            rowOffsets,
                            colOffsets,
                            h_k,      //the h that will determine u_k = M h_k
                            v_k,      // the corresponding feature loadings vector already fixed
                            phs
  );
  parallelReduce( 0, Y.n_cols, hpcac );  
  hess_h_k=hpcac.value;
}
template <typename T>
void rowFamilyPoisson<T>::hessNegLogLL_h_k( const T& Y,
                                            const arma::mat& M,
                                            const arma::mat& Z,
                                            const arma::mat& Zeta,
                                            const arma::mat& umatOthers,
                                            const arma::mat& vmatOthers,
                                            const arma::vec& aphi,
                                            const arma::vec& rowOffsets,
                                            const arma::vec& colOffsets,
                                            const arma::mat& MtM,
                                            const arma::vec& h_k,
                                            const arma::vec& v_k,
                                            arma::mat& hess_h_k

)  {

  const bool phs = this->poissonHessSerial; //this additional step is needed to avoid poissonHessSerial being ODR-used 
  hessPoissonAllCols hpcac( M,
                            Z,
                            Zeta,
                            umatOthers, //columns other than the k^th
                            vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
                            aphi,
                            rowOffsets,
                            colOffsets,
                            h_k,      //the h that will determine u_k = M h_k
                            v_k,      // the corresponding feature loadings vector already fixed
                            phs
  );
  parallelReduce( 0, Y.n_cols, hpcac );  //parallel computation
  hess_h_k=hpcac.value;


}



//a direct tbb version without weights
//for K = 1
template <typename T>
void rowFamilyPoisson<T>::gradNegLogLL_h_k_onColsOf(
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  int n=columns1.n_elem;
  arma::vec cs(n);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y,
                           columns1, //this is the key change we need to make as we compute on colsof
                           M, Z, Zeta,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}
template <typename T>
void rowFamilyPoisson<T>::gradNegLogLL_h_k_onColsOf(
    const T& Y,
    const arma::uvec& columns1,
    const arma::mat& M,
    const arma::mat& Z,
    const arma::mat& Zeta,
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& aphi,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::vec& grad_h_k //reference to which output is written
) {

  //std::vector<double> cs=std::vector<double>(Y.n_cols);
  int n=columns1.n_elem;
  arma::vec cs(n);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         cs(j)=cs_j_gradNegLogLL_h_k(
                           Y,
                           columns1, //this is the key change we need to make as we compute on colsof
                           M, Z, Zeta,
                           umatOthers, vmatOthers,
                           aphi, rowOffsets, colOffsets,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  grad_h_k= M.t() * cs;
}


template <typename T>
void rowFamilyPoisson<T>::hessNegLogLL_h_k_onColsOf( const T& Y,
                                                     const arma::uvec& columns1,
                                                     const arma::mat& M,
                                                     const arma::mat& Z,
                                                     const arma::mat& Zeta,
                                                     const arma::vec& aphi,
                                                     const arma::vec& rowOffsets,
                                                     const arma::vec& colOffsets,
                                                     const arma::mat& MtM,
                                                     const arma::vec& h_k,
                                                     const arma::vec& v_k,
                                                     arma::mat& hess_h_k

)  {

  //This is OK because we only use the columns1 samples in M.
  // and no where in hesspoisson is actual data Y used.
  
  const bool phs = this->poissonHessSerial; //this additional step is needed to avoid poissonHessSerial being ODR-used 
  hessPoissonAllColsK1 hpcac( M,
                            Z,
                            Zeta,
                            aphi,
                            rowOffsets,
                            colOffsets,
                            h_k,      //the h that will determine u_k = M h_k
                            v_k,      // the corresponding feature loadings vector already fixed
                            phs
  );
  
  // if (phs) {
  //   hpcac(0, M.n_rows);
  // } else {
  //   parallelReduce(0, M.n_rows, hpcac);
  // }
  parallelReduce( 0, M.n_rows, hpcac );  
  hess_h_k=hpcac.value;
}
template <typename T>
void rowFamilyPoisson<T>::hessNegLogLL_h_k_onColsOf( const T& Y,
                                                     const arma::uvec& columns1,
                                                     const arma::mat& M,
                                                     const arma::mat& Z,
                                                     const arma::mat& Zeta,
                                                     const arma::mat& umatOthers,
                                                     const arma::mat& vmatOthers,
                                                     const arma::vec& aphi,
                                                     const arma::vec& rowOffsets,
                                                     const arma::vec& colOffsets,
                                                     const arma::mat& MtM,
                                                     const arma::vec& h_k,
                                                     const arma::vec& v_k,
                                                     arma::mat& hess_h_k

)  {

  const bool phs = this->poissonHessSerial; //this additional step is needed to avoid poissonHessSerial being ODR-used 
  hessPoissonAllCols hpcac( M,
                            Z,
                            Zeta,
                            umatOthers, //columns other than the k^th
                            vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
                            aphi,
                            rowOffsets,
                            colOffsets,
                            h_k,      //the h that will determine u_k = M h_k
                            v_k,      // the corresponding feature loadings vector already fixed
                            phs
  );
  // if(phs) {
  //   hpcac(0, M.n_rows);
  // } else {
  //   parallelReduce(0, M.n_rows, hpcac);
  // }
  parallelReduce( 0, M.n_rows, hpcac );  
  hess_h_k=hpcac.value;


}


//////////////////////////////////////////////////////////////////////////////////
/*
 * Calculation of negative log likeslihoods for ROW TYPE OBJECTS.
 * These functions are overloaded for row vectors, matrices
 *  and sparse matrices. Thank you C++,Armadillo developers for
 *  making all of this easy for the rest of us.
 */
// DIMENSIONS: Z nxB, Zeta Bxp, M nxH, h Hxn, umat nxK, vmat pxK
//////////////////////////////////////////////////////////////////////////////////

//For a row vector of y, its means, and offsets, with no weights.
template<typename T>
double rowFamilyPoisson<T>::calcNegLogLL_i( const arma::vec& y,
                                            const arma::vec& eta,
                                            const double& aphi
)  {
  if (!validEta(eta)) {
    return std::numeric_limits<double>::infinity();
  } else {
    arma::vec exp_eta_i = arma::exp(eta);
    double nll = arma::sum(exp_eta_i - y % eta) / aphi;
    return nll; 
  }
}

template<typename T>
double rowFamilyPoisson<T>::calcNegLogLL_j( const arma::vec& y,
                                            const arma::vec& eta,
                                            const arma::vec& aphi //overloaded form with vector
)  {
  if (!validEta(eta)) {
    return std::numeric_limits<double>::infinity();
  } else {
    arma::vec exp_eta_j = arma::exp(eta);
    double nll = arma::sum((exp_eta_j - y % eta) / aphi);
    return nll; 
  }
}

template <typename T>
double rowFamilyPoisson<T>::calcNegLogLLForCols( const T& Y,
                                                  const arma::uvec columns1,
                                                  const arma::mat& Z,
                                                  const arma::mat& Zeta,
                                                  const arma::mat& umat,
                                                  const arma::mat& vmat,
                                                  const arma::vec& aphi,
                                                  const arma::vec& rowOffsets,
                                                  const arma::vec& colOffsets
)  {

  arma::vec nll(Y.n_cols);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_cols),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::vec eta_j = (Zeta.t() * Z.row(j).t()) + (vmat*umat.row(j).t()) + (rowOffsets + colOffsets(j) ) ; //px1 vec corresp to col j
                         arma::vec y_j;
                         y_j=Y.col(columns1(j));
                         nll(j)=calcNegLogLL_j(y_j, eta_j, aphi);
                       }
                     }
  );
  return arma::sum(nll);

}



//////////////////////////////////////////
////////DEVIANCE//////////////////////////
/////////////////////////////////////////


template<typename T>
double rowFamilyPoisson<T>::deviance(const arma::vec& y_i,
                                     const arma::vec& eta_i,
                                     const double& aphi
){

  if (!validEta(eta_i)) {
    return std::numeric_limits<double>::infinity();
  }
  
  arma::vec lambda = arma::exp( eta_i);
  arma::vec y_i_star=y_i;
  for( int i=0; i<y_i.n_elem; i++ ){
    if( y_i(i)==0 ){
      y_i_star(i)=1;
    } else{
      y_i_star(i)=y_i_star(i)/lambda(i);
    }
  }
  arma::vec dev =2*( y_i % log(y_i_star) - (y_i-lambda) );
  return arma::mean( dev );
}

template <typename T>
double rowFamilyPoisson<T>::deviance(const arma::vec& y_j,
                                     const arma::vec& eta_j,
                                     const arma::vec& aphi
){

  if (!validEta(eta_j)) {
    return std::numeric_limits<double>::infinity();
  }
  
  arma::vec lambda = arma::exp( eta_j );
  arma::vec y_j_star=y_j;
  for( int i=0; i<y_j.n_elem; i++ ){
    if( y_j(i)==0 ){
      y_j_star(i)=1;
    } else{
      y_j_star(i)=y_j_star(i)/lambda(i);
    }
  }
  arma::vec dev = 2*( y_j % log(y_j_star) - (y_j-lambda) );
  return arma::mean( dev );
}

template <typename T>
double rowFamilyPoisson<T>::mse(const arma::vec& y_i,
                                const arma::vec& eta_i,
                                const double& aphi
){

  arma::vec lambda = arma::exp( arma::clamp( eta_i, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec resid=arma::clamp( arma::pow(y_i-lambda,2), 0.0, clampThreshUpper);
  return arma::mean(resid);

}

template <typename T>
double rowFamilyPoisson<T>::mse(const arma::vec& y_j,
                                const arma::vec& eta_j,
                                const arma::vec& aphi
){
  //arma::vec lambda = arma::clamp( arma::exp(eta_j), clampThreshLower, clampThreshUpper) ;
  arma::vec lambda = arma::exp( arma::clamp( eta_j, eta_clampThreshLower, eta_clampThreshUpper ) );
  arma::vec resid=arma::clamp( arma::pow(y_j-lambda, 2), 0.0, clampThreshUpper);
  return arma::mean(resid);

}



//////////////////////////////////////////
////////a(phi)=1//////////////////////////
/////////////////////////////////////////
template<typename T>
double rowFamilyPoisson<T>::calc_aphi(const arma::vec& y_i, const arma::vec& eta_i)  {
  return 1;

}












#endif
