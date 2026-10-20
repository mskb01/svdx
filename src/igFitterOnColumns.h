#ifndef IGFITTERONCOLUMNS_H
#define IGFITTERONCOLUMNS_H


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


#include "rowFamily.h"
#include "myutils.h"

#include "dsListC.h"


/*
 * MODELS A DENSE DATASET
 *
 * Acronyms:
 * ULL : unnorm. log likelihood
 * TS  : Thread safe
 */
class igFitterOnColumns {


public:


  /*
   * There are dDense dense matrices.
   *   dSparse sparse matrices.
   *
   * For any index d>=0, if dDense >=d< (dDense+dSparse),
   * we retrive sparseMats[ d-dDense ]
   *
   * If 0<=d < dDense, we return
   */
  const std::vector<arma::mat>& denseMats;
  const std::vector<arma::sp_mat>& sparseMats;
  unsigned int dDense;
  unsigned int dSparse;
  unsigned int D; //D =dDense + dSparse
  const arma::uvec& columns1;

  // //initial conditions
  // std::vector<arma::mat> init_Zeta;                       //Coefficients for Z, Bxp
  // arma::mat init_hmat;                          // h=[h1, h2, hK], h is H x K
  // arma::mat init_umat;                       //Factors U with sample loadings. R^{ nxK }
  // std::vector<arma::mat> init_vmat;                       //Feature scores for factors. R^{pxK}
  // std::vector<arma::vec> init_aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  // std::vector<arma::vec> init_colOffsets;                 //offsets for samples, row vector.
  // std::vector<arma::vec> init_rowOffsets;                 //row-wise offsets.


  //these get updatetd to the most recent call on estimate()
  double last_f1_estimatedWith;
  double last_f2_estimatedWith;



  //modeling
  const std::vector<std::string>& family;                                     //poisson or gaussian
  std::vector<rowFamily<arma::mat>*> denseRowFams;                     //pointer to a rowFamily object
  std::vector<rowFamily<arma::sp_mat>*> sparseRowFams;               //pointer to a rowFamily object
  const std::vector<arma::mat>& Z;                         //Design matrix for batches. nxB
  std::vector<arma::mat> Zeta;                       //Coefficients for Z, Bxp
  const arma::mat& M;                          //U=Mh, M is n x H
  arma::mat hmat;                          // h=[h1, h2, hK], h is H x K
  arma::mat umat;                       //Factors U with sample loadings. R^{ nxK }
  std::vector<arma::mat> vmat;                       //Feature scores for factors. R^{pxK}
  std::vector<arma::vec> aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  const unsigned int K;                                //Number of factors K
  const std::vector<arma::vec>& colOffsets;                 //offsets for samples, row vector.
  const std::vector<arma::vec>& rowOffsets;                 //row-wise offsets.
  std::vector<arma::vec> colOffsets_for_columns;

  const unsigned int r;                                // length(d) in Sd+Rc=Mh, where M=[S;R] , h=[d,c]'
  const arma::vec& wd;
  const unsigned int q;                                //length(c) in above
  const arma::uvec qids;
  arma::uvec rows_for_qxq;
  arma::uvec cols_for_qxq;
  //unsigned int m;                                // dim(Q) but I don't think we need it for proj grad desc.

  const arma::mat& Rtilde;                    //qxq penalty matrix. RtilTheta = R[qids,]lower left block portion of M, with row and corresponding column index r+(1+qids), r+(1+qids).
  //  Subsetting on the arma end is harder because qids are non-contiguous. So
  //  pass from the R side.
  const arma::mat B;                         // qxq; this is full Kernel matrix K(.,.)^{-1} for ortho use, where K=dphi phi^T + R defined over qids.
  const arma::mat Q;                          //n-dimensional. Any n-dimensional vectors for which we need to be orthogonal to.
  arma::mat BQ;


  unsigned int n;                                //number of samples/observations
  std::vector<unsigned int> pd;                                //number of features
  //unsigned int spd;
  double alpha0, rho;
  int inner_maxIter = 25;
  double solved_inner_ftol_abs = 1e-4; // solved-objective scale; set to n * inner_ftol_abs for generalized fitters
  std::vector<arma::mat> Zt;                         //useful for intermediate computations
  std::vector<arma::mat> ZtZ;                        // especially for Gaussian case, instead of recalculating for every i via rowFam
  arma::mat Mt;
  arma::mat MtM;

  //some temp variables for calculations
  arma::uvec seqOfComponents;           //simply stores a vector of 1..K. Useful for intermediate compu.
  std::vector<arma::vec> gradObjh_k; //gradient for each dataset, d=1..D
  arma::vec netGradh_k;             //total gradient
  std::vector<arma::mat> hessObjh_k; //hessian for each dataset, d=1..D
  arma::mat netHessh_k;             //total hessian
  std::vector<arma::vec> plength_vectord;
  std::vector<arma::vec> nlength_vectord;
  arma::vec nlength_vector;

  //negative loglikelihoods
  std::vector<arma::vec> negLogLikesVecd;         //stores (weighted, in weightedcased) negative loglikelihoods!, not loglikelihoods!
  double ULL;                           //unnormalized full data average Log like, so 1/n included
  // double objectiveValue; // not saving this as this is not a property of dataset; it is a property of a fitter, and parmaeters f1 and f2.


  /*For debugging*/
  static constexpr std::size_t DEBUG_NCOLS = 15;
  bool debugTraceEnabled = true;
  int debugIteration = 0;
  std::vector<std::array<double, DEBUG_NCOLS>> debugTrace;

  static constexpr std::size_t H_DEBUG_NCOLS = 22;
  std::vector<std::array<double, H_DEBUG_NCOLS>> hDebugTrace;
  arma::mat getHDebugTrace() const;

  // Minimal diagnostic for the first outer-iteration/component-0 Newton system.
  static constexpr std::size_t H_NEWTON_DEBUG_NCOLS = 14;
  std::vector<std::array<double, H_NEWTON_DEBUG_NCOLS>> hNewtonDebugTrace;
  arma::mat getHNewtonDebugTrace() const;


  static constexpr std::size_t H_PROJECTION_DEBUG_NCOLS = 8;
  std::vector<std::array<double, H_PROJECTION_DEBUG_NCOLS>> hProjectionDebugTrace;
  arma::mat getHProjectionDebugTrace() const;
  
  
  //to avoid oversubscription in nested tbb workers, 
  // we make the inner hessian calculation for poisson serial. 
  // For gausian, we don't use such calculators
  //bool poissonHessSerial=true; we now do this within rowFamilyPoisson - leaving here
  // for future. 
  
  
  ///////////////////////////////
  /*
   * Functions used in estiation
   */
  ///////////////////////////////

  //aphi esetimator
  double calc_aphi_di(const int& d, const int& i,
                      const arma::mat& Zd,
                      const arma::mat& Zetad,
                      const arma::mat& vmatd,
                      const arma::mat& rowOffsetsd,
                      const arma::mat& colOffsetsd);
  void calc_aphi_d(const int& d);
  void update_aphi();


  /*
   * v_k=[v_k1', ... v_kD'] mover
   */

  //void move_v_k_i_withoutNormUkStrategy( int i, int k, arma::vec& vk, double f1, double f2 );
  //void move_v_k_withoutUnitNormConstraint(int k, double f1, double f2);
  double gradNegLogLL_v_dik( const int& d, const arma::vec& y_i,
                             const arma::vec& eta_ik,
                             const arma::vec& u_k, const double& v_ik,
                             const double& aphi_i);
  double hessNegLogLL_v_dik(const int& d, const arma::vec& y_i,
                            const arma::vec& eta_ik,
                            const arma::vec& u_k, const double& v_ik,
                            const double& aphi_i);

  void move_v_d_k_i_normUkStrategy( const int& d,
                               const int& i,
                               const int& k,
                               const double& dk, //=norm(u_k)
                               const arma::vec& u_k_normed, //=u_k / dk, set to zero if dk ~0
                               const arma::mat& Zd,
                               const arma::mat& Zetad,
                               const arma::mat& umatOther,
                               const arma::mat& vmatOtherd,
                               const arma::mat& vmatd,
                               const arma::vec& aphid,
                               const arma::vec& rowOffsetsd,
                               const arma::vec& colOffsetsd,
                               arma::vec& vk, //vector to output to
                               const double& f1,
                               const double& f2 );

  void move_v_d_k_withUnitNormConstraint_normUStrategy(const int& d,
                                                       const int& k,
                                                       const double& dk,
                                                       const arma::vec& u_k_normed,
                                                       arma::vec& vk,
                                                       const double& f1, const double& f2);

  void move_v_k_withUnitNormConstraint_normUStrategy(const int& k,
                                                     const double& f1, const double& f2);


  //h_k mover
  void move_h_k( const int& k, const double& f1,  const double& f2,
                 double alpha0=1,
                 double rho=.5
  );               //uk= M_k h_k, and h_k is moved for a given k=1..K


  //Zeta movers
  void moveZeta_di(const int& d, const int& i,
                   const arma::mat& Zd,
                   const arma::mat& Zetad,
                   const arma::mat& vmatd,
                   const arma::vec& aphid,
                   const arma::vec& rowOffsetsd,
                   const arma::vec& colOffsetsd,
                   const double& f1, const double& f2);
  void moveZeta_d(const int& d, const double& f1, const double& f2);                    //calls all moveZeta_i parallely
  void moveZeta(const double& f1, const double& f2);                    //calls all moveZeta_i parallely



  void moveVU(const double& f1, const double& f2);                      //calls each move_v_k (which in turn calls mov_v_k_i parallely) and h_k serially one after the other.
  void descend();


//public:

  //some getters
  std::vector<arma::mat> get_Zeta();
  arma::mat get_hmat();
  arma::mat get_umat();
  std::vector<arma::mat> get_vmat();
  arma::mat get_gmat();
  std::vector<arma::vec> get_aphi();
  arma::vec get_objective();
  std::vector<arma::mat> get_Zeta_R(); //use on R side (e.g., while wrapping), returns vector of p_d x |Zeta_d| matrices

  //init
  std::vector<arma::mat> get_init_Zeta();
  arma::mat get_init_hmat();
  arma::mat get_init_umat();
  std::vector<arma::mat> get_init_vmat();
  arma::mat get_init_gmat();
  std::vector<arma::vec> get_init_aphi();
  std::vector<arma::mat> get_init_Zeta_R(); //use on R side (e.g., while wrapping), returns vector of p_d x |Zeta_d| matrices



  //////////////////////////////////////////////////////////////
  /*
   * CONSTRUCTORS based on what information is used or not
   * For example: if there are no weights, offsets etc.,
   *        without weights, we can be faster, and we want to
   *        take advantage of that situation.
   */
  //////////////////////////////////////////////////////////////

  explicit igFitterOnColumns( //without Q
    const std::vector<arma::mat>& denseMats,
    const std::vector<arma::sp_mat>& sparseMats,
    const arma::uvec& columns1,
    const std::vector<std::string>& family,
    const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, std::vector<arma::mat> vmat,
    std::vector<arma::vec> aphi,
    const arma::mat& Rtilde,
    const arma::uvec qids,
    const arma::mat B,
    const std::vector<arma::vec>& rowOffsets,
    const std::vector<arma::vec>& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::vec& wd
  );

  explicit igFitterOnColumns( //without Q
    const std::vector<arma::mat>& denseMats,
    const std::vector<arma::sp_mat>& sparseMats,
    const arma::uvec& columns1,
    const std::vector<std::string>& family,
    const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, std::vector<arma::mat> vmat,
    std::vector<arma::vec> aphi,
    const arma::mat& Rtilde,
    const arma::uvec qids,
    const arma::mat B,
    const std::vector<arma::vec>& rowOffsets,
    const std::vector<arma::vec>& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::vec& wd,
    const arma::mat& Q
  );

  explicit igFitterOnColumns( //without Q
      const dsListC& Y,
      const arma::uvec& columns1,
      const std::vector<std::string>& family,
      const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
      const arma::mat& M, arma::mat hmat,
      arma::mat umat, std::vector<arma::mat> vmat,
      std::vector<arma::vec> aphi,
      const arma::mat& Rtilde,
      const arma::uvec qids,
      const arma::mat B,
      const std::vector<arma::vec>& rowOffsets,
      const std::vector<arma::vec>& colOffsets,
      const unsigned int K,
      const unsigned int r,
      const arma::vec& wd
  );

  explicit igFitterOnColumns( //without Q
      const dsListC& Y,
      const arma::uvec& columns1,
      const std::vector<std::string>& family,
      const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
      const arma::mat& M, arma::mat hmat,
      arma::mat umat, std::vector<arma::mat> vmat,
      std::vector<arma::vec> aphi,
      const arma::mat& Rtilde,
      const arma::uvec qids,
      const arma::mat B,
      const std::vector<arma::vec>& rowOffsets,
      const std::vector<arma::vec>& colOffsets,
      const unsigned int K,
      const unsigned int r,
      const arma::vec& wd,
      const arma::mat& Q
  );

  void initializeOthers();

  //~igFitterOnColumns();
  ~igFitterOnColumns()=default;

  //////////////////////////////////////////////////////////////
  /*
   * NegativeLogLike calculations for individual feature & full dataset
   */
  //////////////////////////////////////////////////////////////
  double calcNegLogLL_di(const int& d, const int& i );
  double calcNegLogLL_di(const int& d, const arma::vec& y_di,
                        const arma::vec& eta_di,
                        const double& aphi_di);  //for the ith feature, calculates f1*negll
  double calcNegLogLL_di(const int& d, const int& i,
                  const arma::mat& Zd, const arma::mat& Zetad,
                  const arma::mat& umat, const arma::mat& vmatd,
                  const arma::vec& aphid,
                  const arma::vec& rowOffsetsd, const arma::vec& colOffsetsd
                    );

  void calcNegLogLL_d(const int& d, arma::vec& negLogLikes_forEach_i_in_d
                      ); // fora. given dataset
  double calcNegLogLL(std::vector<arma::vec>& negLogLikes_forEach_di ,
                      const bool updateCurrent=true
                        //this is a p-length vector for each ftr
                        );//for the entire dataset

  //for K=1
  double calcNegLogLL_d_j_with_h_k( //compute j^th samples loglike with a candidate h_k and rest fixed
      const int& d,
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );
  double calcNegLogLL_d_j_with_h_k( //compute j^th samples loglike with a candidate h_k and rest fixed
      const int& d,
      const arma::mat& umatOthers, //columns other than the k^th
      const std::vector<arma::mat>& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );
  //for K=1
  double calcNegLogLL_j_with_h_k( //compute j^th samples loglike with a candidate h_k and rest fixed
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );
  double calcNegLogLL_j_with_h_k( //compute j^th samples loglike with a candidate h_k and rest fixed
      const arma::mat& umatOthers, //columns other than the k^th
      const std::vector<arma::mat>& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );

  //forK=1
  double calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const int& d,
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const arma::vec& v_kd,      // the corresponding feature loadings vector already fixed
      const arma::mat& Zd,
      const arma::mat& Zetad,
      const arma::vec& aphid,
      const arma::vec& rowOffsetsd,
      const arma::vec& colOffsetsd,
      const int j
  );
  double calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
      const int& d,
      const arma::mat& umatOthers, //columns other than the k^th
      const arma::mat& vmatOthersd, //columns other than the k^th in one to one correspondence with umatOther.
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const arma::vec& v_kd,      // the corresponding feature loadings vector already fixed
      const arma::mat& Zd,
      const arma::mat& Zetad,
      const arma::vec& aphid,
      const arma::vec& rowOffsetsd,
      const arma::vec& colOffsetsd,
      const int j
  );

  //for K=1
  double calcNegLogLL_d_with_h_k(
      const int& d,
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      std::vector<arma::vec>& nlength_vectord_tmp
  );
  double calcNegLogLL_d_with_h_k(
      const int& d,
      const arma::mat& umatOthers,
      const std::vector<arma::mat>& vmatOthers,
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      std::vector<arma::vec>& nlength_vectord_tmp
  );

  //for K =1
  double calcNegLogLL_with_h_k( //compute all samples' loglike with a candidate h_k and rest fixed
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k
  );
  double calcNegLogLL_with_h_k( //compute all samples' loglike with a candidate h_k and rest fixed
      const arma::mat& umatOthers,
      const std::vector<arma::mat>& vmatOthers,
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k
  );


  //////////////////////////////////////////////////////////////
  /*
   * Gradient and Hessians for Zeta, the loading v_ik i.e.,
   *    ith feature in kth component, and h_k.
   */
  //////////////////////////////////////////////////////////////
  arma::vec gradNegLogLL_Zeta_di( const int& d, const arma::vec& y_di,
                                 const arma::vec& eta_di,
                                 const arma::vec& zeta_di,
                                 const double& aphi_di
  ); // col vector
  arma::mat hessNegLogLL_Zeta_di( const int& d, const arma::vec& y_di,
                                 const arma::vec& eta_di,
                                 const arma::vec& zeta_di,
                                 const double& aphi_di
  );

  void gradNegLogLL_d_h_k(const int& d, const int& k, const arma::vec& h_k,
                        std::vector<arma::vec>& grad_d_h_k);  //pulling these two out in public; partly also because this enters integrative code.
  void gradNegLogLL_h_k(const int& k, const arma::vec& h_k,
                        std::vector<arma::vec>& grad_d_h_k,
                        arma::vec& netGrad_h_k
                        );  //pulling these two out in public; partly also because this enters integrative code.

  void hessNegLogLL_d_h_k(const int& d, const int& k, const arma::vec& h_k,
                        std::vector<arma::mat>& hess_d_h_k
  );
  void hessNegLogLL_h_k(const int& k, const arma::vec& h_k,
                        std::vector<arma::mat>& hess_d_h_k,
                        arma::mat& netHessh_k
                          );

  // overloaded forms : useful further below in gradient moves
  //for K=1
  void gradNegLogLL_d_h_k(
      const int& d,
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_d_h_k  //to store gradient in
  );
  void gradNegLogLL_d_h_k(
      const int& d,
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_d_h_k  //to store gradient in
  );
  //for K=1
  void gradNegLogLL_h_k(
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      std::vector<arma::vec>& grad_h_k,  //to store gradient in
      arma::vec& netGrad_h_k
  );
  void gradNegLogLL_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      std::vector<arma::vec>& grad_h_k,  //to store gradient in
      arma::vec& netGrad_h_k
  );

  // for K=1
  void hessNegLogLL_d_h_k(
      const int& d,
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& v_k,       // corresponding loading vector
      arma::mat& hess_h_k  //to store gradient in
  );
  void hessNegLogLL_d_h_k(
      const int& d,
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& v_k,       // corresponding loading vector
      arma::mat& hess_h_k  //to store gradient in
  );
  //for K=1
  void hessNegLogLL_h_k(
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      std::vector<arma::mat>& hess_h_k,  //to store gradient in
      arma::mat& netHess_h_k
  );
  void hessNegLogLL_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      std::vector<arma::mat>& hess_h_k,  //to store gradient in
      arma::mat& netHess_h_k
  );

  //////////////////////////////////////////////////////////////
  /*
   * Objective stuff
   */
  //////////////////////////////////////////////////////////////
  void gradPenalizedObjective_h_k(
      const int& k,
      const arma::vec& h_k,
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      arma::vec& netGradh_k,
      const double& f1,
      const double& f2
  );

  void hessPenalizedObjectiveLL_h_k(
      const int& k,
      const arma::vec& h_k,
      arma::mat& netHessh_k,
      const double& f1,
      const double& f2
  );

  // overloaded forms of the two forms above
  //for K=1
  void gradPenalizedObjective_h_k(
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2
  );
  void gradPenalizedObjective_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2
  );
  //for K=1
  void hessPenalizedObjectiveLL_h_k(
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2
  );
  void hessPenalizedObjectiveLL_h_k(
      const arma::mat& umatOthers,
      const std::vector<arma::mat>& vmatOthers,
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2
  );

  double getPenalty();
  double objective(const bool recalculate, const bool updateStored, const double& f1, const double& f2);

  //////////////////////////////////////////////////////////////
  /*
   *  ESTIMATOR
   */
  //////////////////////////////////////////////////////////////
  // void estimateNP(double f1, double f2, //the nonparametric PCA-like form for a single dataset
  //               int maxIter=25, double eps=1e-4);

  void estimate(const double& f1, const double& f2,
                const int& maxIter=25, const double& eps=1e-4,
                const double& xtol_abs=.01,
                const int& inner_maxIter=25,
                const double& inner_ftol_abs=1e-4);


  void enableDebugTrace();
  void recordDebugTrace(
      const int stage,
      const int component,
      const double f2
  );
  arma::mat getDebugTrace() const;


  //MSE AND DEVIANCE AND OBJECTIVE
  void deviance_d(const int& d, std::vector<arma::vec>& nlength_vectord_tmp);
  double deviance();

  double devianceForTest_d(
      const int& d,
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );
  double devianceForTest(
      const arma::uvec& columns1Test,
      const std::vector<arma::mat>& ZTest,
      const arma::mat& MTest,
      const std::vector<arma::vec>& rowOffsetsTest,
      const std::vector<arma::vec>& colOffsetsTest
  );

  void mse_d(const int& d, std::vector<arma::vec>& nlength_vectord_tmp);
  double mse();

  double mseForTest_d(
      const int& d,
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );
  double mseForTest(
      const arma::uvec& columns1Test,
      const std::vector<arma::mat>& ZTest,
      const arma::mat& MTest,
      const std::vector<arma::vec>& rowOffsetsTest,
      const std::vector<arma::vec>& colOffsetsTest
  );

  double objectiveForTest_d(
      const int& d,
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );
  double objectiveForTest(
      const arma::uvec& columns1Test,
      const std::vector<arma::mat>& ZTest,
      const arma::mat& MTest,
      const std::vector<arma::vec>& rowOffsetsTest,
      const std::vector<arma::vec>& colOffsetsTest
  );
};


#endif
