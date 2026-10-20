#ifndef IGFITTERONCOLUMNSGRAPH_H
#define IGFITTERONCOLUMNSGRAPH_H


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


#include "igFitterOnColumns.h"


/*
 * MODELS A DENSE DATASET
 *
 * Acronyms:
 * ULL : unnorm. log likelihood
 * TS  : Thread safe
 */
class igFitterOnColumnsGraph final : public igFitterOnColumns {


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
  using igFitterOnColumns::denseMats;
  using igFitterOnColumns::sparseMats;
  using igFitterOnColumns::dDense;
  using igFitterOnColumns::dSparse;
  using igFitterOnColumns::D; //D =dDense + dSparse
  using igFitterOnColumns::columns1;

  //initial conditions
  // using igFitterOnColumns::init_Zeta;                       //Coefficients for Z, Bxp
  // using igFitterOnColumns::init_hmat;                          // h=[h1, h2, hK], h is H x K
  // using igFitterOnColumns::init_umat;                       //Factors U with sample loadings. R^{ nxK }
  // using igFitterOnColumns::init_vmat;                       //Feature scores for factors. R^{pxK}
  // using igFitterOnColumns::init_aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  // using igFitterOnColumns::init_colOffsets;                 //offsets for samples, row vector.
  // using igFitterOnColumns::init_rowOffsets;                 //row-wise offsets.


  //these get updatetd to the most recent call on estimate()
  using igFitterOnColumns::last_f1_estimatedWith;
  using igFitterOnColumns::last_f2_estimatedWith;
  double last_f3_estimatedWith;


  //modeling
  using igFitterOnColumns::family;                                     //poisson or gaussian
  using igFitterOnColumns::denseRowFams;                     //pointer to a rowFamily object
  using igFitterOnColumns::sparseRowFams;               //pointer to a rowFamily object
  using igFitterOnColumns::Z;                         //Design matrix for batches. nxB
  using igFitterOnColumns::Zeta;                       //Coefficients for Z, Bxp
  using igFitterOnColumns::M;                          //U=Mh, M is n x H
  using igFitterOnColumns::hmat;                          // h=[h1, h2, hK], h is H x K
  using igFitterOnColumns::umat;                       //Factors U with sample loadings. R^{ nxK }
  using igFitterOnColumns::vmat;                       //Feature scores for factors. R^{pxK}
  using igFitterOnColumns::aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  using igFitterOnColumns::K;                                //Number of factors K
  using igFitterOnColumns::colOffsets;                 //offsets for samples, row vector.
  using igFitterOnColumns::rowOffsets;                 //row-wise offsets.
  using igFitterOnColumns::colOffsets_for_columns;

  using igFitterOnColumns::r;                                // length(d) in Sd+Rc=Mh, where M=[S;R] , h=[d,c]'
  using igFitterOnColumns::q;                                //length(c) in above
  using igFitterOnColumns::qids;
  using igFitterOnColumns::rows_for_qxq;
  using igFitterOnColumns::cols_for_qxq;
  //unsigned int m;                                // dim(Q) but I don't think we need it for proj grad desc.

  using igFitterOnColumns::Rtilde;                    //qxq penalty matrix. RtilTheta = R[qids,]lower left block portion of M, with row and corresponding column index r+(1+qids), r+(1+qids).

  const arma::sp_mat& graphDistMat; //qxq graph dist matrix, must be pos semi def/pos def
  const arma::uvec& graphDistOnlyOnPoints;    //the graph distances are defined for these points, in the order
  //  Subsetting on the arma end is harder because qids are non-contiguous. So
  //  pass from the R side.

  using igFitterOnColumns::B;                         // qxq; this is full Kernel matrix K(.,.)^{-1} for ortho use, where K=dphi phi^T + R defined over qids.
  using igFitterOnColumns::Q;                          //n-dimensional. Any n-dimensional vectors for which we need to be orthogonal to.
  using igFitterOnColumns::BQ;


  using igFitterOnColumns::n;                                //number of samples/observations
  using igFitterOnColumns::pd;                                //number of features
  //unsigned int spd;
  using igFitterOnColumns::alpha0;
  using igFitterOnColumns::rho;
  using igFitterOnColumns::Zt;                         //useful for intermediate computations
  using igFitterOnColumns::ZtZ;                        // especially for Gaussian case, instead of recalculating for every i via rowFam
  using igFitterOnColumns::Mt;
  using igFitterOnColumns::MtM;

  arma::mat G; //holds Mq'graphDistMat Mq

  //some temp variables for calculations
  using igFitterOnColumns::seqOfComponents;           //simply stores a vector of 1..K. Useful for intermediate compu.
  using igFitterOnColumns::gradObjh_k; //gradient for each dataset, d=1..D
  using igFitterOnColumns::netGradh_k;             //total gradient
  using igFitterOnColumns::hessObjh_k; //hessian for each dataset, d=1..D
  using igFitterOnColumns::netHessh_k;             //total hessian
  using igFitterOnColumns::plength_vectord;
  using igFitterOnColumns::nlength_vectord;
  using igFitterOnColumns::nlength_vector;

  //negative loglikelihoods
  using igFitterOnColumns::negLogLikesVecd;         //stores (weighted, in weightedcased) negative loglikelihoods!, not loglikelihoods!
  using igFitterOnColumns::ULL;                           //unnormalized full data Log like
  // double objectiveValue; // not saving this as this is not a property of dataset; it is a property of a fitter, and parmaeters f1 and f2.

  ///////////////////////////////
  /*
   * Functions used in estiation
   */
  ///////////////////////////////

  //aphi esetimator
  using igFitterOnColumns::calc_aphi_di;
  using igFitterOnColumns::calc_aphi_d;
  using igFitterOnColumns::update_aphi;


  //Zeta movers
  using igFitterOnColumns::moveZeta_di;
  using igFitterOnColumns::moveZeta_d;                    //calls all moveZeta_i parallely
  using igFitterOnColumns::moveZeta;                    //calls all moveZeta_i parallely

  /*
   * v_k=[v_k1', ... v_kD'] mover
   */

  using igFitterOnColumns::gradNegLogLL_v_dik;
  using igFitterOnColumns::hessNegLogLL_v_dik;
  using igFitterOnColumns::move_v_d_k_i_normUkStrategy;
  using igFitterOnColumns::move_v_d_k_withUnitNormConstraint_normUStrategy;
  using igFitterOnColumns::move_v_k_withUnitNormConstraint_normUStrategy;

  //h_k mover
  void move_h_k( const int& k, const double& f1,  const double& f2,const double& f3,
                 double alpha0=1,
                 double rho=.5
  );               //uk= M_k h_k, and h_k is moved for a given k=1..K
  void moveVU(const double& f1, const double& f2,const double& f3);                      //calls each move_v_k (which in turn calls mov_v_k_i parallely) and h_k serially one after the other.
  void descend();


//public:

  //some getters
  using igFitterOnColumns::get_Zeta;
  using igFitterOnColumns::get_hmat;
  using igFitterOnColumns::get_umat;
  using igFitterOnColumns::get_vmat;
  using igFitterOnColumns::get_gmat;
  using igFitterOnColumns::get_aphi;
  using igFitterOnColumns::get_objective;
  using igFitterOnColumns::get_Zeta_R; //use on R side (e.g., while wrapping), returns vector of p_d x |Zeta_d| matrices

  //init
  // std::vector<arma::mat> get_init_Zeta();
  // arma::mat get_init_hmat();
  // arma::mat get_init_umat();
  // std::vector<arma::mat> get_init_vmat();
  // arma::mat get_init_gmat();
  // std::vector<arma::vec> get_init_aphi();
  // std::vector<arma::mat> get_init_Zeta_R(); //use on R side (e.g., while wrapping), returns vector of p_d x |Zeta_d| matrices



  //////////////////////////////////////////////////////////////
  /*
   * CONSTRUCTORS based on what information is used or not
   * For example: if there are no weights, offsets etc.,
   *        without weights, we can be faster, and we want to
   *        take advantage of that situation.
   */
  //////////////////////////////////////////////////////////////

  explicit igFitterOnColumnsGraph( //without Q
    const std::vector<arma::mat>& denseMats,
    const std::vector<arma::sp_mat>& sparseMats,
    const arma::uvec& columns1,
    const std::vector<std::string>& family,
    const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, std::vector<arma::mat> vmat,
    std::vector<arma::vec> aphi,
    const arma::mat& Rtilde,
    const arma::sp_mat& graphDistMat,
    const arma::uvec& graphDistOnlyOnPoints,
    const arma::uvec qids,
    const arma::mat B,
    const std::vector<arma::vec>& rowOffsets,
    const std::vector<arma::vec>& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::vec& wd
  );

  explicit igFitterOnColumnsGraph( //without Q
    const std::vector<arma::mat>& denseMats,
    const std::vector<arma::sp_mat>& sparseMats,
    const arma::uvec& columns1,
    const std::vector<std::string>& family,
    const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, std::vector<arma::mat> vmat,
    std::vector<arma::vec> aphi,
    const arma::mat& Rtilde,
    const arma::sp_mat& graphDistMat,
    const arma::uvec& graphDistOnlyOnPoints,
    const arma::uvec qids,
    const arma::mat B,
    const std::vector<arma::vec>& rowOffsets,
    const std::vector<arma::vec>& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::vec& wd,
    const arma::mat& Q
  );

  explicit igFitterOnColumnsGraph( //without Q
      const dsListC& Y,
      const arma::uvec& columns1,
      const std::vector<std::string>& family,
      const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
      const arma::mat& M, arma::mat hmat,
      arma::mat umat, std::vector<arma::mat> vmat,
      std::vector<arma::vec> aphi,
      const arma::mat& Rtilde,
      const arma::sp_mat& graphDistMat,
      const arma::uvec& graphDistOnlyOnPoints,
      const arma::uvec qids,
      const arma::mat B,
      const std::vector<arma::vec>& rowOffsets,
      const std::vector<arma::vec>& colOffsets,
      const unsigned int K,
      const unsigned int r,
      const arma::vec& wd
  );

  explicit igFitterOnColumnsGraph( //without Q
      const dsListC& Y,
      const arma::uvec& columns1,
      const std::vector<std::string>& family,
      const std::vector<arma::mat>& Z, std::vector<arma::mat> Zeta,
      const arma::mat& M, arma::mat hmat,
      arma::mat umat, std::vector<arma::mat> vmat,
      std::vector<arma::vec> aphi,
      const arma::mat& Rtilde,
      const arma::sp_mat& graphDistMat,
      const arma::uvec& graphDistOnlyOnPoints,
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

  //~igFitterOnColumnsGraph();
  ~igFitterOnColumnsGraph()=default;

  //////////////////////////////////////////////////////////////
  /*
   * NegativeLogLike calculations for individual feature & full dataset
   */
  //////////////////////////////////////////////////////////////
  using igFitterOnColumns::calcNegLogLL_di;
  using igFitterOnColumns::calcNegLogLL_d;
  using igFitterOnColumns::calcNegLogLL;

  using igFitterOnColumns::calcNegLogLL_d_j_with_h_k;
  using igFitterOnColumns::calcNegLogLL_j_with_h_k;
  using igFitterOnColumns::calcNegLogLL_d_with_h_k;
  using igFitterOnColumns::calcNegLogLL_with_h_k;


  //////////////////////////////////////////////////////////////
  /*
   * Gradient and Hessians for Zeta, the loading v_ik i.e.,
   *    ith feature in kth component, and h_k.
   */
  //////////////////////////////////////////////////////////////
  using igFitterOnColumns::gradNegLogLL_Zeta_di;
  using igFitterOnColumns::hessNegLogLL_Zeta_di;

  using igFitterOnColumns::gradNegLogLL_d_h_k;  //pulling these two out in public; partly also because this enters integrative code.
  using igFitterOnColumns::hessNegLogLL_d_h_k;

  using igFitterOnColumns::gradNegLogLL_h_k;  //pulling these two out in public; partly also because this enters integrative code.
  using igFitterOnColumns::hessNegLogLL_h_k;

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
      const double& f2,
      const double& f3
  );

  void hessPenalizedObjectiveLL_h_k(
      const int& k,
      const arma::vec& h_k,
      arma::mat& netHessh_k,
      const double& f1,
      const double& f2,
      const double& f3
  );

  // overloaded forms of the two forms above
  //for K=1
  void gradPenalizedObjective_h_k(
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2,
      const double& f3
  );
  void gradPenalizedObjective_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const std::vector<arma::vec>& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2,
      const double& f3
  );
  //for K=1
  void hessPenalizedObjectiveLL_h_k(
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );
  void hessPenalizedObjectiveLL_h_k(
      const arma::mat& umatOthers,
      const std::vector<arma::mat>& vmatOthers,
      const arma::vec& h_k,
      const std::vector<arma::vec>& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );

  using igFitterOnColumns::getPenalty;
  double getPenaltyForGraph();
  double objective(const bool recalculate, const bool updateStored,
                   const double& f1, const double& f2, const double& f3);

  //////////////////////////////////////////////////////////////
  /*
   *  ESTIMATOR
   */
  //////////////////////////////////////////////////////////////
  // void estimateNP(double f1, double f2, //the nonparametric PCA-like form for a single dataset
  //               int maxIter=25, double eps=1e-4);

  void estimate(const double& f1, const double& f2,const double& f3,
                const int& maxIter=25, const double& eps=1e-4,
                const double& xtol_abs=.01,
                const int& inner_maxIter=25,
                const double& inner_ftol_abs=1e-4);


  //MSE AND DEVIANCE AND OBJECTIVE
  using igFitterOnColumns::deviance_d;
  using igFitterOnColumns::deviance;

  using igFitterOnColumns::devianceForTest_d;
  using igFitterOnColumns::devianceForTest;

  using igFitterOnColumns::mse_d;
  using igFitterOnColumns::mse;

  using igFitterOnColumns::mseForTest_d;
  using igFitterOnColumns::mseForTest;

  using igFitterOnColumns::objectiveForTest_d; // bad name, this just calculaes LL for dth dataset in base class, not objective.
  double objectiveForTest(
      const arma::uvec& columns1Test,
      const std::vector<arma::mat>& ZTest,
      const arma::mat& MTest,
      const std::vector<arma::vec>& rowOffsetsTest,
      const std::vector<arma::vec>& colOffsetsTest
  );
};


#endif
