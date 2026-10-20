#ifndef GFITTERONCOLUMNS_H
#define GFITTERONCOLUMNS_H


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
#include<limits>
#include<random>
#include<string>
#include<cmath>


#include "rowFamily.h"
#include "myutils.h"


/*
 * MODELS A DENSE DATASET
 *
 * Acronyms:
 * ULL : unnorm. log likelihood
 * TS  : Thread safe
 */
template<typename T>
class gFitterOnColumns {


public:
  
  const T& Y;                         //Data object; pxn.We start with a dense dataset
  const arma::uvec& columns1;

  //these get updaetd to the most recent call on estimate()
  double last_f1_estimatedWith;
  double last_f2_estimatedWith;

  //modeling
  const std::string family;                   //poisson or gaussian
  rowFamily<T>* rowFam;                     //pointer to a rowFamily object
  const arma::mat& Z;                         //Design matrix for batches. nxB
  arma::mat Zeta;                       //Coefficients for Z, Bxp
  const arma::mat& M;                          //U=Mh, M is n x H
  arma::mat hmat;                          // h=[h1, h2, hK], h is H x K
  arma::mat umat;                       //Factors U with sample loadings. R^{ nxK }
  arma::mat vmat;                       //Feature scores for factors. R^{pxK}
  arma::vec aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  const unsigned int K;                                //Number of factors K
  const arma::vec& colOffsets;                 //offsets for samples, row vector.
  const arma::vec& rowOffsets;                 //row-wise offsets.
  arma::vec colOffsets_for_columns; //stores colOffests corresponding to desired columns

  const unsigned int r;                                // length(d) in Sd+Rc=Mh, where M=[S;R] , h=[d,c]'
  const unsigned int q;                                //length(c) in above
  unsigned int m;                                // dim(Q) but I don't think we need it for proj grad desc.
  const arma::uvec qids;
  //std::string knotsType;                //"approx" or "full"
  arma::uvec rows_for_qxq;
  arma::uvec cols_for_qxq;
  //arma::mat& Rtheta;                    //nxq matrix R entering S d + R c, with M=[S; R]
  const arma::mat& Rtilde;                    //qxq penalty matrix. RtilTheta = R[qids,]lower left block portion of M, with row and corresponding column index r+(1+qids), r+(1+qids).
  //  Subsetting on the arma end is harder because qids are non-contiguous. So
  //  pass from the R side.
  const arma::mat B;                         // qxq; this is full Kernel matrix K(.,.)^{-1} for ortho use, where K=dphi phi^T + R defined over qids.
  const arma::mat Q;                          //n-dimensional. Any n-dimensional vectors for which we need to be orthogonal to.
  arma::mat BQ;


  unsigned int n;                                //number of samples/observations
  unsigned int p;                                //number of features
  double alpha0, rho;
  arma::vec hAlphaStart;  // per-component warm start for h_k line searches
  int inner_maxIter = 25;
  double solved_inner_ftol_abs = 1e-4; // solved-objective scale; set to n * inner_ftol_abs for generalized fitters
  arma::mat Zt;                         //useful for intermediate computations
  arma::mat ZtZ;                        // especially for Gaussian case, instead of recalculating for every i via rowFam
  arma::mat Mt;
  arma::mat MtM;

  //some temp variables for calculations
  arma::uvec seqOfComponents;           //simply stores a vector of 1..K. Useful for intermediate compu.
  arma::vec gradObjh_k;
  arma::mat hessObjh_k;
  arma::vec plength_vector;
  arma::vec nlength_vector;

  //negative loglikelihoods
  arma::vec negLogLikesVec;         //stores (weighted, in weightedcased) negative loglikelihoods!, not loglikelihoods!
  double ULL;                           //unnormalized full data Log like
  // double objectiveValue; // not saving this as this is not a property of dataset; it is a property of a fitter, and parmaeters f1 and f2.


  //to avoid oversubscription in nested tbb workers, 
  // we make the inner hessian calculation for poisson serial. 
  // For gausian, we don't use such calculators
  //bool poissonHessSerial=true; we now do this within rowFamilyPoisson - leaving here
  // for future. 
  
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
  
  
  ///////////////////////////////
  /*
   * Functions used in estiation
   */
  ///////////////////////////////
  double calc_aphi_i(const int i);
  void update_aphi();
  void moveZeta_i(const int i, const double& f1, const double& f2);
  double gradNegLogLL_v_ik(const arma::vec& y_i,
                           const arma::vec& eta_ik,
                           const arma::vec& u_k, const double& v_ik,
                           const double& aphi_i);
  double hessNegLogLL_v_ik(const arma::vec& y_i,
                           const arma::vec& eta_ik,
                           const arma::vec& u_k, const double& v_ik,
                           const double& aphi_i);


  //void move_v_k_i_withoutNormUkStrategy( int i, int k, arma::vec& vk, double f1, double f2 );
  void move_v_k_i_normUkStrategy( const int i, const int k, arma::vec& vk, const double& f1, const double& f2 );
  void move_v_k_withUnitNormConstraint_normUStrategy(const int k, const double& f1, const double& f2);
  //void move_v_k_withoutUnitNormConstraint(int k, double f1, double f2);

  void move_v_k_i( const int i, const int k, const double& f1, const double& f2 );
  void move_v_k(const int k, const double& f1, const double& f2);               //same as above but for an entire v
  void move_h_k( const int k, const double& f1,  const double& f2,
                 double alpha0=1,
                 double rho=.5
  );               //projected/damped Newton update for h_k

  void move_h_k_rgd( const int k, const double& f1,  const double& f2,
                     double alpha0=1,
                     double rho=.5
  );               //retained Riemannian/projected-gradient update
  void moveZeta(const double& f1, const double& f2);                    //calls all moveZeta_i parallely
  void moveVU(const double& f1, const double& f2);                      //calls each move_v_k (which in turn calls mov_v_k_i parallely) and h_k serially one after the other.
  void descend();
  
  

//public:

  //some getters
  arma::mat get_Zeta();
  arma::mat get_hmat();
  arma::mat get_umat();
  arma::mat get_vmat();
  arma::mat get_gmat();
  arma::vec get_aphi();
  arma::vec get_objective();


  //////////////////////////////////////////////////////////////
  /*
   * CONSTRUCTORS based on what information is used or not
   * For example: if there are no weights, offsets etc.,
   *        without weights, we can be faster, and we want to
   *        take advantage of that situation.
   */
  //////////////////////////////////////////////////////////////

  

  gFitterOnColumns( //without Q
    const T& Y,
    const arma::uvec& columns1,
    const std::string family,
    const arma::mat& Z, arma::mat Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, arma::mat vmat,
    arma::vec aphi,
    const arma::mat& Rtilde,
    const arma::uvec qids,
    const arma::mat B,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const unsigned int K,
    const unsigned int r
  );

  gFitterOnColumns( //with Q
    const T& Y,
    const arma::uvec& columns1,
    const std::string family,
    const arma::mat& Z, arma::mat Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, arma::mat vmat,
    arma::vec aphi,
    const arma::mat& Rtilde,
    const arma::uvec qids,
    const arma::mat B,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::mat Q
  );

  ~gFitterOnColumns();

  //////////////////////////////////////////////////////////////
  /*
   * NegativeLogLike calculations for individual feature & full dataset
   */
  //////////////////////////////////////////////////////////////
  double calcNegLogLL_i(const int i );

  double calcNegLogLL_i(const arma::vec& y_i,
                        const arma::vec& eta_i,
                        const double& aphi_i);  //for the ith feature, calculates f1*negll

  double calcNegLogLL(arma::vec& negLogLikes_forEach_i ,
                      const bool updateCurrent=true
                      //this is a p-length vector for each ftr

  );                               //for the entire dataset
  double calcNegLogLL_j_with_h_k(// for K =1, //compute j^th samples loglike with a candidate h_k and rest fixed
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );
  double calcNegLogLL_j_with_h_k( //compute j^th samples loglike with a candidate h_k and rest fixed
      const arma::mat& umatOthers, //columns other than the k^th
      const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
      const arma::vec& h_k,      //the h that will determine u_k = M h_k
      const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
      const int j
  );
  double calcNegLogLL_with_h_k( //for K =1, //compute all samples' loglike with a candidate h_k and rest fixed
      const arma::vec& h_k,
      const arma::vec& v_k
  );
  double calcNegLogLL_with_h_k( //compute all samples' loglike with a candidate h_k and rest fixed
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& h_k,
      const arma::vec& v_k
  );


  //////////////////////////////////////////////////////////////
  /*
   * Gradient and Hessians for Zeta, the loading v_ik i.e.,
   *    ith feature in kth component, and h_k.
   */
  //////////////////////////////////////////////////////////////
  arma::vec gradNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::vec& zeta_i,
                                 const double& aphi_i
  ); // col vector
  arma::mat hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                 const arma::vec& eta_i,
                                 const arma::vec& zeta_i,
                                 const double& aphi_i
  );

  void gradNegLogLL_h_k(const int k, const arma::vec& h_k,
                        arma::vec& grad_h_k);  //pulling these two out in public; partly also because this enters integrative code.
  void hessNegLogLL_h_k(const int k, const arma::vec& h_k,
                        arma::mat& hess_h_k
  );


  //////////////////////////////////////////////////////////////
  /*
   * Objective stuff
   */
  //////////////////////////////////////////////////////////////
  void gradPenalizedObjective_h_k(
      const int k,
      const arma::vec& h_k,
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      arma::vec& grad_h_k,
      const double& f1,
      const double& f2
  );

  void hessPenalizedObjectiveLL_h_k(
      const int k,
      const arma::vec& h_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2
  );
  // overloaded forms of the two forms above
  void gradPenalizedObjective_h_k( // for K = 1
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2
  );
  void gradPenalizedObjective_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2
  );
  void hessPenalizedObjectiveLL_h_k( //for K =1 w/o umatOthers, and vmatOthers
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2
  );
  void hessPenalizedObjectiveLL_h_k(
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& h_k,
      const arma::vec& v_k,
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


  //MSE AND DEVIANCE AND OBJECTIVE
  double deviance();
  double devianceForTest(
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );

  double mse();
  double mseForTest(
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );

  double objectiveForTest(
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );
  
  /*
   * For DEBUGGING
   */
  void enableDebugTrace(); 
  void recordDebugTrace(
      const int stage,
      const int component,
      const double f2
  );
  arma::mat getDebugTrace() const;
  
};

template class gFitterOnColumns<arma::mat>;
template class gFitterOnColumns<arma::sp_mat>;


///////////////////////////////////////
/*
 * Constructors for the dense dataset
 */
///////////////////////////////////////

// without weights, with Q
template<typename T>
gFitterOnColumns<T>::gFitterOnColumns(
  const T& Y,
  const arma::uvec& columns1,
  const std::string family,
  const arma::mat& Z, arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  arma::vec aphi,
  const arma::mat& Rtilde,
  const arma::uvec qids,
  const arma::mat B,
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const unsigned int K,
  const unsigned int r,
  const arma::mat Q
):Y(Y), columns1(columns1), family(family),
  Z(Z), Zeta(Zeta), M(M), hmat(hmat), umat(umat), vmat(vmat),
  aphi(aphi),Rtilde(Rtilde), qids(qids), q(qids.size()), B(B),
  rowOffsets(rowOffsets), colOffsets(colOffsets), K(K), r(r), Q(Q)   //Rtheta(Rtheta),
{

  this->p=Y.n_rows;
  this->n=columns1.size();
  this->plength_vector=arma::vec(p, arma::fill::zeros);
  this->nlength_vector=arma::vec(n, arma::fill::zeros);

  // R to C indexing differences
  this->seqOfComponents=arma::regspace<uvec>( 0, K-1 );
  this->rows_for_qxq = r + arma::regspace<arma::uvec>( 0, q-1 );
  this->cols_for_qxq = rows_for_qxq;


  if( rowOffsets.size() != p ){
    throw std::invalid_argument( "Row offsets must be of length equal to number of rows in Y" );
  }
  if( colOffsets.size() != Y.n_cols ){
    throw std::invalid_argument( "Col offsets must be of length equal to number of columns in Y" );
  }
  this->colOffsets_for_columns=colOffsets.elem(columns1);

  if( aphi.size() != p ){
    throw std::invalid_argument( "Argument aphi must be of length equal to number of rows in Y" );
  }
  if( Z.n_rows != n ){
    throw std::invalid_argument( "Argument Z must have rows equal to number of columns in Y" );
  }
  if( M.n_rows != n ){
    throw std::invalid_argument( "Argument M must have rows equal to number of columns in Y" );
  }
  if( M.n_cols != hmat.n_rows ){
    throw std::invalid_argument( "Number of columns in M must match number of rows in h" );
  }
  if( Z.n_cols != Zeta.n_rows ){
    throw std::invalid_argument( "Number of columns in Z must match number of rows in Zeta" );
  }

  // construct family
  if( family=="gaussian" ){
    this->rowFam = new rowFamilyGaussian<T>();
    } else if( family=="poisson" ){
      this->rowFam = new rowFamilyPoisson<T>();
      } else {
    throw std::runtime_error("Family not found");
  }

  this->Zeta=Zeta;
  this->hmat= hmat; this->umat = umat;
  this->vmat=vmat;


  if( Q.n_cols > 0 ){
    // if( knotsType==std::string("approx") ){
    //   this->BQ=B * Q.rows(qids) ; //multiplying by subview OK.
    // } else {
    //   this->BQ=B * Q ;
    // }
    this->BQ=B * Q.rows(qids) ; //multiplying by subview OK.
  }

  // R to C indexing
  this->aphi=aphi;
  this->alpha0=1;
  this->rho=.5;
  this->hAlphaStart = arma::vec(K, arma::fill::value(this->alpha0));
  this->Zt = Z.t();
  this-> ZtZ=Zt*Z;
  this->Mt=M.t();
  this->MtM=Mt*M;

  this->negLogLikesVec=arma::vec( Y.n_rows );

  //others for storing calculation intermediates
  //others for storing calculation intermediates
  this->gradObjh_k = arma::vec( r+q, arma::fill::zeros );
  this->hessObjh_k = arma::mat( r+q, r+q, arma::fill::zeros );


}


// without weights, without Q
template<typename T>
gFitterOnColumns<T>::gFitterOnColumns(
  const T& Y,
  const arma::uvec& columns1,
  const std::string family,
  const arma::mat& Z, arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  arma::vec aphi,
  const arma::mat& Rtilde,
  const arma::uvec qids,
  const arma::mat B,
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const unsigned int K,
  const unsigned int r
):Y(Y), columns1(columns1), family(family),
Z(Z), Zeta(Zeta), M(M), hmat(hmat), umat(umat), vmat(vmat),
aphi(aphi),Rtilde(Rtilde), qids(qids), q(qids.size()), B(B),
rowOffsets(rowOffsets), colOffsets(colOffsets), K(K), r(r)   //Rtheta(Rtheta),
{

  this->p=Y.n_rows;
  this->n=columns1.size();
  this->plength_vector=arma::vec(p, arma::fill::zeros);
  this->nlength_vector=arma::vec(n, arma::fill::zeros);

  //some barriers
  aphi = arma::clamp( aphi, 1e-7, 1e7);


  // R to C indexing differences
  this->seqOfComponents=arma::regspace<uvec>( 0, K-1 ); //arma::regspace includes the last ;
  this->rows_for_qxq = r + arma::regspace<arma::uvec>( 0, q-1 ); //includes the rth index.
  this->cols_for_qxq = rows_for_qxq;


  if( rowOffsets.size() != p ){
    throw std::invalid_argument( "Row offsets must be of length equal to number of rows in Y" );
  }
  if( colOffsets.size() != Y.n_cols ){
    throw std::invalid_argument( "Col offsets must be of length equal to number of columns in Y" );
  }
  this->colOffsets_for_columns=colOffsets.elem(columns1);

  if( aphi.size() != p ){
    throw std::invalid_argument( "Argument aphi must be of length equal to number of rows in Y" );
  }
  if( Z.n_rows != n ){
    throw std::invalid_argument( "Argument Z must have rows equal to number of columns in Y" );
  }
  if( M.n_rows != n ){
    throw std::invalid_argument( "Argument M must have rows equal to number of columns in Y" );
  }
  if( M.n_cols != hmat.n_rows ){
    throw std::invalid_argument( "Number of columns in M must match number of rows in h" );
  }
  if( Z.n_cols != Zeta.n_rows ){
    throw std::invalid_argument( "Number of columns in Z must match number of rows in Zeta" );
  }

  // construct family
  if( family=="gaussian" ){
    this->rowFam = new rowFamilyGaussian<T>();
  } else if( family=="poisson"){
    this->rowFam = new rowFamilyPoisson<T>();
  } else {
    throw std::runtime_error("Family not found");
  }

  this->Zeta=Zeta;
  this->hmat= hmat; this->umat = umat;
  this->vmat=vmat;

  if( Q.n_cols > 0 ){
    // if( knotsType==std::string("approx") ){
    //   this->BQ=B * Q.rows(qids) ; //multiplying by subview OK.
    // } else {
    //   this->BQ=B * Q ;
    // }
    this->BQ=B * Q.rows(qids) ; //multiplying by subview OK.
  }

  this->aphi=aphi;
  this->alpha0=1;
  this->rho=.5;
  this->hAlphaStart = arma::vec(K, arma::fill::value(this->alpha0));
  this->Zt = Z.t();
  this-> ZtZ=Zt*Z;
  this->Mt=M.t();
  this->MtM=Mt*M;

  this->negLogLikesVec=arma::vec( Y.n_rows );

  //others for storing calculation intermediates
  this->gradObjh_k = arma::vec( r+q, arma::fill::zeros );
  this->hessObjh_k = arma::mat( r+q, r+q, arma::fill::zeros );

}

template<typename T>
gFitterOnColumns<T>::~gFitterOnColumns(){
  delete this->rowFam;
}

//////////////////////////////////////////////
/*
 *
 * LOG LIKELIHOOD STUFF
 *
 */
//////////////////////////////////////////////

/*
 * Linesearch for h requires calculating loglike for everything.
 * A parallel implementation for calculation is below.
 *
 */
// overloaded form calculate log likelihood for the ith feature
// used in h descent mover
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_i(const int i ){//for the ith feature

  arma::vec y_i_all;
  y_i_all = Y.row(i).t();
  arma::vec y_i = y_i_all(columns1);
  arma::vec eta_i = (Z * Zeta.col(i) + umat * vmat.row(i).t() + colOffsets_for_columns) + rowOffsets(i);
  return this->rowFam->calcNegLogLL_i( y_i, eta_i, this->aphi(i) );

}

//Overloaded form useful for the vki and zeta_i descent movers, which avoids traversals over vectors at every descent step
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_i(const arma::vec& y_i,
                                           const arma::vec& eta_i,
                                           const double& aphi_i ){//for the ith feature


  return this->rowFam->calcNegLogLL_i( y_i, eta_i, aphi_i );



}

template<typename T>
double gFitterOnColumns<T>::calcNegLogLL(arma::vec& weightedNegLoglikes,
                                         const bool updateCurrent){ //for the entire dataset

  tbb::parallel_for( tbb::blocked_range<size_t>(0,this->p),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i){
                         weightedNegLoglikes(i)=calcNegLogLL_i(i);
                       }
                     }
  );

  double ull = arma::sum( weightedNegLoglikes );

  if(updateCurrent){
    this->negLogLikesVec=weightedNegLoglikes;
    this->ULL=ull;
  }
  return( ull );
}

//for K = 1
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec y_j;
  y_j = Y.col(columns1(j));
  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) +  (rowOffsets + colOffsets_for_columns(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  return( this->rowFam->calcNegLogLL_j( y_j, eta_j, aphi ) );
}
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const arma::mat& umatOthers, //columns other than the k^th
    const arma::mat& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec y_j;
  y_j = Y.col(columns1(j));
  arma::vec otherEtaContrib_j = (Zeta.t() * Z.row(j).t()) + (vmatOthers*(umatOthers.row(j).t())) + (rowOffsets + colOffsets_for_columns(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  return( this->rowFam->calcNegLogLL_j( y_j, eta_j, aphi ) );
}

//a direct tbb version without weights
// for K =1
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_with_h_k(
    const arma::vec& h_k,
    const arma::vec& v_k
) {

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         //cs(j)=
                         nlength_vector(j)=calcNegLogLL_j_with_h_k(
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  return arma::sum(nlength_vector);
}
template<typename T>
double gFitterOnColumns<T>::calcNegLogLL_with_h_k(
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& v_k
) {

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         //cs(j)=
                         nlength_vector(j)=calcNegLogLL_j_with_h_k(
                           umatOthers, vmatOthers,
                           h_k, v_k,
                           j
                         );
                       }
                     }
  );
  
  return arma::sum(nlength_vector);
}


//////////////////////////////////////////////
/*
 *
 * Updating a(phi)
 *
 */
//////////////////////////////////////////////



template<typename T>
double gFitterOnColumns<T>::calc_aphi_i(const int i){
  arma::vec y_i_all;
  y_i_all = Y.row(i).t();
  arma::vec y_i = y_i_all(columns1);
  arma::vec vmat_i = vmat.row(i).t();
  arma::vec eta_i = (Z*Zeta.col(i) + (umat*vmat_i) +  colOffsets_for_columns) + rowOffsets(i);
  return( std::max( this->rowFam->calc_aphi( y_i, eta_i), 1e-7 ) );
}

template<typename T>
void gFitterOnColumns<T>::update_aphi(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,Y.n_rows),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i)
                         aphi(i)=calc_aphi_i(i);
                     }
  );

}


//////////////////////////////////////////////
/*
 *
 * Objective and penalty
 *
 */
//////////////////////////////////////////////
template<typename T>
double gFitterOnColumns<T>::getPenalty(){
  double netPenalty=0;
  arma::mat hmat_qpart=hmat.rows( rows_for_qxq );
  arma::vec h_k_qpart;
  for( int k =0; k<K; k++ ){
    h_k_qpart=hmat_qpart.col(k);
    netPenalty += arma::dot( h_k_qpart, Rtilde * h_k_qpart );
  }
  return netPenalty; //sumovercompoents(f2*sq norm of projection of uk on H1)
}

template<typename T>
double gFitterOnColumns<T>::objective(const bool recalculate,
                                      const bool updateStored,
                                      const double& f1,
                                      const double& f2){
  if( recalculate ){
    arma::vec nll = arma::vec(this->p);
    this->calcNegLogLL( nll,updateStored );
  }

  double netPenalty=0;
  arma::mat hmat_qpart=hmat.rows( rows_for_qxq );
  arma::vec h_k_qpart;
  for( int k =0; k<K; k++ ){
    h_k_qpart=hmat_qpart.col(k);
    netPenalty += arma::dot( h_k_qpart, Rtilde * h_k_qpart );
  }
  double obj =f1* (this->ULL/n) + f2* netPenalty; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return(obj);
}


///////////////////////////////////////////////////////////
/*
 *
 * GRADIENTS AND HESSIANS FOR THE Zeta DESCENT MOVER
 *
 */
//////////////////////////////////////////////////////////
//Grad and Hessians for Zeta, the loading v_ik i.e., ith feature in kth component, and h.

template<typename T>
arma::vec gFitterOnColumns<T>::gradNegLogLL_Zeta_i(
    const arma::vec& y_i,
    const arma::vec& eta_i,
    const arma::vec& zeta_i,
    const double& aphi_i
){

  return( this->rowFam->gradNegLogLL_Zeta_i(y_i, eta_i,
                                            Z, zeta_i,
                                            aphi_i )
  );


}

template<typename T>
arma::mat gFitterOnColumns<T>::hessNegLogLL_Zeta_i( const arma::vec& y_i,
                                        const arma::vec& eta_i,
                                        const arma::vec& zeta_i,
                                        const double& aphi_i
){


  if( family == "gaussian" ){
    return( ZtZ/aphi_i ); //instead of calling the below function for gaussian
  } else {
    return( this->rowFam->hessNegLogLL_Zeta_i(y_i, eta_i,
                                              Z, zeta_i,
                                              aphi_i )
    );
  }

}





//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR ZETA
 *
 *
 * Just to avoid the overhead of checking for i for
 * weights setting, we rewrite for four cases
 * No weights, colweights only, rowweights only,
 * both weights
 *
 */
//////////////////////////////////////////////


//in place move is complicated with RCPP/tbb.
//Movers for individual rows i, depending on weight setting
template<typename T>
void gFitterOnColumns<T>::moveZeta_i(const int i, const double& f1, const double& f2){

  arma::vec zeta_i = arma::vec(this->Zeta.col(i)); // Zeta is Bxp where Z is n x B, so  Z zeta enters eta
  arma::vec zeta_i_new=zeta_i;
  arma::vec y_i_all; y_i_all = Y.row(i).t(); arma::vec y_i = y_i_all(columns1);

  double aphi_i=aphi(i);
  arma::vec otherEtaContrib = ((umat* (vmat.row(i).t()) ) + colOffsets_for_columns) + rowOffsets(i);
  arma::vec eta_i = Z * zeta_i + otherEtaContrib;
  arma::vec eta_i_new=eta_i;

  int iter=0;
  bool conv = false; bool move=false;
  double ll_i = this->calcNegLogLL_i(y_i, eta_i, aphi_i);
  //double ll_i = .5*arma::sum( arma::pow( eta_i-y_i, 2 )  )/aphi_i;
  double ll_i_new, alpha;
  arma::vec gradll_i, p_i;
  arma::mat hessZeta_i;

  while( (!conv) & (iter <= this->inner_maxIter) ){
    iter = iter+1;

    gradll_i = this->gradNegLogLL_Zeta_i( y_i, eta_i, zeta_i, aphi_i);
    hessZeta_i = this->hessNegLogLL_Zeta_i(y_i, eta_i, zeta_i, aphi_i);
    
    p_i = -invpsd(hessZeta_i, 1e-7, false) * gradll_i;
    move =false;
    alpha = alpha0;
    while( (alpha >= 1e-4) & (!move)  ){
      zeta_i_new=zeta_i + alpha*p_i;
      eta_i_new =  Z*zeta_i_new + otherEtaContrib;
      ll_i_new = this->calcNegLogLL_i(y_i, eta_i_new, aphi_i );
      if(  (ll_i_new > ll_i) ||
           (!std::isfinite(ll_i_new))
           ){
        alpha = rho*alpha;
      } else {
        move = true;
      }
    }
    if( move ){
      const double deltaF_inner = f1 * std::abs(ll_i_new - ll_i);
      const double deltaZetaRMS = std::sqrt(arma::mean(arma::square(zeta_i_new - zeta_i)));
      conv = deltaF_inner < this->solved_inner_ftol_abs;
      (void)deltaZetaRMS;
      zeta_i = zeta_i_new;
      eta_i = eta_i_new;
      ll_i = ll_i_new;
      alpha = alpha/rho;
      this->negLogLikesVec(i)=ll_i_new;
      
    } else {
      conv = true; //return with current Zeta, as no step has been made
    }
  }

  this->Zeta.col(i)=zeta_i; //IN PRINCIPLE, THIS IS OK & THREAD SAFE AS NO TWO THREADS ARE WRITING TO SAME MEMORY.

}
//Parallel updater for each row i in the dataset

template<typename T>
void gFitterOnColumns<T>::moveZeta(const double& f1, const double& f2){

  //parallel
  tbb::parallel_for( tbb::blocked_range<size_t>(0,p),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i){
                         moveZeta_i(i, f1, f2); // this is an in place move, a thread safe variant,
                       }
                     }
  );

  this->ULL = arma::sum( this->negLogLikesVec );
}


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR V and U
 *
 *
 */
//////////////////////////////////////////////

template<typename T>
double gFitterOnColumns<T>::gradNegLogLL_v_ik( const arma::vec& y_i,
                                   const arma::vec& eta_ik,
                                   const arma::vec& u_k, const double& v_ik,
                                   const double& aphi_i ){

  return( this->rowFam->gradNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i ));
}
template<typename T>
double gFitterOnColumns<T>::hessNegLogLL_v_ik(const arma::vec& y_i,
                                  const arma::vec& eta_ik,
                                  const arma::vec& u_k,
                                  const double& v_ik,
                                  const double& aphi_i){

  return( this->rowFam->hessNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i ));
}


////////////////////////////////////////////////
/*
 * Mover for v_k_i and v_k with unit norm constraint
 */
////////////////////////////////////////////////

template<typename T>
void gFitterOnColumns<T>::move_v_k_i_normUkStrategy( const int i,
                                                     const int k,
                                                     arma::vec& vk,
                                                     const double& f1,
                                                     const double& f2 ){

  /*
   * IMPORTANT!
   *
   * Prior to calling this function, we assume,
   * with uk = umat.col(k), dk=norm(uk), we have set:
   *  1. uk_norm_vec = uk/dk (or set to zero if dk is ~0),
   *  2. hmat.col(k) /= dk to preserve uk = Mhk equality;
   *  3. vmat.col(k) *= dk .
   */
  arma::vec u_k_normed = umat.col(k); //this is the assumption for this function: uk is already normalized!

  double v_ik=vmat(i,k);
  double v_ik_new=vmat(i,k);
  arma::vec offset_i = this->rowOffsets(i)+ this->colOffsets_for_columns;
  arma::vec otherEtaContrib = this->Z* this->Zeta.col(i) + offset_i;
  if( K>1 ){
    arma::uvec cols_except_k=this->seqOfComponents(find(this->seqOfComponents != k));
    arma::mat umatOther = this->umat.cols(cols_except_k);
    arma::mat vmatOther = this->vmat.cols(cols_except_k);
    otherEtaContrib += umatOther * (vmatOther.row(i).t()) ;
  }

  arma::vec eta_ik = v_ik*u_k_normed + otherEtaContrib; //current mean
  arma::vec eta_ik_new=eta_ik;

  arma::vec y_i_all;y_i_all = Y.row(i).t();
  arma::vec y_i = y_i_all(columns1);
  bool conv = false; bool move=false;
  double aphi_i=this->aphi(i);
  double ll_ik= this->calcNegLogLL_i(y_i, eta_ik, aphi_i ); // this calculation is needed as prior u/h_k moves do not populate negLogLikesVec appropriately
  double ll_ik_new, alpha;
  double gradll_ik, p_ik;
  double hessv_ik;

  int iter=0;
  while( (!conv) & (iter <= this->inner_maxIter) ){
    iter = iter+1;
    gradll_ik =  this->gradNegLogLL_v_ik( y_i, eta_ik, u_k_normed, v_ik, aphi_i ); //scalar
    hessv_ik = std::max( this->hessNegLogLL_v_ik(y_i, eta_ik, u_k_normed, v_ik, aphi_i), 1e-7 ); //scalar
    p_ik = -gradll_ik/hessv_ik; //scalar
    p_ik = std::min( p_ik, 1e7 );
    p_ik = std::max( p_ik, -1e7 );

    //line search, and move
    move =false;
    alpha = alpha0;
    while( (alpha >= 1e-7) & (!move)  ){
      v_ik_new=v_ik + alpha*p_ik;
      v_ik_new = std::min( v_ik_new, 1e7 );
      v_ik_new = std::max( v_ik_new, -1e7 );
      eta_ik_new =  v_ik_new*u_k_normed + otherEtaContrib; ////evaluation ofstats for new proposals should use uk normalized, not umat.col(k).
      ll_ik_new =  this->calcNegLogLL_i(y_i, eta_ik_new, aphi_i );
      //if(  (ll_ik_new > (  ll_ik + alpha*p_ik*gradll_ik )) || //COMMENTING FOR NOW
      if(  (ll_ik_new > (  ll_ik  )) ||
           (!std::isfinite(ll_ik_new))
           ){
        alpha = this->rho * alpha;
      } else {
        move = true;
      }
    }
    if( move ){
      conv = sqrt(pow(v_ik_new- v_ik,2)) < 1e-4;
      v_ik = v_ik_new;
      eta_ik = eta_ik_new;
      ll_ik = ll_ik_new;
      alpha = alpha/rho;
      this->negLogLikesVec(i)=ll_ik_new;
    } else {
      conv = true; //return with current Zeta, as no step has been made
    }
  }
  //likely thread safe as the indices in the vector are independently written to.
  vk(i)=v_ik;

}

template<typename T>
void gFitterOnColumns<T>::
  move_v_k_withUnitNormConstraint_normUStrategy(
    const int k,
    const double& f1,
    const double& f2
  ) {
    
    
    const arma::vec uk = umat.col(k);
    
    if (arma::norm(uk, 2) < 1e-8) {
      return;
    }
    
    const double penalty_term = f2 * n * getPenalty();
    if (!std::isfinite(penalty_term)) {
      return;
    }

    bool converged = false;
    int iter = 0;
    while (!converged && iter <= this->inner_maxIter) {
      ++iter;

      const arma::vec vk_old = vmat.col(k);

      const double objective_old = f1 * this->ULL + penalty_term;
      if (!std::isfinite(objective_old)) {
        break;
      }

      const arma::vec negLogLikesVec_old = negLogLikesVec;
      const double ULL_old = ULL;
    
    arma::vec grad_vk(p, arma::fill::zeros);
    
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, p),
      [&](const tbb::blocked_range<size_t>& ra) {
        for (size_t i = ra.begin(); i != ra.end(); ++i) {
          const double v_ik = vk_old(i);
          
          arma::vec offset_i =
            this->rowOffsets(i) + this->colOffsets_for_columns;
          
          arma::vec eta_i = this->Z * this->Zeta.col(i) + offset_i;
          
          if (K > 1) {
            arma::uvec cols_except_k =
              this->seqOfComponents(
                  find(this->seqOfComponents != k)
              );
            
            arma::mat umat_other = this->umat.cols(cols_except_k);
            arma::mat vmat_other = this->vmat.cols(cols_except_k);
            eta_i += umat_other * vmat_other.row(i).t();
          }
          
          eta_i += v_ik * uk;
          
          //arma::rowvec y_i_row = arma::rowvec(Y.row(i));
          arma::vec y_i_all; 
          y_i_all = Y.row(i).t();
          arma::vec y_i = y_i_all(columns1);
          
          grad_vk(i) =
            f1 * this->gradNegLogLL_v_ik(
                y_i,
                eta_i,
                uk,
                v_ik,
                this->aphi(i)
            );
        }
      }
    );
    
    
    const double radial_coefficient = arma::dot(vk_old, grad_vk);
    const arma::vec grad_vk_tangent =
      grad_vk - radial_coefficient * vk_old;
    const arma::vec direction = -grad_vk_tangent;

    if ((!direction.is_finite()) ||
        (arma::norm(direction, 2) < 1e-12)) {
      break;
    }
    
    bool move = false;
    double alpha = alpha0;
    double objective_new = objective_old;
    arma::vec vk_candidate = vk_old;
    
    while ((alpha >= 1e-4) && (!move)) {
      vk_candidate = vk_old + alpha * direction;
      const double candidate_norm = arma::norm(vk_candidate, 2);
      
      if ((!std::isfinite(candidate_norm)) || (candidate_norm < 1e-12)) {
        alpha = this->rho * alpha;
        continue;
      }
      
      // Euclidean projection onto the unit sphere.
      vk_candidate /= candidate_norm;
      vmat.col(k) = vk_candidate;
      
      objective_new = this->objective(true, true, f1, f2);
      
      if (std::isfinite(objective_new) &&
          (objective_new <= objective_old)) {
        move = true;
      } else {
        // Restore the exact accepted state before trying a smaller step.
        vmat.col(k) = vk_old;
        negLogLikesVec = negLogLikesVec_old;
        ULL = ULL_old;
        alpha = this->rho * alpha;
      }
    }
    
    if (!move) {
      vmat.col(k) = vk_old;
      negLogLikesVec = negLogLikesVec_old;
      ULL = ULL_old;
      break;
    }

    const double deltaF_inner = std::abs(objective_new - objective_old);
    const double deltaVRMS = std::sqrt(
      arma::mean(arma::square(vk_candidate - vk_old))
    );
    converged = deltaF_inner < this->solved_inner_ftol_abs;
    // Optional future parameter-stability criterion:
    // converged = converged || (deltaVRMS < inner_eps);
    (void)deltaVRMS;
    }
  }


template<typename T>
void gFitterOnColumns<T>::moveVU(const double& f1, const double& f2){

  //move each factor
  for( int k=0; k<K; k++ ){ //<= as K has been set to the Total#Components-1 in the cosntructor
    move_h_k(k, f1, f2, alpha0, rho); //STATEMENT
    umat.col(k) = M * hmat.col(k);   //STATEMENT
    
    recordDebugTrace(2, k, f2);

    move_v_k_withUnitNormConstraint_normUStrategy(k, f1, f2); //STATEMENT!

    if( arma::sign( vmat(0,k) ) <0 ){//rotation identifiabilty
      vmat.col(k) *= -1;
      hmat.col(k) *= -1;
      umat.col(k) *= -1;
    }
    
    recordDebugTrace(3, k, f2);

  }

}




////////////////////////////////////////////////
/*
 * Mover for h_k
 */
////////////////////////////////////////////////


////////////////////////////////////////////
/*
 * Gradients of just the negative log likes
 */
////////////////////////////////////////////
template<typename T>
void gFitterOnColumns<T>::gradNegLogLL_h_k(
    const int k,
    const arma::vec& h_k,
    arma::vec& grad_h_k
){

  if( K > 1 ){

    arma::uvec others = find( seqOfComponents != k );
    arma::mat umatOthers = this->umat.cols(others);
    arma::mat vmatOthers = this->vmat.cols(others);
    arma::vec v_k = this->vmat.col(k);
    this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                            umatOthers, vmatOthers,
                                            aphi,rowOffsets, colOffsets_for_columns, //must pass colOffsets_for_columns, not colOffsets
                                            h_k, v_k,
                                            grad_h_k
    );

  } else {

    arma::vec v_k = this->vmat.col(k);
    this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                            aphi,rowOffsets, colOffsets_for_columns, //must pass colOffsets_for_columns, not colOffsets
                                            h_k, v_k,
                                            grad_h_k
    );

  }


}

template<typename T>
void gFitterOnColumns<T>::hessNegLogLL_h_k(
    const int k,
    const arma::vec& h_k,
    arma::mat& hess_h_k
){

  if(family=="gaussian"){
    hess_h_k=arma::sum(arma::pow(vmat.col(k),2)/aphi)*MtM; // a scalar * MtM
  } else {

    if( K > 1 ){
      arma::uvec others = find( seqOfComponents != k );   // TODO: WE ARE DOING THESE COPIES OF UMATS AND VMATS TWICE! ONE IN GRAD FN, and ANOTHER IN HESS FN. FUTURE should minimize this.
      arma::mat umatOthers = this->umat.cols(others);
      arma::mat vmatOthers = this->vmat.cols(others);
      arma::vec v_k = this->vmat.col(k);
      this->rowFam->hessNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                              umatOthers,vmatOthers,
                                              aphi,
                                              rowOffsets, colOffsets_for_columns,
                                              MtM,
                                              h_k, v_k,
                                              hess_h_k
      );
    } else {
      arma::vec v_k = this->vmat.col(k);
      this->rowFam->hessNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                              aphi,
                                              rowOffsets, colOffsets_for_columns,
                                              MtM,
                                              h_k, v_k,
                                              hess_h_k
      );
    }

  }

}


/////////////////////////////////////////
/*
 * Gradients of objectives
 */
/////////////////////////////////////////
template<typename T>
void gFitterOnColumns<T>::gradPenalizedObjective_h_k(
    const int k,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2
){

  if( K > 1 ){

    arma::uvec others = find( seqOfComponents != k );   // TODO: WE ARE DOING THESE COPIES OF UMATS AND VMATS TWICE! ONE IN GRAD FN, and ANOTHER IN HESS FN. FUTURE should minimize this.
    arma::mat umatOthers = this->umat.cols(others);
    arma::mat vmatOthers = this->vmat.cols(others);
    arma::vec v_k = this->vmat.col(k);

    this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                            umatOthers, vmatOthers,
                                            aphi,rowOffsets, colOffsets_for_columns,
                                            h_k, v_k,
                                            grad_h_k
    );
    
  } else {
    arma::vec v_k = this->vmat.col(k);

    this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                            aphi,rowOffsets, colOffsets_for_columns,
                                            h_k, v_k,
                                            grad_h_k
    );

  }

  //alter as appropriate
  grad_h_k *= f1;
  grad_h_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart); // n because of 1/n prefactor in objective
}

template<typename T>
void gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
    const int k,
    const arma::vec& h_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2
){



  //grab hessian of the entire data neg log likelihood
  if(family=="gaussian"){
    hess_h_k=arma::sum(arma::pow(vmat.col(k),2)/aphi)*MtM; // a scalar * MtM
  } else {

    if( K > 1){

      arma::uvec others = find( seqOfComponents != k );   // TODO: WE ARE DOING THESE COPIES OF UMATS AND VMATS TWICE! ONE IN GRAD FN, and ANOTHER IN HESS FN. FUTURE should minimize this.
      arma::mat umatOthers = this->umat.cols(others);
      arma::mat vmatOthers = this->vmat.cols(others);
      arma::vec v_k = this->vmat.col(k);

      this->rowFam->hessNegLogLL_h_k_onColsOf(Y,columns1,
                                              M,
                                              Z, Zeta,
                                              umatOthers,vmatOthers,
                                              aphi,rowOffsets,colOffsets_for_columns,
                                              MtM,
                                              h_k, v_k,
                                              hess_h_k
      );

    } else {

      arma::vec v_k = this->vmat.col(k);

      this->rowFam->hessNegLogLL_h_k_onColsOf(Y,columns1,
                                              M,
                                              Z, Zeta,
                                              aphi,rowOffsets,colOffsets_for_columns,
                                              MtM,
                                              h_k, v_k,
                                              hess_h_k
      );

    }


  }

  //alter for the objective
  hess_h_k *= f1;
  hess_h_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde;
}

//overloaded forms
template<typename T>
void gFitterOnColumns<T>::gradPenalizedObjective_h_k(
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const arma::vec& v_k,
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2
){


  this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                          aphi,rowOffsets, colOffsets_for_columns,
                                          h_k, v_k,
                                          grad_h_k
  );

  //alter as appropriate
  grad_h_k *= f1;
  grad_h_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart);
}

template<typename T>
void gFitterOnColumns<T>::gradPenalizedObjective_h_k(
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const arma::vec& v_k,
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2
){


  this->rowFam->gradNegLogLL_h_k_onColsOf(Y, columns1, M, Z, Zeta,
                                 umatOthers, vmatOthers,
                                 aphi,rowOffsets, colOffsets_for_columns,
                                 h_k, v_k,
                                 grad_h_k
  );

  //alter as appropriate
  grad_h_k *= f1;
  grad_h_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart);
}

//for K =1
template<typename T>
void gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2
){
  //grab hessian of the entire data neg log likelihood
  if(family=="gaussian"){
    hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
  } else {

    this->rowFam->hessNegLogLL_h_k_onColsOf(Y, columns1,
                                            M,
                                            Z, Zeta,
                                            aphi,rowOffsets,colOffsets_for_columns,
                                            MtM,
                                            h_k, v_k,
                                            hess_h_k
    );
  }

  //alter for the objective
  hess_h_k *= f1;
  //std::cout << "adding to In hesspenalized_hk" << std::endl;
  hess_h_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde;
}
template<typename T>
void gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2
){
  //grab hessian of the entire data neg log likelihood
  if(family=="gaussian"){
    hess_h_k=arma::sum(arma::pow(v_k,2)/aphi)*MtM; // a scalar * MtM
  } else {

    this->rowFam->hessNegLogLL_h_k_onColsOf(Y, columns1,
                                   M,
                                   Z, Zeta,
                                   umatOthers,vmatOthers,
                                   aphi,rowOffsets,colOffsets_for_columns,
                                   MtM,
                                   h_k, v_k,
                                   hess_h_k
    );
  }

  //alter for the objective
  hess_h_k *= f1;
  //std::cout << "adding to In hesspenalized_hk" << std::endl;
  hess_h_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde;
}



#if 0
// this needs to incorporate the orth to constant function constraint like move_h_k
template<typename T>
void gFitterOnColumns<T>::move_h_k_rgd(
    const int k,
    const double& f1,
    const double& f2,
    const double alpha0,
    const double rho
){

  double& alphaStart = this->hAlphaStart(k);

  // Diagnostic: the generalized initializer may not initially satisfy u_k = M h_k.
  const double entryUMHDiff =
    arma::abs(
      this->umat.col(k) - this->M * this->hmat.col(k)
    ).max();

  arma::vec h_k = this->hmat.col(k);
  arma::vec v_k = this->vmat.col(k);
  arma::vec h_k_qpart = h_k(rows_for_qxq);

  arma::mat umatOthers;
  arma::mat vmatOthers;

  // P projects coefficient-space vectors onto the feasible subspace
  // implied by the RKHS orthogonality constraints.  For K == 1 there
  // are no other latent functions, so P is the identity.
  arma::mat P;

  if (K > 1) {
    const arma::uvec others = find(seqOfComponents != k);
    umatOthers = this->umat.cols(others);
    vmatOthers = this->vmat.cols(others);

    const arma::mat uothers = umatOthers.rows(qids);
    const arma::mat Buothers = B * uothers;

    arma::mat MtBQ;
    if (BQ.n_cols > 0) {
      MtBQ = M.rows(qids).t() * arma::join_cols(BQ, Buothers);
    } else {
      MtBQ = M.rows(qids).t() * Buothers;
    }

    const arma::mat MtBQ2 = MtBQ.t() * MtBQ;
    const arma::mat HQ =
      MtBQ * invpsd(MtBQ2, 1e-7, false) * MtBQ.t();

    P = arma::eye<arma::mat>(h_k.n_elem, h_k.n_elem) - HQ;

    // Numerical cleanup only.  Once h_k is feasible and every direction
    // lies in range(P), all subsequent candidates remain feasible exactly.
    h_k = P * h_k;
    h_k_qpart = h_k(rows_for_qxq);
  }

  double ll_k;
  if (K == 1) {
    ll_k = this->calcNegLogLL_with_h_k(h_k, v_k);
  } else {
    ll_k = this->calcNegLogLL_with_h_k(
      umatOthers, vmatOthers, h_k, v_k
    );
  }

  bool conv = false;
  int iter = 0;

  while ((!conv) && (iter <= this->inner_maxIter)) {
    ++iter;

    const double obj_k =
      f1 * ll_k +
      f2 *n* arma::dot(h_k_qpart, Rtilde * h_k_qpart);

    if (K == 1) {
      this->gradPenalizedObjective_h_k(
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2
      );
    } else {
      this->gradPenalizedObjective_h_k(
        umatOthers, vmatOthers,
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2
      );
    }

    // Riemannian/projected-gradient direction.
    // For K == 1, this is ordinary negative-gradient descent.
    arma::vec p_k;
    if (K == 1) {
      p_k = -this->gradObjh_k;
    } else {
      p_k = -(P * this->gradObjh_k);
    }

    if (!p_k.is_finite()) {
      throw std::runtime_error(
        "move_h_k: non-finite projected gradient"
      );
    }

    const double projectedGradNorm = arma::norm(p_k, 2);
    if (projectedGradNorm < 1e-10) {
      conv = true;
      break;
    }

    // Since p_k = -P g and P is an orthogonal projector,
    // g' p_k = -||P g||^2 <= 0.  This is the derivative of the
    // one-dimensional objective F(h_k + alpha p_k) at alpha = 0.
    const double gradDotDirection =
      arma::dot(this->gradObjh_k, p_k);

    if (!(gradDotDirection < 0.0) ||
        !std::isfinite(gradDotDirection)) {
      conv = true;
      break;
    }

    bool move = false;
    double alpha = alphaStart;
    double ll_k_new = ll_k;
    double obj_k_new = obj_k;
    arma::vec h_k_new = h_k;
    arma::vec h_k_new_qpart = h_k_qpart;
    int lineSearchAttempt = 0;

    while ((alpha >= 1e-4) && (!move)) {
      ++lineSearchAttempt;

      // Because h_k is feasible and p_k lies in the same linear subspace,
      // this candidate remains RKHS-feasible for every alpha.  Do not clamp:
      // elementwise clamping would generally violate the linear constraints.
      h_k_new = h_k + alpha * p_k;

      bool candidateFinite = h_k_new.is_finite();
      if (candidateFinite) {
        h_k_new_qpart = h_k_new(rows_for_qxq);

        if (K == 1) {
          ll_k_new = this->calcNegLogLL_with_h_k(h_k_new, v_k);
        } else {
          ll_k_new = this->calcNegLogLL_with_h_k(
            umatOthers, vmatOthers, h_k_new, v_k
          );
        }

        obj_k_new =
          f1 * ll_k_new +
          f2 *n* arma::dot(
            h_k_new_qpart,
            Rtilde * h_k_new_qpart
          );
      } else {
        obj_k_new = std::numeric_limits<double>::infinity();
      }

      const arma::vec actualStep = h_k_new - h_k;
      const arma::vec stepMinusAlphaP = actualStep - alpha * p_k;
      const double stepIdentityError =
        arma::norm(stepMinusAlphaP, 2);
      const double gradDotActualStep =
        arma::dot(this->gradObjh_k, actualStep);
      const double gradDotAlphaP =
        alpha * gradDotDirection;

      // Strict monotone backtracking.  Armijo can be added later once the
      // projected-gradient implementation has been validated.
      const bool accepted =
        std::isfinite(obj_k_new) &&
        (obj_k_new <= obj_k);

      if (debugTraceEnabled) {
        hDebugTrace.push_back({
          static_cast<double>(debugIteration),
          static_cast<double>(k),
          static_cast<double>(iter),
          static_cast<double>(lineSearchAttempt),
          alpha,
          obj_k,
          obj_k_new,
          obj_k_new - obj_k,
          gradDotActualStep,
          gradDotAlphaP,
          stepIdentityError,
          arma::abs(p_k).max(),
          arma::abs(actualStep).max(),
          accepted ? 1.0 : 0.0,
          entryUMHDiff
        });
      }

      if (accepted) {
        move = true;
      } else {
        alpha *= rho;
      }
    }

    if (move) {
      conv =
        std::sqrt(
          as_scalar(
            arma::mean(arma::pow(h_k_new - h_k, 2))
          )
        ) < 1e-4;

      h_k = h_k_new;
      h_k_qpart = h_k_new_qpart;
      ll_k = ll_k_new;
      alphaStart = alpha;
    } else {
      alphaStart =
        ((rho * alphaStart) > 1e-4)
          ? (rho * alphaStart)
          : 1e-4;
      conv = true;
    }
  }

  hmat.col(k) = h_k;
  umat.col(k) = M * h_k;
  this->ULL = ll_k;
}
#endif


template<typename T>
void gFitterOnColumns<T>::move_h_k(
    const int k,
    const double& f1,
    const double& f2,
    const double alpha0,
    const double rho
){
  
  
  
  double& alphaStart = this->hAlphaStart(k);

  const double entryUMHDiff =
    arma::abs(
      this->umat.col(k) - this->M * this->hmat.col(k)
    ).max();

  const arma::vec h_k_entry = this->hmat.col(k);
  const arma::vec u_k_entry = this->umat.col(k);
  const double ULL_entry = this->ULL;
  bool acceptedAnyMove = false;

  arma::vec h_k = h_k_entry;
  arma::vec v_k = this->vmat.col(k);
  arma::vec h_k_qpart = h_k(rows_for_qxq);

  const arma::vec uBeforeProjection = this->M * h_k;

  
  arma::mat umatOthers;
  arma::mat vmatOthers;

  arma::mat constraintMat;
  arma::vec u0 = arma::vec(n, arma::fill::ones);
  arma::mat rawConstraintMat = M.rows(qids).t() * B * u0(qids);

  arma::mat P =
    arma::eye<arma::mat>(h_k.n_elem, h_k.n_elem);

  if (K > 1) {
    const arma::uvec others = find(seqOfComponents != k);
    umatOthers = this->umat.cols(others);
    vmatOthers = this->vmat.cols(others);

    const arma::mat uothers = umatOthers.rows(qids);
    const arma::mat Buothers = B * uothers;
    
    if (BQ.n_cols > 0) {
      rawConstraintMat =
        arma::join_rows(rawConstraintMat,
                        M.rows(qids).t() * arma::join_rows(BQ, Buothers) 
                          ) ;
    } else {
      rawConstraintMat = arma::join_rows(rawConstraintMat, 
                                         M.rows(qids).t() * Buothers
                                           );
    }
  }
  
  constraintMat = arma::orth(rawConstraintMat);

    if (constraintMat.n_cols > 0) {
      P -= constraintMat * constraintMat.t();

      h_k = P * h_k;
      h_k_qpart = h_k(rows_for_qxq);
    }

  const double projectionChangeRMS =
    std::sqrt(arma::mean(arma::square(h_k - h_k_entry)));

  double ll_k;
  if (K == 1) {
    ll_k = this->calcNegLogLL_with_h_k(h_k, v_k);
  } else {
    ll_k = this->calcNegLogLL_with_h_k(
      umatOthers, vmatOthers, h_k, v_k
    );
  }

  const arma::vec uAfterProjection = this->M * h_k;
  hProjectionDebugTrace.push_back({
    static_cast<double>(debugIteration),
    static_cast<double>(k),
    uBeforeProjection.min(),
    uBeforeProjection.max(),
    uAfterProjection.min(),
    uAfterProjection.max(),
    ll_k,
    std::isfinite(ll_k) ? 1.0 : 0.0
  });

  bool conv = false;
  int iter = 0;

  while ((!conv) && (iter <= this->inner_maxIter)) {
    ++iter;

    const double obj_k =
      f1 * ll_k +
      f2 * n* arma::dot(h_k_qpart, Rtilde * h_k_qpart);

    if (K == 1) {
      this->gradPenalizedObjective_h_k(
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2
      );
      this->hessPenalizedObjectiveLL_h_k(
        h_k, v_k,
        this->hessObjh_k, f1, f2
      );
    } else {
      this->gradPenalizedObjective_h_k(
        umatOthers, vmatOthers,
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2
      );
      this->hessPenalizedObjectiveLL_h_k(
        umatOthers, vmatOthers,
        h_k, v_k,
        this->hessObjh_k, f1, f2
      );
    }

    if ((!this->gradObjh_k.is_finite()) ||
        (!this->hessObjh_k.is_finite())) {
      throw std::runtime_error(
        "move_h_k: non-finite gradient or Hessian"
      );
    }

    arma::vec p_k;
    bool haveNewtonDirection = false;
    int dampingAttempts = 0;
    double dampingUsed =
      std::numeric_limits<double>::quiet_NaN();

    const arma::mat hessSym =
      0.5 * (this->hessObjh_k + this->hessObjh_k.t());

    const double meanAbsDiagonal =
      arma::mean(arma::abs(hessSym.diag()));
    double damping =
      1e-8 * std::max(1.0, meanAbsDiagonal);

    for (int dampingAttempt = 0;
         dampingAttempt < 10 && !haveNewtonDirection;
         ++dampingAttempt) {

      ++dampingAttempts;
      const double currentDamping = damping;

      const arma::mat hessDamped =
        hessSym +
        damping *
        arma::eye<arma::mat>(hessSym.n_rows, hessSym.n_cols);

      const arma::vec absHessDiag = arma::abs(hessDamped.diag());
      const double diagScaleFloor =
        1e-12 * std::max(1.0, absHessDiag.max());
      const arma::vec newtonScale =
        1.0 / arma::sqrt(arma::clamp(
          absHessDiag,
          diagScaleFloor,
          std::numeric_limits<double>::max()
        ));
      const arma::mat D = arma::diagmat(newtonScale);
      const arma::mat hessScaled = D * hessDamped * D;
      const arma::vec gradScaled = D * this->gradObjh_k;

      arma::vec candidateDirection;
      bool solved = false;

      if (constraintMat.n_cols == 0) {
        arma::vec candidateScaled;
        solved = arma::solve(
          candidateScaled,
          hessScaled,
          -gradScaled,
          arma::solve_opts::likely_sympd +
          arma::solve_opts::no_approx
        );
        if (solved && candidateScaled.is_finite()) {
          candidateDirection = D * candidateScaled;
        } else {
          solved = false;
        }
      } else {
        const arma::uword nCoef = hessScaled.n_rows;
        const arma::uword nConstraints = constraintMat.n_cols;
        const arma::mat constraintScaled = D * constraintMat;

        arma::mat KKT(
          nCoef + nConstraints,
          nCoef + nConstraints,
          arma::fill::zeros
        );

        KKT.submat(0, 0, nCoef - 1, nCoef - 1) = hessScaled;
        KKT.submat(
          0, nCoef,
          nCoef - 1, nCoef + nConstraints - 1
        ) = constraintScaled;
        KKT.submat(
          nCoef, 0,
          nCoef + nConstraints - 1, nCoef - 1
        ) = constraintScaled.t();

        arma::vec rhs(
          nCoef + nConstraints,
          arma::fill::zeros
        );
        rhs.head(nCoef) = -gradScaled;

        arma::vec solution;
        solved = arma::solve(
          solution,
          KKT,
          rhs,
          arma::solve_opts::no_approx
        );

        if (solved && solution.n_elem >= nCoef &&
            solution.head(nCoef).is_finite()) {
          candidateDirection = D * solution.head(nCoef);
        } else {
          solved = false;
        }
      }

      if (solved && candidateDirection.is_finite()) {
        const double feasibilityError =
          (constraintMat.n_cols > 0)
            ? arma::norm(
                constraintMat.t() * candidateDirection, 2
              )
            : 0.0;

        const double directionNorm =
          arma::norm(candidateDirection, 2);
        const double feasibilityTolerance =
          1e-8 * std::max(1.0, directionNorm);
        const double gradDotCandidate =
          arma::dot(this->gradObjh_k, candidateDirection);

        if (debugTraceEnabled && debugIteration == 1 && k == 0) {
          hNewtonDebugTrace.push_back({
            static_cast<double>(debugIteration), static_cast<double>(k),
            static_cast<double>(iter), static_cast<double>(dampingAttempt + 1),
            currentDamping, arma::norm(this->gradObjh_k, 2),
            hessSym.diag().min(), hessSym.diag().max(), meanAbsDiagonal,
            static_cast<double>(constraintMat.n_cols), 1.0,
            feasibilityError, gradDotCandidate, arma::abs(candidateDirection).max()
          });
        }

        if ((feasibilityError <= feasibilityTolerance) &&
            std::isfinite(gradDotCandidate) &&
            (gradDotCandidate < 0.0)) {
          p_k = candidateDirection;
          haveNewtonDirection = true;
          dampingUsed = currentDamping;
        }
      }

      if (debugTraceEnabled && debugIteration == 1 && k == 0 &&
          !(solved && candidateDirection.is_finite())) {
        hNewtonDebugTrace.push_back({
          static_cast<double>(debugIteration), static_cast<double>(k),
          static_cast<double>(iter), static_cast<double>(dampingAttempt + 1),
          currentDamping, arma::norm(this->gradObjh_k, 2),
          hessSym.diag().min(), hessSym.diag().max(), meanAbsDiagonal,
          static_cast<double>(constraintMat.n_cols), 0.0,
          arma::datum::nan, arma::datum::nan, arma::datum::nan
        });
      }
      damping *= 10.0;
    }

    if (!haveNewtonDirection) {
      if (constraintMat.n_cols == 0) {
        p_k = -this->gradObjh_k;
      } else {
        p_k = -(P * this->gradObjh_k);
      }

      const arma::vec fallbackDeltaU = this->M * p_k;
      const double fallbackMaxAbsDeltaU = arma::abs(fallbackDeltaU).max();
      const double fallbackUStepMax = 20.0;
      if (std::isfinite(fallbackMaxAbsDeltaU) &&
          fallbackMaxAbsDeltaU > fallbackUStepMax) {
        p_k *= fallbackUStepMax / fallbackMaxAbsDeltaU;
      }
    }

    if (!p_k.is_finite()) {
      throw std::runtime_error(
        "move_h_k: non-finite Newton/RGD direction"
      );
    }

    const double directionNorm = arma::norm(p_k, 2);
    if (directionNorm < 1e-10) {
      conv = true;
      break;
    }

    const double gradDotDirection =
      arma::dot(this->gradObjh_k, p_k);

    if (!(gradDotDirection < 0.0) ||
        !std::isfinite(gradDotDirection)) {
      conv = true;
      break;
    }

    bool move = false;
    double alpha = alphaStart;
    double ll_k_new = ll_k;
    double obj_k_new = obj_k;
    arma::vec h_k_new = h_k;
    arma::vec h_k_new_qpart = h_k_qpart;
    int lineSearchAttempt = 0;

    while ((alpha >= 1e-4) && (!move)) {
      ++lineSearchAttempt;

      h_k_new = h_k + alpha * p_k;

      bool candidateFinite = h_k_new.is_finite();
      if (candidateFinite) {
        h_k_new_qpart = h_k_new(rows_for_qxq);

        if (K == 1) {
          ll_k_new = this->calcNegLogLL_with_h_k(h_k_new, v_k);
        } else {
          ll_k_new = this->calcNegLogLL_with_h_k(
            umatOthers, vmatOthers, h_k_new, v_k
          );
        }

        obj_k_new =
          f1 * ll_k_new +
          f2 * n* arma::dot(
            h_k_new_qpart,
            Rtilde * h_k_new_qpart
          );
      } else {
        obj_k_new = std::numeric_limits<double>::infinity();
      }

      const arma::vec actualStep = h_k_new - h_k;
      const arma::vec stepMinusAlphaP = actualStep - alpha * p_k;
      const double stepIdentityError =
        arma::norm(stepMinusAlphaP, 2);
      const double gradDotActualStep =
        arma::dot(this->gradObjh_k, actualStep);
      const double gradDotAlphaP =
        alpha * gradDotDirection;
      const double deltaHRMS =
        std::sqrt(arma::mean(arma::square(actualStep)));
      const arma::vec deltaU = this->M * actualStep;
      const double deltaURMS =
        std::sqrt(arma::mean(arma::square(deltaU)));
      const double projectedGradNorm =
        arma::norm(P * this->gradObjh_k, 2);

      // Retain the validated strict-monotone backtracking rule.
      const bool accepted =
        std::isfinite(obj_k_new) &&
        (obj_k_new <= obj_k);

      if (debugTraceEnabled) {
        hDebugTrace.push_back({
          static_cast<double>(debugIteration),
          static_cast<double>(k),
          static_cast<double>(iter),
          static_cast<double>(lineSearchAttempt),
          alpha,
          obj_k,
          obj_k_new,
          obj_k_new - obj_k,
          gradDotActualStep,
          gradDotAlphaP,
          stepIdentityError,
          arma::abs(p_k).max(),
          arma::abs(actualStep).max(),
          accepted ? 1.0 : 0.0,
          entryUMHDiff,
          static_cast<double>(dampingAttempts),
          dampingUsed,
          haveNewtonDirection ? 1.0 : 0.0,
          projectionChangeRMS,
          deltaHRMS,
          deltaURMS,
          projectedGradNorm
        });
      }

      if (accepted) {
        move = true;
      } else {
        alpha *= rho;
      }
    }

    if (move) {
      const double deltaF_inner = std::abs(obj_k_new - obj_k);
      const double deltaURMS = std::sqrt(
        as_scalar(arma::mean(arma::pow(M * (h_k_new - h_k), 2)))
      );
      conv = deltaF_inner < this->solved_inner_ftol_abs;
      // Optional future parameter-stability criterion:
      // conv = conv || (deltaURMS < inner_eps);
      (void)deltaURMS;

      h_k = h_k_new;
      h_k_qpart = h_k_new_qpart;
      ll_k = ll_k_new;
      //alphaStart = alpha;
      alphaStart = std::min(alpha0, alpha/rho); 
    } else {
      alphaStart =
        ((rho * alphaStart) > 1e-4)
          ? (rho * alphaStart)
          : 1e-4;
      conv = true;
    }
  }

  hmat.col(k) = h_k;
  umat.col(k) = M * h_k;
  this->ULL = ll_k;
}

template<typename T>
arma::mat gFitterOnColumns<T>::get_Zeta(){
  return( Zeta );
}
template<typename T>
arma::mat gFitterOnColumns<T>::get_hmat(){
  return( hmat );
}
template<typename T>
arma::mat gFitterOnColumns<T>::get_umat(){
  return( umat );
}
template<typename T>
arma::mat gFitterOnColumns<T>::get_vmat(){
  return( vmat );
}
template<typename T>
arma::vec gFitterOnColumns<T>::get_aphi(){
  return( aphi );
}

template<typename T>
arma::mat gFitterOnColumns<T>::get_gmat(){ // u = f + g = Sd + Rc
  return( M.cols(r, r+q-1) *  hmat.rows( r, r+q-1) );
}



//////////////////////////////////////////////
/*
 *
 * FULL ESTIMATION FOR THE DATASET
 *
 */
//////////////////////////////////////////////
template<typename T>
void gFitterOnColumns<T>::estimate(const double& f1,
                                   const double& f2,
                                   const int& maxIter,
                                   const double& eps,
                                   const double& xtol_abs,
                                   const int& inner_maxIter,
                                   const double& inner_ftol_abs){ //M, Z, qids, Rtilde, Q, B, BQ

  (void)xtol_abs; // reserved for future parameter-change convergence

  this->inner_maxIter = inner_maxIter;
  this->solved_inner_ftol_abs = n * inner_ftol_abs;


  arma::vec nllvec=this->negLogLikesVec; 
  this->calcNegLogLL(nllvec,true);
  double obj = f1* this->ULL + f2*n*getPenalty(); //objective
  double obj_old = obj;

  
  debugIteration = 0;
  recordDebugTrace(0, -1, f2);
  
  //run until convergence
  int iter=0;
  bool conv=false;
  while( (!conv) & (iter < maxIter)  ){
    iter = iter+1;
    debugIteration = iter;
    
    //calculate aphi
    update_aphi(); //aphi independent of f1, f2
    
    //move Zeta_i for each i
    moveZeta(f1, f2);
    recordDebugTrace(1, -1, f2);

    //move factors
    moveVU(f1, f2);
    obj = f1* this->ULL + f2*n*getPenalty();

    recordDebugTrace(4, -1, f2);
    //update convgence metric
    const double solved_eps = n * eps;
    conv = std::abs(obj - obj_old) < solved_eps;
    obj_old = obj;

  }

  //udpate aphi, loglike and objective to reflect new pars
  update_aphi();
  this->calcNegLogLL(nllvec, true);
  obj=f1* this->ULL + f2*n*getPenalty();
  last_f1_estimatedWith = f1;
  last_f2_estimatedWith = f2;
  
  recordDebugTrace(5, -1, f2);

}



//////////////////////////////////////////////
/*
 *
 * DEVIANCE AND MSES
 *
 *
 * MSEs are calculated as mean of y-g(eta) where g is the
 * needed transformation to the raw scale.
 *
 */
//////////////////////////////////////////////

template<typename T>
double gFitterOnColumns<T>::deviance(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1(j);
                         arma::vec y_j; y_j = Y.col(colj);
                         arma::vec eta_j = (Zeta.t() * Z.row(j).t()) + (vmat*(umat.row(j).t())) + (rowOffsets + colOffsets_for_columns(j) ) ;
                         nlength_vector(j)=this->rowFam->deviance(y_j,eta_j,aphi);
                       }
                     }
  );
  return arma::mean(nlength_vector);
}

template<typename T>
double gFitterOnColumns<T>::mse(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1(j);
                         arma::vec y_j; y_j = Y.col(colj);
                         arma::vec eta_j = (Zeta.t() * Z.row(j).t()) + (vmat*(umat.row(j).t())) + (rowOffsets + colOffsets_for_columns(j) ) ;
                         nlength_vector(j)=rowFam->mse(y_j,eta_j,aphi);
                       }
                     }
  );
  return arma::mean(nlength_vector);
}



template<typename T>
double gFitterOnColumns<T>::devianceForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  if(  colOffsetsTest.size() != Y.n_cols  ){
    throw std::invalid_argument("colOffsetsTest must have the same number of elements as the total number of columns in matrix Y");
  }

  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec mse_swiseTest( ntest, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1Test(j);
                         arma::vec yTest_j; yTest_j = Y.col(colj);
                         arma::vec etaTest_j = (Zeta.t() * ZTest.row(j).t()) + (vmat*(umatTest.row(j).t())) + (rowOffsetsTest + colOffsetsTest(colj) ) ;
                         mse_swiseTest(j)=this->rowFam->deviance(yTest_j,etaTest_j,aphi);
                       }
                     }
  );
  return arma::mean(mse_swiseTest);
}


template<typename T>
double gFitterOnColumns<T>::mseForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  if(  colOffsetsTest.size() != Y.n_cols  ){
    throw std::invalid_argument("colOffsetsTest must have the same number of elements as the total number of columns in matrix Y");
  }

  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec mse_swiseTest( ntest, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1Test(j);
                         arma::vec yTest_j; yTest_j = Y.col(colj);
                         arma::vec etaTest_j = (Zeta.t() * ZTest.row(j).t()) + (vmat*(umatTest.row(j).t())) + (rowOffsetsTest + colOffsetsTest(colj) ) ;
                         mse_swiseTest(j)=this->rowFam->mse(yTest_j,etaTest_j,aphi);
                       }
                     }
                     );
  return arma::mean(mse_swiseTest);
}


template<typename T>
double gFitterOnColumns<T>::objectiveForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  if(  colOffsetsTest.size() != Y.n_cols  ){
    throw std::invalid_argument("colOffsetsTest must have the same number of elements as the total number of columns in matrix Y");
  }

  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec LL_swiseTest( ntest, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1Test(j);
                         arma::vec yTest_j; yTest_j = Y.col(colj);
                         arma::vec etaTest_j = (Zeta.t() * ZTest.row(j).t()) + (vmat*(umatTest.row(j).t())) + (rowOffsetsTest + colOffsetsTest(colj) ) ;
                         LL_swiseTest(j)=this->rowFam->calcNegLogLL_j(yTest_j,etaTest_j,aphi);
                       }
                     }
  );
  double testULL = arma::mean(LL_swiseTest);
  double penalty = getPenalty();
  double obj =last_f1_estimatedWith* testULL + last_f2_estimatedWith*penalty ; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return obj;
}


/*
 * RECORDERS
 */
template<typename T>
void gFitterOnColumns<T>::enableDebugTrace() {
  debugTraceEnabled = true;
  debugTrace.clear();
  hDebugTrace.clear();
  hNewtonDebugTrace.clear();
  hProjectionDebugTrace.clear();
}
template<typename T>
void gFitterOnColumns<T>::recordDebugTrace(
    const int stage,
    const int component,
    const double f2
) {
  if (!debugTraceEnabled) {
    return;
  }
  
  // Linear predictor on the currently fitted columns: p x n
  arma::mat uv = vmat * umat.t();
  arma::mat eta = uv;
  
  if (Z.n_cols > 0 && Zeta.n_rows > 0) {
    eta += (Z * Zeta).t();
  }
  
  eta.each_col() += rowOffsets;
  eta.each_row() += colOffsets_for_columns.t();
  
  // Recompute the objective from the current parameter state.
  arma::vec nllvec = negLogLikesVec;
  calcNegLogLL(nllvec, true);
  const double currentObjective = ULL + f2 * getPenalty();
  
  std::size_t belowClamp = 0;
  std::size_t aboveClamp = 0;
  
  if (family == "poisson") {
    belowClamp = arma::accu(eta < -20.0);
    aboveClamp = arma::accu(eta >  20.0);
  }
  
  debugTrace.push_back({
    static_cast<double>(debugIteration), // 0
    static_cast<double>(stage),          // 1
    static_cast<double>(component),      // 2
    currentObjective,                    // 3
    umat.min(),                          // 4
    umat.max(),                          // 5
    hmat.min(),                          // 6
    hmat.max(),                          // 7
    vmat.min(),                          // 8
    vmat.max(),                          // 9
    uv.min(),                            // 10
    uv.max(),                            // 11
    eta.min(),                           // 12
    eta.max(),                           // 13
    static_cast<double>(belowClamp + aboveClamp) // 14
  });
}

template<typename T>
arma::mat gFitterOnColumns<T>::getDebugTrace() const {
  arma::mat out(debugTrace.size(), DEBUG_NCOLS, arma::fill::zeros);
  
  for (std::size_t i = 0; i < debugTrace.size(); ++i) {
    for (std::size_t j = 0; j < DEBUG_NCOLS; ++j) {
      out(i, j) = debugTrace[i][j];
    }
  }
  
  return out;
}

template<typename T>
arma::mat gFitterOnColumns<T>::getHDebugTrace() const {
  
  arma::mat out(
      hDebugTrace.size(),
      H_DEBUG_NCOLS,
      arma::fill::zeros
  );
  
  for (std::size_t i = 0; i < hDebugTrace.size(); ++i) {
    for (std::size_t j = 0; j < H_DEBUG_NCOLS; ++j) {
      out(i, j) = hDebugTrace[i][j];
    }
  }
  
  return out;
}

template<typename T>
arma::mat gFitterOnColumns<T>::getHNewtonDebugTrace() const {
  arma::mat out(hNewtonDebugTrace.size(), H_NEWTON_DEBUG_NCOLS, arma::fill::zeros);
  for (std::size_t i = 0; i < hNewtonDebugTrace.size(); ++i)
    for (std::size_t j = 0; j < H_NEWTON_DEBUG_NCOLS; ++j) out(i,j)=hNewtonDebugTrace[i][j];
  return out;
}

template<typename T>
arma::mat gFitterOnColumns<T>::getHProjectionDebugTrace() const {
  arma::mat out(
      hProjectionDebugTrace.size(),
      H_PROJECTION_DEBUG_NCOLS,
      arma::fill::zeros
  );

  for (std::size_t i = 0; i < hProjectionDebugTrace.size(); ++i) {
    for (std::size_t j = 0; j < H_PROJECTION_DEBUG_NCOLS; ++j) {
      out(i, j) = hProjectionDebugTrace[i][j];
    }
  }

  return out;
}


#endif
