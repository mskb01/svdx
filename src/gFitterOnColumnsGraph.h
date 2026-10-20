#ifndef GFITTERONCOLUMNSGRAPH_H
#define GFITTERONCOLUMNSGRAPH_H

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


#include "gFitterOnColumns.h"
#include "myutils.h"


/*
 * MODELS A DENSE DATASET
 *
 * Acronyms:
 * ULL : unnorm. log likelihood
 * TS  : Thread safe
 */
template<typename T>
class gFitterOnColumnsGraph final : public gFitterOnColumns<T> {


public:
  using gFitterOnColumns<T>::Y;                         //Data object; pxn.We start with a dense dataset
  using gFitterOnColumns<T>::columns1;

  //these get updaetd to the most recent call on estimate()
  using gFitterOnColumns<T>::last_f1_estimatedWith;
  using gFitterOnColumns<T>::last_f2_estimatedWith;
  double last_f3_estimatedWith;

  //modeling
  using gFitterOnColumns<T>::family;                   //poisson or gaussian
  using gFitterOnColumns<T>::rowFam;                     //pointer to a rowFamily object
  using gFitterOnColumns<T>::Z;                         //Design matrix for batches. nxB
  using gFitterOnColumns<T>::Zeta;                       //Coefficients for Z, Bxp
  using gFitterOnColumns<T>::M;                          //U=Mh, M is n x H
  using gFitterOnColumns<T>::hmat;                          // h=[h1, h2, hK], h is H x K
  using gFitterOnColumns<T>::umat;                       //Factors U with sample loadings. R^{ nxK }
  using gFitterOnColumns<T>::vmat;                       //Feature scores for factors. R^{pxK}
  using gFitterOnColumns<T>::aphi;                       //p-length col vector. For poisson this is all 1, for Guassian, s2.
  using gFitterOnColumns<T>::K;                                //Number of factors K
  using gFitterOnColumns<T>::colOffsets;                 //offsets for samples, row vector.
  using gFitterOnColumns<T>::rowOffsets;                 //row-wise offsets.
  using gFitterOnColumns<T>::colOffsets_for_columns; //stores colOffests corresponding to desired columns

  using gFitterOnColumns<T>::r;                                // length(d) in Sd+Rc=Mh, where M=[S;R] , h=[d,c]'
  using gFitterOnColumns<T>::q;                                //length(c) in above
  using gFitterOnColumns<T>::m;                                // dim(Q) but I don't think we need it for proj grad desc.
  using gFitterOnColumns<T>::qids;
  using gFitterOnColumns<T>::rows_for_qxq;
  using gFitterOnColumns<T>::cols_for_qxq;
  using gFitterOnColumns<T>::Rtilde;                    //qxq penalty matrix. RtilTheta = R[qids,]lower left block portion of M, with row and corresponding column index r+(1+qids), r+(1+qids).

  const arma::sp_mat& graphDistMat; //qxq graph dist matrix, must be pos semi def/pos def
  const arma::uvec& graphDistOnlyOnPoints;    //the graph distances are defined for these points, in the order

  //  Subsetting on the arma end is harder because qids are non-contiguous. So
  //  pass from the R side.
  using gFitterOnColumns<T>::B;                         // qxq; this is full Kernel matrix K(.,.)^{-1} for ortho use, where K=dphi phi^T + R defined over qids.
  using gFitterOnColumns<T>::Q;                          //n-dimensional. Any n-dimensional vectors for which we need to be orthogonal to.
  using gFitterOnColumns<T>::BQ;


  using gFitterOnColumns<T>::n;                                //number of samples/observations
  using gFitterOnColumns<T>::p;                                //number of features
  using gFitterOnColumns<T>::alpha0;
  using gFitterOnColumns<T>::rho;
  using gFitterOnColumns<T>::Zt;                         //useful for intermediate computations
  using gFitterOnColumns<T>::ZtZ;                        // especially for Gaussian case, instead of recalculating for every i via rowFam
  using gFitterOnColumns<T>::Mt;
  using gFitterOnColumns<T>::MtM;

  arma::mat G;

  //some temp variables for calculations
  using gFitterOnColumns<T>::seqOfComponents;           //simply stores a vector of 1..K. Useful for intermediate compu.
  using gFitterOnColumns<T>::gradObjh_k;
  using gFitterOnColumns<T>::hessObjh_k;
  using gFitterOnColumns<T>::plength_vector;
  using gFitterOnColumns<T>::nlength_vector;

  //negative loglikelihoods
  using gFitterOnColumns<T>::negLogLikesVec;         //stores (weighted, in weightedcased) negative loglikelihoods!, not loglikelihoods!
  using gFitterOnColumns<T>::ULL;                           //unnormalized full data Log like
  // double objectiveValue; // not saving this as this is not a property of dataset; it is a property of a fitter, and parmaeters f1 and f2.

  ///////////////////////////////
  /*
   * Functions used in estiation
   */
  ///////////////////////////////

  // -- these functions don't have f3 / G dependence
  using gFitterOnColumns<T>::calc_aphi_i;
  using gFitterOnColumns<T>::update_aphi;
  using gFitterOnColumns<T>::moveZeta_i;
  using gFitterOnColumns<T>::gradNegLogLL_v_ik;
  using gFitterOnColumns<T>::hessNegLogLL_v_ik;
  using gFitterOnColumns<T>::move_v_k_i_normUkStrategy;
  using gFitterOnColumns<T>::move_v_k_withUnitNormConstraint_normUStrategy;
  using gFitterOnColumns<T>::move_v_k_i;
  using gFitterOnColumns<T>::move_v_k;               //same as above but for an entire v
  using gFitterOnColumns<T>::moveZeta;                    //calls all moveZeta_i parallely

  // these functions have f3 and G depdence
  void move_h_k( const int k, const double& f1,  const double& f2,
                 const double& f3,
                 double alpha0=1,
                 double rho=.5
  );               //uk= M_k h_k, and h_k is moved for a given k=1..K
  void moveVU(const double& f1, const double& f2,
              const double& f3
              );                      //calls each move_v_k (which in turn calls mov_v_k_i parallely) and h_k serially one after the other.
  void descend();



  //some getters
  using gFitterOnColumns<T>::get_Zeta;
  using gFitterOnColumns<T>::get_hmat;
  using gFitterOnColumns<T>::get_umat;
  using gFitterOnColumns<T>::get_vmat;
  using gFitterOnColumns<T>::get_gmat;
  using gFitterOnColumns<T>::get_aphi;

  //override
  arma::vec get_objective();


  //////////////////////////////////////////////////////////////
  /*
   * CONSTRUCTORS based on what information is used or not
   * For example: if there are no weights, offsets etc.,
   *        without weights, we can be faster, and we want to
   *        take advantage of that situation.
   */
  //////////////////////////////////////////////////////////////

  explicit gFitterOnColumnsGraph( //without Q
    const T& Y,
    const arma::uvec& columns1,
    const std::string family,
    const arma::mat& Z, arma::mat Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, arma::mat vmat,
    arma::vec aphi,
    const arma::mat& Rtilde,
    const arma::sp_mat& graphDistMat,
    const arma::uvec& graphDistOnlyOnPoints,
    const arma::uvec qids,
    const arma::mat B,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const unsigned int K,
    const unsigned int r
  );

  explicit gFitterOnColumnsGraph( //with Q
    const T& Y,
    const arma::uvec& columns1,
    const std::string family,
    const arma::mat& Z, arma::mat Zeta,
    const arma::mat& M, arma::mat hmat,
    arma::mat umat, arma::mat vmat,
    arma::vec aphi,
    const arma::mat& Rtilde,
    const arma::sp_mat& graphDistMat,
    const arma::uvec& graphDistOnlyOnPoints,
    const arma::uvec qids,
    const arma::mat B,
    const arma::vec& rowOffsets,
    const arma::vec& colOffsets,
    const unsigned int K,
    const unsigned int r,
    const arma::mat Q
  );

  ~gFitterOnColumnsGraph()=default;

  //////////////////////////////////////////////////////////////
  /*
   * NegativeLogLike calculations for individual feature & full dataset
   */
  //////////////////////////////////////////////////////////////
  using gFitterOnColumns<T>::calcNegLogLL_i; // all the overloaded forms are inherited
  using gFitterOnColumns<T>::calcNegLogLL_j_with_h_k;
  using gFitterOnColumns<T>::calcNegLogLL_with_h_k;
  using gFitterOnColumns<T>::calcNegLogLL;


  //////////////////////////////////////////////////////////////
  /*
   * Gradient and Hessians for Zeta, the loading v_ik i.e.,
   *    ith feature in kth component, and h_k.
   */
  //////////////////////////////////////////////////////////////
  using gFitterOnColumns<T>::gradNegLogLL_Zeta_i; // col vector
  using gFitterOnColumns<T>::hessNegLogLL_Zeta_i;
  using gFitterOnColumns<T>::gradNegLogLL_h_k;  //pulling these two out in public; partly also because this enters integrative code.
  using gFitterOnColumns<T>::hessNegLogLL_h_k;


  //////////////////////////////////////////////////////////////
  /*
   * Objective stuff : all have dependence on f3 and G, so cannot use base class
   */
  //////////////////////////////////////////////////////////////
  void gradPenalizedObjective_h_k(
      const int k,
      const arma::vec& h_k,
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      arma::vec& grad_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );

  void hessPenalizedObjectiveLL_h_k(
      const int k,
      const arma::vec& h_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );
  // overloaded forms of the two forms above
  void gradPenalizedObjective_h_k( // for K = 1
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2,
      const double& f3
  );
  void gradPenalizedObjective_h_k(
      const arma::mat& umatOthers, //containing columns of umat except for the k^th
      const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
      const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
      const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      const arma::vec& v_k,       // corresponding loading vector
      arma::vec& grad_h_k,  //to store gradient in
      const double& f1,
      const double& f2,
      const double& f3
  );
  void hessPenalizedObjectiveLL_h_k( //for K =1 w/o umatOthers, and vmatOthers
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );
  void hessPenalizedObjectiveLL_h_k(
      const arma::mat& umatOthers,
      const arma::mat& vmatOthers,
      const arma::vec& h_k,
      const arma::vec& v_k,
      arma::mat& hess_h_k,
      const double& f1,
      const double& f2,
      const double& f3
  );

  using gFitterOnColumns<T>::getPenalty;
  double getPenaltyForGraph();
  double objective(const bool recalculate, const bool updateStored,
                   const double& f1, const double& f2, const double& f3 );

  //////////////////////////////////////////////////////////////
  /*
   *  ESTIMATOR
   */
  //////////////////////////////////////////////////////////////
  // void estimateNP(double f1, double f2, //the nonparametric PCA-like form for a single dataset
  //               int maxIter=25, double eps=1e-4);

  void estimate(const double& f1, const double& f2, const double& f3,
                const int& maxIter=25, const double& eps=1e-4,
                const double& xtol_abs=.01,
                const int& inner_maxIter=25,
                const double& inner_ftol_abs=1e-4);


  //MSE AND DEVIANCE AND OBJECTIVE
  using gFitterOnColumns<T>::deviance;
  using gFitterOnColumns<T>::devianceForTest;

  using gFitterOnColumns<T>::mse;
  using gFitterOnColumns<T>::mseForTest;

  //this depends on f3 and G
  double objectiveForTest(
      const arma::uvec& columns1Test,
      const arma::mat& ZTest,
      const arma::mat& MTest,
      const arma::vec& rowOffsetsTest,
      const arma::vec& colOffsetsTest
  );
};

template class gFitterOnColumnsGraph<arma::mat>;
template class gFitterOnColumnsGraph<arma::sp_mat>;


///////////////////////////////////////
/*
 * Constructors for the dense dataset
 */
///////////////////////////////////////

// without weights, with Q
template<typename T>
gFitterOnColumnsGraph<T>::gFitterOnColumnsGraph(
  const T& Y,
  const arma::uvec& columns1,
  const std::string family,
  const arma::mat& Z, arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  arma::vec aphi,
  const arma::mat& Rtilde,
  const arma::sp_mat& graphDistMat,
  const arma::uvec& graphDistOnlyOnPoints,
  const arma::uvec qids,
  const arma::mat B,
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const unsigned int K,
  const unsigned int r,
  const arma::mat Q
):gFitterOnColumns<T>(
    Y,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat,vmat,
    aphi,
    Rtilde,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r,
    Q
), graphDistMat(graphDistMat), graphDistOnlyOnPoints(graphDistOnlyOnPoints)
{
  this->G =Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
}


// without weights, without Q
template<typename T>
gFitterOnColumnsGraph<T>::gFitterOnColumnsGraph(
  const T& Y,
  const arma::uvec& columns1,
  const std::string family,
  const arma::mat& Z, arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  arma::vec aphi,
  const arma::mat& Rtilde,
  const arma::sp_mat& graphDistMat,
  const arma::uvec& graphDistOnlyOnPoints,
  const arma::uvec qids,
  const arma::mat B,
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const unsigned int K,
  const unsigned int r
):gFitterOnColumns<T>(
    Y,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat,vmat,
    aphi,
    Rtilde,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r
), graphDistMat(graphDistMat), graphDistOnlyOnPoints(graphDistOnlyOnPoints)
  {
  this->G =Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
  }

//////////////////////////////////////////////
/*
 *
 * All LOG LIKELIHOOD STUFF inherited from base class
 *
 */
//////////////////////////////////////////////



//////////////////////////////////////////////
/*
 *
 * Objective and penalty
 *
 */
//////////////////////////////////////////////

template<typename T>
double gFitterOnColumnsGraph<T>::getPenaltyForGraph(){

  //TODO : Number of threads must be controlled.
  double penalty = 0;
  arma::vec hk(hmat.n_rows);
  for( int k=0; k<K; k++ ){
    hk = hmat.col(k);
    penalty += arma::dot( hk, G * hk  );
  }
  return penalty;
}


template<typename T>
double gFitterOnColumnsGraph<T>::objective(const bool recalculate,
                                      const bool updateStored,
                                      const double& f1,
                                      const double& f2,
                                      const double& f3
                                      ){


  //get non-G part
  double nonG_objectivePart = this->gFitterOnColumns<T>::objective( recalculate, updateStored, f1, f2 );
  double gpenalty = this->getPenaltyForGraph();
  double obj =nonG_objectivePart + f3*gpenalty; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return obj;
}


///////////////////////////////////////////////////////////
/*
 *
 * GRADIENTS AND HESSIANS FOR THE Zeta DESCENT MOVER
 *
 * inherited
 */
//////////////////////////////////////////////////////////
//Grad and Hessians for Zeta, the loading v_ik i.e., ith feature in kth component, and h.


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR ZETA
 *
 *     INnherited from base class.
 *
 *
 */
//////////////////////////////////////////////


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR V and U
 *
 *
 *. Mover for V, inherited from Base Class
 */
//////////////////////////////////////////////


template<typename T>
void gFitterOnColumnsGraph<T>::moveVU(const double& f1, const double& f2, const double& f3){

  
  //move each factor
  for( int k=0; k<K; k++ ){ //<= as K has been set to the Total#Components-1 in the cosntructor
    //move u_k ie., h_k with line search

    
    move_h_k(k, f1, f2, f3); //STATEMENT
    umat.col(k) = M * hmat.col(k);   //STATEMENT

    
    move_v_k_withUnitNormConstraint_normUStrategy(k, f1, f2); //STATEMENT!

    
    if( arma::as_scalar(arma::sign( vmat(0,k) )) <0 ){//rotation identifiabilty
      vmat.col(k) *= -1;
      hmat.col(k) *= -1;
      umat.col(k) *= -1;
    }

  }

}




////////////////////////////////////////////////
/*
 * Mover for h_k
 */
////////////////////////////////////////////////


////////////////////////////////////////////
/*
 * Gradients of hk just the negative log likes
 *  inherited from base class
 */
////////////////////////////////////////////


/////////////////////////////////////////
/*
 * Gradients of objectives
 */
/////////////////////////////////////////
template<typename T>
void gFitterOnColumnsGraph<T>::gradPenalizedObjective_h_k(
    const int k,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  //get the non G part from base
  this->gFitterOnColumns<T>::gradPenalizedObjective_h_k(
    k,
    h_k,
    h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    grad_h_k,
    f1,
    f2);

  //add the remaining G part of the gradient
  grad_h_k += 2*f3*n*(G*h_k);
}

template<typename T>
void gFitterOnColumnsGraph<T>::hessPenalizedObjectiveLL_h_k(
    const int k,
    const arma::vec& h_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  //grab non-G part of the hessian frm base
  this->gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
    k, h_k,hess_h_k,f1,f2);

  //add the remaining G part of the hessian
  hess_h_k += 2*f3*n*G;
}

//overloaded forms
template<typename T>
void gFitterOnColumnsGraph<T>::gradPenalizedObjective_h_k(
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const arma::vec& v_k,
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){


  //grab non-G part from base
  this->gFitterOnColumns<T>::gradPenalizedObjective_h_k(
      h_k,
      h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      v_k,
      grad_h_k,
      f1,
      f2
  );
  // add G part
  grad_h_k += 2*f3*n*(G*h_k);
}

template<typename T>
void gFitterOnColumnsGraph<T>::gradPenalizedObjective_h_k(
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const arma::vec& v_k,
    arma::vec& grad_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){


  //grab non-G portion from base
  this->gFitterOnColumns<T>::gradPenalizedObjective_h_k(
      umatOthers,
      vmatOthers,
      h_k,
      h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      v_k,
      grad_h_k,
      f1,
      f2
  );
  grad_h_k += 2*f3*n*(G*h_k);
}

//for K =1
template<typename T>
void gFitterOnColumnsGraph<T>::hessPenalizedObjectiveLL_h_k(
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){
  //grab non-G part from base
  this->gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
      h_k,
      v_k,
      hess_h_k,
      f1,
      f2
      );

  //add the G part
  hess_h_k += 2*f3*n*G;
}
template<typename T>
void gFitterOnColumnsGraph<T>::hessPenalizedObjectiveLL_h_k(
    const arma::mat& umatOthers,
    const arma::mat& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& v_k,
    arma::mat& hess_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){
  //grab non-G part
  this->gFitterOnColumns<T>::hessPenalizedObjectiveLL_h_k(
      umatOthers,
      vmatOthers,
      h_k,
      v_k,
      hess_h_k,
      f1,
      f2
      );
  // add the G part
  hess_h_k += 2*f3*n*G;
}


#if 0
template<typename T>
void gFitterOnColumnsGraph<T>::move_h_k(
    const int k,
    const double& f1,
    const double& f2,
    const double& f3,
    const double alpha0,
    const double rho
){


  arma::vec h_k = this->hmat.col(k);
  double ll_k;
  if( K ==1 ){

    //Separate parts due to k and the rest.
    // TODO : check if K must be >1 for this to work.
    arma::vec v_k = this->vmat.col(k);
    arma::vec h_k_new=h_k;
    arma::vec h_k_qpart=h_k(rows_for_qxq);
    arma::vec h_k_new_qpart=h_k_qpart;

    bool conv = false; bool move=false;
    double ll_k_new, alpha;
    double obj_k, obj_k_new; //this is formed from ll_k
    //gradll_k and hessh_k are around in this-> which we can reuse w/o regenerating.
    arma::vec p_k;
    // std::cout << "first : attempting calcneglogll_with_hk before while loop" << std::endl;
    ll_k = this->calcNegLogLL_with_h_k( h_k, v_k ); //this is current,
    // std::cout << "done with that" << std::endl;

    //TODO : check if iterations diverge and restart once from zero, as in gTCA0

    int iter=0;
    while( (!conv) & (iter <= this->inner_maxIter) ){
      iter = iter+1;

      //current stats
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart ) + f3*n*arma::dot( h_k, G * h_k );
      // std::cout << "second : attempting gradPenalized in while loop" << std::endl;
      this->gradPenalizedObjective_h_k(h_k, h_k_qpart, v_k, this->gradObjh_k, f1, f2, f3);
      // std::cout << "done with that. " << std::endl;
      // std::cout << "third : attempting hess in while loop" << std::endl;
      this->hessPenalizedObjectiveLL_h_k(h_k, v_k, this->hessObjh_k, f1, f2, f3);
      // std::cout << "done with that" << std::endl;
      p_k = -invpsd(this->hessObjh_k, 1e-7, false) * gradObjh_k; //col vector
      // std::cout << "fourth : done with inversion" << std::endl;
      p_k.clamp( -1e7, 1e7 );

      //line search, and move
      move =false;
      alpha = alpha0;
      while( (alpha >= 1e-4) & (!move)  ){

        h_k_new=h_k + alpha*p_k;
        h_k_new.clamp(-1e7,1e7);
        // std::cout << "fifth : in inner while. attempting projection" << std::endl;
        //h_k_new = h_k_new - HQ * h_k_new; //ortho projec
        // std::cout << "done with that" << std::endl;
        h_k_new_qpart=h_k_new(rows_for_qxq);
        // std::cout << "sixth : done extracting q part, attempting calcneglogll_with_hk" << std::endl;
        ll_k_new = this->calcNegLogLL_with_h_k( h_k_new, v_k ); //this is current,
        // std::cout << "done with thta." << std::endl;
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart ) +  f3*n*arma::dot( h_k_new, G * h_k_new );
        // std::cout << "seventh : done calculating penalty" << std::endl;
        if(  obj_k_new > (  obj_k + alpha*arma::dot(p_k, gradObjh_k) ) ){
          alpha = rho*alpha;
        } else {
          move = true;
        }
      }

      if( move ){
        conv = std::sqrt(as_scalar(arma::mean(arma::pow(h_k_new-h_k,2)))) < 1e-4;
        h_k = h_k_new; //hmat.col(k)=h_k_new, and umat.col(k)=Mh_k_new is not needed as we already changed em in the prev block.
        h_k_qpart=h_k_new_qpart;
        ll_k=ll_k_new;
        obj_k = obj_k_new;
        alpha = alpha/rho;
      } else {
        conv = true; //return with current Zeta, as no step has been made
      }
    }


  } else { //K >1

    //Separate parts due to k and the rest.
    // TODO : check if K must be >1 for this to work.
    arma::uvec others = find( seqOfComponents != k );
    arma::mat umatOthers = this->umat.cols(others);
    arma::mat vmatOthers = this->vmat.cols(others);
    arma::vec v_k = this->vmat.col(k);

    arma::vec h_k_new=h_k;
    arma::vec h_k_qpart=h_k(rows_for_qxq);
    arma::vec h_k_new_qpart=h_k_qpart;

    arma::mat MtBQ;
    //for orthogonal projectiom
    // TODO : check if K must be >1 for this to work.
    // if(knotsType=="approx"){
    //   arma::mat uothers = umatOthers.rows(qids);
    //   arma::mat Buothers=B*uothers; // B doesn't need subset; it is defined over qids at the input
    //   if(BQ.n_cols>0){
    //     MtBQ = M.rows(qids).t() * arma::join_cols( BQ, Buothers );
    //   } else {
    //     MtBQ = M.rows(qids).t() * Buothers;
    //   }
    // } else {
    //   arma::mat Buothers= B * umatOthers;
    //   if( BQ.n_cols > 0 ){
    //     MtBQ = M.t() * arma::join_cols( BQ, Buothers);
    //   } else {
    //     MtBQ = M.t() * Buothers;
    //   }
    // }
    arma::mat uothers = umatOthers.rows(qids);
    arma::mat Buothers=B*uothers; // B doesn't need subset; it is defined over qids at the input
    if(BQ.n_cols>0){
      MtBQ = M.rows(qids).t() * arma::join_cols( BQ, Buothers );
    } else {
      MtBQ = M.rows(qids).t() * Buothers;
    }
    arma::mat MtBQ2=MtBQ.t() * MtBQ;
    arma::mat HQ = MtBQ * invpsd( MtBQ2, 1e-7, false ) * MtBQ.t(); //not a big mat (r+q+m x r+q+m)

    bool conv = false; bool move=false;
    double ll_k_new, alpha;
    double obj_k, obj_k_new; //this is formed from ll_k
    //gradll_k and hessh_k are around in this-> which we can reuse w/o regenerating.
    arma::vec p_k;
    // std::cout << "first : attempting calcneglogll_with_hk before while loop" << std::endl;
    ll_k = this->calcNegLogLL_with_h_k( umatOthers, vmatOthers, h_k, v_k ); //this is current,
    // std::cout << "done with that" << std::endl;

    //TODO : check if iterations diverge and restart once from zero, as in gTCA0

    int iter=0;
    while( (!conv) & (iter <= this->inner_maxIter) ){
      iter = iter+1;

      //current stats
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart ) + f3*n*arma::dot( h_k, G * h_k );
      // std::cout << "second : attempting gradPenalized in while loop" << std::endl;
      this->gradPenalizedObjective_h_k(umatOthers, vmatOthers, h_k, h_k_qpart, v_k, this->gradObjh_k, f1, f2, f3);
      // std::cout << "done with that. " << std::endl;
      // std::cout << "third : attempting hess in while loop" << std::endl;
      this->hessPenalizedObjectiveLL_h_k(umatOthers, vmatOthers, h_k, v_k, this->hessObjh_k, f1, f2, f3);
      // std::cout << "done with that" << std::endl;
      p_k = -invpsd(this->hessObjh_k, 1e-7, false) * gradObjh_k; //col vector
      // std::cout << "fourth : done with inversion" << std::endl;
      p_k.clamp( -1e7, 1e7 );

      //line search, and move
      move =false;
      alpha = alpha0;
      while( (alpha >= 1e-4) & (!move)  ){

        h_k_new=h_k + alpha*p_k;
        h_k_new.clamp(-1e7,1e7);
        // std::cout << "fifth : in inner while. attempting projection" << std::endl;
        h_k_new = h_k_new - HQ * h_k_new; //ortho projec
        // std::cout << "done with that" << std::endl;
        h_k_new_qpart=h_k_new(rows_for_qxq);
        // std::cout << "sixth : done extracting q part, attempting calcneglogll_with_hk" << std::endl;
        ll_k_new = this->calcNegLogLL_with_h_k( umatOthers, vmatOthers, h_k_new, v_k ); //this is current,
        // std::cout << "done with thta." << std::endl;
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart ) + f3*n*arma::dot( h_k_new, G * h_k_new );
        // std::cout << "seventh : done calculating penalty" << std::endl;
        if(  obj_k_new > (  obj_k + alpha*arma::dot(p_k, gradObjh_k) ) ){
          alpha = rho*alpha;
        } else {
          move = true;
        }
      }

      if( move ){
        conv = std::sqrt(as_scalar(arma::mean(arma::pow(h_k_new-h_k,2)))) < 1e-4;
        h_k = h_k_new; //hmat.col(k)=h_k_new, and umat.col(k)=Mh_k_new is not needed as we already changed em in the prev block.
        h_k_qpart=h_k_new_qpart;
        ll_k=ll_k_new;
        obj_k = obj_k_new;
        alpha = alpha/rho;
      } else {
        conv = true; //return with current Zeta, as no step has been made
      }
    }


  }


  //likely thread safe as the indices in the vector are independently written to.
  hmat.col(k)=h_k;       //this resets hmat to h_k from h_k_new in case no move has happened.
  umat.col(k)= M*h_k;     // same as above but for u_k
  this->ULL = ll_k;
}
#endif

template<typename T>
void gFitterOnColumnsGraph<T>::move_h_k(
    const int k,
    const double& f1,
    const double& f2,
    const double& f3,
    const double alpha0,
    const double rho
){

  double& alphaStart = this->hAlphaStart(k);

  
  const double entryUMHDiff =
    arma::abs(
      this->umat.col(k) - this->M * this->hmat.col(k)
    ).max();

  arma::vec h_k = this->hmat.col(k);
  arma::vec v_k = this->vmat.col(k);
  arma::vec h_k_qpart = h_k(rows_for_qxq);

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
      rawConstraintMat =arma::join_rows( rawConstraintMat, 
                                         M.rows(qids).t() * arma::join_rows(BQ, Buothers)
                                         );
        
    } else {
      rawConstraintMat = arma::join_rows(rawConstraintMat, 
                                         M.rows(qids).t() * Buothers);
      
    }
  }
  
  constraintMat = arma::orth(rawConstraintMat);
  
  if (constraintMat.n_cols > 0) {
    P -= constraintMat * constraintMat.t();
    
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
      f2 *n* arma::dot(h_k_qpart, Rtilde * h_k_qpart) +
      f3 *n* arma::dot(h_k, G * h_k);

    if (K == 1) {
      this->gradPenalizedObjective_h_k(
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2, f3
      );
      this->hessPenalizedObjectiveLL_h_k(
        h_k, v_k,
        this->hessObjh_k, f1, f2, f3
      );
    } else {
      this->gradPenalizedObjective_h_k(
        umatOthers, vmatOthers,
        h_k, h_k_qpart, v_k,
        this->gradObjh_k, f1, f2, f3
      );
      this->hessPenalizedObjectiveLL_h_k(
        umatOthers, vmatOthers,
        h_k, v_k,
        this->hessObjh_k, f1, f2, f3
      );
    }

    if ((!this->gradObjh_k.is_finite()) ||
        (!this->hessObjh_k.is_finite())) {
      throw std::runtime_error(
        "gFitterOnColumnsGraph::move_h_k: non-finite gradient or Hessian"
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

        if ((feasibilityError <= feasibilityTolerance) &&
            std::isfinite(gradDotCandidate) &&
            (gradDotCandidate < 0.0)) {
          p_k = candidateDirection;
          haveNewtonDirection = true;
          dampingUsed = currentDamping;
        }
      }

      damping *= 10.0;
    }

    if (!haveNewtonDirection) {
      if (constraintMat.n_cols==0) {
        p_k = -this->gradObjh_k;
      } else {
        p_k = -(P * this->gradObjh_k);
      }

      const arma::vec fallbackDeltaU = this->M * p_k;
      const double fallbackMaxAbsDeltaU =
        arma::abs(fallbackDeltaU).max();
      const double fallbackUStepMax = 20.0;
      if (std::isfinite(fallbackMaxAbsDeltaU) &&
          fallbackMaxAbsDeltaU > fallbackUStepMax) {
        p_k *= fallbackUStepMax / fallbackMaxAbsDeltaU;
      }
    }

    if (!p_k.is_finite()) {
      throw std::runtime_error(
        "gFitterOnColumnsGraph::move_h_k: non-finite Newton/RGD direction"
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
          f2 *n* arma::dot(
            h_k_new_qpart,
            Rtilde * h_k_new_qpart
          ) +
          f3 *n* arma::dot(h_k_new, G * h_k_new);
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

      // Retain the validated strict-monotone backtracking rule.
      const bool accepted =
        std::isfinite(obj_k_new) &&
        (obj_k_new <= obj_k);

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
      alphaStart = std::min( alpha0, alpha/rho);
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



//////////////////////////////////////////////
/*
 *
 * FULL ESTIMATION FOR THE DATASET
 *
 */
//////////////////////////////////////////////
template<typename T>
void gFitterOnColumnsGraph<T>::estimate(const double& f1,
                                   const double& f2,
                                   const double& f3,
                                   const int& maxIter,
                                   const double& eps,
                                   const double& xtol_abs,
                                   const int& inner_maxIter,
                                   const double& inner_ftol_abs){ //M, Z, qids, Rtilde, Q, B, BQ

  (void)xtol_abs; // reserved for future parameter-change convergence

  this->inner_maxIter = inner_maxIter;
  this->solved_inner_ftol_abs = n * inner_ftol_abs;


  
  //update NLL : necessary before calling any of the below!
  arma::vec nllvec=this->negLogLikesVec; 
  this->calcNegLogLL(nllvec,true);
  double obj = f1* this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph(); //objective
  double obj_old = obj;

  //run until convergence
  int iter=0;
  bool conv=false;
  while( (!conv) & (iter < maxIter)  ){
    iter = iter+1;
    
    //calculate aphi
    update_aphi(); //aphi independent of f1, f2
    
    //move Zeta_i for each i
    moveZeta(f1, f2);
    
    //move factors
    moveVU(f1, f2, f3);
    
    obj = f1* this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph();

    const double solved_eps = n * eps;
    conv = std::abs(obj - obj_old) < solved_eps;
    
    obj_old = obj;

  }

  //udpate aphi, loglike and objective to reflect new pars
  update_aphi();
  this->calcNegLogLL(nllvec, true);
  obj=f1* this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph();
  last_f1_estimatedWith = f1;
  last_f2_estimatedWith = f2;
  last_f3_estimatedWith = f3;

  
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
 * Nearly all, except objective, INHERITED AS IS from Base class
 *
 */
//////////////////////////////////////////////

template<typename T>
double gFitterOnColumnsGraph<T>::objectiveForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  //get nonGpart
  double nonGpartObjective = this->gFitterOnColumns<T>::objectiveForTest( columns1Test,
                                                                            ZTest,
                                                                          MTest,
                                                                          rowOffsetsTest,
                                                                          colOffsetsTest );
  // add G part
  double gpenalty = getPenaltyForGraph();
  double obj =nonGpartObjective + last_f3_estimatedWith*gpenalty ;
  return obj;
}



#endif
