#include "igFitterOnColumnsGraph.h"


///////////////////////////////////////
/*
 * Constructors for the dense dataset
 */
///////////////////////////////////////

// without weights, with Q
igFitterOnColumnsGraph::igFitterOnColumnsGraph(
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
):graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
igFitterOnColumns( denseMats, sparseMats,
                   columns1, family, Z, Zeta,
                   M, hmat, umat, vmat,
                   aphi,
                   Rtilde, qids,
                   B,
                   rowOffsets,
                   colOffsets,
                   K, r, wd, Q
                   )
{

  this->initializeOthers();

}


// without weights, without Q

// without weights, with Q
igFitterOnColumnsGraph::igFitterOnColumnsGraph(
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
):graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
igFitterOnColumns( denseMats, sparseMats,
                   columns1, family, Z, Zeta,
                   M, hmat, umat, vmat,
                   aphi,
                   Rtilde, qids,
                   B,
                   rowOffsets,
                   colOffsets,
                   K, r,wd
                     ) {

  this->initializeOthers();

}

igFitterOnColumnsGraph::igFitterOnColumnsGraph(
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
):igFitterOnColumnsGraph(
    Y.denseY,
    Y.sparseY,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat, vmat,
    aphi,
    Rtilde,
    graphDistMat,
    graphDistOnlyOnPoints,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r,
    wd,
    Q
)
{}


// without weights, without Q

// without weights, with Q
igFitterOnColumnsGraph::igFitterOnColumnsGraph(
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
):igFitterOnColumnsGraph(
    Y.denseY,
    Y.sparseY,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat, vmat,
    aphi,
    Rtilde,
    graphDistMat,
    graphDistOnlyOnPoints,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r,
    wd
) {}


void igFitterOnColumnsGraph::initializeOthers(){
  this->igFitterOnColumns::initializeOthers();
  this->G =Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints); //no prefactor of 2 unlike igFitterOnColumnsGraph; 2 enters in the grad and hess eqns.
}



//////////////////////////////////////////////
/*
 *
 * ALL LOG LIKELIHOOD STUFF : inherited from base class
 *
 */
//////////////////////////////////////////////


//////////////////////////////////////////////
/*
 *
 * Updating a(phi) : base class inherited
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
double igFitterOnColumnsGraph::getPenaltyForGraph(){
  double netPenalty=0;
  arma::vec h_k;
  for( int k =0; k<K; k++ ){
    h_k=hmat.col(k);
    netPenalty += arma::dot( h_k, G * h_k );
  }
  return netPenalty; //sumovercompoents(f2*sq norm of projection of uk on H1)
}

double igFitterOnColumnsGraph::objective(const bool recalculate,
                                    const bool updateStored,
                                    const double& f1,
                                    const double& f2,
                                    const double& f3
                                    ){
  
  double nonG_objectivePart = this->igFitterOnColumns::objective( recalculate, updateStored, f1, f2 );
  double gpenalty = this->getPenaltyForGraph();
  double obj =nonG_objectivePart + f3*gpenalty; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return obj;

}


///////////////////////////////////////////////////////////
/*
 *
 * GRADIENTS AND HESSIANS FOR THE Zeta DESCENT MOVER
 *
 * INHERITED
 *
 */
//////////////////////////////////////////////////////////
//Grad and Hessians for Zeta, the loading v_ik i.e., ith feature in kth component, and h.


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR ZETA : INHERITED
 *
 *
 * Just to avoid the overhead of checking for i for
 * weights setting, we rewrite for four cases
 * No weights, colweights only, rowweights only,
 * both weights
 *
 */
//////////////////////////////////////////////


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR V and U :
 *
 * V : all inherited
 *
 * U : defined new
 *
 *
 */
//////////////////////////////////////////////

void igFitterOnColumnsGraph::moveVU(const double& f1, const double& f2, const double& f3){

  //move each factor
  for( int k=0; k<K; k++ ){ //<= as K has been set to the Total#Components-1 in the cosntructor
    //move u_k ie., h_k with line search

    //MOVE u_K, h_k
    
    move_h_k(k, f1, f2, f3);
    umat.col(k) = M * hmat.col(k);

    //MOVE V_k
    
    move_v_k_withUnitNormConstraint_normUStrategy(k, f1, f2);
    

    //check if any nans come by
    for( int d=0; d < D; d++ ){
      arma::vec& nld =this->negLogLikesVecd[d];
      if( nld.has_inf() | nld.has_nan() | nld.has_nonfinite() ){
        throw std::invalid_argument( "Nope. Nans found in dataset: " + std::to_string(d) );
      }
    }


    if( arma::as_scalar(arma::sign( vmat[0].at(0,k) )) <0 ){//rotation identifiabilty
      
      for( int d=0; d<D; d++ ){
        vmat[d].col(k) *= -1;
      }
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
 * Gradients of just the negative log likes
 *  -- all these are inherited from base class.
 */
////////////////////////////////////////////

/////////////////////////////////////////
/*
 * Gradients of objectives
 */
/////////////////////////////////////////
void igFitterOnColumnsGraph::gradPenalizedObjective_h_k(
    const int& k,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    arma::vec& netGradh_k,
    const double& f1,
    const double& f2,
    const double& f3
){
  //get the non G part from base
  this->igFitterOnColumns::gradPenalizedObjective_h_k(
      k,
      h_k,
      h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      netGradh_k,
      f1,
      f2);

  //add the remaining G part of the gradient
  netGradh_k += 2*f3*n*(G*h_k);
}

void igFitterOnColumnsGraph::hessPenalizedObjectiveLL_h_k(
    const int& k,
    const arma::vec& h_k,
    arma::mat& netHessh_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  //grab non-G part of the hessian frm base
  this->igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
      k, h_k,netHessh_k,f1,f2);

  //add the remaining G part of the hessian
  netHessh_k += 2*f3*n*G;
}

//overloaded forms
//for K=1
void igFitterOnColumnsGraph::gradPenalizedObjective_h_k(
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const std::vector<arma::vec>& v_k,
    arma::vec& netGrad_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  //grab non-G part from base
  this->igFitterOnColumns::gradPenalizedObjective_h_k(
      h_k,
      h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      v_k,
      netGrad_h_k,
      f1,
      f2
  );
  // add G part
  netGrad_h_k += 2*f3*n*(G*h_k);
}
void igFitterOnColumnsGraph::gradPenalizedObjective_h_k(
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const std::vector<arma::vec>& v_k,
    arma::vec& netGrad_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  //grab non-G portion from base
  this->igFitterOnColumns::gradPenalizedObjective_h_k(
      umatOthers,
      vmatOthers,
      h_k,
      h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
      v_k,
      netGrad_h_k,
      f1,
      f2
  );

  //add G part
  netGrad_h_k += 2*f3*n*(G*h_k);
}

//for K=1
void igFitterOnColumnsGraph::hessPenalizedObjectiveLL_h_k(
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    arma::mat& netHess_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){
  //get nonG part
  this->igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
      h_k,
      v_k,
      netHess_h_k,
      f1,
      f2
  );

  //add G part
  netHess_h_k += 2*f3*n*G;
}
void igFitterOnColumnsGraph::hessPenalizedObjectiveLL_h_k(
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    arma::mat& netHess_h_k,
    const double& f1,
    const double& f2,
    const double& f3
){

  this->igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
      umatOthers,
      vmatOthers,
      h_k,
      v_k,
      netHess_h_k,
      f1,
      f2
  );

  //add G part
  netHess_h_k += 2*f3*n*G;
}


#if 0
// Previous igFitterOnColumnsGraph::move_h_k() retained for reference.
void igFitterOnColumnsGraph::move_h_k(
    const int& k,
    const double& f1,
    const double& f2,
    const double& f3,
    const double alpha0,
    const double rho
){

  arma::vec h_k = this->hmat.col(k);
  double ll_k;

  if( K==1 ){

    //Separate parts due to k and the rest.
    // TODO : check if K must be >1 for this to work.
    std::vector<arma::vec> v_k(D);
    for(int d=0; d<D; d++){
      v_k[d] = this->vmat[d].col(k);
    }

    arma::vec h_k_new=h_k;
    arma::vec h_k_qpart=h_k(rows_for_qxq);
    arma::vec h_k_new_qpart=h_k_qpart;

    bool conv = false; bool move=false;
    double ll_k_new, alpha;
    double obj_k, obj_k_new; //this is formed from ll_k
    arma::vec p_k;
    ll_k = this->calcNegLogLL_with_h_k( h_k, v_k ); //this is current,
    

    //TODO : check if iterations diverge and restart once from zero, as in gTCA0

    int iter=0;
    while( (!conv) & (iter <= this->inner_maxIter) ){
      iter = iter+1;

      //current stats
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart ) + f3*n*arma::dot( h_k, G * h_k ); //n because of 1/n factor in objective
      
      this->gradPenalizedObjective_h_k(h_k, h_k_qpart, v_k, this->netGradh_k, f1, f2, f3);
      this->hessPenalizedObjectiveLL_h_k(h_k, v_k, this->netHessh_k, f1, f2, f3);
      p_k = -invpsd(this->netHessh_k, 1e-7, false) * netGradh_k; //col vector
      p_k.clamp( -1e7, 1e7 );

      move =false;
      alpha = alpha0;
      while( (alpha >= 1e-4) & (!move)  ){

        h_k_new=h_k + alpha*p_k;
        h_k_new.clamp(-1e7,1e7);
        h_k_new_qpart=h_k_new(rows_for_qxq);
        
        ll_k_new = this->calcNegLogLL_with_h_k( h_k_new, v_k ); //this is current,
        
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart ) + f3*n*arma::dot( h_k_new, G * h_k_new );;
        
        if(  obj_k_new > (  obj_k + alpha*arma::dot(p_k, this->netGradh_k) ) ){
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


  } else {

    
    arma::uvec others = find( seqOfComponents != k );
    arma::mat umatOthers = this->umat.cols(others);
    std::vector<arma::mat> vmatOthers(D);
    std::vector<arma::vec> v_k(D);
    for(int d=0; d<D; d++){
      vmatOthers[d] = this->vmat[d].cols(others);
      v_k[d] = this->vmat[d].col(k);
    }


    arma::vec h_k_new=h_k;
    arma::vec h_k_qpart=h_k(rows_for_qxq);
    arma::vec h_k_new_qpart=h_k_qpart;

    arma::mat MtBQ;
    
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
    
    arma::vec p_k;
    
    ll_k = this->calcNegLogLL_with_h_k( umatOthers, vmatOthers, h_k, v_k ); //this is current,
    
    int iter=0;
    while( (!conv) & (iter <= this->inner_maxIter) ){
      iter = iter+1;

      //current stats
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart ) + f3*n*arma::dot( h_k, G * h_k );;
      this->gradPenalizedObjective_h_k(umatOthers, vmatOthers, h_k, h_k_qpart, v_k, this->netGradh_k, f1, f2, f3);
      this->hessPenalizedObjectiveLL_h_k(umatOthers, vmatOthers, h_k, v_k, this->netHessh_k, f1, f2, f3);
      p_k = -invpsd(this->netHessh_k, 1e-7, false) * netGradh_k; //col vector
      p_k.clamp( -1e7, 1e7 );

      //line search, and move
      move =false;
      alpha = alpha0;
      while( (alpha >= 1e-4) & (!move)  ){

        h_k_new=h_k + alpha*p_k;
        h_k_new.clamp(-1e7,1e7);
        h_k_new = h_k_new - HQ * h_k_new; //ortho projec
        
        h_k_new_qpart=h_k_new(rows_for_qxq);
        
        ll_k_new = this->calcNegLogLL_with_h_k( umatOthers, vmatOthers, h_k_new, v_k ); //this is current,
        
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart ) + f3*n*arma::dot( h_k_new, G * h_k_new );;
        
        if(  obj_k_new > (  obj_k + alpha*arma::dot(p_k, this->netGradh_k) ) ){
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


  hmat.col(k)=h_k;       //this resets hmat to h_k from h_k_new in case no move has happened.
  umat.col(k)= M*h_k;     // same as above but for u_k
  this->ULL = ll_k;
}
#endif

void igFitterOnColumnsGraph::move_h_k(
    const int& k,
    const double& f1,
    const double& f2,
    const double& f3,
    const double alpha0,
    const double rho
){
  arma::vec h_k = hmat.col(k);
  arma::vec h_k_qpart = h_k(rows_for_qxq);

  const arma::uvec others = find(seqOfComponents != k);
  const arma::mat umatOthers = umat.cols(others);
  std::vector<arma::mat> vmatOthers(D);
  std::vector<arma::vec> v_k(D);
  for (int d = 0; d < D; ++d) {
    vmatOthers[d] = vmat[d].cols(others);
    v_k[d] = vmat[d].col(k);
  }

  arma::mat constraintMat;
  arma::vec u0 = arma::vec(n, arma::fill::ones);
  arma::mat rawConstraintMat = M.rows(qids).t() * B * u0(qids);

  arma::mat P = arma::eye<arma::mat>(h_k.n_elem, h_k.n_elem);

  if (K > 1) {
    const arma::mat uothers = umatOthers.rows(qids);
    const arma::mat Buothers = B * uothers;

    if (BQ.n_cols > 0) {
      rawConstraintMat = arma::join_rows(
        rawConstraintMat,
        M.rows(qids).t() * arma::join_rows(BQ, Buothers)
      );
    } else {
      rawConstraintMat = arma::join_rows(
        rawConstraintMat, 
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

  double nll_k = (K == 1)
    ? calcNegLogLL_with_h_k(h_k, v_k)
    : calcNegLogLL_with_h_k(umatOthers, vmatOthers, h_k, v_k);

  if (!std::isfinite(nll_k)) {
    return;
  }

  bool converged = false;
  int iter = 0;
  while (!converged && iter <= this->inner_maxIter) {
    ++iter;

    const double obj_k =
      f1 * nll_k +
      f2 * n* arma::dot(h_k_qpart, Rtilde * h_k_qpart) +
      f3 * n* arma::dot(h_k, G * h_k);

    if (K == 1) {
      gradPenalizedObjective_h_k(
        h_k, h_k_qpart, v_k, netGradh_k, f1, f2, f3
      );
      hessPenalizedObjectiveLL_h_k(
        h_k, v_k, netHessh_k, f1, f2, f3
      );
    } else {
      gradPenalizedObjective_h_k(
        umatOthers, vmatOthers, h_k, h_k_qpart,
        v_k, netGradh_k, f1, f2, f3
      );
      hessPenalizedObjectiveLL_h_k(
        umatOthers, vmatOthers, h_k, v_k,
        netHessh_k, f1, f2, f3
      );
    }

    if (netGradh_k.has_nonfinite() || netHessh_k.has_nonfinite()) {
      break;
    }

    const arma::mat hessSym = 0.5 * (netHessh_k + netHessh_k.t());
    double damping = 1e-8 * std::max(
      1.0,
      arma::mean(arma::abs(hessSym.diag()))
    );

    arma::vec p_k;
    bool haveNewtonDirection = false;

    for (int dampingAttempt = 0;
         dampingAttempt < 10 && !haveNewtonDirection;
         ++dampingAttempt) {
      const arma::mat hessDamped =
        hessSym + damping * arma::eye<arma::mat>(
          hessSym.n_rows, hessSym.n_cols
        );

      const arma::vec absHessDiag = arma::abs(hessDamped.diag());
      const double diagScaleFloor =
        1e-12 * std::max(1.0, absHessDiag.max());
      const arma::vec newtonScale =
        1.0 / arma::sqrt(arma::clamp(
          absHessDiag,
          diagScaleFloor,
          std::numeric_limits<double>::max()
        ));
      const arma::mat Dscale = arma::diagmat(newtonScale);
      const arma::mat hessScaled =
        Dscale * hessDamped * Dscale;
      const arma::vec gradScaled = Dscale * netGradh_k;

      bool solved = false;
      arma::vec candidateDirection;

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
          candidateDirection = Dscale * candidateScaled;
        } else {
          solved = false;
        }
      } else {
        const arma::uword coefficientDim = hessScaled.n_rows;
        const arma::uword constraintDim = constraintMat.n_cols;
        const arma::mat constraintScaled =
          Dscale * constraintMat;
        arma::mat KKT(
          coefficientDim + constraintDim,
          coefficientDim + constraintDim,
          arma::fill::zeros
        );
        KKT.submat(0, 0, coefficientDim - 1, coefficientDim - 1) =
          hessScaled;
        KKT.submat(
          0, coefficientDim,
          coefficientDim - 1,
          coefficientDim + constraintDim - 1
        ) = constraintScaled;
        KKT.submat(
          coefficientDim, 0,
          coefficientDim + constraintDim - 1,
          coefficientDim - 1
        ) = constraintScaled.t();

        arma::vec rhs(coefficientDim + constraintDim, arma::fill::zeros);
        rhs.head(coefficientDim) = -gradScaled;

        arma::vec solution;
        solved = arma::solve(
          solution, KKT, rhs, arma::solve_opts::no_approx
        );
        if (solved && solution.n_elem == rhs.n_elem &&
            solution.head(coefficientDim).is_finite()) {
          candidateDirection =
            Dscale * solution.head(coefficientDim);
        } else {
          solved = false;
        }
      }

      if (solved && candidateDirection.n_elem == h_k.n_elem &&
          !candidateDirection.has_nonfinite()) {
        const double feasibilityError =
          (constraintMat.n_cols == 0)
            ? 0.0
            : arma::norm(constraintMat.t() * candidateDirection, "inf");
        const double directionalDerivative =
          arma::dot(netGradh_k, candidateDirection);

        if (std::isfinite(feasibilityError) &&
            feasibilityError <= 1e-7 &&
            std::isfinite(directionalDerivative) &&
            directionalDerivative < 0.0) {
          p_k = candidateDirection;
          haveNewtonDirection = true;
        }
      }

      if (!haveNewtonDirection) {
        damping *= 10.0;
      }
    }

    if (!haveNewtonDirection) {
      p_k = -P * netGradh_k;

      const arma::vec fallbackDeltaU = this->M * p_k;
      const double fallbackMaxAbsDeltaU =
        arma::abs(fallbackDeltaU).max();
      const double fallbackUStepMax = 20.0;
      if (std::isfinite(fallbackMaxAbsDeltaU) &&
          fallbackMaxAbsDeltaU > fallbackUStepMax) {
        p_k *= fallbackUStepMax / fallbackMaxAbsDeltaU;
      }

      const double directionalDerivative = arma::dot(netGradh_k, p_k);
      if (p_k.has_nonfinite() ||
          !std::isfinite(directionalDerivative) ||
          directionalDerivative >= 0.0) {
        break;
      }
    }

    bool accepted = false;
    double alpha = alpha0;
    arma::vec h_k_new = h_k;
    double nll_k_new = nll_k;
    double obj_k_new = obj_k;

    while (alpha >= 1e-4 && !accepted) {
      h_k_new = h_k + alpha * p_k;
      const arma::vec h_k_new_qpart = h_k_new(rows_for_qxq);

      nll_k_new = (K == 1)
        ? calcNegLogLL_with_h_k(h_k_new, v_k)
        : calcNegLogLL_with_h_k(
            umatOthers, vmatOthers, h_k_new, v_k
          );

      obj_k_new =
        f1 * nll_k_new +
        f2 * n * arma::dot(h_k_new_qpart, Rtilde * h_k_new_qpart) +
        f3 * n * arma::dot(h_k_new, G * h_k_new);

      if (std::isfinite(nll_k_new) &&
          std::isfinite(obj_k_new) &&
          obj_k_new <= obj_k) {
        accepted = true;
      } else {
        alpha *= rho;
      }
    }

    if (!accepted) {
      break;
    }

    const double deltaF_inner = std::abs(obj_k_new - obj_k);
    const double deltaURMS = std::sqrt(
      arma::mean(arma::square(M * (h_k_new - h_k)))
    );
    converged = deltaF_inner < this->solved_inner_ftol_abs;
    (void)deltaURMS;
    h_k = h_k_new;
    h_k_qpart = h_k(rows_for_qxq);
    nll_k = nll_k_new;
  }

  hmat.col(k) = h_k;
  umat.col(k) = M * h_k;
  ULL = nll_k;
}


//////////////////////////////////////////////
/*
 *
 * FULL ESTIMATION FOR THE DATASET
 *
 *
 * Negative loglikelihoods are minimized w.r.t f1, f2, and Rtilde
 * Weighted Negative Loglikelihoods(weighted NLL) are minimized when eweights are
 * given. WeightedNLL_ij = w_ij * nll_ij, w_ij = rowWeights(i)*colWeights(j)
 *
 */
//////////////////////////////////////////////
void igFitterOnColumnsGraph::estimate(const double& f1,
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
  std::vector<arma::vec> nllvec=this->negLogLikesVecd; //this stores weighted negative log likelihoods; unscaled by f1!
  this->calcNegLogLL(nllvec,true);
  double obj = f1*this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph(); //objective
  double obj_old = obj;

  //run until convergence
  int iter=0;
  bool conv=false;
  while( (!conv) & (iter < maxIter)  ){
    iter = iter+1;
    
    //calculate aphi
    update_aphi(); //aphi independent of f1, f2

    moveZeta(f1, f2);
    
    moveVU(f1, f2, f3);
    
    obj = f1*this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph();

    
    const double solved_eps = n * eps;
    conv = std::abs(obj - obj_old) < solved_eps;
    

    obj_old = obj;
    

  }

  
  update_aphi();
  this->calcNegLogLL(nllvec, true);
  obj=f1*this->ULL + f2*n*getPenalty() + f3*n*getPenaltyForGraph();
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

double igFitterOnColumnsGraph::objectiveForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){

  //get nonG part
  double nonGPartObjective = this->igFitterOnColumns::objectiveForTest(columns1Test, ZTest, MTest, rowOffsetsTest, colOffsetsTest);

  //add G part
  double gpenalty = getPenaltyForGraph();
  double obj = nonGPartObjective + last_f3_estimatedWith*gpenalty ; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return obj;
}


