#include "inpFitterOnColumnsGraph.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * CONSTRUCTORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * With Q
 */
inpFitterOnColumnsGraph::inpFitterOnColumnsGraph(  const dsListC& Y,   //without Q, orthogonal space to avoid
                                         const arma::uvec& columns,
                                         const std::vector<arma::mat>& Z,
                                         const std::vector<arma::mat> Zeta,
                                         const arma::mat& M, arma::mat hmat,
                                         arma::mat umat, std::vector<arma::mat> vmat,
                                         const arma::mat& Rtilde,
                                         const arma::sp_mat& graphDistMat,
                                         const arma::uvec& graphDistOnlyOnPoints,
                                         const arma::uvec& qids,
                                         const arma::mat& B,
                                         const std::vector<arma::vec>& rowOffsets,
                                         const std::vector<arma::vec>& colOffsets,
                                         const int K,
                                         const int r,
                                         const arma::vec& wd,
                                         const arma::mat& Q,
                                         const bool fitZeta
): graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
inpFitterOnColumns(Y, columns, Z, Zeta, M, hmat, umat, vmat,
                   Rtilde, qids, B, rowOffsets, colOffsets, K, r, wd, Q, fitZeta)  {
  this->G =Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
  // The matrix Aex (not exponential! poor name) i.e., A_expanded, is dealt with separately in the estimate() function
  //  as Aex is dependent on f1 and f2.

}


/*
 * Without Q
 */
inpFitterOnColumnsGraph::inpFitterOnColumnsGraph( const dsListC& Y,   //without Q, orthogonal space to avoid
                                        const arma::uvec& columns,
                                        const std::vector<arma::mat>& Z,
                                        const std::vector<arma::mat> Zeta,
                                        const arma::mat& M, arma::mat hmat, // our hmat will be a copy, we will use the same memory as output. Otherwise the R side input gets screwed up too! So would umat, and vmat  that we will generate.
                                        arma::mat umat, std::vector<arma::mat> vmat,
                                        const arma::mat& Rtilde,
                                        const arma::sp_mat& graphDistMat,
                                        const arma::uvec& graphDistOnlyOnPoints,
                                        const arma::uvec& qids,
                                        const arma::mat& B,
                                        const std::vector<arma::vec> rowOffsets,
                                        const std::vector<arma::vec> colOffsets,
                                        const int K,
                                        const int r,
                                        const arma::vec& wd,
                                        const bool fitZeta
): graphDistMat(graphDistMat),graphDistOnlyOnPoints(graphDistOnlyOnPoints),
inpFitterOnColumns(Y, columns, Z, Zeta, M, hmat, umat, vmat,
                   Rtilde, qids, B, rowOffsets, colOffsets, K, r, wd, fitZeta)
  {
  this->G =Mt.cols(graphDistOnlyOnPoints) *  graphDistMat * M.rows(graphDistOnlyOnPoints);
  // The matrix Aex (not exponential! poor name) i.e., A_expanded, is dealt with separately in the estimate() function
  //  as Aex is dependent on f1 and f2.

}



void inpFitterOnColumnsGraph::estimate(double f1, double f2, double f3, double maxIter, double eps, double xtol_abs){
  (void)xtol_abs; // reserved for future parameter-change convergence

  if( Y.size()==0 ){
    throw std::runtime_error("Y has not been assigned. Has the right constructor been called? ");
  }

  //It is important that we restart the internal structures here, and not
  // let estimate() be called with a previously generated Aex1 Aex2 and related inverses.
  this->setAexToAandLock();  //does this->Aex=this->A;
  this->Aex += 2*(n*f3/f1)*G; //this is the only addition; n arises because of 1/n prefactor in the objective
  this->estimateHelper(f1, f2, maxIter, eps);
  last_f3_estimatedWith=f3; //cprresonding lastf1, lastf2 updated within base class estimateHelper

}

void inpFitterOnColumnsGraph::estimate(const estimatePar& ep){
  this->estimate( ep.f1, ep.f2, ep.f3, ep.maxIter, ep.eps, .01 );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KEY: USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
double inpFitterOnColumnsGraph::getPenalty(){

  //TODO : Number of threads must be controlled.
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

double inpFitterOnColumnsGraph::getPenaltyForGraph(){

  //TODO : Number of threads must be controlled.
  double penalty = 0;
  arma::vec hk(hmat.n_rows);
  for( int k=0; k<K; k++ ){
    hk = hmat.col(k);
    penalty += arma::dot( hk, G * hk  );
  }
  return penalty;
}


// KEY: USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
double inpFitterOnColumnsGraph::objective(double f1, double f2, double f3){
  //TODO : Number of threads must be controlled.
  double lf = imse(Y, wd, Z,Zeta,umat,vmat,columns, rowOffsets, colOffsets); //must pass colOffsets and not colOffsets_for_columns
  double penalty = getPenalty();
  double gpenalty = this->getPenaltyForGraph();
  return (f1*lf)+(f2*penalty)+(f3*gpenalty); //imse is the objective term
}

double inpFitterOnColumnsGraph::objectiveForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){

  double lf = this->mseForTest( columns1Test, ZTest, MTest,
                                rowOffsetsTest, colOffsetsTest); //mse
  double penalty = this->getPenalty();
  double gpenalty = this->getPenaltyForGraph();
  double obj= last_f1_estimatedWith* lf + last_f2_estimatedWith* penalty + last_f3_estimatedWith*gpenalty ; //as before for cv, we would like to average over samples, so no multiplication by test columns size 
  return obj;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * GETTERS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<arma::mat> inpFitterOnColumnsGraph::get_Zeta_R() const{
  std::vector<arma::mat> Zeta_R(Y.size());
  for( int d=0; d < Y.size(); d++ ){
    Zeta_R[d] = Zeta[d].t();
  }
  return( Zeta_R );
}
