#include "inpFitterOnColumns.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * CONSTRUCTORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * With Q
 */
inpFitterOnColumns::inpFitterOnColumns(  const dsListC& Y,   //without Q, orthogonal space to avoid
                                         const arma::uvec& columns,
                                         const std::vector<arma::mat>& Z,
                                         const std::vector<arma::mat> Zeta,
                                         const arma::mat& M, arma::mat hmat,
                                         arma::mat umat, std::vector<arma::mat> vmat,
                                         const arma::mat& Rtilde,
                                         const arma::uvec& qids,
                                         const arma::mat& B,
                                         const std::vector<arma::vec>& rowOffsets,
                                         const std::vector<arma::vec>& colOffsets,
                                         const int K,
                                         const int r,
                                         const arma::vec& wd, 
                                         const arma::mat& Q,
                                         const bool fitZeta
): Y(Y),columns(columns), Z(Z), Zeta(Zeta),
M(M), hmat(hmat), umat(umat), vmat(vmat),
Rtilde(Rtilde), qids(qids), B(B),
rowOffsets(rowOffsets),colOffsets(colOffsets),
K(K), r(r),wd(wd), Q(Q), fitZeta(fitZeta), n(columns.n_elem)  {


  //TODO enforce a check for length(qids)==nrow(Rtilde)==ncol(Rtilde)
  //this->K = K;
  //this->r = r;
  this->q = qids.size();
  this->seqOfComponents=arma::regspace<uvec>(0,K-1);
  this->qindx= arma::regspace<arma::uvec>( r, r+q-1 );
  this->rqindx= arma::regspace<arma::uvec>( 0, r+q-1 );

  //initialize Z, Zeta and vmat
  this->D = Y.size();
  this->Zt = std::vector<arma::mat>(D);
  this->ZtZ = std::vector<arma::mat>(D);
  this->Zls = std::vector<arma::mat>(D);
  this->YZlst = std::vector<arma::mat>(D);
  this->ZtYtilde = std::vector<arma::mat>(D);
  this->pd = std::vector<int>(D);
  this->ones_pd = std::vector<arma::vec>( D );
  this->ones_n = arma::vec( n, arma::fill::ones );
  this->colOffsets_for_columns = std::vector<arma::vec>(D);
  for(int d=0; d<D; d++){

    this->Zt[d] = Z[d].t();
    this->ZtZ[d] = Zt[d] * Z[d];
    this->Zls[d] = invpsd( ZtZ[d], 1e-6, false ) * Zt[d];
    colOffsets_for_columns[d] = colOffsets[d].elem(columns);

    //first term same as using Y.postMultiplyWithMatReturnMat( d, (Zls[d].t()) );
    arma::mat Zlsdt = Zls[d].t();
    this->pd[d]=Y.nrow(d);
    this->ones_pd[d] = arma::vec( this->pd[d], arma::fill::ones );
    this->YZlst[d] = ((Y.postMultiplyWithMatReturnMat(d, Zlsdt, columns)).t()).eval(); //(( yd  * (Zls[d].t())).t()).eval(); // (pxn x  nx|Zeta|)' = |Zeta| x p ; adjusting YZlst for offsets outside of the block
    this->YZlst[d] = this->YZlst[d] - ((( Zls[d] * colOffsets_for_columns[d] ).eval())*(ones_pd[d].t())) - ((( Zls[d] * ones_n ).eval())*(rowOffsets[d].t())); //accounting for offsets
    
    arma::mat Zd = Z[d];
    this->ZtYtilde[d] = (Y.postMultiplyWithMatReturnMat(d, Zd, columns).t()).eval()
      - (Zt[d] * colOffsets_for_columns[d]) * ones_pd[d].t()
      - (Zt[d] * ones_n) * rowOffsets[d].t();
  }
  this->spd= std::accumulate( pd.begin(), pd.end(), 0  );

  //construct A, Z
  this->Mt = M.t();
  this->A =2* Mt * M;
  // The matrix Aexp (not exponential! poor name) i.e., A_expanded, is dealt with separately in the estimate() function
  //  as Aexp is dependent on f1 and f2.
  this->w_hat = arma::vec( n, arma::fill::zeros );
  this->v_score = std::vector<arma::vec>(D);
  this->lf_swise=arma::vec(n, arma::fill::zeros);
}


/*
 * Without Q
 */
inpFitterOnColumns::inpFitterOnColumns( const dsListC& Y,   //without Q, orthogonal space to avoid
                                        const arma::uvec& columns,
                                        const std::vector<arma::mat>& Z,
                                        const std::vector<arma::mat> Zeta,
                                        const arma::mat& M, arma::mat hmat, // our hmat will be a copy, we will use the same memory as output. Otherwise the R side input gets screwed up too! So would umat, and vmat  that we will generate.
                                        arma::mat umat, std::vector<arma::mat> vmat,
                                        const arma::mat& Rtilde,
                                        const arma::uvec& qids,
                                        const arma::mat& B,
                                        const std::vector<arma::vec> rowOffsets,
                                        const std::vector<arma::vec> colOffsets,
                                        const int K,
                                        const int r,
                                        const arma::vec& wd, 
                                        const bool fitZeta
): Y(Y), columns(columns), Z(Z), Zeta(Zeta),
M(M), hmat(hmat), umat(umat), vmat(vmat),
Rtilde(Rtilde), qids(qids), B(B),
rowOffsets(rowOffsets),colOffsets(colOffsets),
K(K), r(r), wd(wd), fitZeta(fitZeta), n(columns.n_elem)  {


  
  this->q = qids.size();
  this->seqOfComponents=arma::regspace<uvec>(0,K-1);
  this->qindx= arma::regspace<arma::uvec>( r, r+q-1 );
  this->rqindx= arma::regspace<arma::uvec>( 0, r+q-1 );

  //initialize Z, Zeta and vmat
  this->D = Y.size();
  this->Zt = std::vector<arma::mat>(D);
  this->ZtZ = std::vector<arma::mat>(D);
  this->Zls = std::vector<arma::mat>(D);
  this->YZlst = std::vector<arma::mat>(D);
  this->ZtYtilde = std::vector<arma::mat>(D);
  this->pd = std::vector<int>(D);
  this->ones_pd = std::vector<arma::vec>( D );
  this->ones_n = arma::vec( n, arma::fill::ones );
  this->colOffsets_for_columns = std::vector<arma::vec>(D);
  for(int d=0; d<D; d++){

    this->Zt[d] = Z[d].t();
    this->ZtZ[d] = Zt[d] * Z[d];
    this->Zls[d] = invpsd( ZtZ[d], 1e-6, false ) * Zt[d];
    colOffsets_for_columns[d]=colOffsets[d].elem(columns);

    //first term same as using Y.postMultiplyWithMatReturnMat( d, (Zls[d].t()) );
    arma::mat Zlsdt = Zls[d].t();
    this->pd[d]=Y.nrow(d);
    this->ones_pd[d] = arma::vec( this->pd[d], arma::fill::ones );
    this->YZlst[d] = ((Y.postMultiplyWithMatReturnMat(d, Zlsdt, columns)).t()).eval(); //(( yd  * (Zls[d].t())).t()).eval(); // (pxn x  nx|Zeta|)' = |Zeta| x p
    this->YZlst[d] = this->YZlst[d] - ((( Zls[d] * colOffsets_for_columns[d] ).eval())*(ones_pd[d].t())) - ((( Zls[d] * ones_n ).eval())*(rowOffsets[d].t())); //accounting for offsets
    
    arma::mat Zd = Z[d];
    this->ZtYtilde[d] = (Y.postMultiplyWithMatReturnMat(d, Zd, columns).t()).eval()
      - (Zt[d] * colOffsets_for_columns[d]) * ones_pd[d].t()
      - (Zt[d] * ones_n) * rowOffsets[d].t();

  }
  this->spd= std::accumulate( pd.begin(), pd.end(), 0  );

  //construct A, Z
  this->Mt = M.t();
  this->A =2* Mt * M;
  // The matrix Aexp (not exponential! poor name) i.e., A_expanded, is dealt with separately in the estimate() function
  //  as Aexp is dependent on f1 and f2.
  this->w_hat = arma::vec(n, arma::fill::zeros);
  this->v_score = std::vector<arma::vec>(D);
  this->lf_swise=arma::vec(n, arma::fill::zeros);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT ZETA
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////
/*
 * For the first component (k=0) only, without any prior
 * components fitted.
 */
//////////////////////////////////////////////////////////////
void inpFitterOnColumns::move_Zeta(arma::vec& u_k, std::vector<arma::vec>& v_k){
  for( int d=0; d < this->D; d++  ){
    Zeta[d] = YZlst[d] - ( (Zls[d] * u_k).eval() * v_k[d].t() ) ; //already accounted for offsets in YZlst[d]
  }
}


/////////////////////////////////////////////////////////////////////
/*
 * For subsequent components with k>0
 * with k-1 components fitted and stored in umatOthers and vmatOthers.
 */
/////////////////////////////////////////////////////////////////////
void inpFitterOnColumns::move_Zeta( arma::vec& u_k, std::vector<arma::vec>& v_k,
                                    arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers){

  // Kxp
  for(int d=0; d< this->D; d++){
    Zeta[d] = YZlst[d] - ( (Zls[d] * u_k).eval() * v_k[d].t() )   - ( (Zls[d] * umatOthers).eval() * vmatOthers[d].t() )     ; //already accounted for offsets in YZlst[d]
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT V_K
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
////OVERLOADED FORM WITHOUT umatOthers and vmatOthers
//// useful for example, for moving k=1
////////////////////////////////////////////////////////////
void inpFitterOnColumns::move_v_k(arma::vec& uk, std::vector<arma::vec>& vk){

  //treat like single concat dat
  double dk = 0;
  for( int d=0; d<this->D; d++ ){
    arma::vec Yduk=Y.postMultiplyWithVecReturnVec( d, uk, columns );
    vk[d]=Yduk - ( Zeta[d].t() * (Zt[d]*uk).eval() );
    vk[d] = vk[d] - arma::dot(ones_n, uk)*rowOffsets[d] - arma::dot( colOffsets_for_columns[d], uk )*ones_pd[d] ;  //accounting for offsets
    v_score[d] = vk[d]; // retain already-computed unnormalised score
    dk += wd[d]*arma::sum( arma::pow( vk[d], 2 ) );
  }
  dk = std::sqrt(dk);

  if( dk < 1e-7 ){
    for( int d=0; d<this->D; d++ ){
      vk[d]*=0;
    }
  } else {
    for( int d=0; d<this->D; d++ ){
      vk[d]=vk[d]/dk;
    }
  }
}


/////////////////////////////////////
////MOVE REST OF THE COMPONENTS//////
/////////////////////////////////////
void inpFitterOnColumns::move_v_k(arma::vec& uk, std::vector<arma::vec>& vk, arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers ){

  double dk = 0;
  for( int d=0; d<this->D; d++ ){
    arma::vec Yduk=Y.postMultiplyWithVecReturnVec( d, uk, columns );
    vk[d]=Yduk - vmatOthers[d]*( umatOthers.t() * uk ).eval() - ( Zeta[d].t() * (Zt[d]*uk).eval() );
    vk[d] = vk[d] - arma::dot(ones_n, uk)*rowOffsets[d] - arma::dot( colOffsets_for_columns[d], uk )*ones_pd[d] ;  //accounting for offsets
    v_score[d] = vk[d]; // retain already-computed unnormalised score
    dk += wd[d]*arma::sum( arma::pow( vk[d], 2 ) );
  }
  dk = std::sqrt(dk);

  if( dk < 1e-7 ){
    for( int d=0; d<this->D; d++ ){
      vk[d]*=0;
    }
  } else {
    for( int d=0; d<this->D; d++ ){
      vk[d]=vk[d]/dk;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT u_K
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// For the first comp, k=0
////////////////////////////////////////////////////////////
void inpFitterOnColumns::populate_w_hat_for_k(std::vector<arma::vec>& vk){

  this->w_hat = 0* this->w_hat;
  for( int d=0; d< this->D; d++ ){
    arma::rowvec vtYd = (Y.preMultiplyWithVecTransposeReturnRowvec( d, vk[d], columns ));//.eval(); //i.e., (vk[d].t()*Yd).eval()
    vtYd = vtYd - arma::dot( vk[d], ones_pd[d] )*colOffsets_for_columns[d].t() - arma::dot( vk[d], rowOffsets[d] )*ones_n.t(); //accounting for offsets
    arma::mat Zetavd= ((Zeta[d] * vk[d])).eval();  //|Zeta|xp x px1
    this->w_hat += wd[d]*(vtYd.t() - (Z[d]*Zetavd) );            //nx1  - nx|Zeta| x |Zeta| x1
  }
}

void inpFitterOnColumns::move_h_k(arma::vec& hk, arma::vec& uk, std::vector<arma::vec>& vk){ //moves and updates Aex for the next component
  this->populate_w_hat_for_k( vk );          //construct w_hat=((Y - Zb) %*% v) \in Rn
  hk =2* Aex_pinv_rPlusq_submat * ( Mt * w_hat); //solving for the h portion alone from the normal equation for h & gamma
  uk = M*hk;
}

////////////////////////////////////////////////////////////
// Overloaded forms. For the subsequents, k>0
////////////////////////////////////////////////////////////
void inpFitterOnColumns::populate_w_hat_for_k( std::vector<arma::vec>& vk,
                                               arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers
){
  this->w_hat = 0* this->w_hat;
  for( int d=0; d< this->D; d++ ){
    arma::rowvec vtYd = (Y.preMultiplyWithVecTransposeReturnRowvec( d, vk[d], columns ));//.eval(); //i.e., (vk[d].t()*Yd).eval()
    vtYd = vtYd - arma::dot( vk[d], ones_pd[d] )*colOffsets_for_columns[d].t() - arma::dot( vk[d], rowOffsets[d] )*ones_n.t(); //accounting for offsets
    arma::mat Zetavd= ((Zeta[d] * vk[d])).eval();  //|Zeta|xp x px1
    arma::mat vv=(vmatOthers[d].t() * vk[d]).eval();
    this->w_hat += wd[d]*(vtYd.t() - umatOthers*vv  - Z[d] * Zetavd);            //nx1  - nx|Zeta| x |Zeta| x1
  }
}


void inpFitterOnColumns::move_h_k(arma::vec& hk, arma::vec& uk, std::vector<arma::vec>& vk,
                                  arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers){ //moves and updates Aex for the next component

  this->populate_w_hat_for_k( vk, umatOthers, vmatOthers );   //construct w_hat=((Y - Zb) %*% v) \in Rn
  hk= 2* Aex_pinv_rPlusq_submat * ( Mt * w_hat);
  uk = M*hk;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * ESTIMATORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void inpFitterOnColumns::append_to_Aex_and_update_Aexpinv( const arma::vec& u ){

  //Transform u for the Kernel inn Prod
  arma::vec Qq = this->B * u.elem(qids); // B must be of |qids| x |qids|

  //update Aex for next component
  arma::rowvec rowAppendage = arma::rowvec( this->Aex.n_cols, fill::zeros );
  rowAppendage.elem( rqindx ) =  Qq.t() * M.rows(qids);
  arma::mat Aex0 = arma::join_vert(Aex, rowAppendage);
  arma::vec zeroBlock=arma::vec(1,fill::zeros);
  arma::vec colAppendage = arma::join_vert(rowAppendage.t(),  zeroBlock );
  this->Aex = arma::join_horiz( Aex0, colAppendage  );
  this->Aex_pinv=invpsd( Aex, 1e-6, false );
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );

}

void inpFitterOnColumns::append_to_Aex_and_update_Aexpinv( const arma::mat& Q1 ){

  //Transform Q1 for the Kernel inn Prod
  arma::mat Qq = this->B * Q1.rows(qids); // B must be of |qids| x |qids|

  //update Aex for next component
  arma::mat rowAppendage = arma::mat( Q1.n_cols, Aex.n_cols, fill::zeros );
  rowAppendage( arma::span::all, arma::span( 0, r+q-1 ) ) =  Qq.t() * M.rows(qids);
  arma::mat Aex0 = arma::join_vert(Aex, rowAppendage);
  arma::mat zeroBlock = arma::mat(Q1.n_cols,Q1.n_cols,fill::zeros);
  arma::mat colAppendage = join_vert( rowAppendage.t().eval(), zeroBlock );
  this->Aex = join_horiz( Aex0, colAppendage  );
  this->Aex_pinv=invpsd( Aex, 1e-6, false );
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );

}

// Estimate the first component, when there are no
//   umatOthers and vmatOthers to account for.
void inpFitterOnColumns::estimate_k0( int maxIter, double eps){

  int k =0;

  arma::vec hk=hmat.col(k);
  arma::vec uk=umat.col(k);
  std::vector<arma::vec> vk=std::vector<arma::vec>(D);
  for(int d=0; d<D; d++){
    vk[d]=vmat[d].col(k);
  }

  // Reused bookkeeping objects; values are refreshed immediately before each block move.
  std::vector<arma::mat> Zeta_old = Zeta;
  arma::vec hk_old = hk;
  std::vector<arma::vec> vk_old = vk;

  bool conv = false;
  int iter = 0;
  const arma::mat Hhalf = 0.5 * Aex.submat(0, 0, r+q-1, r+q-1);
  const arma::mat Hpen = Hhalf - 0.5 * A;
  while( !conv && (iter < maxIter) ){
    iter = iter+1;
    double deltaF = 0.0;

    Zeta_old = Zeta;
    move_Zeta(uk, vk);
    for(int d=0; d<D; d++){
      arma::mat dZeta = Zeta[d] - Zeta_old[d];
      arma::mat ZtR_old = ZtYtilde[d] - (Zt[d] * uk) * vk[d].t() - ZtZ[d] * Zeta_old[d];
      deltaF += wd[d] * (
        -2.0 * arma::accu(dZeta % ZtR_old)
        + arma::accu(dZeta % (ZtZ[d] * dZeta))
      );
    }

    hk_old = hk;
    double weighted_vnorm2 = 0.0;
    for(int d=0; d<D; d++){
      weighted_vnorm2 += wd[d] * arma::dot(vk[d], vk[d]);
    }
    move_h_k(hk, uk, vk);
    arma::vec dh = hk - hk_old;
    deltaF += -2.0 * arma::dot(dh, Mt * w_hat)
              + weighted_vnorm2 * arma::dot(dh, A * hk_old)
              + 0.5 * weighted_vnorm2 * arma::dot(dh, A * dh)
              + 2.0 * arma::dot(dh, Hpen * hk_old)
              + arma::dot(dh, Hpen * dh);

    vk_old = vk;
    move_v_k(uk, vk);
    double weighted_norm_change = 0.0;
    double weighted_score_change = 0.0;
    for(int d=0; d<D; d++){
      weighted_norm_change += wd[d] * (arma::dot(vk[d], vk[d]) - arma::dot(vk_old[d], vk_old[d]));
      weighted_score_change += wd[d] * arma::dot(v_score[d], vk[d] - vk_old[d]);
    }
    deltaF += arma::dot(uk, uk) * weighted_norm_change - 2.0 * weighted_score_change;

    if( arma::as_scalar(arma::sign( vk[0](0) )) <0 ){//rotation identifiabilty, 1st datset's 1st feature constrained to be non-negative.
      for( int d=0; d<D; d++ ){vk[d] = -vk[d];}
      hk = -hk;
      uk = -uk;
    }

    conv = std::abs(deltaF) < eps;
  }

  if(!conv){
    std::cout << "Component " + std::to_string(k) + " fails to converge. Increase maxIter?" << std::endl;
  }

  //update the internal datastructures
  hmat.col(k)=hk;
  umat.col(k)=uk;
  for( int d=0; d<D; d++ ){
    vmat[d].col(k)=vk[d];
  }


}

void inpFitterOnColumns::estimate_k(int k, arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers, int maxIter, double eps){

  arma::vec hk=hmat.col(k);
  arma::vec uk=umat.col(k);
  std::vector<arma::vec> vk=std::vector<arma::vec>(D);
  for(int d=0; d<D; d++){
    vk[d]=vmat[d].col(k);
  }

  // Reused bookkeeping objects; values are refreshed immediately before each block move.
  std::vector<arma::mat> Zeta_old = Zeta;
  arma::vec hk_old = hk;
  std::vector<arma::vec> vk_old = vk;

  bool conv = false;
  int iter = 0;
  const arma::mat Hhalf = 0.5 * Aex.submat(0, 0, r+q-1, r+q-1);
  const arma::mat Hpen = Hhalf - 0.5 * A;
  while( !conv && (iter < maxIter) ){
    iter = iter+1;
    double deltaF = 0.0;

    Zeta_old = Zeta;
    move_Zeta(uk, vk, umatOthers, vmatOthers);
    for(int d=0; d<D; d++){
      arma::mat dZeta = Zeta[d] - Zeta_old[d];
      arma::mat ZtR_old = ZtYtilde[d]
        - (Zt[d] * uk) * vk[d].t()
        - (Zt[d] * umatOthers) * vmatOthers[d].t()
        - ZtZ[d] * Zeta_old[d];
      deltaF += wd[d] * (
        -2.0 * arma::accu(dZeta % ZtR_old)
        + arma::accu(dZeta % (ZtZ[d] * dZeta))
      );
    }

    hk_old = hk;
    double weighted_vnorm2 = 0.0;
    for(int d=0; d<D; d++){
      weighted_vnorm2 += wd[d] * arma::dot(vk[d], vk[d]);
    }
    move_h_k(hk, uk, vk, umatOthers, vmatOthers);
    arma::vec dh = hk - hk_old;
    deltaF += -2.0 * arma::dot(dh, Mt * w_hat)
              + weighted_vnorm2 * arma::dot(dh, A * hk_old)
              + 0.5 * weighted_vnorm2 * arma::dot(dh, A * dh)
              + 2.0 * arma::dot(dh, Hpen * hk_old)
              + arma::dot(dh, Hpen * dh);

    vk_old = vk;
    move_v_k(uk, vk, umatOthers, vmatOthers);
    double weighted_norm_change = 0.0;
    double weighted_score_change = 0.0;
    for(int d=0; d<D; d++){
      weighted_norm_change += wd[d] * (arma::dot(vk[d], vk[d]) - arma::dot(vk_old[d], vk_old[d]));
      weighted_score_change += wd[d] * arma::dot(v_score[d], vk[d] - vk_old[d]);
    }
    deltaF += arma::dot(uk, uk) * weighted_norm_change - 2.0 * weighted_score_change;

    if( arma::sign( vk[0](0) ) <0 ){//rotation identifiabilty, 1st datset's 1st feature constrained to be non-negative.
      for( int d=0; d<D; d++ ){vk[d] = -vk[d];}
      hk = -hk;
      uk = -uk;
    }

    conv = std::abs(deltaF) < eps;
  }

  if(!conv){
    std::cout << "Component " + std::to_string(k) + " fails to converge. Increase maxIter?" << std::endl;
  }

  //update the internal datastructures
  hmat.col(k)=hk;
  umat.col(k)=uk;
  for( int d=0; d<D; d++ ){
    vmat[d].col(k)=vk[d];
  }

}


void inpFitterOnColumns::estimateHelper(double f1, double f2, double maxIter, double eps){

  //Guard for making sure this function is never called by itself without estimate!
  if( !this->estimateHelperCalledViaEstimate ){ // the private boolean is set to true by a protected setAexToAandLock() function.
    throw std::runtime_error( "FATAL: estimateHelper() invoked directly or out-of-order."
                                "Calculations must be initiated via the public estimate() interface." );
  }


  //It is important that we restart the internal structures here, and not
  // let estimate() be called with a previously generated Aex1 Aex2 and related inverses.
  // we have already reset Aex to A in estimate() via which this function is called.

  this->Aex.submat( r, r, r+q-1, r+q-1 ) += 2*n*(f2/f1)*Rtilde; //n arises because of 1/n prefactor in the objective
  this->Aex_pinv=invpsd(Aex, 1e-6, false);
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );
  if( this->Q.n_cols > 0 ){
    this->append_to_Aex_and_update_Aexpinv(this->Q); //orth with k-1
  }
  
  //orth to constant
  arma::vec u0 = arma::vec(n, arma::fill::ones);
  this->append_to_Aex_and_update_Aexpinv(u0); //orth with k-1
  
  /////////////
  //ESTIMATION
  ////////////

  //First component
  const double solved_eps = (n/f1) * eps; // eps is on the canonical-objective scale
  this->estimate_k0(maxIter, solved_eps);

  //Subsequent components.
  if( K > 1 ){

    arma::uvec before_k;
    arma::mat umatOthers;
    std::vector<arma::mat> vmatOthers=std::vector<arma::mat>(this->D);
    for( int k=1; k < K; k++ ){
      //Append orth with the just constructed k-1 st.
      //  Components prior to the previous have been already appended in previous cycles.
      arma::vec ukm1 = umat.col(k-1); //instantiation of ukm1 is necessary b/c umat.col(k-1) is a subview that cannot be passed as input to the next line's function.
      this->append_to_Aex_and_update_Aexpinv(ukm1); //orth with k-1

      //Fit the k^th while djusting the mean for
      //   the already reconstructed 0 .. k-1 components.
      before_k = arma::find( this->seqOfComponents < k ) ;
      umatOthers = umat.cols(before_k);
      for(int d=0; d<D; d++){
        vmatOthers[d]=vmat[d].cols( before_k );
      }
      this->estimate_k(k, umatOthers, vmatOthers, maxIter, solved_eps);
    }
  }

  last_f1_estimatedWith=f1;
  last_f2_estimatedWith=f2;

}

void inpFitterOnColumns::setAexToAandLock(){
  this->Aex=A;
  this->estimateHelperCalledViaEstimate=true;
}

void inpFitterOnColumns::estimate(double f1, double f2, double maxIter, double eps, double xtol_abs){
  (void)xtol_abs; // reserved for future parameter-change convergence

  if( Y.size()==0 ){
    throw std::runtime_error("Y has not been assigned. Has the right constructor been called? ");
  }
  this->setAexToAandLock(); //same strategy as npFitterOnColumns.h - refer that file.
  this->estimateHelper(f1, f2, maxIter, eps);
}

void inpFitterOnColumns::estimate(const estimatePar& ep){
  this->estimate( ep.f1, ep.f2, ep.maxIter, ep.eps, .01 );
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * OBJECTIVE: JUST IMPLEMENTING IT NOW, HAVEN'T USED IT YET : PERHAPS MOVE THE CONVERGENCE CRITERION TO BE BASED
 *  ON THIS. BUT CAN BE EXPENSIVE TO COMPUTE, AND CAN SLOW US DOWN.
 *
 *  BOTH FUNCTIONS USE CURRENTLY STORED VALUES OF U AND V and ZETA.
 *  If f1 and f2 do not reflect the currently stored values, estimate must be rerun for f1 and f2!
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KEY: USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
double inpFitterOnColumns::getPenalty(){

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

// KEY: USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
double inpFitterOnColumns::objective(double f1, double f2){
  //TODO : Number of threads must be controlled.
  double lf = imse(Y, wd, Z,Zeta,umat,vmat,columns, rowOffsets, colOffsets); //must pass colOffsets and not colOffsets_for_columns
  double penalty = getPenalty();
  return (f1*lf)+(f2*penalty); //our objective is f1*imse
}

double inpFitterOnColumns::mseForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){

  for( int d=0; d < this->D; d++ ){
    if( colOffsetsTest[d].n_elem != Y.ncol(0)  ){
      throw std::invalid_argument( "colOffsets must have the same length as the number of columns in data matrices Y " );
    }
  }
  arma::mat umatTest = MTest* this->hmat;
  return imse(Y, wd, ZTest,Zeta,umatTest,vmat,columns1Test, rowOffsetsTest, colOffsetsTest);

}

double inpFitterOnColumns::objectiveForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){

  double lf = this->mseForTest( columns1Test, ZTest, MTest,
                                rowOffsetsTest, colOffsetsTest); //mse
  double penalty = this->getPenalty();
  double obj= last_f1_estimatedWith* lf + last_f2_estimatedWith* penalty; //we leave objective for test as average as we want to divide by number of samples for cv
  return obj;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * GETTERS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<arma::mat> inpFitterOnColumns::get_Zeta() const{
  return( Zeta );
}

arma::mat inpFitterOnColumns::get_hmat() const{
  return( hmat );
}

arma::mat inpFitterOnColumns::get_umat() const{
  return( umat );
}

std::vector<arma::mat> inpFitterOnColumns::get_vmat() const{
  return( vmat );
}
