#include "igFitterOnColumns.h"


///////////////////////////////////////
/*
 * Constructors for the dense dataset
 */
///////////////////////////////////////

// without weights, with Q
igFitterOnColumns::igFitterOnColumns(
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
):denseMats(denseMats), sparseMats(sparseMats), columns1(columns1), family(family),
Z(Z), Zeta(Zeta), M(M), hmat(hmat), umat(umat), vmat(vmat),
aphi(aphi),Rtilde(Rtilde), qids(qids), q(qids.size()), B(B),
rowOffsets(rowOffsets), colOffsets(colOffsets), K(K), r(r), wd(wd), Q(Q),
dDense(denseMats.size()), dSparse(sparseMats.size()), D(denseMats.size()+sparseMats.size()) //Rtheta(Rtheta),
{
  initializeOthers();
}


// without weights, without Q

// without weights, with Q
igFitterOnColumns::igFitterOnColumns(
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
):denseMats(denseMats), sparseMats(sparseMats), columns1(columns1), family(family),
Z(Z), Zeta(Zeta), M(M), hmat(hmat), umat(umat), vmat(vmat),
aphi(aphi),Rtilde(Rtilde), qids(qids), q(qids.size()), B(B),
rowOffsets(rowOffsets), colOffsets(colOffsets), K(K), r(r), wd(wd),
dDense(denseMats.size()), dSparse(sparseMats.size()), D(denseMats.size()+sparseMats.size()) //Rtheta(Rtheta),
{



  // Rcpp::Rcout << "Before initialize others! " << std::endl;
  initializeOthers();

  // Rcpp::Rcout << "All initialized! " << std::endl;

}


igFitterOnColumns::igFitterOnColumns(
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
):igFitterOnColumns(
    Y.denseY,
    Y.sparseY,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat, vmat,
    aphi,
    Rtilde,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r,
    wd,
    Q
) {}


// without weights, without Q

// without weights, with Q
igFitterOnColumns::igFitterOnColumns(
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
):igFitterOnColumns(
    Y.denseY,
    Y.sparseY,
    columns1,
    family,
    Z, Zeta,
    M, hmat,
    umat, vmat,
    aphi,
    Rtilde,
    qids,
    B,
    rowOffsets,
    colOffsets,
    K,
    r,
    wd
){}




void igFitterOnColumns::initializeOthers(){

  //some barriers
  for(int d=0; d<D; d++){
    aphi[d]=arma::clamp(aphi[d], 1e-7, 1e7);
  }

  //initialize coloffsets_for_columns
  this->colOffsets_for_columns = std::vector<arma::vec>(D);
  for(int d=0; d<D; d++){
    if( d < dDense ){
      if( colOffsets[d].size() != denseMats[d].n_cols  ){
        throw std::invalid_argument( "Column offsets for matrix d must equal the number of columns in matrix d" );
      }
    } else {
      if( colOffsets[d].size() != sparseMats[d-dDense].n_cols  ){
        throw std::invalid_argument( "Column offsets for matrix d must equal the number of columns in matrix d" );
      }
    }
    colOffsets_for_columns[d] = colOffsets[d].elem(columns1);
  }


  this->pd=std::vector<unsigned int>(D);
  this->n=columns1.size();
  this->nlength_vector=arma::vec(n, arma::fill::zeros);
  this->plength_vectord=std::vector<arma::vec>(D);
  this->nlength_vectord=std::vector<arma::vec>(D);

  denseRowFams = std::vector<rowFamily<arma::mat>*>(D);
  sparseRowFams = std::vector<rowFamily<arma::sp_mat>*>(D);
  gradObjh_k=std::vector<arma::vec>(D);
  hessObjh_k=std::vector<arma::mat>(D);
  netGradh_k=arma::vec( r+q, arma::fill::zeros );
  netHessh_k=arma::mat( r+q, r+q, arma::fill::zeros );
  //this->spd = std::accumulate( pd.begin(), pd.end(), 0 );

  negLogLikesVecd=std::vector<arma::vec>(D);
  //negLogLikesVecd_tmp=std::vector<arma::vec>(D);
  Zt = std::vector<arma::mat>(D);
  ZtZ = std::vector<arma::mat>(D);

  // Rcpp::Rcout << "First round of initiali" << std::endl;

  for( int d=0; d < D; d++ ){
    if( d < dDense ){

      pd[d]=denseMats[d].n_rows;

      if( family[d]=="gaussian" ){
        this->denseRowFams[d] = new rowFamilyGaussian<arma::mat>();
      } else if( family[d]=="poisson" ){
        this->denseRowFams[d] = new rowFamilyPoisson<arma::mat>();
      } else {
        throw std::runtime_error("Family not found");
      }

    } else {
      pd[d]=sparseMats[d-dDense].n_rows;

      if( family[d]=="gaussian" ){
        this->sparseRowFams[d-dDense] = new rowFamilyGaussian<arma::sp_mat>();
      } else if( family[d]=="poisson" ){
        this->sparseRowFams[d-dDense] = new rowFamilyPoisson<arma::sp_mat>();
      } else {
        throw std::runtime_error("Family not found");
      }
    }

    // Rcpp::Rcout << "Initialized the family for d" << std::endl;

    gradObjh_k[d] = arma::vec( r+q, arma::fill::zeros );
    hessObjh_k[d] = arma::mat( r+q, r+q, arma::fill::zeros );
    plength_vectord[d]=arma::vec( pd[d], arma::fill::zeros);
    nlength_vectord[d]=arma::vec(n, arma::fill::zeros);

    // Rcpp::Rcout << "Initialized the grad, hess, plength, nlength vectors for d" << std::endl;

    Zt[d] = Z[d].t();
    ZtZ[d]=Zt[d]*Z[d];
    negLogLikesVecd[d]=arma::vec( pd[d], arma::fill::zeros);
  }


  // R to C indexing differences
  this->seqOfComponents=arma::regspace<uvec>( 0, K-1 );
  this->rows_for_qxq = r + arma::regspace<arma::uvec>( 0, q-1 );
  this->cols_for_qxq = rows_for_qxq;

  if( Q.n_cols > 0 ){
    this->BQ=B * Q.rows(qids) ; //multiplying by subview OK.
  }

  // R to C indexing
  this->alpha0=1;
  this->rho=.5;

  this->Mt=M.t();
  this->MtM=Mt*M;

  // Rcpp::Rcout << "Initialized the rest of rest" << std::endl;



}




//////////////////////////////////////////////
/*
 *
 * LOG LIKELIHOOD STUFF
 * 
 * ONLY THE AGGREGATION across datasets are WEIGHTED by wd. 
 * REST are not. 
 * 
 *
 */
//////////////////////////////////////////////

double igFitterOnColumns::calcNegLogLL_di(const int& d, const int& i ){//for the ith feature

  arma::vec eta_i = (Z[d] * Zeta[d].col(i) + umat * (vmat[d].row(i)).t() + colOffsets_for_columns[d]) + rowOffsets[d](i);
  arma::vec y_i_all;
  if( d < dDense){
    y_i_all = denseMats[d].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return this->denseRowFams[d]->calcNegLogLL_i( y_i, eta_i, this->aphi[d].at(i) );
  } else {
    y_i_all = sparseMats[d-dDense].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return this->sparseRowFams[d-dDense]->calcNegLogLL_i( y_i, eta_i, this->aphi[d].at(i) );
  }

}

//Overloaded form useful for the vki and zeta_i descent movers, which avoids traversals over vectors at every descent step
double igFitterOnColumns::calcNegLogLL_di(const int& d, const arma::vec& y_i,
                                          const arma::vec& eta_i,
                                          const double& aphi_i ){//for the ith feature


  if( d < dDense ){
    return this->denseRowFams[d]->calcNegLogLL_i( y_i, eta_i, aphi_i );
  } else {
    return this->sparseRowFams[d-dDense]->calcNegLogLL_i( y_i, eta_i, aphi_i );
  }

}

double igFitterOnColumns::calcNegLogLL_di(const int& d, const int& i,
                                          const arma::mat& Zd, const arma::mat& Zetad,
                                          const arma::mat& umat, const arma::mat& vmatd,
                                          const arma::vec& aphid,
                                          const arma::vec& rowOffsetsd, const arma::vec& colOffsets_for_columnsd
){//for the ith feature

  arma::vec eta_i = (Zd * Zetad.col(i) + umat * (vmatd.row(i)).t() + colOffsets_for_columnsd) + rowOffsetsd(i);
  arma::vec y_i_all;
  if( d < dDense){
    y_i_all = denseMats[d].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return this->denseRowFams[d]->calcNegLogLL_i( y_i, eta_i, aphid(i) );
  } else {
    y_i_all = sparseMats[d-dDense].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return this->sparseRowFams[d-dDense]->calcNegLogLL_i( y_i, eta_i, aphid(i) );
  }

}


// NOTE: THESE ARE NOT WEIGHTED BY wd
void igFitterOnColumns::calcNegLogLL_d(const int& d,
                                       arma::vec& nlld //pd length vector storing neg loglikes for dth dataset
){ //for the entire dataset

  const arma::mat& Zd=Z[d];
  const arma::mat& Zetad=Zeta[d];
  const arma::mat& vmatd=vmat[d];
  const arma::vec& aphid=aphi[d];
  const arma::vec& rowOffsetsd=rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd=colOffsets_for_columns[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,pd[d]),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i){
                         nlld(i)=calcNegLogLL_di(d,i,
                              Zd, Zetad, umat, vmatd,
                              aphid,
                              rowOffsetsd, colOffsets_for_columnsd
                         );
                       }
                     }
  );

}


// NOTE: the ull will be weighted by wd, while the weightedNegLogLikes won't be. 
double igFitterOnColumns::calcNegLogLL(
    std::vector<arma::vec>& weightedNegLoglikes,
    const bool updateCurrent
){

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         calcNegLogLL_d(d,weightedNegLoglikes[d]);
                       }
                     }
  );
  double ull =0;
  for( int d=0; d<D; d++ ){

    ull += wd[d]* arma::sum(weightedNegLoglikes[d]); //weighted!
    if(updateCurrent){
      this->negLogLikesVecd[d] = weightedNegLoglikes[d]; //unweighted, name is bad!
    }
  }
  if(updateCurrent){
    this->ULL=ull;//update net neg log lik
  }
  return ull;
}

//// new set end;
//for K=1
double igFitterOnColumns::calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const int& d,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta[d].t() * Z[d].row(j).t()) +  (rowOffsets[d] + colOffsets_for_columns[d].at(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k[d] * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec y_j;
  if( d < dDense ){
    y_j = denseMats[d].col(columns1(j));
    return this->denseRowFams[d]->calcNegLogLL_j( y_j, eta_j, aphi[d] ) ;
  } else {
    y_j = sparseMats[d-dDense].col(columns1(j));
    return( this->sparseRowFams[d-dDense]->calcNegLogLL_j( y_j, eta_j, aphi[d] ) );
  }
}

double igFitterOnColumns::calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const int& d,
    const arma::mat& umatOthers, //columns other than the k^th
    const std::vector<arma::mat>& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {

  arma::vec otherEtaContrib_j = (Zeta[d].t() * Z[d].row(j).t()) + (vmatOthers[d]*(umatOthers.row(j).t())) + (rowOffsets[d] + colOffsets_for_columns[d].at(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_k[d] * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec y_j;
  if( d < dDense ){
    y_j = denseMats[d].col(columns1(j));
    return this->denseRowFams[d]->calcNegLogLL_j( y_j, eta_j, aphi[d] ) ;
  } else {
    y_j = sparseMats[d-dDense].col(columns1(j));
    return( this->sparseRowFams[d-dDense]->calcNegLogLL_j( y_j, eta_j, aphi[d] ) );
  }
}



//for K=1
double igFitterOnColumns::calcNegLogLL_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra){
                       for(size_t d=ra.begin(); d<D; ++d ){
                         nlength_vectord[d].at(j)=calcNegLogLL_d_j_with_h_k(d, h_k, v_k, j);
                       }
                     });
  double ull=0;
  for(int d=0; d<D; d++){
    ull += wd[d]* arma::sum( nlength_vectord[d] ); //weighted! 
  }
  return ull;
}
double igFitterOnColumns::calcNegLogLL_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const arma::mat& umatOthers, //columns other than the k^th
    const std::vector<arma::mat>& vmatOthers, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const std::vector<arma::vec>& v_k,      // the corresponding feature loadings vector already fixed
    const int j
) {
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra){
                       for(size_t d=ra.begin(); d<D; ++d ){
                         nlength_vectord[d].at(j)=calcNegLogLL_d_j_with_h_k(d, umatOthers, vmatOthers, h_k, v_k, j);
                       }
                     });
  double ull=0;
  for(int d=0; d<D; d++){
    ull += wd[d]* arma::sum( nlength_vectord[d] ); //weighted!
  }
  return ull;
}


//forK=1
double igFitterOnColumns::calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const int& d,
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_kd,      // the corresponding feature loadings vector already fixed
    const arma::mat& Zd,
    const arma::mat& Zetad,
    const arma::vec& aphid,
    const arma::vec& rowOffsetsd,
    const arma::vec& colOffsets_for_columnsd,
    const int j
) {

  arma::vec otherEtaContrib_j = (Zetad.t() * Zd.row(j).t()) + (rowOffsetsd + colOffsets_for_columnsd(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_kd * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec y_j;
  if( d < dDense ){
    y_j = denseMats[d].col(columns1(j));
    return this->denseRowFams[d]->calcNegLogLL_j( y_j, eta_j, aphid ) ;
  } else {
    y_j = sparseMats[d-dDense].col(columns1(j));
    return( this->sparseRowFams[d-dDense]->calcNegLogLL_j( y_j, eta_j, aphid ) );
  }
}
double igFitterOnColumns::calcNegLogLL_d_j_with_h_k( //compute a type of j^th columnsum needed for grad of negLL for h_k, without weights
    const int& d,
    const arma::mat& umatOthers, //columns other than the k^th
    const arma::mat& vmatOthersd, //columns other than the k^th in one to one correspondence with umatOther.
    const arma::vec& h_k,      //the h that will determine u_k = M h_k
    const arma::vec& v_kd,      // the corresponding feature loadings vector already fixed
    const arma::mat& Zd,
    const arma::mat& Zetad,
    const arma::vec& aphid,
    const arma::vec& rowOffsetsd,
    const arma::vec& colOffsets_for_columnsd,
    const int j
) {

  arma::vec otherEtaContrib_j = (Zetad.t() * Zd.row(j).t()) + (vmatOthersd*(umatOthers.row(j).t())) + (rowOffsetsd + colOffsets_for_columnsd(j) ) ; //px1 vec corresp to col j
  arma::vec eta_j= v_kd * (M.row(j) * h_k)  + otherEtaContrib_j; //adding the k^th components contribution
  arma::vec y_j;
  if( d < dDense ){
    y_j = denseMats[d].col(columns1(j));
    return this->denseRowFams[d]->calcNegLogLL_j( y_j, eta_j, aphid ) ;
  } else {
    y_j = sparseMats[d-dDense].col(columns1(j));
    return( this->sparseRowFams[d-dDense]->calcNegLogLL_j( y_j, eta_j, aphid ) );
  }
}

//for K=1
double igFitterOnColumns::calcNegLogLL_d_with_h_k(
    const int& d,
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    std::vector<arma::vec>& nlvd
) {

  const arma::mat& Zd=Z[d];
  const arma::mat& Zetad=Zeta[d];
  const arma::mat& v_kd=v_k[d];
  const arma::vec& aphid=aphi[d];
  const arma::vec& rowOffsetsd=rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd=colOffsets_for_columns[d];

  arma::vec& nlld=nlvd[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         //cs(j)=
                         nlld(j)=calcNegLogLL_d_j_with_h_k( d,
                              h_k, v_kd,
                              Zd, Zetad, aphid,
                              rowOffsetsd, colOffsets_for_columnsd, j
                         );
                       }
                     }
  );

  return arma::sum(nlld);


}
double igFitterOnColumns::calcNegLogLL_d_with_h_k(
    const int& d,
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    std::vector<arma::vec>& nlvd
) {

  const arma::mat& Zd=Z[d];
  const arma::mat& Zetad=Zeta[d];
  const arma::mat& vmatOthersd=vmatOthers[d];
  const arma::mat& v_kd=v_k[d];
  const arma::vec& aphid=aphi[d];
  const arma::vec& rowOffsetsd=rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd=colOffsets_for_columns[d];

  arma::vec& nlld=nlvd[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         //cs(j)=
                         nlld(j)=calcNegLogLL_d_j_with_h_k( d, umatOthers, vmatOthersd,
                              h_k, v_kd,
                              Zd, Zetad, aphid,
                              rowOffsetsd, colOffsets_for_columnsd, j
                         );
                       }
                     }
  );

  return arma::sum(nlld);


}


//a direct tbb version without weights
//for K=1
double igFitterOnColumns::calcNegLogLL_with_h_k(
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k
) {

  arma::vec Dlength_vector(D, arma::fill::zeros);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         //cs(j)=
                         Dlength_vector(d)=wd[d] * calcNegLogLL_d_with_h_k(d,
                                        h_k, v_k, nlength_vectord );  // weighted!
                       }
                     }
  );
  return arma::sum(Dlength_vector);
}

double igFitterOnColumns::calcNegLogLL_with_h_k(
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k
) {

  arma::vec Dlength_vector(D, arma::fill::zeros);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         //cs(j)=
                         Dlength_vector(d)= wd[d]* calcNegLogLL_d_with_h_k(d,
                                        umatOthers, vmatOthers, h_k, v_k, nlength_vectord );
                       }
                     }
  );
  return arma::sum(Dlength_vector);
}




//////////////////////////////////////////////
/*
 *
 * Updating a(phi)
 *
 */
//////////////////////////////////////////////



double igFitterOnColumns::calc_aphi_di(const int& d, const int& i,
                                       const arma::mat& Zd,
                                       const arma::mat& Zetad,
                                       const arma::mat& vmatd,
                                       const arma::mat& rowOffsetsd,
                                       const arma::mat& colOffsets_for_columnsd
){

  arma::vec eta_i = (Zd*Zetad.col(i) + (umat*(vmatd.row(i).t())) +  colOffsets_for_columnsd) + rowOffsetsd(i);

  arma::vec y_i_all;
  if( d < dDense){
    y_i_all = denseMats[d].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return std::max( this->denseRowFams[d]->calc_aphi( y_i, eta_i), 1e-7 ) ;

  } else {
    y_i_all = sparseMats[d-dDense].row(i).t();
    arma::vec y_i = y_i_all(columns1);
    return std::max( this->sparseRowFams[d-dDense]->calc_aphi( y_i, eta_i), 1e-7 );

  }

}

void igFitterOnColumns::calc_aphi_d(const int& d){

  const arma::mat& Zd=Z[d];
  const arma::mat& Zetad=Zeta[d];
  const arma::mat& vmatd=vmat[d];
  const arma::vec& rowOffsetsd=rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd=colOffsets_for_columns[d];


  arma::vec& aphid = aphi[d];
  tbb::parallel_for( tbb::blocked_range<size_t>(0,pd[d]),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i)
                         aphid(i)=calc_aphi_di(d,i,Zd, Zetad, vmatd, rowOffsetsd, colOffsets_for_columnsd);
                     }
  );
  //aphi[d]=aphid;
}

void igFitterOnColumns::update_aphi(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d)
                         calc_aphi_d(d);
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
double igFitterOnColumns::getPenalty(){
  double netPenalty=0;
  arma::mat hmat_qpart=hmat.rows( rows_for_qxq );
  arma::vec h_k_qpart;
  for( int k =0; k<K; k++ ){
    h_k_qpart=hmat_qpart.col(k);
    netPenalty += arma::dot( h_k_qpart, Rtilde * h_k_qpart );
  }
  return netPenalty; //sumovercompoents(f2*sq norm of projection of uk on H1)
}

double igFitterOnColumns::objective(const bool recalculate,
                                    const bool updateStored,
                                    const double& f1,
                                    const double& f2){
  if( recalculate ){
    this->calcNegLogLL( this->negLogLikesVecd, updateStored );
  }

  double netPenalty=0;
  arma::mat hmat_qpart=hmat.rows( rows_for_qxq );
  arma::vec h_k_qpart;
  for( int k =0; k<K; k++ ){
    h_k_qpart=hmat_qpart.col(k);
    netPenalty += arma::dot( h_k_qpart, Rtilde * h_k_qpart );
  }
  double obj =f1* (this->ULL /n) + f2* netPenalty; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
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

arma::vec igFitterOnColumns::gradNegLogLL_Zeta_di(
    const int& d,
    const arma::vec& y_i,
    const arma::vec& eta_i,
    const arma::vec& zeta_i,
    const double& aphi_i
){

  if( d < dDense  ){

    return this->denseRowFams[d]->gradNegLogLL_Zeta_i(y_i, eta_i,
                                                      Z[d], zeta_i,
                                                      aphi_i );

  } else {
    return this->sparseRowFams[d-dDense]->gradNegLogLL_Zeta_i(y_i, eta_i,
                                                              Z[d], zeta_i,
                                                              aphi_i );
  }




}

arma::mat igFitterOnColumns::hessNegLogLL_Zeta_di( const int& d,
                                                   const arma::vec& y_i,
                                                   const arma::vec& eta_i,
                                                   const arma::vec& zeta_i,
                                                   const double& aphi_i
){

  if( family[d] == std::string("gaussian") ){
    return ZtZ[d]/aphi_i; //instead of calling the below function for gaussian
  } else {
    if( d < dDense ){
      return this->denseRowFams[d]->hessNegLogLL_Zeta_i(y_i, eta_i,
                                                        Z[d], zeta_i,
                                                        aphi_i );
    } else {
      return this->sparseRowFams[d-dDense]->hessNegLogLL_Zeta_i(y_i, eta_i,
                                                                Z[d], zeta_i,
                                                                aphi_i );
    }
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
void igFitterOnColumns::moveZeta_di(const int& d, const int& i,
                                    const arma::mat& Zd,
                                    const arma::mat& Zetad,
                                    const arma::mat& vmatd,
                                    const arma::vec& aphid,
                                    const arma::vec& rowOffsetsd,
                                    const arma::vec& colOffsets_for_columnsd,
                                    const double& f1, const double& f2){

  arma::vec& nlld = this->negLogLikesVecd[d];

  arma::vec zeta_i = arma::vec(Zetad.col(i)); // Zeta is Bxp where Z is n x B, so  Z zeta enters eta
  arma::vec zeta_i_new=zeta_i;
  arma::vec y_i_all;
  if( d < dDense ){
    y_i_all = denseMats[d].row(i).t();
  } else {
    y_i_all = sparseMats[d-dDense].row(i).t();
  }
  arma::vec y_i = y_i_all(columns1);


  double aphi_i=aphid(i);
  arma::vec otherEtaContrib = ((umat* (vmatd.row(i).t()) ) + colOffsets_for_columnsd) + rowOffsetsd(i);
  arma::vec eta_i = Zd * zeta_i + otherEtaContrib;
  arma::vec eta_i_new=eta_i;

  int iter=0;
  bool conv = false; bool move=false;
  double ll_i = this->calcNegLogLL_di(d, y_i, eta_i, aphi_i);
  //double ll_i = .5*arma::sum( arma::pow( eta_i-y_i, 2 )  )/aphi_i;
  double ll_i_new, alpha;
  arma::vec gradll_i, p_i;
  arma::mat hessZeta_i;

  while( (!conv) & (iter <= this->inner_maxIter) ){
    iter = iter+1;

    gradll_i = this->gradNegLogLL_Zeta_di(d, y_i, eta_i, zeta_i, aphi_i);
    hessZeta_i = this->hessNegLogLL_Zeta_di(d, y_i, eta_i, zeta_i, aphi_i);
    
    p_i = -invpsd(hessZeta_i, 1e-7, false) * gradll_i;
    move =false;
    alpha = alpha0;
    while( (alpha >= 1e-4) & (!move)  ){
      zeta_i_new=zeta_i + alpha*p_i;
      eta_i_new =  Zd*zeta_i_new + otherEtaContrib;
      ll_i_new = this->calcNegLogLL_di(d, y_i, eta_i_new, aphi_i );
      if(  (ll_i_new > ll_i) ||
           (!std::isfinite(ll_i_new))
           ){
        alpha = rho*alpha;
      } else {
        move = true;
      }
    }
    if( move ){
      const double deltaF_inner = f1 * wd[d] * std::abs(ll_i_new - ll_i);
      const double deltaZetaRMS = std::sqrt(arma::mean(arma::square(zeta_i_new - zeta_i)));
      conv = deltaF_inner < this->solved_inner_ftol_abs;
      (void)deltaZetaRMS;
      zeta_i = zeta_i_new;
      eta_i = eta_i_new;
      ll_i = ll_i_new;
      alpha = alpha/rho;
      nlld(i)=ll_i_new; //update negloglikesvecd[d](i)s
      
    } else {
      conv = true; //return with current Zeta, as no step has been made
    }
  }

  this->Zeta[d].col(i)=zeta_i; 

}


void igFitterOnColumns::moveZeta_d(const int& d, const double& f1, const double& f2){

  const arma::mat& Zd=Z[d];
  const arma::mat& Zetad=Zeta[d];
  const arma::mat& vmatd=vmat[d];
  const arma::vec& aphid=aphi[d];
  const arma::vec& rowOffsetsd=rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd=colOffsets_for_columns[d];

  //parallel
  tbb::parallel_for( tbb::blocked_range<size_t>(0,pd[d]),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i){
                         moveZeta_di(d, i, Zd, Zetad, vmatd, aphid, rowOffsetsd, colOffsets_for_columnsd, f1, f2); // this is an in place move, a thread safe variant,
                         
                       }
                     }
  );
}

void igFitterOnColumns::moveZeta(const double& f1, const double& f2){

  //parallel
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         moveZeta_d(d, f1, f2); // this is an in place move, a thread safe variant,
                         
                       }
                     }
  );

  double ull=0;
  for( int d=0; d<D; d++ ){
    ull+=wd[d]*arma::sum( this->negLogLikesVecd[d] );
  }
  this->ULL = ull;
}


//////////////////////////////////////////////
/*
 *
 * GRADIENT DESCENT MOVER FOR V and U
 *
 *
 */
//////////////////////////////////////////////

double igFitterOnColumns::gradNegLogLL_v_dik( const int& d, const arma::vec& y_i,
                                              const arma::vec& eta_ik,
                                              const arma::vec& u_k, const double& v_ik,
                                              const double& aphi_i ){

  if( d < dDense ){
    return this->denseRowFams[d]->gradNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i );
  } else {
    return this->sparseRowFams[d-dDense]->gradNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i );
  }

}

double igFitterOnColumns::hessNegLogLL_v_dik(const int& d, const arma::vec& y_i,
                                             const arma::vec& eta_ik,
                                             const arma::vec& u_k,
                                             const double& v_ik,
                                             const double& aphi_i){
  if( d < dDense ){
    return this->denseRowFams[d]->hessNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i );
  } else {
    return this->sparseRowFams[d-dDense]->hessNegLogLL_v_ik(y_i, eta_ik, u_k, v_ik, aphi_i );
  }
}


////////////////////////////////////////////////
/*
 * Mover for v_k_i and v_k with unit norm constraint
 */
////////////////////////////////////////////////

void igFitterOnColumns::move_v_d_k_i_normUkStrategy( const int& d,
                                                     const int& i,
                                                     const int& k,
                                                     const double& dk, //=norm(u_k)
                                                     const arma::vec& u_k_normed,
                                                     const arma::mat& Zd,
                                                     const arma::mat& Zetad,
                                                     const arma::mat& umatOther,
                                                     const arma::mat& vmatOtherd,
                                                     const arma::mat& vmatd,
                                                     const arma::vec& aphid,
                                                     const arma::vec& rowOffsetsd,
                                                     const arma::vec& colOffsets_for_columnsd,
                                                     arma::vec& vdk, //output object
                                                     const double& f1,
                                                     const double& f2 ){


  /*
   * 
   *
   * Prior to calling this function, we assume,
   * with uk = umat.col(k), dk=norm(uk), we have set:
   *  1. uk_norm_vec = uk/dk (or set to zero if dk is ~0),
   *  2. hmat.col(k) /= dk to preserve uk = Mhk equality;
   *  3. vmat[d].col(k) *= dk for all d=1...D.
   */

  //we update negloglikes of the moves as we go by, through this reference variable
  arma::vec& nlld = this->negLogLikesVecd[d];


  //double v_ik = dk* vmatd(i,k); // we assume we have already scaled vmatd by dk
  double v_ik = vmatd(i,k); // Zeta is pxB where Z is n x B, so  zeta Z' enters eta
  double v_ik_new=v_ik;

  arma::vec otherEtaContrib = Zd* Zetad.col(i) + umatOther * (vmatOtherd.row(i).t()) + colOffsets_for_columnsd + rowOffsetsd(i);
  arma::vec eta_ik = v_ik*u_k_normed + otherEtaContrib; //current mean
  arma::vec eta_ik_new=eta_ik;

  arma::vec y_i_all;
  if( d < dDense ){
    y_i_all = denseMats[d].row(i).t();
  } else {
    y_i_all = sparseMats[d-dDense].row(i).t();
  }
  arma::vec y_i = y_i_all(columns1);

  bool conv = false; bool move=false;
  double aphi_i=aphid(i);
  double ll_ik= this->calcNegLogLL_di(d, y_i, eta_ik, aphi_i ); // this calculation is needed as prior u/h_k moves do not populate negLogLikesVec appropriately
  double ll_ik_new, alpha;
  double gradll_ik, p_ik;
  double hessv_ik;

  int iter=0;
  while( (!conv) & (iter <= this->inner_maxIter) ){

    iter = iter+1;

    //current stats
    gradll_ik =  this->gradNegLogLL_v_dik( d, y_i, eta_ik, u_k_normed, v_ik, aphi_i ); //scalar
    hessv_ik = std::max( this->hessNegLogLL_v_dik(d, y_i, eta_ik, u_k_normed, v_ik, aphi_i), 1e-7 ); //scalar
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
      ll_ik_new =  this->calcNegLogLL_di(d, y_i, eta_ik_new, aphi_i );
      if(  ll_ik_new > (  ll_ik + alpha*p_ik*gradll_ik ) ){
        alpha = this->rho * alpha;
      } else {
        move = true;
      }
    }

    //make a move
    if( move ){
      conv = sqrt(pow(v_ik_new- v_ik,2)) < 1e-4;
      v_ik = v_ik_new;
      eta_ik = eta_ik_new;
      ll_ik = ll_ik_new;
      alpha = alpha/rho;
      nlld(i)=ll_ik_new;
      //this->ULL = this->ULL - ll_i + ll_i_new; //this is a problem when threading unless ULL satsifies some propties. so we separately deal with this.
    } else {
      conv = true; //return with current Zeta, as no step has been made
    }
  }

  //likely thread safe as the indices in the vector are independently written to.
  vdk(i)=v_ik;

}

void igFitterOnColumns::move_v_d_k_withUnitNormConstraint_normUStrategy(const int& d, const int& k,
                                                                        const double& dk,
                                                                        const arma::vec& u_k_normed,
                                                                        arma::vec& vdk,
                                                                        const double& f1,
                                                                        const double& f2){

  const arma::mat& Zd = Z[d];
  const arma::mat& Zetad = Zeta[d];
  arma::uvec cols_except_k=this->seqOfComponents(find(this->seqOfComponents != k));
  const arma::mat& umatOther = this->umat.cols(cols_except_k);
  const arma::mat& vmatd = this->vmat[d];
  const arma::mat& vmatOtherd = vmatd.cols(cols_except_k);
  const arma::vec& aphid = aphi[d];
  const arma::vec& rowOffsetsd = rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd = colOffsets_for_columns[d];


  tbb::parallel_for( tbb::blocked_range<size_t>(0,pd[d]),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t i=ra.begin(); i!=ra.end(); ++i){
                         move_v_d_k_i_normUkStrategy(d, i,k,
                                                     dk, u_k_normed,
                                                     Zd, Zetad, umatOther, vmatOtherd,
                                                     vmatd, aphid,
                                                     rowOffsetsd, colOffsets_for_columnsd,
                                                     vdk,
                                                     f1, f2);
                       }
                     }
  );
}

void igFitterOnColumns::move_v_k_withUnitNormConstraint_normUStrategy(
    const int& k,
    const double& f1,
    const double& f2
){
  
  const double penalty_term = f2 * n * getPenalty();
  if (!std::isfinite(penalty_term)) {
    return;
  }

  std::vector<arma::vec> v_k(D);
  std::vector<arma::vec> grad_v_k(D);
  std::vector<arma::vec> p_k(D);

  double normSquared = 0.0;
  for (int d = 0; d < D; ++d) {
    v_k[d] = vmat[d].col(k);
    grad_v_k[d].set_size(pd[d]);
    p_k[d].set_size(pd[d]);
    normSquared += wd[d]*arma::dot(v_k[d], v_k[d]);
  }

  if (!std::isfinite(normSquared) || normSquared <= 1e-16) {
    
    hmat.col(k).zeros();
    umat.col(k).zeros();
    for (int d = 0; d < D; ++d) {
      vmat[d].col(k).zeros();
    }
    return;
  }

  const double normVk = std::sqrt(normSquared);
  if (std::abs(normVk - 1.0) > 1e-12) {
    for (int d = 0; d < D; ++d) {
      vmat[d].col(k) /= normVk;
      v_k[d] = vmat[d].col(k);
    }
    umat.col(k) *= normVk;
    hmat.col(k) *= normVk;
  }

  bool converged = false;
  int iter = 0;
  while (!converged && iter <= this->inner_maxIter) {
    ++iter;

    const arma::vec u_k = umat.col(k);
    const arma::uvec others = find(seqOfComponents != k);
    const arma::mat umatOthers = umat.cols(others);

    for (int d = 0; d < D; ++d) {
      const arma::mat vmatOthersd = vmat[d].cols(others);
      const arma::vec& aphid = aphi[d];
      const arma::vec& rowOffsetsd = rowOffsets[d];
      const arma::vec& colOffsetsd = colOffsets_for_columns[d];

      tbb::parallel_for(
        tbb::blocked_range<size_t>(0, pd[d]),
        [&](const tbb::blocked_range<size_t>& ra) {
          for (size_t i = ra.begin(); i != ra.end(); ++i) {
            arma::vec y_i_all;
            if (d < dDense) {
              y_i_all = denseMats[d].row(i).t();
            } else {
              y_i_all = sparseMats[d - dDense].row(i).t();
            }
            const arma::vec y_i = y_i_all(columns1);

            const arma::vec otherEtaContrib =
              Z[d] * Zeta[d].col(i) +
              umatOthers * vmatOthersd.row(i).t() +
              colOffsetsd + rowOffsetsd(i);
            const arma::vec eta_ik =
              vmat[d](i, k) * u_k + otherEtaContrib;

            grad_v_k[d](i) = f1 * gradNegLogLL_v_dik(
              d, y_i, eta_ik, u_k, vmat[d](i, k), aphid(i)
            ); 
          }
        }
      );
    }

    double vdotGradient = 0.0;
    for (int d = 0; d < D; ++d) {
      if (grad_v_k[d].has_nonfinite()) {
        return;
      }
      vdotGradient += wd[d]*arma::dot(v_k[d], grad_v_k[d]); 
    }

    double gradientDotDirection = 0.0;
    double directionNormSquared = 0.0;
    for (int d = 0; d < D; ++d) {
      p_k[d] = -grad_v_k[d] + vdotGradient * v_k[d];
      gradientDotDirection += wd[d]*arma::dot(grad_v_k[d], p_k[d]);
      directionNormSquared += arma::dot(p_k[d], p_k[d]);
    }

    if (!std::isfinite(gradientDotDirection) ||
        !std::isfinite(directionNormSquared) ||
        directionNormSquared <= 1e-20 ||
        gradientDotDirection >= 0.0) {
      break;
    }

    std::vector<arma::vec> nllCurrent(D);
    std::vector<arma::vec> nllCandidate(D);
    for (int d = 0; d < D; ++d) {
      nllCurrent[d].set_size(pd[d]);
      nllCandidate[d].set_size(pd[d]);
    }
    const double nllCurrentValue = ULL;
    const double objectiveCurrentValue = f1 * nllCurrentValue + penalty_term;
    if (!std::isfinite(objectiveCurrentValue)) {
      break;
    }

    bool accepted = false;
    double alpha = alpha0;
    double nllCandidateValue = nllCurrentValue;
    std::vector<arma::vec> candidate(D);

    while (alpha >= 1e-4 && !accepted) {
      double candidateNormSquared = 0.0;
      
      for (int d = 0; d < D; ++d) {
        candidate[d] = v_k[d] + alpha * p_k[d];
        candidateNormSquared += wd[d]*arma::dot(candidate[d], candidate[d]);
      }

      if (!std::isfinite(candidateNormSquared) ||
          candidateNormSquared <= 1e-20) {
        alpha *= rho;
        continue;
      }

      const double candidateNorm = std::sqrt(candidateNormSquared);
      for (int d = 0; d < D; ++d) {
        candidate[d] /= candidateNorm;
        vmat[d].col(k) = candidate[d];
      }

      nllCandidateValue = calcNegLogLL(nllCandidate, false);
      const double objectiveCandidateValue =
        f1 * nllCandidateValue + penalty_term;
      if (std::isfinite(objectiveCandidateValue) &&
          objectiveCandidateValue <= objectiveCurrentValue) {
        accepted = true;
      } else {
        for (int d = 0; d < D; ++d) {
          vmat[d].col(k) = v_k[d];
        }
        alpha *= rho;
      }
    }

    if (!accepted) {
      for (int d = 0; d < D; ++d) {
        vmat[d].col(k) = v_k[d];
      }
      break;
    }

    double realizedStepSquared = 0.0;
    for (int d = 0; d < D; ++d) {
      realizedStepSquared += arma::accu(
        arma::square(candidate[d] - v_k[d])
      );
      v_k[d] = candidate[d];
      negLogLikesVecd[d] = nllCandidate[d];
    }
    ULL = nllCandidateValue;

    unsigned int totalFeatures = 0;
    for (int d = 0; d < D; ++d) {
      totalFeatures += pd[d];
    }
    const double objectiveCandidateValue =
      f1 * nllCandidateValue + penalty_term;
    const double deltaF_inner =
      std::abs(objectiveCandidateValue - objectiveCurrentValue);
    const double deltaVRMS = std::sqrt(
      realizedStepSquared /
      static_cast<double>(std::max<unsigned int>(1, totalFeatures))
    );
    converged = deltaF_inner < this->solved_inner_ftol_abs;
    (void)deltaVRMS;
  }
}

void igFitterOnColumns::moveVU(const double& f1, const double& f2){

  //move each factor
  for( int k=0; k<K; k++ ){ //<= as K has been set to the Total#Components-1 in the cosntructor
    //move u_k ie., h_k with line search

    move_h_k(k, f1, f2,this->alpha0, this->rho);
    umat.col(k) = M * hmat.col(k);

    recordDebugTrace(2, k, f2);

    //MOVE V_k
    move_v_k_withUnitNormConstraint_normUStrategy(k, f1, f2);
    
    
    for( int d=0; d < D; d++ ){
      arma::vec& nld =this->negLogLikesVecd[d];
      if( nld.has_inf() | nld.has_nan() | nld.has_nonfinite() ){
        throw std::invalid_argument( "Nope. Nans found in dataset: " + std::to_string(d) );
      }
    }


    if( arma::sign( vmat[0].at(0,k) ) <0 ){//rotation identifiabilty
      for( int d=0; d<D; d++ ){
        vmat[d].col(k) *= -1;
      }
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
void igFitterOnColumns::gradNegLogLL_d_h_k(
    const int& d,
    const int& k,
    const arma::vec& h_k,
    std::vector<arma::vec>& grad_h_k
){

  arma::vec v_k = this->vmat[d].col(k);
  if( K > 1 ){
    arma::uvec others = find( seqOfComponents != k );
    arma::mat umatOthers = this->umat.cols(others);
    arma::mat vmatOthers = this->vmat[d].cols(others);
    if( d < dDense ){
      this->denseRowFams[d]->gradNegLogLL_h_k_onColsOf(denseMats[d], columns1, M, Z[d], Zeta[d],
                                                       umatOthers, vmatOthers,
                                                       aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                    h_k, v_k,
                                                                                                    grad_h_k[d]
      );
    } else{
      this->sparseRowFams[d-dDense]->gradNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1, M, Z[d], Zeta[d],
                                                               umatOthers, vmatOthers,
                                                               aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                            h_k, v_k,
                                                                                                            grad_h_k[d]
      );
    }
  } else {
    // K =1
    if( d < dDense ){
      this->denseRowFams[d]->gradNegLogLL_h_k_onColsOf(denseMats[d], columns1, M, Z[d], Zeta[d],
                                                       aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                    h_k, v_k,
                                                                                                    grad_h_k[d]
      );
    } else{
      this->sparseRowFams[d-dDense]->gradNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1, M, Z[d], Zeta[d],
                                                               aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                            h_k, v_k,
                                                                                                            grad_h_k[d]
      );
    }

  }

}

void igFitterOnColumns::gradNegLogLL_h_k(
    const int& k,
    const arma::vec& h_k,
    std::vector<arma::vec>& grad_d_h_k,
    arma::vec& netGrad_h_k
){

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         gradNegLogLL_d_h_k(d,k,
                                            h_k,
                                            grad_d_h_k);
                       }
                     }
  );

  //calculate net gradient
  netGrad_h_k = 0*grad_d_h_k[0];
  for( int d=0; d<D; d++ ){
    netGrad_h_k += wd[d]*grad_d_h_k[d]; //weighted! 
  }
}


// K=1
void igFitterOnColumns::gradNegLogLL_d_h_k(
    const int& d,
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const arma::vec& v_k,       // corresponding loading vector
    arma::vec& grad_d_h_k  //to store gradient in
){
  if( d < dDense ){
    this->denseRowFams[d]->gradNegLogLL_h_k_onColsOf(denseMats[d], columns1, M, Z[d], Zeta[d],
                                                     aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                  h_k, v_k,
                                                                                                  grad_d_h_k
    );
  } else{
    this->sparseRowFams[d-dDense]->gradNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1, M, Z[d], Zeta[d],
                                                             aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                                          h_k, v_k,
                                                                                                          grad_d_h_k
    );
  }

}
void igFitterOnColumns::gradNegLogLL_d_h_k(
    const int& d,
    const arma::mat& umatOthers, //containing columns of umat except for the k^th
    const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const arma::vec& v_k,       // corresponding loading vector
    arma::vec& grad_d_h_k  //to store gradient in
){
  if( d < dDense ){
    this->denseRowFams[d]->gradNegLogLL_h_k_onColsOf(denseMats[d], columns1, M, Z[d], Zeta[d],
                                                     umatOthers, vmatOthers,
                                                     aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                      h_k, v_k,
                                                                                      grad_d_h_k
    );
  } else{
    this->sparseRowFams[d-dDense]->gradNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1, M, Z[d], Zeta[d],
                                                             umatOthers, vmatOthers,
                                                             aphi[d],rowOffsets[d], colOffsets_for_columns[d],
                                                                                              h_k, v_k,
                                                                                              grad_d_h_k
    );
  }

}

//K=1
void igFitterOnColumns::gradNegLogLL_h_k(
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const std::vector<arma::vec>& v_k,       // corresponding loading vector
    std::vector<arma::vec>& grad_h_k,  //to store gradient in
    arma::vec& netGrad_h_k
){

  //assign gradients dataset-wise
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         gradNegLogLL_d_h_k(d,h_k,
                                            v_k[d],grad_h_k[d]);
                       }
                     }
  );
  //net gradient
  netGrad_h_k = 0*grad_h_k[0];
  for( int d=0; d<D; d++ ){
    netGrad_h_k += wd[d]*grad_h_k[d]; //weighted!
  }

}
void igFitterOnColumns::gradNegLogLL_h_k(
    const arma::mat& umatOthers, //containing columns of umat except for the k^th
    const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const std::vector<arma::vec>& v_k,       // corresponding loading vector
    std::vector<arma::vec>& grad_h_k,  //to store gradient in
    arma::vec& netGrad_h_k
){

  //assign gradients dataset-wise
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         gradNegLogLL_d_h_k(d,umatOthers,
                                            vmatOthers[d],
                                                      h_k,
                                                      v_k[d],
                                                         grad_h_k[d]);
                       }
                     }
  );

  //net gradient
  netGrad_h_k = 0*grad_h_k[0];
  for( int d=0; d<D; d++ ){
    netGrad_h_k += wd[d]* grad_h_k[d]; //weighted!
  }

}

void igFitterOnColumns::hessNegLogLL_d_h_k(
    const int& d,
    const int& k,
    const arma::vec& h_k,
    std::vector<arma::mat>& hess_d_h_k
){

  arma::vec v_k = this->vmat[d].col(k);

  if(family[d]==std::string("gaussian")){
    hess_d_h_k[d]= arma::sum(arma::pow(vmat[d].col(k),2)/aphi[d]) *MtM; // a scalar * MtM
  } else {

    if( K == 1 ){

      if( d < dDense ){

        this->denseRowFams[d]->hessNegLogLL_h_k_onColsOf(denseMats[d], columns1,
                                                         M, Z[d], Zeta[d],
                                                                      aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                    MtM, h_k, v_k, hess_d_h_k[d]
        );

      } else {
        this->sparseRowFams[d]->hessNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1,
                                                          M, Z[d], Zeta[d],
                                                                       aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                     MtM, h_k, v_k, hess_d_h_k[d]
        );

      }

    } else {

      arma::uvec others = find( seqOfComponents != k );   // TODO: WE ARE DOING THESE COPIES OF UMATS AND VMATS TWICE! ONE IN GRAD FN, and ANOTHER IN HESS FN. FUTURE should minimize this.
      arma::mat umatOthers = this->umat.cols(others);
      arma::mat vmatOthers = this->vmat[d].cols(others);
      if( d < dDense ){

        this->denseRowFams[d]->hessNegLogLL_h_k_onColsOf(denseMats[d], columns1,
                                                         M, Z[d], Zeta[d],
                                                                      umatOthers, vmatOthers,
                                                                      aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                    MtM, h_k, v_k, hess_d_h_k[d]
        );

      } else {
        this->sparseRowFams[d]->hessNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1,
                                                          M, Z[d], Zeta[d],
                                                                       umatOthers, vmatOthers,
                                                                       aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                     MtM, h_k, v_k, hess_d_h_k[d]
        );

      }


    }





  }

}

void igFitterOnColumns::hessNegLogLL_h_k(
    const int& k,
    const arma::vec& h_k,
    std::vector<arma::mat>& hess_d_h_k,
    arma::mat& netHessh_k
){

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         hessNegLogLL_d_h_k(d,k,h_k,hess_d_h_k);
                       }
                     }
  );

  //calculate net hessian
  netHessh_k = 0*hess_d_h_k[0];
  for( int d=0; d<D; d++ ){
    netHessh_k += wd[d]* hess_d_h_k[d]; //weighted!
  }

}

// for K =1
void igFitterOnColumns::hessNegLogLL_d_h_k(
    const int& d,
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const arma::vec& v_k,       // corresponding loading vector
    arma::mat& hess_h_k  //to store gradient in
){

  if( d < dDense ){

    this->denseRowFams[d]->hessNegLogLL_h_k_onColsOf(denseMats[d], columns1,
                                                     M, Z[d], Zeta[d],
                                                                  aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                MtM, h_k, v_k, hess_h_k
    );

  } else {
    this->sparseRowFams[d]->hessNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1,
                                                      M, Z[d], Zeta[d],
                                                                   aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                                 MtM, h_k, v_k, hess_h_k
    );
  }

}
void igFitterOnColumns::hessNegLogLL_d_h_k(
    const int& d,
    const arma::mat& umatOthers, //containing columns of umat except for the k^th
    const arma::mat& vmatOthers, //same config as umatOthers except here we have the vmat
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const arma::vec& v_k,       // corresponding loading vector
    arma::mat& hess_h_k  //to store gradient in
){

  if( d < dDense ){

    this->denseRowFams[d]->hessNegLogLL_h_k_onColsOf(denseMats[d], columns1,
                                                     M, Z[d], Zeta[d],
                                                                  umatOthers, vmatOthers,
                                                                  aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                    MtM, h_k, v_k, hess_h_k
    );

  } else {
    this->sparseRowFams[d]->hessNegLogLL_h_k_onColsOf(sparseMats[d-dDense], columns1,
                                                      M, Z[d], Zeta[d],
                                                                   umatOthers, vmatOthers,
                                                                   aphi[d], rowOffsets[d], colOffsets_for_columns[d],
                                                                                                     MtM, h_k, v_k, hess_h_k
    );
  }

}

//for K=1
void igFitterOnColumns::hessNegLogLL_h_k(
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const std::vector<arma::vec>& v_k,       // corresponding loading vector
    std::vector<arma::mat>& hess_h_k,  //to store gradient in
    arma::mat& netHess_h_k
){

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         hessNegLogLL_d_h_k(d,h_k, v_k[d], hess_h_k[d]);
                       }
                     }
  );

  //calculate net hessian
  netHessh_k = 0*hess_h_k[0];
  for( int d=0; d<D; d++ ){
    netHessh_k += wd[d]* hess_h_k[d]; //weighted!
  }

}
void igFitterOnColumns::hessNegLogLL_h_k(
    const arma::mat& umatOthers, //containing columns of umat except for the k^th
    const std::vector<arma::mat>& vmatOthers, //same config as umatOthers except here we have the vmat
    const arma::vec& h_k,        //the h_k tha entires the k^th comp u_k = M h_k
    const std::vector<arma::vec>& v_k,       // corresponding loading vector
    std::vector<arma::mat>& hess_h_k,  //to store gradient in
    arma::mat& netHess_h_k
){

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         hessNegLogLL_d_h_k(d,umatOthers, vmatOthers[d], h_k, v_k[d], hess_h_k[d]);
                       }
                     }
  );

  //calculate net hessian
  netHessh_k = 0*hess_h_k[0];
  for( int d=0; d<D; d++ ){
    netHessh_k += wd[d]* hess_h_k[d];//weighted
  }

}

/////////////////////////////////////////
/*
 * Gradients of objectives
 */
/////////////////////////////////////////
void igFitterOnColumns::gradPenalizedObjective_h_k(
    const int& k,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    arma::vec& netGradh_k,
    const double& f1,
    const double& f2
){
  //gradient of negloglikelihood
  gradNegLogLL_h_k( k, h_k,
                    this->gradObjh_k,
                    this->netGradh_k  );

  //alter as appropriate
  netGradh_k *= f1;
  netGradh_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart);//n because of 1/n prefactor in objective
}

void igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
    const int& k,
    const arma::vec& h_k,
    arma::mat& netHessh_k,
    const double& f1,
    const double& f2
){
  //grab hessian of the entire data neg log likelihood
  hessNegLogLL_h_k( k, h_k,
                    this->hessObjh_k,
                    this->netHessh_k  );


  //alter for the objective
  netHessh_k *= f1;
  //std::cout << "adding to In hesspenalized_hk" << std::endl;
  netHessh_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde; //n because of 1/n prefactor in objective
}

//overloaded forms
//for K=1
void igFitterOnColumns::gradPenalizedObjective_h_k(
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const std::vector<arma::vec>& v_k,
    arma::vec& netGrad_h_k,
    const double& f1,
    const double& f2
){


  //grab gradient of the entire data negative log like
  this->gradNegLogLL_h_k( h_k,        //the h_k tha entires the k^th comp u_k = M h_k
                          v_k,       // corresponding loading vector
                          this->gradObjh_k,  //to store gradient in
                          netGrad_h_k );


  //alter as appropriate
  netGrad_h_k *= f1;
  netGrad_h_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart); //n because of 1/n prefactor in objective
}
void igFitterOnColumns::gradPenalizedObjective_h_k(
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const arma::vec& h_k_qpart, //i.e., hmat.col(k).rows(rows_for_qxq) if this is allowed in armadillo
    const std::vector<arma::vec>& v_k,
    arma::vec& netGrad_h_k,
    const double& f1,
    const double& f2
){


  //grab gradient of the entire data negative log like
  this->gradNegLogLL_h_k( umatOthers, //containing columns of umat except for the k^th
                          vmatOthers, //same config as umatOthers except here we have the vmat
                          h_k,        //the h_k tha entires the k^th comp u_k = M h_k
                          v_k,       // corresponding loading vector
                          this->gradObjh_k,  //to store gradient in
                          netGrad_h_k );


  //alter as appropriate
  netGrad_h_k *= f1;
  netGrad_h_k( this->rows_for_qxq ) += 2*f2*n*(Rtilde*h_k_qpart); //n because of 1/n prefactor in objective
}

//for K=1
void igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    arma::mat& netHess_h_k,
    const double& f1,
    const double& f2
){
  hessNegLogLL_h_k( h_k,        //the h_k tha entires the k^th comp u_k = M h_k
                    v_k,       // corresponding loading vector
                    this->hessObjh_k,  //to store gradient in
                    netHess_h_k );


  //alter for the objective
  netHess_h_k *= f1;
  //std::cout << "adding to In hesspenalized_hk" << std::endl;
  netHess_h_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde; // n because of 1/n prefactor in objective
}
void igFitterOnColumns::hessPenalizedObjectiveLL_h_k(
    const arma::mat& umatOthers,
    const std::vector<arma::mat>& vmatOthers,
    const arma::vec& h_k,
    const std::vector<arma::vec>& v_k,
    arma::mat& netHess_h_k,
    const double& f1,
    const double& f2
){
  hessNegLogLL_h_k( umatOthers, //containing columns of umat except for the k^th
                    vmatOthers, //same config as umatOthers except here we have the vmat
                    h_k,        //the h_k tha entires the k^th comp u_k = M h_k
                    v_k,       // corresponding loading vector
                    this->hessObjh_k,  //to store gradient in
                    netHess_h_k );


  //alter for the objective
  netHess_h_k *= f1;
  //std::cout << "adding to In hesspenalized_hk" << std::endl;
  netHess_h_k( rows_for_qxq, cols_for_qxq ) += 2*f2*n*Rtilde; //n because of 1/n prefactor in objective
}


#if 0
// Previous h_k implementation retained for reference.  It computed an
// unconstrained Newton direction, clamped it, and projected the candidate
// afterward.  The active implementation below instead solves the constrained
// quadratic model directly through a damped KKT system.
void igFitterOnColumns::move_h_k(
    const int& k,
    const double& f1,
    const double& f2,
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
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart );
      // std::cout << "second : attempting gradPenalized in while loop" << std::endl;
      this->gradPenalizedObjective_h_k(h_k, h_k_qpart, v_k, this->netGradh_k, f1, f2);
      // std::cout << "done with that. " << std::endl;
      // std::cout << "third : attempting hess in while loop" << std::endl;
      this->hessPenalizedObjectiveLL_h_k(h_k, v_k, this->netHessh_k, f1, f2);
      // std::cout << "done with that" << std::endl;
      p_k = -invpsd(this->netHessh_k, 1e-7, false) * netGradh_k; //col vector
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
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart );
        // std::cout << "seventh : done calculating penalty" << std::endl;
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

    //Separate parts due to k and the rest.
    // TODO : check if K must be >1 for this to work.
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
      obj_k =f1*ll_k + f2*n*arma::dot( h_k_qpart, Rtilde * h_k_qpart );
      // std::cout << "second : attempting gradPenalized in while loop" << std::endl;
      this->gradPenalizedObjective_h_k(umatOthers, vmatOthers, h_k, h_k_qpart, v_k, this->netGradh_k, f1, f2);
      // std::cout << "done with that. " << std::endl;
      // std::cout << "third : attempting hess in while loop" << std::endl;
      this->hessPenalizedObjectiveLL_h_k(umatOthers, vmatOthers, h_k, v_k, this->netHessh_k, f1, f2);
      // std::cout << "done with that" << std::endl;
      p_k = -invpsd(this->netHessh_k, 1e-7, false) * netGradh_k; //col vector
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
        obj_k_new = f1*ll_k_new + f2*n*arma::dot( h_k_new_qpart, Rtilde * h_k_new_qpart );
        // std::cout << "seventh : done calculating penalty" << std::endl;
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



  //likely thread safe as the indices in the vector are independently written to.
  hmat.col(k)=h_k;       //this resets hmat to h_k from h_k_new in case no move has happened.
  umat.col(k)= M*h_k;     // same as above but for u_k
  this->ULL = ll_k;
}

#endif

void igFitterOnColumns::move_h_k(
    const int& k,
    const double& f1,
    const double& f2,
    const double alpha0,
    const double rho
){
  const double entryUMHDiff =
    arma::abs(
      this->umat.col(k) - this->M * this->hmat.col(k)
    ).max();
  const arma::vec h_k_entry = hmat.col(k);

  arma::vec h_k = h_k_entry;
  arma::vec h_k_qpart = h_k(rows_for_qxq);

  const arma::vec uBeforeProjection = this->M * h_k;

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
  
  // P projects coefficient-space vectors onto the feasible tangent
  // subspace.  It is initialized to the identity so that a numerically
  // rank-zero constraint matrix correctly yields an unconstrained update.
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
      rawConstraintMat = arma::join_rows(rawConstraintMat, 
                                         M.rows(qids).t() * Buothers
                                           );
    }
    
  }
  
  // Replace the raw constraints by an orthonormal basis for the same
  // column space.  This preserves the feasible tangent space because
  // null(rawConstraintMat.t()) == null(constraintMat.t()).  It also
  // removes zero and linearly dependent constraint columns, which create
  // singular KKT multiplier directions even though the geometric
  // constraint space itself is well defined.
  constraintMat = arma::orth(rawConstraintMat);
  
  if (constraintMat.n_cols > 0) {
    // In general,
    //   P = I - C(C.t() C)^{-1}C.t().
    // Because C has orthonormal columns after arma::orth(), C.t() C = I,
    // and therefore P = I - C C.t().
    P -= constraintMat * constraintMat.t();
    
    // Numerical cleanup only.  Subsequent KKT directions satisfy
    // C.t() p = 0, so all accepted linear steps remain feasible.
    h_k = P * h_k;
    h_k_qpart = h_k(rows_for_qxq);
  }

  const double projectionChangeRMS =
    std::sqrt(arma::mean(arma::square(h_k - h_k_entry)));

  double nll_k = (K == 1)
    ? calcNegLogLL_with_h_k(h_k, v_k)
    : calcNegLogLL_with_h_k(umatOthers, vmatOthers, h_k, v_k);

  const arma::vec uAfterProjection = this->M * h_k;
  hProjectionDebugTrace.push_back({
    static_cast<double>(debugIteration),
    static_cast<double>(k),
    uBeforeProjection.min(),
    uBeforeProjection.max(),
    uAfterProjection.min(),
    uAfterProjection.max(),
    nll_k,
    std::isfinite(nll_k) ? 1.0 : 0.0
  });

  if (!std::isfinite(nll_k)) {
    return;
  }

  bool converged = false;
  int iter = 0;
  while (!converged && iter <= this->inner_maxIter) {
    ++iter;

    const double obj_k =
      f1 * nll_k + f2*n * arma::dot(h_k_qpart, Rtilde * h_k_qpart);

    if (K == 1) {
      gradPenalizedObjective_h_k(
        h_k, h_k_qpart, v_k, netGradh_k, f1, f2
      );
      hessPenalizedObjectiveLL_h_k(
        h_k, v_k, netHessh_k, f1, f2
      );
    } else {
      gradPenalizedObjective_h_k(
        umatOthers, vmatOthers, h_k, h_k_qpart,
        v_k, netGradh_k, f1, f2
      );
      hessPenalizedObjectiveLL_h_k(
        umatOthers, vmatOthers, h_k, v_k,
        netHessh_k, f1, f2
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
    int dampingAttempts = 0;
    double dampingUsed =
      std::numeric_limits<double>::quiet_NaN();

    for (int dampingAttempt = 0;
         dampingAttempt < 10 && !haveNewtonDirection;
         ++dampingAttempt) {
      ++dampingAttempts;
      const double currentDamping = damping;
      const arma::mat hessDamped =
        hessSym + damping * arma::eye<arma::mat>(
          hessSym.n_rows, hessSym.n_cols
        );

      // Diagonal (Jacobi) rescaling of the Newton/KKT system,
      // mirroring gFitterOnColumns. Write p = D q, solve in the
      // scaled coordinates, then map back to the original h coordinates.
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

        if (debugTraceEnabled && debugIteration == 1 && k == 0) {
          hNewtonDebugTrace.push_back({
            static_cast<double>(debugIteration), static_cast<double>(k),
            static_cast<double>(iter), static_cast<double>(dampingAttempt + 1),
            currentDamping, arma::norm(netGradh_k, 2),
            hessSym.diag().min(), hessSym.diag().max(),
            arma::mean(arma::abs(hessSym.diag())),
            static_cast<double>(constraintMat.n_cols), 1.0,
            feasibilityError, directionalDerivative, arma::abs(candidateDirection).max()
          });
        }

        if (std::isfinite(feasibilityError) &&
            feasibilityError <= 1e-7 &&
            std::isfinite(directionalDerivative) &&
            directionalDerivative < 0.0) {
          p_k = candidateDirection;
          haveNewtonDirection = true;
          dampingUsed = currentDamping;
        }
      }

      if (!haveNewtonDirection) {
        if (debugTraceEnabled && debugIteration == 1 && k == 0 &&
            !(solved && candidateDirection.n_elem == h_k.n_elem &&
              !candidateDirection.has_nonfinite())) {
          hNewtonDebugTrace.push_back({
            static_cast<double>(debugIteration), static_cast<double>(k),
            static_cast<double>(iter), static_cast<double>(dampingAttempt + 1),
            currentDamping, arma::norm(netGradh_k, 2),
            hessSym.diag().min(), hessSym.diag().max(),
            arma::mean(arma::abs(hessSym.diag())),
            static_cast<double>(constraintMat.n_cols), 0.0,
            arma::datum::nan, arma::datum::nan, arma::datum::nan
          });
        }
        damping *= 10.0;
      }
    }

    // Robust fallback: projected gradient descent in the same feasible
    // tangent space if every damped Newton attempt fails.
    if (!haveNewtonDirection) {
      p_k = -P * netGradh_k;

      // Mirror gFitterOnColumns: trust-limit only the projected-gradient
      // fallback in U-space. Successful Newton/KKT directions are unchanged.
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
    int lineSearchAttempt = 0;

    while (alpha >= 1e-4 && !accepted) {
      ++lineSearchAttempt;
      h_k_new = h_k + alpha * p_k;
      const arma::vec h_k_new_qpart = h_k_new(rows_for_qxq);

      nll_k_new = (K == 1)
        ? calcNegLogLL_with_h_k(h_k_new, v_k)
        : calcNegLogLL_with_h_k(
            umatOthers, vmatOthers, h_k_new, v_k
          );

      obj_k_new =
        f1 * nll_k_new +
        f2*n * arma::dot(h_k_new_qpart, Rtilde * h_k_new_qpart);

      const arma::vec actualStep = h_k_new - h_k;
      const arma::vec stepMinusAlphaP = actualStep - alpha * p_k;
      const double stepIdentityError =
        arma::norm(stepMinusAlphaP, 2);
      const double gradDotActualStep =
        arma::dot(netGradh_k, actualStep);
      const double gradDotAlphaP =
        alpha * arma::dot(netGradh_k, p_k);
      const double deltaHRMS =
        std::sqrt(arma::mean(arma::square(actualStep)));
      const arma::vec deltaU = this->M * actualStep;
      const double deltaURMS =
        std::sqrt(arma::mean(arma::square(deltaU)));
      const double projectedGradNorm =
        arma::norm(P * netGradh_k, 2);

      const bool candidateAccepted =
        std::isfinite(nll_k_new) &&
        std::isfinite(obj_k_new) &&
        obj_k_new <= obj_k;

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
          candidateAccepted ? 1.0 : 0.0,
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

      if (candidateAccepted) {
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
    // Optional future parameter-stability criterion:
    // converged = converged || (deltaURMS < inner_eps);
    (void)deltaURMS;
    h_k = h_k_new;
    h_k_qpart = h_k(rows_for_qxq);
    nll_k = nll_k_new;
  }

  hmat.col(k) = h_k;
  umat.col(k) = M * h_k;
  ULL = nll_k;
}


/*
 * RECORDERS
 */
void igFitterOnColumns::enableDebugTrace() {
  debugTraceEnabled = true;
  debugTrace.clear();
  hDebugTrace.clear();
  hNewtonDebugTrace.clear();
  hProjectionDebugTrace.clear();
}

void igFitterOnColumns::recordDebugTrace(
    const int stage,
    const int component,
    const double f2
) {
  if (!debugTraceEnabled) {
    return;
  }

  std::vector<arma::vec> nllvec = negLogLikesVecd;
  calcNegLogLL(nllvec, true);
  const double currentObjective = ULL + f2 * getPenalty();

  double vMin = std::numeric_limits<double>::infinity();
  double vMax = -std::numeric_limits<double>::infinity();
  double uvMin = std::numeric_limits<double>::infinity();
  double uvMax = -std::numeric_limits<double>::infinity();
  double etaMin = std::numeric_limits<double>::infinity();
  double etaMax = -std::numeric_limits<double>::infinity();
  std::size_t outsideClamp = 0;

  for (unsigned int d = 0; d < D; ++d) {
    vMin = std::min(vMin, vmat[d].min());
    vMax = std::max(vMax, vmat[d].max());

    arma::mat uv = vmat[d] * umat.t();
    arma::mat eta = uv;

    if (Z[d].n_cols > 0 && Zeta[d].n_rows > 0) {
      eta += (Z[d] * Zeta[d]).t();
    }

    eta.each_col() += rowOffsets[d];
    eta.each_row() += colOffsets_for_columns[d].t();

    uvMin = std::min(uvMin, uv.min());
    uvMax = std::max(uvMax, uv.max());
    etaMin = std::min(etaMin, eta.min());
    etaMax = std::max(etaMax, eta.max());

    if (family[d] == "poisson") {
      outsideClamp += arma::accu(eta < -20.0);
      outsideClamp += arma::accu(eta >  20.0);
    }
  }

  debugTrace.push_back({
    static_cast<double>(debugIteration),
    static_cast<double>(stage),
    static_cast<double>(component),
    currentObjective,
    umat.min(),
    umat.max(),
    hmat.min(),
    hmat.max(),
    vMin,
    vMax,
    uvMin,
    uvMax,
    etaMin,
    etaMax,
    static_cast<double>(outsideClamp)
  });
}

arma::mat igFitterOnColumns::getDebugTrace() const {
  arma::mat out(debugTrace.size(), DEBUG_NCOLS, arma::fill::zeros);

  for (std::size_t i = 0; i < debugTrace.size(); ++i) {
    for (std::size_t j = 0; j < DEBUG_NCOLS; ++j) {
      out(i, j) = debugTrace[i][j];
    }
  }

  return out;
}

arma::mat igFitterOnColumns::getHDebugTrace() const {
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


arma::mat igFitterOnColumns::getHNewtonDebugTrace() const {
  arma::mat out(hNewtonDebugTrace.size(), H_NEWTON_DEBUG_NCOLS, arma::fill::zeros);
  for (std::size_t i = 0; i < hNewtonDebugTrace.size(); ++i)
    for (std::size_t j = 0; j < H_NEWTON_DEBUG_NCOLS; ++j) out(i,j)=hNewtonDebugTrace[i][j];
  return out;
}

arma::mat igFitterOnColumns::getHProjectionDebugTrace() const {
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


std::vector<arma::mat> igFitterOnColumns::get_Zeta(){
  return Zeta;
}

std::vector<arma::mat> igFitterOnColumns::get_Zeta_R(){
  std::vector<arma::mat> Zeta_R(D);
  for( int d=0; d < D; d++ ){
    Zeta_R[d] = Zeta[d].t();
  }
  return Zeta_R;
}

arma::mat igFitterOnColumns::get_hmat(){
  return hmat;
}

arma::mat igFitterOnColumns::get_umat(){
  return umat;
}

std::vector<arma::mat> igFitterOnColumns::get_vmat(){
  return vmat;
}

std::vector<arma::vec> igFitterOnColumns::get_aphi(){
  return aphi;
}

arma::mat igFitterOnColumns::get_gmat(){ // u = f + g = Sd + Rc
  return M.cols(r, r+q-1) *  hmat.rows( r, r+q-1);
}


//init
// std::vector<arma::mat> igFitterOnColumns::get_init_Zeta(){
//   return init_Zeta;
// }
//
// std::vector<arma::mat> igFitterOnColumns::get_init_Zeta_R(){
//   std::vector<arma::mat> Zeta_R(D);
//   for( int d=0; d < D; d++ ){
//     Zeta_R[d] = init_Zeta[d].t();
//   }
//   return Zeta_R;
// }
//
// arma::mat igFitterOnColumns::get_init_hmat(){
//   return init_hmat;
// }
//
// arma::mat igFitterOnColumns::get_init_umat(){
//   return init_umat;
// }
//
// std::vector<arma::mat> igFitterOnColumns::get_init_vmat(){
//   return init_vmat;
// }
//
// std::vector<arma::vec> igFitterOnColumns::get_init_aphi(){
//   return init_aphi;
// }
//
//


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
void igFitterOnColumns::estimate(const double& f1,
                                 const double& f2,
                                 const int& maxIter,
                                 const double& eps,
                                 const double& xtol_abs,
                                 const int& inner_maxIter,
                                   const double& inner_ftol_abs){ //M, Z, qids, Rtilde, Q, B, BQ

  (void)xtol_abs; // reserved for future parameter-change convergence

  this->inner_maxIter = inner_maxIter;
  this->solved_inner_ftol_abs = n * inner_ftol_abs;


  // std::cout << "----------igFitterOnColumns-------------" << std::endl;
  // std::cout << "maxIter: " << std::endl;
  // std::cout << maxIter << std::endl;
  // std::cout << "--------------------------------" << std::endl;

  // init_aphi=aphi;
  // init_hmat=hmat;
  // init_umat=umat;
  // init_vmat=vmat;
  // init_Zeta = Zeta;
  // init_rowOffsets=rowOffsets;
  // init_colOffsets=colOffsets;


  //update NLL : necessary before calling any of the below!
  // Also we may need to clamp for Poisson families
  std::vector<arma::vec> nllvec=this->negLogLikesVecd; //this stores weighted negative log likelihoods; unscaled by f1!
  this->calcNegLogLL(nllvec,true);
  double obj = f1*this->ULL + f2*n*getPenalty(); //objective
  double obj_old = obj;

  //debugging
  debugIteration = 0;
  recordDebugTrace(0, -1, f2);

  // std::vector<arma::vec> aphi_old = aphi;
  // arma::mat hmat_old = hmat;
  // std::vector<arma::mat> vmat_old = vmat;


  // std::cout << "Initial stats: " << std::endl;
  // std::cout << " + NegLogLike: " + std::to_string(this->ULL) << std::endl;
  // std::cout << " + Objective: " + std::to_string(obj) << std::endl;

  // for( int d=0; d<D; d++ ){
  //   if( (vmat[d].has_inf()) | (vmat[d].has_nan()) | (vmat[d].has_nonfinite()) ){
  //     throw std::invalid_argument("Bad initials ... nllvec has non finite");
  //   }
  //   if( (Zeta[d].has_inf()) | (Zeta[d].has_nan()) | (Zeta[d].has_nonfinite()) ){
  //     throw std::invalid_argument("Bad initials ... nllvec has non finite");
  //   }
  //   if( (aphi[d].has_inf()) | (aphi[d].has_nan()) | (aphi[d].has_nonfinite()) ){
  //     throw std::invalid_argument("Bad initials ... nllvec has non finite");
  //   }
  //   if( (umat.has_inf()) | (umat.has_nan()) | (umat.has_nonfinite()) ){
  //     throw std::invalid_argument("Bad initials ... nllvec has non finite");
  //   }
  //
  //   Rcpp::Rcout << vmat[d] << std::endl;
  //   Rcpp::Rcout << Zeta[d] << std::endl;
  //   Rcpp::Rcout << aphi[d] << std::endl;
  //   Rcpp::Rcout << nllvec[d] << std::endl;
  //
  //   if( (nllvec[d].has_inf()) | (nllvec[d].has_nan()) | (nllvec[d].has_nonfinite()) ){
  //     throw std::invalid_argument("Bad initials ... nllvec has non finite");
  //   }
  // }


  //run until convergence
  int iter=0;
  bool conv=false;
  while( (!conv) & (iter < maxIter)  ){
    iter = iter+1;
    debugIteration = iter;
    //make sure ULL and loglikes_forEach_i are updated as subsequent function is dpendent

    // std::cout << "gFitterOnColumns : Iteration: " + std::to_string(iter) << std::endl;
    // std::cout << "f2: " + std::to_string(f2) << std::endl;
    // std::cout << "iter: " + std::to_string(iter) << std::endl;
    // std::cout << "but allowed maxiter: " + std::to_string(maxIter) << std::endl;

    // std::cout << " + + Updating aphi ...";
    //calculate aphi
    update_aphi(); //aphi independent of f1, f2

    // std::cout << " done. " << std::endl;

    //move Zeta_i for each i
    // std::cout << " + + Updating Zeta matrix ...";
    moveZeta(f1, f2);
    recordDebugTrace(1, -1, f2);
    // std::cout << " done. " << std::endl;

    //make sure ULL and loglikes_forEach_i are updated as subsequent function is dpendent

    //move factors
    // std::cout << " + + Moving U and V ... Moving U/H is over i, and therefore does not preserve the negLogLikesVec structure /";
    moveVU(f1, f2);
    // std::cout << " done. Calculating netglogll" << std::endl;

    // Calculate the current objective directly from ULL. Accepted h/v moves
    // maintain the scalar weighted ULL, so a full per-dataset/per-feature
    // likelihood-cache refresh is unnecessary for outer convergence.
    //
    // Individual entries of negLogLikesVecd may be stale after an h update,
    // because h changes fitted values across features/datasets while the h
    // optimizer updates the scalar total ULL rather than every cache entry.
    // moveZeta() recomputes the feature-specific likelihood values it needs on
    // the next pass, so the stale cache does not affect outer convergence.
    //
    // Uncomment the refresh below if future code requires negLogLikesVecd to be
    // synchronized immediately after moveVU(), or if accepted h/v moves no
    // longer maintain ULL.
    // this->calcNegLogLL(nllvec, true);
    obj = f1*this->ULL + f2*n*getPenalty();
    
    recordDebugTrace(4, -1, f2);

    // std::cout << "Objective at iteration: " +std::to_string(iter) << std::endl;
    // std::cout << " + NegLogLike: " + std::to_string(this->ULL) << std::endl;
    // std::cout << " + Objective: " + std::to_string(obj) << std::endl;


    //update convgence metric : the objective based convergence yielded much better results
    // Generalized fitters solve an objective equal to n times the canonical
    // objective. In contrast, non-family fitters solve an objective scaled by n/f1.
    // Generalized fitters solve n * objective_canonical, unlike non-family fitters
    // whose solved objective is (n/f1) * objective_canonical.
    const double solved_eps = n * eps;
    conv = std::abs(obj - obj_old) < solved_eps;
    //conv = (mse_VectorOfVecs(aphi, aphi_old)+mse_mats(hmat, hmat_old)+mse_VectorOfMats(vmat, vmat_old))/3;

    obj_old = obj;
    // aphi_old = aphi;
    // hmat_old=hmat;
    // vmat_old=vmat;

  }

  //udpate aphi, loglike and objective to reflect new pars
  update_aphi();
  this->calcNegLogLL(nllvec, true);
  obj=f1*this->ULL + f2*n*getPenalty();
  last_f1_estimatedWith = f1;
  last_f2_estimatedWith = f2;
  
  recordDebugTrace(5, -1, f2);

  // std::cout << "Final stats: " << std::endl;
  // std::cout << " + NegLogLike: " + std::to_string(this->ULL) << std::endl;
  // std::cout << " + Objective: " + std::to_string(obj) << std::endl;

}



//////////////////////////////////////////////
/*
 *
 * DEVIANCE AND MSES
 *
 * ONLY aggregators across dataests are weighted by wd. 
 * 
 *
 * MSEs are calculated as mean of y-g(eta) where g is the
 * needed transformation to the raw scale.
 *
 */
//////////////////////////////////////////////

void igFitterOnColumns::deviance_d(const int& d, std::vector<arma::vec>& nlength_vectord_tmp){

  const arma::mat& Zetad = Zeta[d];
  const arma::mat& Zd = Z[d];
  const arma::mat& vmatd = vmat[d];
  const arma::vec& rowOffsetsd = rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd = colOffsets_for_columns[d];
  const arma::vec& aphid = aphi[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1(j);
                         arma::vec eta_j = (Zetad.t() * Zd.row(j).t()) + (vmatd*(umat.row(j).t())) + (rowOffsetsd + colOffsets_for_columnsd(j) ) ;
                         arma::vec y_j;
                         if( d < dDense ){
                           y_j = denseMats[d].col(colj);
                           nlength_vectord_tmp[d].at(j)=this->denseRowFams[d]->deviance(y_j,eta_j,aphid);
                         } else {
                           y_j = sparseMats[d-dDense].col(colj);
                           nlength_vectord_tmp[d].at(j)=this->sparseRowFams[d-dDense]->deviance(y_j,eta_j,aphid);
                         }
                       }

                     });

}

double igFitterOnColumns::deviance(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         deviance_d( d, nlength_vectord );
                       }
                     }
  );

  double dev=0;
  for( int d=0; d < D; d++){
    dev +=wd[d]*arma::mean(nlength_vectord[d]); //weighted!
  }
  return dev;
}


void igFitterOnColumns::mse_d(const int& d, std::vector<arma::vec>& nlength_vectord_tmp){

  const arma::mat& Zetad = Zeta[d];
  const arma::mat& Zd = Z[d];
  const arma::mat& vmatd = vmat[d];
  const arma::vec& rowOffsetsd = rowOffsets[d];
  const arma::vec& colOffsets_for_columnsd = colOffsets_for_columns[d];
  const arma::vec& aphid = aphi[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,n),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1(j);
                         arma::vec eta_j = (Zetad.t() * Zd.row(j).t()) + (vmatd*(umat.row(j).t())) + (rowOffsetsd + colOffsets_for_columnsd(j) ) ;
                         arma::vec y_j;
                         if( d < dDense ){
                           y_j = denseMats[d].col(colj);
                           nlength_vectord_tmp[d].at(j)=this->denseRowFams[d]->mse(y_j,eta_j,aphid);
                         } else {
                           y_j = sparseMats[d-dDense].col(colj);
                           nlength_vectord_tmp[d].at(j)=this->sparseRowFams[d-dDense]->mse(y_j,eta_j,aphid);
                         }
                       }

                     });

}

double igFitterOnColumns::mse(){
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         mse_d( d, nlength_vectord );
                       }
                     }
  );

  double mse=0;
  for( int d=0; d < D; d++){
    mse +=wd[d]*arma::mean(nlength_vectord[d]); //weighted!
  }
  return mse;
}


double igFitterOnColumns::devianceForTest_d(
    const int& d,
    const arma::uvec& columns1Test,
    const arma::mat& ZTestd,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTestd,
    const arma::vec& colOffsetsTestd
){
  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec mse_swiseTest( ntest, arma::fill::zeros );
  arma::mat& Zetad = Zeta[d];
  arma::mat& vmatd = vmat[d];
  arma::mat& aphid = aphi[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1Test(j);
                         arma::vec etaTest_j = (Zetad.t() * ZTestd.row(j).t()) + (vmatd*(umatTest.row(j).t())) + (rowOffsetsTestd + colOffsetsTestd(colj) ) ;

                         arma::vec yTest_j;
                         if( d < dDense ){
                           yTest_j = denseMats[d].col(colj);
                           mse_swiseTest(j)=this->denseRowFams[d]->deviance(yTest_j,etaTest_j,aphid);
                         } else {
                           yTest_j = sparseMats[d-dDense].col(colj);
                           mse_swiseTest(j)=this->sparseRowFams[d-dDense]->deviance(yTest_j,etaTest_j,aphid);
                         }
                       }
                     });
  return arma::mean(mse_swiseTest);
}

double igFitterOnColumns::devianceForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){
  arma::mat umatTest = MTest* this->hmat;
  arma::vec mse_swiseTest( D, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         const arma::mat& ZTest_d = ZTest[d];
                         const arma::vec& rowOffsetsTest_d = rowOffsetsTest[d];
                         const arma::vec& colOffsetsTest_d = colOffsetsTest[d];
                         mse_swiseTest(d)=devianceForTest_d( d, columns1Test, ZTest_d, MTest, rowOffsetsTest_d, colOffsetsTest_d );
                       }
                     }
  );
  return arma::sum( wd % mse_swiseTest);
}


double igFitterOnColumns::mseForTest_d(
    const int& d,
    const arma::uvec& columns1Test,
    const arma::mat& ZTestd,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTestd,
    const arma::vec& colOffsetsTestd
){
  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec mse_swiseTest( ntest, arma::fill::zeros );
  arma::mat& Zetad = Zeta[d];
  arma::mat& vmatd = vmat[d];
  arma::mat& aphid = aphi[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         arma::uword colj = columns1Test(j);
                         arma::vec etaTest_j = (Zetad.t() * ZTestd.row(j).t()) + (vmatd*(umatTest.row(j).t())) + (rowOffsetsTestd + colOffsetsTestd(colj) ) ;

                         arma::vec yTest_j;
                         if( d < dDense ){
                           yTest_j = denseMats[d].col(colj);
                           mse_swiseTest(j)=this->denseRowFams[d]->mse(yTest_j,etaTest_j,aphid);
                         } else {
                           yTest_j = sparseMats[d-dDense].col(colj);
                           mse_swiseTest(j)=this->sparseRowFams[d-dDense]->mse(yTest_j,etaTest_j,aphid);
                         }
                       }
                     });
  return arma::mean(mse_swiseTest);
}

double igFitterOnColumns::mseForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){
  arma::mat umatTest = MTest* this->hmat;
  arma::vec mse_swiseTest( D, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         const arma::mat& ZTest_d = ZTest[d];
                         const arma::vec& rowOffsetsTest_d = rowOffsetsTest[d];
                         const arma::vec& colOffsetsTest_d = colOffsetsTest[d];
                         mse_swiseTest(d)=mseForTest_d( d, columns1Test, ZTest_d, MTest, rowOffsetsTest_d, colOffsetsTest_d );
                       }
                     }
  );
  return arma::sum(wd % mse_swiseTest);
}


double igFitterOnColumns::objectiveForTest_d(
    const int& d,
    const arma::uvec& columns1Test,
    const arma::mat& ZTestd,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTestd,
    const arma::vec& colOffsetsTestd
){
  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec LL_swiseTest( ntest, arma::fill::zeros );

  arma::mat& Zetad = Zeta[d];
  arma::mat& vmatd = vmat[d];
  arma::mat& aphid = aphi[d];

  tbb::parallel_for( tbb::blocked_range<size_t>(0,ntest),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t j=ra.begin(); j!=ra.end(); ++j){
                         // arma::uword colj = columns1Test(j);
                         // arma::vec yTest_j; yTest_j = Y.col(colj);
                         // arma::vec etaTest_j = (Zeta.t() * ZTest.row(j).t()) + (vmat*(umatTest.row(j).t())) + (rowOffsetsTest + colOffsetsTest(j) ) ;
                         //
                         arma::uword colj = columns1Test(j);
                         arma::vec etaTest_j = (Zetad.t() * ZTestd.row(j).t()) + (vmatd*(umatTest.row(j).t())) + (rowOffsetsTestd + colOffsetsTestd(colj) ) ;
                         arma::vec yTest_j;
                         if( d < dDense ){
                           yTest_j = denseMats[d].col(colj);
                           LL_swiseTest(j)=this->denseRowFams[d]->calcNegLogLL_j(yTest_j,etaTest_j,aphid);
                         } else {
                           yTest_j = sparseMats[d-dDense].col(colj);
                           LL_swiseTest(j)=this->sparseRowFams[d-dDense]->calcNegLogLL_j(yTest_j,etaTest_j,aphid);
                         }
                       }
                     });

  return arma::mean(LL_swiseTest);
}

double igFitterOnColumns::objectiveForTest(
    const arma::uvec& columns1Test,
    const std::vector<arma::mat>& ZTest,
    const arma::mat& MTest,
    const std::vector<arma::vec>& rowOffsetsTest,
    const std::vector<arma::vec>& colOffsetsTest
){
  arma::mat umatTest = MTest* this->hmat;
  arma::vec LL_dwiseTest( D, arma::fill::zeros );

  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         const arma::mat& ZTest_d = ZTest[d];
                         const arma::vec& rowOffsetsTest_d = rowOffsetsTest[d];
                         const arma::vec& colOffsetsTest_d = colOffsetsTest[d];
                         LL_dwiseTest(d)=objectiveForTest_d( d, columns1Test, ZTest_d, MTest, rowOffsetsTest_d, colOffsetsTest_d );
                       }
                     }
  );

  double testULL = arma::sum(wd % LL_dwiseTest);
  double penalty = getPenalty();
  double obj =last_f1_estimatedWith* testULL + last_f2_estimatedWith*penalty ; //f1*NEGLL + sumovercompoents(f2*sq norm of projection of uk on H1)
  return obj;
}



