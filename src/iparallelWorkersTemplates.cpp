#include "iparallelWorkersTemplates.h"

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;



//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FOR objective calculations in a single dataset
 *
 * npSMLFC does it for a given matrix Y
 * npSMLFCOnColumns does for a given matrix Y and a given set of columns `columns`
 *
 */
//////////////////////////////////////////////////////////////////////////////////////////////////////////
inpSMLFC::inpSMLFC( const dsListC& Y,
                    const arma::vec& wd, 
                    const std::vector<arma::mat>& Z,
                    const std::vector<arma::mat>& Zeta,
                    const arma::mat& umat,
                    const std::vector<arma::mat>& vmat,
                    arma::mat& objs,
                    const std::vector<arma::vec>& rowOffsets,
                    const std::vector<arma::vec>& colOffsets
): Y(Y), wd(wd), n(Y.ncol(0)), D(Y.size()), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objColwise(objs), rowOffsets(rowOffsets),colOffsets(colOffsets){

  if( objColwise.n_rows != D ){
    throw std::invalid_argument( "objs variable must have the same number of columns as number of datasets" );
  }
  if( objColwise.n_cols != n ){
    throw std::invalid_argument( "Number of columns in Y must equal the number of colums in objs variable" );
  }
}

void inpSMLFC::operator()( std::size_t begin, std::size_t end ) {
  for( std::size_t d=begin; d < end; d++   ){
    arma::mat Zd=Z[d];
    arma::mat Zetad=Zeta[d];
    arma::mat vmatd=vmat[d];
    arma::vec lf_swise(n, arma::fill::zeros);
    Y.dispatchWorkersOfType_npSMLFC_OnColsOf( d, Zd, Zetad, umat, vmatd, lf_swise, rowOffsets, colOffsets );
    objColwise.row(d)=wd[d]* (lf_swise.t());
  }
}


///////////////////////////////////////////////////////////////////////
/*
 * Same as above but only on columns
 * the "objs" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
inpSMLFCOnColumns::inpSMLFCOnColumns( const dsListC& Y,
                                      const arma::vec& wd,
                                      const arma::uvec& columns,
                                      const std::vector<arma::mat>& Z,
                                      const std::vector<arma::mat>& Zeta,
                                      const arma::mat& umat,
                                      const std::vector<arma::mat>& vmat,
                                      arma::mat& objs,
                                      const std::vector<arma::vec>& rowOffsets,
                                      const std::vector<arma::vec>& colOffsets
): Y(Y), wd(wd), n(Y.ncol(0)), D(Y.size()), columns(columns), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objColwise(objs),
rowOffsets(rowOffsets), colOffsets(colOffsets) {
  if( objColwise.n_rows != D ){
    throw std::invalid_argument( "objs variable must have the same number of columns as number of datasets" );
  }
  if( objColwise.n_cols != n ){
    throw std::invalid_argument( "Length of columns variable  must equal the number of columns in objs variable" );
  }
}

void inpSMLFCOnColumns::operator()( std::size_t begin, std::size_t end ) {
  for( std::size_t d=begin; d < end; d++   ){
    arma::mat Zd=Z[d];
    arma::mat Zetad=Zeta[d];
    arma::mat vmatd=vmat[d];
    arma::vec lf_swise(n, arma::fill::zeros);
    Y.dispatchWorkersOfType_npSMLFC_OnColsOf( d, Zd, Zetad, umat, vmatd, lf_swise, columns, rowOffsets, colOffsets );
    objColwise.row(d)=wd[d]*(lf_swise.t());
  }
}


///////////////////////////////////////////////////////////////////////
/*
 * This is for calculating MSE based on the above worker objects.
 * One for all matrices, one for a subset of columns in matrices.
 */
///////////////////////////////////////////////////////////////////////
double imse(   const dsListC& Y,
               const arma::vec& wd, 
               const std::vector<arma::mat>& Z,
               const std::vector<arma::mat>& Zeta,
               const arma::mat& umat,
               const std::vector<arma::mat>& vmat,
               const std::vector<arma::vec>& rowOffsets,
               const std::vector<arma::vec>& colOffsets
){
  const int D = Y.size();
  const int n=Y.ncol(0);
  arma::mat lf_swise( D, n, arma::fill::zeros);
  inpSMLFC LFC( Y, wd, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, D, LFC );
  //return arma::sum( arma::sum(lf_swise) )/(D*n);
  return arma::sum( arma::sum(lf_swise) )/n;
}

//Now over a subset of columns
double imse(    const dsListC& Y,
                const arma::vec& wd, 
               const std::vector<arma::mat>& Z,
               const std::vector<arma::mat>& Zeta,
               const arma::mat& umat,
               const std::vector<arma::mat>& vmat,
               const arma::uvec& columns,
               const std::vector<arma::vec>& rowOffsets,
               const std::vector<arma::vec>& colOffsets
){
  const int D = Y.size();
  const int n=Y.ncol(0);
  arma::mat lf_swise( D, n, arma::fill::zeros);
  inpSMLFCOnColumns LFC( Y, wd, columns, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, D, LFC );
  //return arma::sum( arma::sum(lf_swise) )/(D*columns.n_elem); // as we only evaluate for columns. 
  return arma::sum( arma::sum(lf_swise) )/(columns.n_elem); // as we only evaluate for columns. 
}

