#ifndef PARALLELWORKERSTEMPLATES_H
#define PARALLELWORKERSTEMPLATES_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

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
 * npSMLFCRowWise does it for a given matrix Y, and returns rowwise MSEs
 * npSMLFCOnColumnsRowWise does for a given matrix Y and a given set of columns `columns`, and returns rowwise.
 *
 */
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> //sp_mat or mat
class npSMLFC : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)
  const T& Y;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umat;
  const arma::mat& vmat;
  arma::vec& objColwise; //nlength, we will store the objective elements here; stores MSE only
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
public:
  npSMLFC( const T& Y,
          const arma::mat& Z,
          const arma::mat& Zeta,
          const arma::mat& umat,
          const arma::mat& vmat,
          arma::vec& objs,
          const arma::vec& rowOffsets,
          const arma::vec& colOffsets
          ): Y(Y), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objColwise(objs), rowOffsets(rowOffsets), colOffsets(colOffsets) {};
  void operator()( std::size_t begin, std::size_t end ) {
    for( std::size_t j=begin; j < end; j++   ){
      arma::vec yj; yj=Y.col(j); //two step splitting, in case we have an sp_mat as Y
      yj = yj - rowOffsets - colOffsets(j); //accounting for offsets
      objColwise(j) =  arma::mean( arma::pow( yj - Zeta.t() * Z.row(j).t() -  vmat * umat.row(j).t() , 2 ) ); //we have overloaded col() for dsList object; obj0_swise of length n initialized in the constructors
    }
  }
};
template class npSMLFC<arma::mat>;
template class npSMLFC<arma::sp_mat>;

///////////////////////////////////////////////////////////////////////
/*
 * Same as above but only on columns
 * the "objs" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
template<typename T> //sp_mat or mat
class npSMLFCOnColumns : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)
  const T& Y;
  const arma::uvec& columns;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umat;
  const arma::mat& vmat;
  arma::vec& objColwise; //nlength, we will store the objective elements here; stores MSE only.
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
public:
  npSMLFCOnColumns( const T& Y,
           const arma::uvec& columns,
           const arma::mat& Z,
           const arma::mat& Zeta,
           const arma::mat& umat,
           const arma::mat& vmat,
           arma::vec& objs,
           const arma::vec& rowOffsets,
           const arma::vec& colOffsets
           ): Y(Y), columns(columns), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objColwise(objs), rowOffsets(rowOffsets), colOffsets(colOffsets) {
    if( colOffsets.n_elem != Y.n_cols ){
      throw std::invalid_argument("Column offsets must equal the total number of columns Y" );
    }
    // if(objs.n_elem != columns.n_elem){
    //   throw std::invalid_argument( "length of objColWise should equal the length of columns variable given" );
    // }
  };
  void operator()( std::size_t begin, std::size_t end ) {
    for( std::size_t j=begin; j < end; j++   ){
      arma::vec yj; yj=Y.col( columns(j) ); //two step splitting, in case we have an sp_mat as Y
      yj = yj - rowOffsets - colOffsets(columns(j)); //accounting for offsets
      objColwise(j) =  arma::mean( arma::pow( yj - Zeta.t() * Z.row(j).t() -  vmat * umat.row(j).t() , 2 ) ); //we have overloaded col() for dsList object; obj0_swise of length n initialized in the constructors
    }
  }
};
template class npSMLFCOnColumns<arma::mat>;
template class npSMLFCOnColumns<arma::sp_mat>;

///////////////////////////////////////////////////////////////////////
/*
 * This is for calculating MSE based on the above worker objects.
 * One for all matrices, one for a subset of columns in matrices.
 */
///////////////////////////////////////////////////////////////////////
template<typename T>
double mse(    const T& Y,
               const arma::mat& Z,
               const arma::mat& Zeta,
               const arma::mat& umat,
               const arma::mat& vmat,
               const arma::vec& rowOffsets,
               const arma::vec& colOffsets
){
  const int n=Y.n_cols;
  arma::vec lf_swise(n, arma::fill::zeros);
  npSMLFC<T> LFC( Y, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  return arma::mean( lf_swise );
}

//Now over a subset of columns
template<typename T>
double mseOnColumns(    const T& Y,
                        const arma::uvec& columns,
                        const arma::mat& Z,
                        const arma::mat& Zeta,
                        const arma::mat& umat,
                        const arma::mat& vmat,
                        const arma::vec& rowOffsets,
                        const arma::vec& colOffsets
){
  //const int n=Y.n_cols;
  const int n=columns.n_elem;
  arma::vec lf_swise(n, arma::fill::zeros);
  npSMLFCOnColumns<T> LFC( Y, columns, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  return arma::mean( lf_swise );
}


///////////////////////////////////////////////////////////////////////
/*
 * Rowwise MSEs.
 *
 * First two are worker objects, followed by appllication.
 */
///////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////
/*
 * Same as the previous section but for each row
 * the "objs" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
template<typename T> //sp_mat or mat
class npSMLFCRowWise : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)
  const T& Y;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umat;
  const arma::mat& vmat;
  arma::vec& objRowWise; //nlength, we will store the objective elements here; stores MSE only
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
public:
  npSMLFCRowWise( const T& Y,
                  const arma::mat& Z,
                  const arma::mat& Zeta,
                  const arma::mat& umat,
                  const arma::mat& vmat,
                  arma::vec& objs,
                  const arma::vec& rowOffsets,
                  const arma::vec& colOffsets
                  ): Y(Y), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objRowWise(objs), rowOffsets(rowOffsets), colOffsets(colOffsets) {
    // if( objRowWise.n_elem != Y.n_rows ){
    //   throw std::invalid_argument( "objRowWise variable must have the same length as number of rows in Y " )
    // }
  };
  void operator()( std::size_t begin, std::size_t end ) {
    for( std::size_t i=begin; i < end; i++   ){
      arma::vec yi; yi=(Y.row(i)).t(); //two step splitting, in case we have an sp_mat as Y
      yi = yi - colOffsets - rowOffsets(i); //accounting for offsets
      objRowWise(i) =  arma::mean( arma::pow( yi - Z * Zeta.col(i) -  umat * (vmat.row(i)).t() , 2 ) ); //we have overloaded col() for dsList object; obj0_swise of length n initialized in the constructors
    }
  }
};
template class npSMLFCRowWise<arma::mat>;
template class npSMLFCRowWise<arma::sp_mat>;


///////////////////////////////////////////////////////////////////////
/*
 * Rowwise MSEs, buto only on columns
 *
 */
///////////////////////////////////////////////////////////////////////
template<typename T> //sp_mat or mat
class npSMLFCOnColumnsRowWise : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)
  const T& Y;
  const arma::uvec& columns;
  const arma::mat& Z;
  const arma::mat& Zeta;
  const arma::mat& umat;
  const arma::mat& vmat;
  arma::vec& objRowWise; //nlength, we will store the objective elements here; stores MSE only.
  const arma::vec& rowOffsets;
  const arma::vec& colOffsets;
  arma::vec colOffsets_at_columns; //stores the column offsets just at the desired columns
public:
  npSMLFCOnColumnsRowWise( const T& Y,
                           const arma::uvec& columns,
                           const arma::mat& Z,
                           const arma::mat& Zeta,
                           const arma::mat& umat,
                           const arma::mat& vmat,
                           arma::vec& objs,
                           const arma::vec& rowOffsets,
                           const arma::vec& colOffsets
                           ): Y(Y), columns(columns), Z(Z), Zeta(Zeta), umat(umat), vmat(vmat), objRowWise(objs),
                           rowOffsets(rowOffsets), colOffsets(colOffsets) {
    colOffsets_at_columns=colOffsets.elem( columns );
  };
  void operator()( std::size_t begin, std::size_t end ) {
    for( std::size_t i=begin; i < end; i++   ){
      arma::vec yi; yi=(Y.row(i)).t(); yi=yi(columns); //two step splitting, in case we have an sp_mat as Y
      yi = yi - colOffsets_at_columns - rowOffsets(i); //accounting for offsets
      objRowWise(i) =  arma::mean( arma::pow( yi - Z * Zeta.col(i) -  umat*(vmat.row(i)).t() , 2 ) ); //we have overloaded col() for dsList object; obj0_swise of length n initialized in the constructors
    }
  }
};
template class npSMLFCOnColumnsRowWise<arma::mat>;
template class npSMLFCOnColumnsRowWise<arma::sp_mat>;


/////////////////////////////////////////////////////////////////
/*
 This is for calculating MSE based on the above worker objects, .
 * One for all matrices, one for a subset of columns in matrices.
 */
/////////////////////////////////////////////////////////////////
template<typename T>
arma::vec mseRowWise( const T& Y,
               const arma::mat& Z,
               const arma::mat& Zeta,
               const arma::mat& umat,
               const arma::mat& vmat,
               const arma::vec& rowOffsets,
               const arma::vec& colOffsets
){
  const int p=Y.n_rows;
  arma::vec lf_rowwise(p, arma::fill::zeros);
  npSMLFCRowWise<T> LFC( Y, Z, Zeta, umat, vmat, lf_rowwise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, p, LFC );
  return lf_rowwise;
}



//Now over a subset of columns
template<typename T>
arma::vec mseOnColumnsRowWise(    const T& Y,
                        const arma::uvec& columns,
                        const arma::mat& Z,
                        const arma::mat& Zeta,
                        const arma::mat& umat,
                        const arma::mat& vmat,
                        const arma::vec& rowOffsets,
                        const arma::vec& colOffsets
){
  const int p=Y.n_rows;
  arma::vec lf_rowWise(p, arma::fill::zeros);
  npSMLFCOnColumnsRowWise<T> LFC( Y, columns, Z, Zeta, umat, vmat, lf_rowWise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, p, LFC );
  return lf_rowWise;
}


//Now, mses col wise
template<typename T>
arma::vec mseColWise( const T& Y,
                      const arma::mat& Z,
                      const arma::mat& Zeta,
                      const arma::mat& umat,
                      const arma::mat& vmat,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets
){
  const int n=Y.n_cols;
  arma::vec lf_colwise(n, arma::fill::zeros);
  npSMLFC<T> LFC( Y, Z, Zeta, umat, vmat, lf_colwise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  return lf_colwise;
}

//Now over a subset of columns, and colwise
template<typename T>
arma::vec mseOnColumnsColWise(    const T& Y,
                                  const arma::uvec& columns,
                                  const arma::mat& Z,
                                  const arma::mat& Zeta,
                                  const arma::mat& umat,
                                  const arma::mat& vmat,
                                  const arma::vec& rowOffsets,
                                  const arma::vec& colOffsets
){
  const int n=columns.size();
  arma::vec lf_colWise(n, arma::fill::zeros);
  npSMLFCOnColumns<T> LFC( Y, columns, Z, Zeta, umat, vmat, lf_colWise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  return lf_colWise;
}

#endif
