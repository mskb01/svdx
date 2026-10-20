#ifndef DSLISTC_H
#define DSLISTC_H

//The end C is for copy!

/*
 * Models an R list with dense or sparse (currently only dgCMatrix) matrices.
 *
 *
 */


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <Rinternals.h>
#include <tbb/tbb.h>

#include "parallelWokersTemplates.h"
//#include "iparallelWokersTemplates.h"

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

class dsListC {  // same as dsList but copies if given a RList / reads in if file names given.
                 //dense and/or sparse matrix list;

// private:

  // std::vector<arma::mat> denseY;
  // std::vector<arma::sp_mat> sparseY;
  // int dDense;
  // int dSparse;
  // int D; //D=dDense + dSparse

  //const Rcpp::List& Ys; // an R list of length D containing various matrices
  //const std::vector<std::string> denseFiles;
  //const std::vector<std::string> sparseFiles;

public:

  std::vector<arma::mat> denseY;
  std::vector<arma::sp_mat> sparseY;
  int dDense;
  int dSparse;
  int D; //D=dDense + dSparse


  //MUST HAVE : all dense matrices first, all sparse matrices next. !
  explicit dsListC( const Rcpp::List& denseThenSparseYsList //can have dense or sparse or just one type!
  ): D( denseThenSparseYsList.size() )
  {

    //index 0 ... dDense-1 will be considered dense matrices
    // index dDense ... D-1 will be considered sparse matrirces
    // so some rules :
    //  : d^th dataset is obtained as follows:
    //     - if d >= dDense, we extract the d-dDense th sparse matrix.
    //     - if d < dDense, we exrtract the d^th Dense matrix.

    //sanity check on the list: checks if dense matrices if present appear before sparse matrices.
    bool sparseDetected=false, denseDetected=false;
    dDense=0; dSparse=0;
    int dtil=0;
    while( dtil<D){
      SEXP md=denseThenSparseYsList[dtil];
      if( Rf_isS4( md  )  ){ //sparse?
        if(Rf_inherits(md, "dgCMatrix")) {
          if( !sparseDetected ){
            sparseDetected = true;
          }
          dSparse++;
        } else {
          std::cout << "Unknown S4 matrix type" << std::endl;
          throw std::invalid_argument( "Unknown S4 matrix type" );
        }
      } else {             //else dense. assumption.
        if( sparseDetected ){ //we have detected a sparse before! We shouldn't be detectdd dense s anymore!
          std::cout << "For the current implementation, all dense matrices must be placed before sparse matrices in the input list. " << std::endl;
          throw std::invalid_argument( "For the current implementation, all dense matrices must be placed before sparse matrices in the input list. " );
        }
        denseDetected=true;
        dDense++;
      }
      dtil++;
    }
    if( dDense > 0 ){
      denseY=std::vector<arma::mat>( dDense );
      for( int d=0; d<dDense; d++  ){ //we copy!!! All so that we don't have those stack imbalance from R
        denseY[d]=Rcpp::as<arma::mat>( denseThenSparseYsList[d] );
      }
    }
    if( dSparse > 0 ){
      sparseY=std::vector<arma::sp_mat>( dSparse );
      for( int d=dDense; d<D; d++  ){ //we copy!!!
        sparseY[d-dDense] = Rcpp::as<arma::sp_mat>( denseThenSparseYsList[d] )  ;
      }
    }
  };


  explicit dsListC( const Rcpp::List& denseYs,
                   const Rcpp::List& sparseYs
                     ): dDense(denseYs.size()), dSparse(sparseYs.size()),
                     D( denseYs.size() + sparseYs.size() )
    {

    //index 0 ... dDense-1 will be considered dense matrices
    // index dDense ... D-1 will be considered sparse matrirces
    // so some rules :
    //  : d^th dataset is obtained as follows:
    //     - if d >= dDense, we extract the d-dDense th sparse matrix.
    //     - if d < dDense, we exrtract the d^th Dense matrix.


    denseY=std::vector<arma::mat>( dDense );
    sparseY=std::vector<arma::sp_mat>( dSparse );
    for( int d=0; d<dDense; d++  ){ //we copy!!! All so that we don't have those stack imbalance from R
      denseY[d] = Rcpp::as<arma::mat>( denseYs[d] )  ;
    }
    for( int d=0; d<dSparse; d++  ){ //we copy!!!
      sparseY[d] = Rcpp::as<arma::sp_mat>( sparseYs[d] )  ;
    }
  };



  //TODO

  // //dense matrices must be in a format detected by armadillo; headers
  // //sparse matrices must be in a 3 column format with a space: row column value as
  // explicit dsListC( const Rcpp::ListOf<Rcpp::String>& denseYFiles,
  //                  const Rcpp::ListOf<Rcpp::String>& sparseYFiles,
  //                  const bool headerPresent
  // ): dDense(denseYFiles.size()), dSparse(sparseYFiles.size()),
  // D( denseYFiles.size() + sparseYFiles.size() )
  // {
  //
  //   //index 0 ... dDense-1 will be considered dense matrices
  //   // index dDense ... D-1 will be considered sparse matrirces
  //   // so some rules :
  //   //  : d^th dataset is obtained as follows:
  //   //     - if d >= dDense, we extract the d-dDense th sparse matrix.
  //   //     - if d < dDense, we exrtract the d^th Dense matrix.
  //
  //
  //   denseY=std::vector<arma::mat>( dDense );
  //   sparseY=std::vector<arma::sp_mat>( dSparse );
  //   for( int d=0; d<dDense; d++  ){ //we copy!!! All so that we don't have those stack imbalance from R
  //     std::string infile_d=denseYFiles[d];
  //     arma::csv_name( denseY[d], infile_d, headerPresent  );
  //   }
  //   for( int d=0; d<dSparse; d++  ){ //we copy!!!
  //     //direct loading of sparse seems to be slow!
  //     // we follow:
  //
  //     sparseY[d] = Rcpp::as<arma::sp_mat>( sparseYs[d] )  ;
  //   }
  // };

  //Get the size of the list
  int size() const{
    return( D );
  }

  //Get the number of columns of a matrix in list index
  int ncol(int d) const {

    // if( d>=D ){
    //   throw std::invalid_argument( "index exceeds!" )
    // }

    if( d >= dDense ){ //sparse
      return( sparseY[d-dDense].n_cols );
    } else { //dense
      return( denseY[d].n_cols );
    }
  }

  //Get column j in dataset d d.
  arma::vec col(int d, int j) const {
    if( d >= dDense ){ // sparse
      arma::vec yj;
      yj =sparseY[d-dDense].col(j); //splitting declaration and storage for sparse, as noted above in things to remember.
      return( yj  );
    } else {
      return( denseY[d].col(j) );
    }
  }

  //Get row i in dataset d d. KEY! Returns a rowvec!!
  arma::rowvec row(int d, int i) const {
    if( d >= dDense ){
      arma::rowvec yi;//splitting declaration and storage for sparse, as noted above in things to remember.
      yi=sparseY[d-dDense].row(i);
      return( yi  );
    } else {
      return( denseY[d].row(i) );
    }
  }


  //Get the number of rows of a matrix in list index
  int nrow(int d) const {
    if( d >= dDense ){
      return( sparseY[d-dDense].n_rows );
    } else {
      return( denseY[d].n_rows );
    }
  }

  bool isSparseMat(int& d) const{
    return( d >= dDense );
  }

  const arma::mat& get_dth_dataset_knownToBeDense( const int& d ){
    return denseY[d];
  }

  const arma::sp_mat& get_dth_dataset_knownToBeSparse( const int& d ){
    return sparseY[d-dDense];
  }


  //Performs Ys[d]*v, where v is a vector
  //Haven't yet implemented any check for whether the result is a proper arma::vec for return.
  // So mindful.
  //Following function used in moving vk
  arma::vec postMultiplyWithVecReturnVec(int& d, arma::vec& v ) const{
    if( d >= dDense ){
      return( sparseY[d-dDense] * v );
    } else {
      return( denseY[d] * v );
    }
  }


  //Performs Ys[d].cols(columns)*v, where v is a vector
  //Haven't yet implemented any check for whether the result is a proper arma::vec for return.
  // So mindful.
  //Following function used in moving vk in inpfitterOnColumns
  arma::vec postMultiplyWithVecReturnVec(int& d, arma::vec& v, const arma::uvec& columns ) const{
    if( d >= dDense ){
      return( sparseY[d-dDense].cols(columns) * v );
    } else {
      return( denseY[d].cols(columns) * v );
    }
  }

  //Performs Ys[d]*v, where v is a vector
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So mindful.
  arma::mat postMultiplyWithVecReturnMat(int& d, arma::vec& v )const {
    if( d >= dDense ){
      return( sparseY[d-dDense] * v );
    } else {
      return( denseY[d] * v );
    }
  }

  //Performs m*Ys[d], where m is a matrix
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So mindful.
  arma::mat tcrossprodMatReturnMat(int& d, arma::mat& m ) const {
    if( d >= dDense ){
      return( sparseY[d-dDense] * m.t() );
    } else {
      return( denseY[d] * m.t() );
    }
  }

  //Performs m*Ys[d], where m is a matrix
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So mindful.
  arma::mat tcrossprodMatReturnMatForDataCols(int& d, arma::mat& m, const arma::uvec& columns ) const {
    if( d >= dDense ){
      return( ((sparseY[d-dDense]).cols(columns)) * m.t() );
    } else {
      return( (denseY[d].cols(columns)) * m.t() );
    }
  }


  //Performs Ys[d]*m, where m is a matrix
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So mindful.
  arma::mat postMultiplyWithMatReturnMat(int& d, arma::mat& m ) const {
    if( d >= dDense ){
      return( sparseY[d-dDense] * m );
    } else {
      return( denseY[d] * m );
    }
  }


  arma::mat postMultiplyWithMatReturnMat(int& d, arma::mat& m, const arma::uvec& columns ) const {
    if( d >= dDense ){
      return( sparseY[d-dDense].cols(columns) * m );
    } else {
      return( denseY[d].cols(columns) * m );
    }
  }



  //Performs m*Ys[d], where m is a matrix
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So careful when calling.
  //Following fun used when updating what in hk move.
  arma::rowvec preMultiplyWithVecTransposeReturnRowvec(int& d, arma::vec& v ) const{
    if( d >= dDense ){
      return( v.t() * sparseY[d-dDense] );
    } else {
      return( v.t() * denseY[d]  );
    }
  }


  //Performs m*Ys[d].cols(columns), where m is a matrix
  //Haven't yet implemented any check for whether the result is a proper arma::mat for return.
  // So careful when calling.
  //Following fun used when updating what in hk move.
  arma::rowvec preMultiplyWithVecTransposeReturnRowvec(int& d, arma::vec& v, const arma::uvec& columns ) const{
    if( d >= dDense ){
      return( v.t() * sparseY[d-dDense].cols(columns) );
    } else {
      return( v.t() * denseY[d].cols(columns)  );
    }
  }

  //Similarly we can extend for others.
  // We have only implemented for what we need.


  
  void dispatchWorkersOfType_npSMLFC_OnColsOf(
      const int& d,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umat,
      const arma::mat& vmat,
      arma::vec& colWiseStorer, //NOTE! Must be of length equal to ncol(Ys[d]). No checks are done for this in code!
      const std::vector<arma::vec>& rowOffsets,
      const std::vector<arma::vec>& colOffsets
  ) const {

    if( d >= dDense ){
      npSMLFC<arma::sp_mat> LFC_d( sparseY[d-dDense], Z, Zeta, umat, vmat, colWiseStorer, rowOffsets[d-dDense], colOffsets[d-dDense] ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
      RcppParallel::parallelFor( 0, sparseY[d-dDense].n_cols, LFC_d );

    } else {
      npSMLFC<arma::mat> LFC_d( denseY[d], Z, Zeta, umat, vmat, colWiseStorer, rowOffsets[d], colOffsets[d] ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
      RcppParallel::parallelFor( 0, denseY[d].n_cols, LFC_d );
    }
  }


  //calculates loss function only on columns 'columns' of the d^th dataset.
  void dispatchWorkersOfType_npSMLFC_OnColsOf(
      const int& d,
      const arma::mat& Z,
      const arma::mat& Zeta,
      const arma::mat& umat,
      const arma::mat& vmat,
      arma::vec& lf_swise, //NOTE! Must be of length equal to columns for which loss function must be calculated. No checks are done for this in code!
      const arma::uvec& columns,
      const std::vector<arma::vec>& rowOffsets,
      const std::vector<arma::vec>& colOffsets
  ) const {

    const int nd=columns.size();
    if( d >= dDense ){
      npSMLFCOnColumns<arma::sp_mat> LFC( sparseY[d-dDense], columns, Z, Zeta, umat, vmat, lf_swise,
                                          rowOffsets[d-dDense], colOffsets[d-dDense]
                                          ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
      RcppParallel::parallelFor( 0, nd, LFC );
    } else {
      npSMLFCOnColumns<arma::mat> LFC( denseY[d], columns, Z, Zeta, umat, vmat, lf_swise,
                                       rowOffsets[d], colOffsets[d]
                                       ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
      RcppParallel::parallelFor( 0, nd, LFC );
    }
  }

};




#endif
