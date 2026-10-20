// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

#include "parallelWokersTemplates.h"

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;



// [[Rcpp::export]]
double mseC( SEXP& Y,
             arma::mat& Z,
             arma::mat& Zeta,
             arma::mat& umat,
             arma::mat& vmat,
             arma::vec& rowOffsets,
             arma::vec& colOffsets
             ){
  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mse<arma::sp_mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat=Rcpp::as<arma::mat>(Y);
    return mse<arma::mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}

// [[Rcpp::export]]
double mseOnColumnsC( SEXP& Y,
                      arma::uvec& columnsR,
             arma::mat& Z,
             arma::mat& Zeta,
             arma::mat& umat,
             arma::mat& vmat,
             arma::vec& rowOffsets,
             arma::vec& colOffsets
){

  arma::uvec columnsC = columnsR -1; //R to C indexing

  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mseOnColumns<arma::sp_mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    return mseOnColumns<arma::mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}


// [[Rcpp::export]]
arma::vec mseRowWiseC( SEXP& Y,
                       arma::mat& Z,
                       arma::mat& Zeta,
                       arma::mat& umat,
                       arma::mat& vmat,
                       arma::vec& rowOffsets,
                       arma::vec& colOffsets
){
  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mseRowWise<arma::sp_mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    return mseRowWise<arma::mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}

// [[Rcpp::export]]
arma::vec mseRowWiseOnColumnsC( SEXP& Y,
                      arma::uvec& columnsR,
                      arma::mat& Z,
                      arma::mat& Zeta,
                      arma::mat& umat,
                      arma::mat& vmat,
                      arma::vec& rowOffsets,
                      arma::vec& colOffsets
){

  arma::uvec columnsC = columnsR-1;

  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mseOnColumnsRowWise<arma::sp_mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    return mseOnColumnsRowWise<arma::mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}



// [[Rcpp::export]]
arma::vec mseColWiseC( SEXP& Y,
                       arma::mat& Z,
                       arma::mat& Zeta,
                       arma::mat& umat,
                       arma::mat& vmat,
                       arma::vec& rowOffsets,
                       arma::vec& colOffsets
){
  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mseColWise<arma::sp_mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    return mseColWise<arma::mat>( Ymat, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}

// [[Rcpp::export]]
arma::vec mseColWiseOnColumnsC( SEXP& Y,
                                arma::uvec& columnsR,
                                arma::mat& Z,
                                arma::mat& Zeta,
                                arma::mat& umat,
                                arma::mat& vmat,
                                arma::vec& rowOffsets,
                                arma::vec& colOffsets
){

  arma::uvec columnsC = columnsR-1;

  if( Rf_isS4(Y) ){
    if(Rf_inherits(Y , "dgCMatrix")) {
      arma::sp_mat Ymat=Rcpp::as<arma::sp_mat>(Y);
      return mseOnColumnsColWise<arma::sp_mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
    }
    else {
      throw std::invalid_argument("Unknown matrix type for data");
    }
  } else {
    arma::mat Ymat = Rcpp::as<arma::mat>(Y);
    return mseOnColumnsColWise<arma::mat>( Ymat, columnsC, Z, Zeta, umat, vmat, rowOffsets, colOffsets );
  }
}
