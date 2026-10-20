#ifndef IMR_H
#define IMR_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>
#include <memory>
#include "nlopt.hpp"
#include<vector>
#include<string>
#include<cmath>
#include<stdexcept>

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

///////////////////////////////////////////////////////////////////////
/*
 * READER for single data model from R side.
 *
 * Compared to single data analysis, each dataset in integrative
 * can have its own Z, Zeta, vmat, row and col offsets, aphi and family.
 *
 * Applies to both train and test
 *
 *
 */
///////////////////////////////////////////////////////////////////////
class imr {
public:

  //all are copies from R side
  arma::uvec columns1c;
  std::vector<arma::mat> Zc;       //Z may or may not be present
  std::vector<arma::mat> Zlsc;       //Z may or may not be present
  std::vector<arma::mat> Zetac;
  arma::mat hmatc; //initial conditions
  arma::mat umatc;//initial conditions
  std::vector<arma::mat> vmatc;//initial conditions
  arma::uvec qidsc;
  int qc;   // saves number of qids for each fold
  arma::mat Sc;
  arma::mat SStqc;
  arma::mat RsCboundc;
  std::vector<arma::vec> rowOffsetsc;
  std::vector<arma::vec> colOffsetsc;

  //graph specific additions : first two do not alter through the optimization
  arma::sp_mat* graphDistMatc = nullptr;
  arma::uvec* graphDistOnlyOnPointsc = nullptr;
  // -- stack vars to store references
  arma::sp_mat gdm;
  arma::uvec gdop;

  //family
  std::vector<std::string>* family=nullptr;
  std::vector<std::string> fam;

  std::vector<arma::vec>* aphi=nullptr;
  std::vector<arma::vec> ap;

  int r,q; //r =ncol(S), q = length(qids)
  unsigned int nTheta;
  arma::vec wd; 


  // After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
  imr( const Rcpp::List& trainList) {

    //check for input
    std::vector<std::string> names = {"columns",
                                      "Z", "Zls", "Zeta",
                                      "umat", "vmat", "hmat",
                                      "qids", "nr", "nq",
                                      "S", "SStq", "RsCbound",
                                      "rowOffsets", "colOffsets", "wd" };
    for( int i=0; i < names.size(); i++ ){
      if( !trainList.containsElementNamed( (names[i]).c_str() ) ){
        throw std::invalid_argument( "Not found in the input list: " + names[i] ) ;
      }
    }

    columns1c = Rcpp::as<arma::uvec>( trainList["columns"] )-1;
    Zc = Rcpp::as<std::vector<arma::mat>>( trainList["Z"] );
    Zlsc = Rcpp::as<std::vector<arma::mat>>( trainList["Zls"] );
    Zetac = Rcpp::as<std::vector<arma::mat>>( trainList["Zeta"] );
    umatc = Rcpp::as<arma::mat>( trainList["umat"] );
    vmatc = Rcpp::as<std::vector<arma::mat>>( trainList["vmat"] );
    hmatc = Rcpp::as<arma::mat>( trainList["hmat"] );
    qidsc= Rcpp::as<arma::uvec>( trainList["qids"] )-1; //initilize qids
    wd= Rcpp::as<arma::vec>( trainList["wd"] ); //dataset weights
    r = Rcpp::as<int>( trainList["nr"] );
    q=Rcpp::as<int>( trainList["nq"] );
    qc = qidsc.size();
    Sc = Rcpp::as<arma::mat>( trainList["S"] );
    SStqc = Rcpp::as<arma::mat>( trainList["SStq"] );
    RsCboundc=Rcpp::as<arma::mat>( trainList["RsCbound"] );
    rowOffsetsc = Rcpp::as<std::vector<arma::vec>>( trainList["rowOffsets"] );
    colOffsetsc = Rcpp::as<std::vector<arma::vec>>( trainList["colOffsets"] );

    //Load graph if present stuff
    //if( trainList.containsElementNamed( "graphDistMat" ) ){
    bool isGraphOn = Rcpp::as<bool>( trainList["isGraphOn"] );
    if( isGraphOn ){
      gdm = arma::sp_mat( Rcpp::as<arma::sp_mat>(trainList["graphDistMat"]) );
      gdop = arma::uvec( Rcpp::as<arma::uvec>(trainList["graphDistOnlyOnPoints"]))-1;
      graphDistMatc = &gdm ;
      graphDistOnlyOnPointsc =&gdop ;
    }

    if( trainList.containsElementNamed("family") ){
      this->fam =Rcpp::as<std::vector<std::string>>( trainList[ "family"] );
      this->family = &fam;
      this->ap = Rcpp::as<std::vector<arma::vec>>( trainList["aphi"] );
      this->aphi = &ap;
    }

  }

  ~imr()=default;

};




#endif
