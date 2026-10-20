#ifndef SMR_H
#define SMR_H

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
 */
///////////////////////////////////////////////////////////////////////
class smr {
public:

  //all are copies from R side
  arma::uvec columns1c;
  arma::mat Zc;       //Z may or may not be present
  arma::mat Zlsc;       //Z may or may not be present
  arma::mat Zetac;
  arma::mat hmatc; //initial conditions
  arma::mat umatc;//initial conditions
  arma::mat vmatc;//initial conditions
  arma::uvec qidsc;
  int qc;   // saves number of qids for each fold
  arma::mat Sc;
  arma::mat SStqc;
  arma::mat RsCboundc;
  arma::vec rowOffsetsc;
  arma::vec colOffsetsc;

  //graph specific additions : first two do not alter through the optimization
  arma::sp_mat* graphDistMatc = nullptr;
  arma::uvec* graphDistOnlyOnPointsc = nullptr;
  // -- stack vars to store references
  arma::sp_mat gdm;
  arma::uvec gdop;

  //family
  std::string* family=nullptr;
  std::string fam;

  arma::vec* aphi=nullptr;
  arma::vec ap;

  int r,q; //r =ncol(S), q = length(qids)
  unsigned int nTheta;


  // After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
  smr( const Rcpp::List& trainList) {

    //check for input
    std::vector<std::string> names = {"columns",
                                      "Z", "Zls", "Zeta",
                                      "umat", "vmat", "hmat",
                                      "qids", "nr", "nq",
                                      "S", "SStq", "RsCbound",
                                      "rowOffsets", "colOffsets"
                                      };
    for( int i=0; i < names.size(); i++ ){
      if( !trainList.containsElementNamed( (names[i]).c_str() ) ){
        throw std::invalid_argument( "Not found in the input list: " + names[i] ) ;
      }
    }

    columns1c = Rcpp::as<arma::uvec>( trainList["columns"] )-1;
    Zc = Rcpp::as<arma::mat>( trainList["Z"] );
    Zlsc = Rcpp::as<arma::mat>( trainList["Zls"] );
    Zetac = Rcpp::as<arma::mat>( trainList["Zeta"] );
    umatc = Rcpp::as<arma::mat>( trainList["umat"] );
    vmatc = Rcpp::as<arma::mat>( trainList["vmat"] );
    hmatc = Rcpp::as<arma::mat>( trainList["hmat"] );
    qidsc= Rcpp::as<arma::uvec>( trainList["qids"] )-1; //initilize qids
    r = Rcpp::as<int>( trainList["nr"] );
    q=Rcpp::as<int>( trainList["nq"] );
    qc = qidsc.size();
    Sc = Rcpp::as<arma::mat>( trainList["S"] );
    SStqc = Rcpp::as<arma::mat>( trainList["SStq"] );
    RsCboundc=Rcpp::as<arma::mat>( trainList["RsCbound"] );
    rowOffsetsc = Rcpp::as<arma::vec>( trainList["rowOffsets"] );
    colOffsetsc = Rcpp::as<arma::vec>( trainList["colOffsets"] );

    //Load graph if present stuff
    bool isGraphOn = Rcpp::as<bool>( trainList["isGraphOn"] );
    if( isGraphOn ){
      gdm = arma::sp_mat( Rcpp::as<arma::sp_mat>(trainList["graphDistMat"]) );
      gdop = arma::uvec( Rcpp::as<arma::uvec>(trainList["graphDistOnlyOnPoints"]))-1;
      graphDistMatc = &gdm ;
      graphDistOnlyOnPointsc =&gdop ;
    }

    if( trainList.containsElementNamed("family") ){
      this->fam =Rcpp::as<std::string>( trainList[ "family"] );
      this->family = &fam;
      this->ap = Rcpp::as<arma::vec>( trainList["aphi"] );
      this->aphi = &ap;
    }

  }

  ~smr()=default;

};




#endif
