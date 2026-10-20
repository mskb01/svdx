#ifndef IMTTFR_H
#define IMTTFR_H


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
#include<cmath>
#include<algorithm>
#include<stdexcept>

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

///////////////////////////////////////////////////////////////////////
/*
 * READER for single data train test folds model from R side.
 *
 * Compared to single data analysis, each dataset in integrative
 * can have its own Z, Zeta, vmat, row and col offsets, aphi and family.
 *
 * Applies to both train and test
 *
 *
 */
///////////////////////////////////////////////////////////////////////

class imttfr {


public:
  //All lists below mst be of the same length.
  //mses should have the same length.

  // vector of vectors of matrices/vectors/string appear below.
  //    -- outer vector runs through folds, inner through datasests for
  //      integrative analysis.

  std::vector<arma::uvec> columns1c;
  std::vector< std::vector<arma::mat> > Zc;       //Z may or may not be present
  std::vector< std::vector<arma::mat> > Zetac;
  std::vector<arma::mat> hmatc; //initial conditions
  std::vector<arma::mat> umatc;//initial conditions
  std::vector< std::vector<arma::mat> > vmatc;//initial conditions
  std::vector<arma::uvec> qidsc;
  std::vector<int> qForFoldsc;   // saves number of qids for each fold
  std::vector<arma::mat> Sc;
  std::vector<arma::mat> SStqc;
  std::vector<arma::mat> RsCboundc;
  std::vector< std::vector<arma::vec> > rowOffsetsc;
  std::vector< std::vector<arma::vec> > colOffsetsc;

  std::vector<arma::uvec> columns1Testc;
  std::vector<std::vector<arma::mat>> ZTestc;       //Z may or may not be present
  std::vector<arma::mat> STestc;
  std::vector<arma::mat> RsCboundTestc;
  std::vector<std::vector<arma::vec>> rowOffsetsTestc;
  std::vector<std::vector<arma::vec>> colOffsetsTestc;

  //graph specific additions : first two do not alter through the optimization
  std::vector<arma::sp_mat>* graphDistMatc = nullptr;
  std::vector<arma::uvec>* graphDistOnlyOnPointsc = nullptr;
  std::vector<arma::sp_mat> gdm;
  std::vector<arma::uvec> gdop;

  //exponential family
  std::vector< std::string >* family = nullptr;
  std::vector<std::string> fam; //all folds have the same given family
  std::vector< std::vector<arma::vec> >* aphi = nullptr;
  std::vector< std::vector<arma::vec> > ap; //one vector of estimates for each fold

  int r;
  unsigned int nTheta;
  arma::vec wd;
  unsigned int nfolds;

  double f2Init=1.0;
  double f3ValIfG = 1.0;
  double* f3Init_ptr=nullptr;
  arma::vec thetaInit;

  // After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
  imttfr( const Rcpp::List& foldMods) {

    //read overall structure
    std::vector<std::string> names = {"train", "test", "nTheta"};
    for( int i=0; i < names.size(); i++ ){
      if( !foldMods.containsElementNamed( (names[i]).c_str() ) ){
        throw std::invalid_argument( "Not found in the input list: " + names[i] ) ;
      }
    }
    const Rcpp::List& trainFolds = Rcpp::as<Rcpp::List>(foldMods["train"]);
    const Rcpp::List& testFolds = Rcpp::as<Rcpp::List>(foldMods["test"]);
    nTheta = Rcpp::as<unsigned int>( foldMods["nTheta"] );
    wd = Rcpp::as<arma::vec>( foldMods["wd"] );

    //family
    if( trainFolds.containsElementNamed( "family" ) ){
      this->fam=Rcpp::as< std::vector<std::string> >( trainFolds["family"] );
      this->family= &fam;
    }

    //read within train folds
    names = {"R0traceList", "columnsList", "ZList", "ZetaList", "SList", "RsCboundList", "qidsList", "SStqList",
             "hmatList", "umatList", "vmatList", "rowOffsetsList", "colOffsetsList"
             };
    for( int i=0; i < names.size(); i++ ){
      if( !trainFolds.containsElementNamed( (names[i]).c_str() ) ){
        throw std::invalid_argument( "Not found in the input list, under train subset: " + names[i] ) ;
      }
    }
    const Rcpp::ListOf<Rcpp::NumericVector>& R0trace=Rcpp::as<Rcpp::ListOf<Rcpp::NumericVector> >(trainFolds["R0traceList"]);    //folds
    arma::vec R0tracesForFirstFold= Rcpp::as<arma::vec>(R0trace[0]);
    for( int i=0; i<nTheta; i++ ){//clip, so that inverse of these variables, use as initial conditions don't explode.
      R0tracesForFirstFold(i)=std::max( std::abs(R0tracesForFirstFold(i)), 0.001);
    }
    const Rcpp::ListOf<Rcpp::NumericVector>& columns1=Rcpp::as< Rcpp::ListOf<Rcpp::NumericVector> >( trainFolds["columnsList"] );    //folds
    nfolds = columns1.size();
    if( nfolds == 0 ){
      throw std::runtime_error("Empty folds for crossvalidation" );
    }
    const Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericMatrix> >& Z=Rcpp::as< Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericMatrix>> >( trainFolds["ZList"] );          //Z may or may not be present
    const Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericMatrix> >& Zeta=Rcpp::as< Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericMatrix> > >( trainFolds["ZetaList"] );
    const Rcpp::ListOf<Rcpp::NumericMatrix>& S=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( trainFolds["SList"] );
    const Rcpp::ListOf<Rcpp::NumericMatrix>& RsCbound=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( trainFolds["RsCboundList"] );
    const Rcpp::ListOf<Rcpp::NumericVector>& qids=Rcpp::as< Rcpp::ListOf<Rcpp::NumericVector> >( trainFolds["qidsList"] ); //don't forget to subtract!!
    const Rcpp::ListOf<Rcpp::NumericMatrix>& SStq= Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( trainFolds["SStqList"] ); //SStq here is for qidsxqids only
    const Rcpp::ListOf<Rcpp::NumericMatrix>& hmat=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( trainFolds["hmatList"] ); //don't forget to subtract!! ;
    const Rcpp::ListOf<Rcpp::NumericMatrix>& umat=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( trainFolds["umatList"] ) ;
    const Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericMatrix>>& vmat=Rcpp::as< Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericMatrix>> >( trainFolds["vmatList"] ) ;
    const Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericVector>>& rowOffsets=Rcpp::as< Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericVector>> >( trainFolds["rowOffsetsList"] ) ;
    const Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericVector>>& colOffsets= Rcpp::as< Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericVector>> >( trainFolds["colOffsetsList"] ) ;

    //read within test folds
    names = {"columnsList", "ZList", "SList", "RsCboundList","rowOffsetsList", "colOffsetsList"};
    for( int i=0; i < names.size(); i++ ){
      if( !testFolds.containsElementNamed( (names[i]).c_str() ) ){
        throw std::invalid_argument( "Not found in the input list, under test subset: " + names[i] ) ;
      }
    }
    const Rcpp::ListOf<Rcpp::NumericVector>& columns1Test=Rcpp::as< Rcpp::ListOf<Rcpp::NumericVector> >( testFolds["columnsList"] ) ;
    const Rcpp::ListOf< Rcpp::ListOf< Rcpp::NumericMatrix > >& ZTest=Rcpp::as< Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericMatrix> > >( testFolds["ZList"] )  ;       //Z may or may not be present
    const Rcpp::ListOf<Rcpp::NumericMatrix>& STest=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( testFolds["SList"] )  ;
    const Rcpp::ListOf<Rcpp::NumericMatrix>& RsCboundTest=Rcpp::as< Rcpp::ListOf<Rcpp::NumericMatrix> >( testFolds["RsCboundList"] )  ;
    const Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericVector> >& rowOffsetsTest=Rcpp::as< Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericVector>> >( testFolds["rowOffsetsList"] )  ;
    const Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericVector> >& colOffsetsTest=Rcpp::as< Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericVector>> >( testFolds["colOffsetsList"] )  ;

    //C datastructures with copies
    columns1c.resize( nfolds );
    Zc.resize( nfolds );
    Zetac.resize( nfolds );
    Sc.resize( nfolds );
    RsCboundc.resize( nfolds );
    hmatc.resize( nfolds );
    umatc.resize( nfolds );
    vmatc.resize( nfolds );
    qidsc.resize( nfolds );
    qForFoldsc.resize(nfolds);   // saves number of qids for each fold
    SStqc.resize( nfolds );
    rowOffsetsc.resize( nfolds );
    colOffsetsc.resize( nfolds );

    columns1Testc.resize(nfolds);
    ZTestc.resize(nfolds);      //Z may or may not be present
    STestc.resize(nfolds);
    RsCboundTestc.resize(nfolds);
    rowOffsetsTestc.resize(nfolds);
    colOffsetsTestc.resize(nfolds);

    for( int i=0; i<nfolds; i++ ){
      columns1c[i]=(Rcpp::as<arma::uvec>(columns1[i]))-1; //converting to 0 index based for C side
      Zc[i]=Rcpp::as<std::vector<arma::mat>>(Z[i]);
      Zetac[i]=Rcpp::as<std::vector<arma::mat>>(Zeta[i]);
      Sc[i]=Rcpp::as<arma::mat>(S[i]);
      RsCboundc[i]=Rcpp::as<arma::mat>(RsCbound[i]);
      hmatc[i]=Rcpp::as<arma::mat>(hmat[i]);
      umatc[i]=Rcpp::as<arma::mat>(umat[i]);
      vmatc[i]=Rcpp::as<std::vector<arma::mat>>(vmat[i]);
      qidsc[i]=(Rcpp::as<arma::uvec>(qids[i]))-1; //convert to 0 intdex based
      qForFoldsc[i]=qidsc[i].size();
      SStqc[i]=Rcpp::as<arma::mat>(SStq[i]);
      rowOffsetsc[i]=Rcpp::as<std::vector<arma::vec>>(rowOffsets[i]);
      colOffsetsc[i]=Rcpp::as<std::vector<arma::vec>>(colOffsets[i]);

      columns1Testc[i]=(Rcpp::as<arma::uvec>(columns1Test[i]))-1;
      ZTestc[i]=Rcpp::as<std::vector<arma::mat>>(ZTest[i]);
      STestc[i]=Rcpp::as<arma::mat>(STest[i]);
      RsCboundTestc[i]=Rcpp::as<arma::mat>(RsCboundTest[i]);
      rowOffsetsTestc[i]=Rcpp::as<std::vector<arma::vec>>(rowOffsetsTest[i]);
      colOffsetsTestc[i]=Rcpp::as<std::vector<arma::vec>>(colOffsetsTest[i]);

    }
    this->r=Sc[0].n_cols;

    //Load graph if present
    if( trainFolds.containsElementNamed( "graphDistMatList" ) ){
      gdm.resize(nfolds); gdop.resize(nfolds);
      const Rcpp::List graphDistMat=Rcpp::as<Rcpp::List>(trainFolds["graphDistMatList"]);
      if( !trainFolds.containsElementNamed( "graphDistOnlyOnPointsList" ) ){
        throw std::invalid_argument( "graphDistOnlyOnPointsList not found in the trainFolds" ) ;
      }
      const Rcpp::ListOf<Rcpp::NumericVector>& graphDistOnlyOnPoints=Rcpp::as< Rcpp::ListOf<Rcpp::NumericVector> >( trainFolds["graphDistOnlyOnPointsList"] ); //don't forget to subtract!!
      for( int i=0; i < nfolds; i++ ){
        gdm[i]=Rcpp::as<arma::sp_mat>(graphDistMat[i]);
        gdop[i]=(Rcpp::as<arma::uvec>(graphDistOnlyOnPoints[i]))-1; //R to C subscript
      }
      this->graphDistMatc = &gdm;
      this->graphDistOnlyOnPointsc = &gdop;
      this->f3Init_ptr = &(this->f3ValIfG);
    }

    //if aphi present, copy to stack
    if( trainFolds.containsElementNamed("aphiList") ){
      this->ap.resize( nfolds );
      const Rcpp::ListOf<Rcpp::ListOf<Rcpp::NumericVector>> apListR = Rcpp::as<Rcpp::ListOf< Rcpp::ListOf<Rcpp::NumericVector>> >( trainFolds["aphiList"] );
      for( int i=0; i < nfolds; i++ ){
        ap[i]=Rcpp::as<std::vector<arma::vec>>( apListR[i] );
      }
      this->aphi = &ap;
    }
    this->f2Init=1.0;
    this->thetaInit = 1.0/R0tracesForFirstFold;
    //this->thetaInit = arma::ones<arma::vec>(R0tracesForFirstFold.n_elem);


    /*
     * If svdx with exponential family setting, we have initial conditions,
     * in train test fold that we could read
     *
     */
    if( trainFolds.containsElementNamed("theta") ){
      this->thetaInit = Rcpp::as<arma::vec>( trainFolds["theta"] );
      //this->thetaInit = arma::ones<arma::vec>(R0tracesForFirstFold.n_elem);
    }
    if( trainFolds.containsElementNamed("f2") ){
      this->f2Init = Rcpp::as<double>( trainFolds["f2"] );
    }
    if( trainFolds.containsElementNamed("graphDistOnlyOnPointsList") ){
      if( trainFolds.containsElementNamed("f3") ){
        this->f3ValIfG = Rcpp::as<double>(trainFolds["f3"]);
        this->f3Init_ptr = &(this->f3ValIfG);
      }
    }

  }

  ~imttfr() = default;

};



#endif
