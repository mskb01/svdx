#ifndef IMLFCONCOLSLIST_H
#define IMLFCONCOLSLIST_H



// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

#include <memory>


#include<unordered_map>
#include<list>
#include<vector>
#include<array>
#include<random>
#include<string>
#include<cmath>

#include "inpFitterOnColumnsGraph.h"
#include "igFitterOnColumnsGraph.h"

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

///////////////////////////////////////////////////////////////////////
/*
 * Trains on object, and evaluates MSE on folds.
 * Same as above but only on columns
 * the "mses" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
class IMLFCOnColsListTT : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data


  //every dataset can have its own set of following variables: Z, Zeta, vmat, offsets, aphi, family
  //.   own set of matrices: Z, ZTest, Zeta, vmat
  //.   own set of vectors: colOffsets, rowOffsets, colOffsetsTest, rowOffsetsTest, aphi
  //.   own family


  static constexpr double FAILED_CV_SCORE = 1e100;
  
  //All lists below mst be of the same length.
  //mses should have the same length.

  const dsListC& Y;
  const std::vector<arma::uvec>& columns1;
  const std::vector<std::vector<arma::mat>>& Z;       //Z may or may not be present
  const std::vector<std::vector<arma::mat>>& Zeta;
  const std::vector<arma::mat>& M;
  const std::vector<arma::mat>& hmat;
  const std::vector<arma::mat>& umat;
  const std::vector<std::vector<arma::mat>>& vmat;
  const std::vector<arma::mat>& Rtilde; // qxq penalty
  const std::vector<arma::uvec>& qids1;
  const std::vector<arma::mat>& B;      //qxq full kernel inverse
  const std::vector<std::vector<arma::vec>>& rowOffsets;
  const std::vector<std::vector<arma::vec>>& colOffsets;

  const std::vector<arma::uvec>& columns1Test;
  const std::vector<std::vector<arma::mat>>& ZTest;       //Z may or may not be present
  const std::vector<arma::mat>& MTest;
  const std::vector<std::vector<arma::vec>>& rowOffsetsTest;
  const std::vector<std::vector<arma::vec>>& colOffsetsTest;

  const int& K;
  const int& r;
  const arma::vec& f1;
  const arma::vec& f2;
  const int& maxIter;
  const double& eps;
  const double& xtol_abs;
  const int& inner_maxIter;
  const double& inner_ftol_abs;
  
  const arma::vec& wd; 

  arma::vec* mses; //size must equal the length of list columns1Test i.e., samples for test data
  arma::vec* pens; //ditto; stores sum of penalty values for each component
  std::vector<arma::mat>* fittedHmats = nullptr; // optional fitted shared hmat capture, one per fold


  //graph specific additions
  const std::vector<arma::sp_mat>* graphDistMat = nullptr; // qxq penalty
  const std::vector<arma::uvec>* graphDistOnlyOnPoints = nullptr;
  const arma::vec* f3 = nullptr;

  //family for generalized
  const std::vector<std::string>* family = nullptr;
  const std::vector<std::vector<arma::vec>>* aphi=nullptr;

  int dispatch; // 0 if inpfitter, 1 if inpfittergraph, 2 if igfitter, 3 if igfittergraph

public:

  explicit IMLFCOnColsListTT( const dsListC& Y,
                     const std::vector<arma::uvec>& columns1,
                     const std::vector<std::vector<arma::mat>>& Z,       //Z may or may not be present
                     const std::vector<std::vector<arma::mat>>& Zeta,
                     const std::vector<arma::mat>& M,
                     const std::vector<arma::mat>& hmat,
                     const std::vector<arma::mat>& umat,
                     const std::vector<std::vector<arma::mat>>& vmat,
                     const std::vector<arma::mat>& Rtilde, // qxq penalty
                     const std::vector<arma::uvec>& qids1, //qids must be a subset of columns1 for each case to avoid leakage into test !
                     const std::vector<arma::mat>& B,      //qxq full kernel inverse
                     const std::vector<std::vector<arma::vec>>& rowOffsets,
                     const std::vector<std::vector<arma::vec>>& colOffsets,

                     const std::vector<arma::sp_mat>* graphDistMat, // qxq graph distances post semi def at least
                     const std::vector<arma::uvec>* graphDistOnlyOnPoints, // qxq graph distances post semi def at least

                     const std::vector<std::string>* family,
                     const std::vector<std::vector<arma::vec>>* aphi,

                     const std::vector<arma::uvec>& columns1Test,
                     const std::vector<std::vector<arma::mat>>& ZTest,      //Z may or may not be present
                     const std::vector<arma::mat>& MTest,
                     const std::vector<std::vector<arma::vec>>& rowOffsetsTest,
                     const std::vector<std::vector<arma::vec>>& colOffsetsTest,

                     const int& K,
                     const int& r,
                     arma::vec& f1,
                     arma::vec& f2,
                     arma::vec* f3,
                     
                     const arma::vec& wd, 

                     const int& maxIter,
                     const double& eps,
                     const double& xtol_abs,
                     const int& inner_maxIter,
                     const double& inner_ftol_abs,
                     arma::vec* mses,
                     arma::vec* pens,
                     std::vector<arma::mat>* fittedHmats = nullptr

  );

  void operator()( std::size_t begin, std::size_t end );

  void execute();
  
  void executeSerial();

};


#endif
