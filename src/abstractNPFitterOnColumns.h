// #ifndef ABSTRACTNPFITTERONCOLUMNS_H
// #define ABSTRACTNPFITTERONCOLUMNS_H
//
//
// // [[Rcpp::depends(RcppParallel)]]
// // [[Rcpp::depends(RcppArmadillo)]]
//
// //#include <Rcpp.h>
// #include <RcppParallel.h>
// #include <RcppArmadillo.h>
// #include <tbb/tbb.h>
//
// #include "estimatePar.h"
//
// using namespace Rcpp;
// using namespace arma;
// using namespace RcppParallel;
//
// #include<unordered_map>
// #include<list>
// #include<vector>
// #include<array>
// #include<random>
// #include<string>
// #include<cmath>
//
// template<typename T>
// class abstractNPFitterOnColumns {
// public:
//   virtual ~abstractNPFitterOnColumns();
//   virtual void estimate(const estimatePar& par)=0;
//   virtual double mseForTest(const arma::uvec& columns1Test,
//                     const arma::mat& ZTest,
//                     const arma::mat& MTest,
//                     const arma::vec& rowOffsetsTest,
//                     const arma::vec& colOffsetsTest
//   )=0;
//   virtual double getPenalty()=0;
//
//
// };
//
// #endif
