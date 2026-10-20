#ifndef ESTIMATEPAR_H
#define ESTIMATEPAR_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>



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

class estimatePar {
public:
  const double& f1;
  const double& f2;
  const int& maxIter;
  const double& eps;
  double f3;  //used only for graph

  estimatePar( const double& f1, const double& f2, const int& maxIter, const double& eps ):
    f1(f1), f2(f2), maxIter(maxIter), eps(eps){f3=0;}
  estimatePar( const double& f1, const double& f2, const double& f3, const int& maxIter, const double& eps ):
    f1(f1), f2(f2), maxIter(maxIter), eps(eps){
    this->f3 = f3;
  }

};

#endif
