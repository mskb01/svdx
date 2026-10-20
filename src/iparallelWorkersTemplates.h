#ifndef IPARALLELWOKERSTEMPLATES_H
#define IPARALLELWOKERSTEMPLATES_H


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

#include "dsListC.h"
#include "parallelWokersTemplates.h"

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
class inpSMLFC : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)

private:
  const dsListC& Y;
  const arma::vec& wd; 
  const int n,D;
  const arma::mat& umat;
  const std::vector<arma::mat>& Z, Zeta, vmat;
  arma::mat& objColwise; // number of datasets x number of samples, we will store the objective elements here; stores MSE only
  const std::vector<arma::vec>& rowOffsets;
  const std::vector<arma::vec>& colOffsets;

public:
  inpSMLFC( const dsListC& Y,
            const arma::vec& wd, 
            const std::vector<arma::mat>& Z,
            const std::vector<arma::mat>& Zeta,
            const arma::mat& umat,
            const std::vector<arma::mat>& vmat,
            arma::mat& objs,
            const std::vector<arma::vec>& rowOffsets,
            const std::vector<arma::vec>& colOffsets
  );
  void operator()( std::size_t begin, std::size_t end );
};

///////////////////////////////////////////////////////////////////////
/*
 * Same as above but only on columns
 * the "objs" input must be of "columns" length i.e.,
 * the number of samples for which the objective is calculated
 */
///////////////////////////////////////////////////////////////////////
class inpSMLFCOnColumns : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective)

private:
  const dsListC& Y;
  const arma::vec& wd; 
  const int n, D;
  const arma::uvec& columns;
  const std::vector<arma::mat>& Z, Zeta, vmat;
  const arma::mat& umat;
  arma::mat& objColwise; //nlength, we will store the objective elements here; stores MSE only.
  const std::vector<arma::vec>& rowOffsets;
  const std::vector<arma::vec>& colOffsets;
public:
  inpSMLFCOnColumns( const dsListC& Y,
                     const arma::vec& wd, 
                     const arma::uvec& columns,
                     const std::vector<arma::mat>& Z,
                     const std::vector<arma::mat>& Zeta,
                     const arma::mat& umat,
                     const std::vector<arma::mat>& vmat,
                     arma::mat& objs,
                     const std::vector<arma::vec>& rowOffsets,
                     const std::vector<arma::vec>& colOffsets
  );
  void operator()( std::size_t begin, std::size_t end );
};

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
);

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
);

#endif
