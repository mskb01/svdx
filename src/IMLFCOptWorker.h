#ifndef IMLFCOPT_H
#define IMLFCOPT_H

//KEY FUNCTION USED FOR CV OF NPFITTER!

// cvWorkersTemplates.h
/*
 * VERY IMPORTANT CODING NOTES TO SELF
 *
 * When X is of type sp_mat,
 * arma::vec y = X.row(i) (and similarly col) won't work!
 * We need to separate as follows:
 * arma::vec y;
 * y = X.row(i). I have no idea why!
 *
 * ListOf only works with NumericMatrix, NumericVector currently; So use as<> to convert to arma types!
 *
 */


/*
 * TODO :
 * 1. BORDERING METHOD FOR AEX_PINV1 AND A2 RELATED INVERSES
 * 2. Sign flips : u, v
 * 3. Objective calc
 * 4. Offsets
 *
 */


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

#include <memory>
#include "nlopt.hpp"


#include<unordered_map>
#include<list>
#include<vector>
#include<array>
#include<random>
#include<string>
#include<cmath>

#include "IMLFCOnColsListTT.h"
#include "imttfr.h"

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
class IMLFCOpt { //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data

public:
  //All lists below mst be of the same length.
  //mses should have the same length.

  static double constexpr FAILED_CV_SCORE = 1e100; 
  /*
   * Grid workers
   */
  template<typename U>
  struct gridWorker:public Worker {
  public:
    const U& grid;
    arma::vec& grid_testMses; //holds average mse across folds for each grid point.
    IMLFCOpt* so;

    gridWorker(
      IMLFCOpt* so,
      const U& grid,
      arma::vec& grid_testMses): so(so), grid(grid), grid_testMses(grid_testMses){
      if( grid.size() != grid_testMses.size() ){
        throw std::invalid_argument( "bad grid, grid size must equal grid_testMses" );
      }
    }

    //IMPORTANT: returns a double, must cast ont the
    // receiver's end appropriately.
    double getMinPoint(){

      // Rcpp::Rcout << "grid, grid_testMSES" << std::endl;
      // Rcpp::Rcout << grid << std::endl;
      // Rcpp::Rcout << grid_testMses << std::endl;
      // Rcpp::Rcout << grid_testMses.index_min() << std::endl;
      // Rcpp::Rcout << grid(grid_testMses.index_min()) << std::endl;
      return grid(grid_testMses.index_min());
    };
    // virtual void operator()(std::size_t begin, std::size_t end)=0;
    //virtual ~gridWorker()=0;
    //~gridWorker(){}
    
    void execute(){
      parallelFor(0, this->grid.size(), *this);
    }
    
    void executeSerial(){
      (*this)(0, this->grid.size());
    }
  };

  struct KWorker:public gridWorker<arma::uvec> {

    using gridWorker<uvec>::grid;
    using gridWorker<uvec>::grid_testMses;
    using gridWorker<uvec>::so;
    using gridWorker<uvec>::getMinPoint;

    KWorker(IMLFCOpt* so);
    void operator()(std::size_t begin, std::size_t end) override;
    virtual ~KWorker();
  };

  struct f2Worker:public gridWorker<arma::vec> {

    using gridWorker<arma::vec>::grid;
    using gridWorker<arma::vec>::grid_testMses;
    using gridWorker<arma::vec>::so;
    using gridWorker<arma::vec>::getMinPoint;

    f2Worker(IMLFCOpt* so);
    void operator()(std::size_t begin, std::size_t end) override;
    virtual ~f2Worker();
  };


  /*
   * Variables
   */
  const dsListC& Y;
  const std::vector<arma::uvec>& columns1;
  const std::vector< std::vector<arma::mat> >& Z;       //Z may or may not be present
  // Owned fold-specific source initializers. These are copied from imttfr
  // in the constructor. For family fits they are replaced exactly once by
  // initializeGeneralizedFolds(), before any CV optimization begins.
  std::vector< std::vector<arma::mat> > Zeta;
  std::vector<arma::mat> hmat; //initial conditions
  std::vector<arma::mat> umat; //initial conditions
  std::vector< std::vector<arma::mat> > vmat; //initial conditions
  const std::vector<arma::uvec>& qids1;
  const std::vector<int> qForFolds;   // saves number of qids for each fold
  const std::vector<arma::mat>& S;
  const std::vector<arma::mat>& SStq;
  const std::vector<arma::mat>& RsCbound;
  const std::vector< std::vector<arma::vec> >& rowOffsets;
  const std::vector< std::vector<arma::vec> >& colOffsets;

  const std::vector<arma::uvec>& columns1Test;
  const std::vector<std::vector<arma::mat>>& ZTest;       //Z may or may not be present
  const std::vector<arma::mat>& STest;
  const std::vector<arma::mat>& RsCboundTest;
  const std::vector< std::vector<arma::vec> >& rowOffsetsTest;
  const std::vector< std::vector<arma::vec> >& colOffsetsTest;

  const unsigned int& nfolds;
  const int& r;
  const int& optCV_maxIter;
  const double& optCV_xtol_abs;
  const double& optCV_ftol_abs;
  const int& fitter_maxIter;
  const double& fitter_ftol_abs;
  const double& fitter_xtol_abs;
  const int& fitter_inner_maxIter;
  const double& fitter_inner_ftol_abs;

  arma::vec mses; //size must equal the length of list columns1Test i.e., samples for test data
  arma::vec pens; //ditto; stores sum of penalty values for each component

  // Debugging only: [evaluation, f2, theta..., fold scores..., mean CV score].
  std::vector<arma::vec> cvObjectiveTrace;
  unsigned long cvObjectiveEvalCounter = 0;

  // PARAMETERS, and dependents
  double f1_w;
  arma::vec f1vec_w; //replicated version of f1 for nfolds

  std::unique_ptr<KWorker> KWp;
  unsigned int K_w; //working value
  std::vector<arma::mat> hmat_w; //hmat subsetted for working values of K_w
  std::vector<arma::mat> umat_w; //ditto for umat
  std::vector<std::vector<arma::mat>> vmat_w; //ditto for vmat
  arma::uvec KGrid; //optimizing over grid
  arma::uvec default_KGrid;
  arma::vec KGrid_aveMses; //mses for each gridpoint

  std::unique_ptr<f2Worker> F2Wp;
  double f2_w;   //working value
  arma::vec f2vec_w; //replicated version of f2_w
  arma::vec f2Grid; //optimizing over grid
  arma::vec default_f2Grid;
  arma::vec f2Grid_aveMses; //corresponding mse for grid

  arma::vec theta_w;
  const unsigned int& nTheta;
  const arma::vec& wd; 
  std::vector<arma::mat> RTheta_w; // working values qxq penalty
  std::vector<arma::mat> MTheta_w; // qxq penalty
  std::vector<arma::mat> RtildeTheta_w; // qxq penalty
  std::vector<arma::mat> BTheta_w;      //qxq full kernel inverse
  std::vector<arma::mat> RThetaTest_w;
  std::vector<arma::mat> MThetaTest_w;

  //graph specific additions : first two do not alter through the optimization
  const std::vector<arma::sp_mat>* graphDistMat = nullptr;
  const std::vector<arma::uvec>* graphDistOnlyOnPoints = nullptr;
  double f3_w=1;
  arma::vec f3vec_w; //replicated version of f3, like f2vec_w
  arma::vec* f3vec_w_ptr = nullptr;

  //for exponential families
  const std::vector<std::string>* family = nullptr;
  const std::vector<std::vector<arma::vec>>* aphi = nullptr;

  int dispatch; //0 if single-np, 1 if single-graph, 2 if single-generalized, 3, if single-generalized graph
  arma::vec ones_w;
  int KMAX;
  bool f1set, f2set, f3set, Kset, thetaset, KGridset, f2Gridset; //tracks if all the respective values & their dependents have been initialized

  const double log10LB=-7, LB=1e-7;
  const double log10UB=7, UB=1e7;
  
  //to prevent oversubscription with nested workers, 
  // we are making all but one level serial. So we need these booleans. 
  bool outerGridSerial;   // K/f2 grid level
  bool innerFoldsSerial;  // SMLFCOnColsListTT / IMLFCOnColsListTT
  bool estimateCVFnSerial;  // optTheta & optf2 

  // After initialization, call setKGrid, setf2Grid, setf3 before calling the respective optimizers.
  explicit IMLFCOpt( const dsListC& Y,
            const imttfr& ttf,
            const double& f1,
            const int& optCV_maxIter,
            const double& optCV_xtol_abs,
            const double& optCV_ftol_abs,
            const int& fitter_maxIter,
            const double& fitter_ftol_abs,
            const double& fitter_xtol_abs,
            const int& fitter_inner_maxIter,
            const double& fitter_inner_ftol_abs
  );
  /*
   * Optimization grid settings
   */
  void setKGrid( const arma::uvec& KGrid );

  void setf2Grid( const arma::vec& f2Grid );

  /*
   * Optimization helpers
   */
  void setK( const unsigned int& K );
  void setf2( const double& f2 );
  void setf3( const double& f3 );
  void setTheta( const arma::vec& theta );

  // Reconcile the non-family fold initializers with the requested family at
  // the supplied initial hyperparameters, once per training fold.
  void initializeGeneralizedFolds();

  arma::mat getCVObjectiveTrace() const;
  double estimateCVScore();

  // Gu-style reduced-basis theta/f2 initialization helpers.
  arma::vec thetaTraceBalanced() const;
  arma::vec thetaCommonScaled(const arma::vec& theta);
  std::vector<arma::mat> fitFoldHmatsAtCurrentState();
  arma::vec thetaSignalAdaptive(
      const arma::vec& thetaTilde,
      const std::vector<arma::mat>& fittedHmats) const;
  /*
   * Optimizing engines
   */
  // K optimization over a grid
  void optK();

  // f2 optimization over a grid
  void optf2();

  /*
   * theta and f3 univariate opt for SERIAL computations only!
   *    via nlopt with nlopt wrappers.
   *
   *    Did not implement them thread safe, as nlopt works serially on these functions.
   *    If future changes are made, must need to revisit this.
   */
  double thetaObjective(arma::vec& th);
  double f3Objective( double& f3 );

  // NLOPT wrapper for arma::vec based objective
  static double nlopt_log10_thetaWrapper(unsigned n, const double* x, double* grad, void* data) {
    IMLFCOpt* instance = static_cast<IMLFCOpt*>(data);
    // Wrap the raw pointer 'x' into an arma::vec without copying
    arma::vec l10_current_x(const_cast<double*>(x), n, false, true);
    arma::vec current_x = arma::exp10( l10_current_x );
    return instance->thetaObjective(current_x);
  }

  // Wrapper for the double f3 objective
  static double nlopt_log10_f3Wrapper(unsigned n, const double* x, double* grad, void* data) {
    IMLFCOpt* instance = static_cast<IMLFCOpt*>(data);
    // Even if it's just one double, NLopt passes it as an array of size 1
    double current_f3 = std::pow(10,x[0]);
    return instance->f3Objective(current_f3);
  }


  //TODO: initial values for f2, f3, theta?
  void optimize(
      unsigned int KInit,
      double f2Init,
      arma::vec thetaInit,
      double* f3Init=nullptr,
      bool optForK=false,
                bool optForf2 = true,
                bool optForf3 = false,
                bool optFortheta = true,
                double nlopt_xtol_abs = .01,
                double nlopt_ftol_abs = .001,
                int nlopt_maxeval = 0
                );

};








#endif
