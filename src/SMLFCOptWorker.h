#ifndef SMLFCOPT_H
#define SMLFCOPT_H

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

//#include <Rcpp.h>
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <tbb/tbb.h>
#include <Rinternals.h>

#include <memory>
#include <algorithm>
#include "nlopt.hpp"



#include<unordered_map>
#include<list>
#include<vector>
#include<array>
#include<random>
#include<string>
#include<cmath>

#include "SMLFCOnColsListTT.h"
#include "smttfr.h"

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
template<typename T> //sp_mat or mat
class SMLFCOpt { //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data

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
    SMLFCOpt<T>* so;

    gridWorker(
      SMLFCOpt* so,
      const U& grid,
      arma::vec& grid_testMses): so(so), grid(grid), grid_testMses(grid_testMses){
      if( grid.size() != grid_testMses.size() ){
        throw std::invalid_argument( "bad grid, grid size must equal grid_testMses" );
      }
    }

    double getMinPoint(){

      return grid(grid_testMses.index_min());
    };
    
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
    using gridWorker<uvec>::execute;
    using gridWorker<uvec>::executeSerial;

    KWorker(SMLFCOpt* so
    ): gridWorker<uvec>::gridWorker( so, so->KGrid, so->KGrid_aveMses ){}

    virtual ~KWorker();

    void operator()(std::size_t begin, std::size_t end) override{
      for( std::size_t i = begin; i<end; i++ ){

        //For K optimization, separate objects for each K are needed to avoid race conditions
        unsigned int K_i = grid(i);
        std::vector<arma::mat> umatc_i; umatc_i.resize( so->nfolds );
        std::vector<arma::mat> hmatc_i; hmatc_i.resize( so->nfolds );
        std::vector<arma::mat> vmatc_i; vmatc_i.resize( so->nfolds );
        for( int f=0; f< so->nfolds; f++ ){
          hmatc_i[f]=so->hmat[f].cols(0,K_i-1);
          umatc_i[f]=so->umat[f].cols(0,K_i-1);
          vmatc_i[f]=so->vmat[f].cols(0,K_i-1);
        }

        //holds mses and penalties for each train fold
        // its the average of these that we return from the objective function below.
        arma::vec msesc_i(so->nfolds, arma::fill::zeros);
        arma::vec pensc_i(so->nfolds, arma::fill::zeros);
        SMLFCOnColsListTT<T> soi( so->Y,
                                  so->columns1,
                                  so->Z,       //Z may or may not be present
                                  so->Zeta,
                                  so->MTheta_w,

                                  hmatc_i,
                                  umatc_i,
                                  vmatc_i,

                                  so->RtildeTheta_w, // qxq penalty
                                  so->qids1, //qids must be a subset of columns1 for each case to avoid leakage into test !
                                  so->BTheta_w,      //qxq full kernel inverse
                                  so->rowOffsets,
                                  so->colOffsets,

                                  so->graphDistMat, // qxq graph distances post semi def at least
                                  so->graphDistOnlyOnPoints, // qxq graph distances post semi def at least

                                  so->family,
                                  so->aphi,

                                  so->columns1Test,
                                  so->ZTest,      //Z may or may not be present
                                  so->MThetaTest_w,
                                  so->rowOffsetsTest,
                                  so->colOffsetsTest,

                                  K_i,

                                  so->r,
                                  so->f1vec_w,
                                  so->f2vec_w,
                                  so->f3vec_w_ptr,
                                  so->fitter_maxIter,
                                  so->fitter_ftol_abs,
                                  so->fitter_xtol_abs,
                                  so->fitter_inner_maxIter,
                                  so->fitter_inner_ftol_abs,

                                  &msesc_i,
                                  &pensc_i

        );
        
        if( so->innerFoldsSerial ){
          soi.executeSerial(); //we parallelize over the grid, but serially over folds for every grid point
          // we had to do this to avoid the large number of threads and CPU usage due to optf2() and optK()
          //. despite using RcppParallel::setNumThreads & tbb::global_control & task.arena
        } else {
          soi.execute();
        }
        grid_testMses(i) = arma::mean( msesc_i );
      }
    }
  };

  struct f2Worker:public gridWorker<arma::vec> {

    using gridWorker<arma::vec>::grid;
    using gridWorker<arma::vec>::grid_testMses;
    using gridWorker<arma::vec>::so;
    using gridWorker<arma::vec>::getMinPoint;
    using gridWorker<arma::vec>::execute;
    using gridWorker<arma::vec>::executeSerial;

    f2Worker(SMLFCOpt* so
    ): gridWorker<arma::vec>::gridWorker( so, so->f2Grid, so->f2Grid_aveMses ){}

    virtual ~f2Worker();

    void operator()(std::size_t begin, std::size_t end) override{
      for( std::size_t i = begin; i<end; i++ ){

        //For K optimization, separate objects for each K are needed to avoid race conditions
        //construct f2_vec for this w,
        arma::vec f2vec_i = grid(i)*(so->ones_w);

        //holds mses and penalties for each train fold
        // its the average of these that we return from the objective function below.
        arma::vec msesc_i(so->nfolds, arma::fill::zeros);
        arma::vec pensc_i(so->nfolds, arma::fill::zeros);
        SMLFCOnColsListTT<T> soi( so->Y,
                                  so->columns1,
                                  so->Z,       //Z may or may not be present
                                  so->Zeta,
                                  so->MTheta_w,
                                  so->hmat_w,
                                  so->umat_w,
                                  so->vmat_w,
                                  so->RtildeTheta_w, // qxq penalty
                                  so->qids1, //qids must be a subset of columns1 for each case to avoid leakage into test !
                                  so->BTheta_w,      //qxq full kernel inverse
                                  so->rowOffsets,
                                  so->colOffsets,

                                  so->graphDistMat, // qxq graph distances post semi def at least
                                  so->graphDistOnlyOnPoints, // qxq graph distances post semi def at least

                                  so->family,
                                  so->aphi,

                                  so->columns1Test,
                                  so->ZTest,      //Z may or may not be present
                                  so->MThetaTest_w,
                                  so->rowOffsetsTest,
                                  so->colOffsetsTest,

                                  so->K_w,
                                  so->r,
                                  so->f1vec_w,

                                  f2vec_i,

                                  so->f3vec_w_ptr,
                                  so->fitter_maxIter,
                                  so->fitter_ftol_abs,
                                  so->fitter_xtol_abs,
                                  so->fitter_inner_maxIter,
                                  so->fitter_inner_ftol_abs,

                                  &msesc_i,
                                  &pensc_i
        );
        if( so->innerFoldsSerial ){
          soi.executeSerial(); //we parallelize over the grid, but serially over folds for every grid point
          // we had to do this to avoid the large number of threads and CPU usage due to optf2() and optK()
          //. despite using RcppParallel::setNumThreads & tbb::global_control & task.arena
        } else {
          soi.execute();
        }
        grid_testMses(i) = arma::mean( msesc_i );
      }
    }
  };


  /*
   * Variables
   */
  const T& Y;
  const std::vector<arma::uvec>& columns1;
  const std::vector<arma::mat>& Z;       //Z may or may not be present
  // Fold-specific source initializers. These are copied from smttfr in the
  // constructor. For family models, initializeGeneralizedFolds() replaces
  // them once before hyperparameter optimization; existing optimization
  // routines then continue to consume these same members unchanged.
  std::vector<arma::mat> Zeta;
  std::vector<arma::mat> hmat; //initial conditions
  std::vector<arma::mat> umat; //initial conditions
  std::vector<arma::mat> vmat; //initial conditions
  const std::vector<arma::uvec>& qids1;
  const std::vector<int> qForFolds;   // saves number of qids for each fold
  const std::vector<arma::mat>& S;
  const std::vector<arma::mat>& SStq;
  const std::vector<arma::mat>& RsCbound;
  const std::vector<arma::vec>& rowOffsets;
  const std::vector<arma::vec>& colOffsets;

  const std::vector<arma::uvec>& columns1Test;
  const std::vector<arma::mat>& ZTest;       //Z may or may not be present
  const std::vector<arma::mat>& STest;
  const std::vector<arma::mat>& RsCboundTest;
  const std::vector<arma::vec>& rowOffsetsTest;
  const std::vector<arma::vec>& colOffsetsTest;

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

  // Debugging only. Existing columns are preserved in place:
  // [evaluation, f2, theta..., fold scores..., mean CV score, trace_type, K1_mse, ..., KMAX_mse]
  // trace_type: 0 = ordinary CV objective; 1 = completed K-grid evaluation; 2 = f2-grid candidate.
  std::vector<arma::vec> cvObjectiveTrace;
  unsigned long cvObjectiveEvalCounter = 0;

  // PARAMETERS, and dependents
  double f1_w;
  arma::vec f1vec_w; //replicated version of f1 for nfolds

  std::unique_ptr<KWorker> KWp;
  unsigned int K_w; //working value
  std::vector<arma::mat> hmat_w; //hmat subsetted for working values of K_w
  std::vector<arma::mat> umat_w; //ditto for umat
  std::vector<arma::mat> vmat_w; //ditto for vmat
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
  const std::string* family = nullptr;
  const std::vector<arma::vec>* aphi = nullptr;

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
  SMLFCOpt( const T& Y,
            const smttfr& ttf,
            const double& f1,
            const int& optCV_maxIter,
            const double& optCV_xtol_abs,
            const double& optCV_ftol_abs,
            const int& fitter_maxIter,
            const double& fitter_ftol_abs,
            const double& fitter_xtol_abs,
            const int& fitter_inner_maxIter,
            const double& fitter_inner_ftol_abs
  ): Y(Y), columns1(ttf.columns1c),
  Z(ttf.Zc), Zeta(ttf.Zetac),
  S(ttf.Sc),SStq(ttf.SStqc), RsCbound(ttf.RsCboundc),
  hmat(ttf.hmatc), umat(ttf.umatc),vmat(ttf.vmatc),
  qids1(ttf.qidsc), qForFolds(ttf.qForFoldsc),
  rowOffsets(ttf.rowOffsetsc), colOffsets(ttf.colOffsetsc),
  graphDistMat(ttf.graphDistMatc), graphDistOnlyOnPoints(ttf.graphDistOnlyOnPointsc),
  family(ttf.family), aphi(ttf.aphi),
  columns1Test(ttf.columns1Testc),
  ZTest(ttf.ZTestc),
  STest(ttf.STestc), RsCboundTest(ttf.RsCboundTestc),
  rowOffsetsTest(ttf.rowOffsetsTestc), colOffsetsTest(ttf.colOffsetsTestc),
  f1_w(f1),r(ttf.r), nfolds(ttf.nfolds), nTheta(ttf.nTheta),
  optCV_maxIter(optCV_maxIter), optCV_xtol_abs(optCV_xtol_abs), optCV_ftol_abs(optCV_ftol_abs),
  fitter_maxIter(fitter_maxIter), fitter_ftol_abs(fitter_ftol_abs), fitter_xtol_abs(fitter_xtol_abs), fitter_inner_maxIter(fitter_inner_maxIter), fitter_inner_ftol_abs(fitter_inner_ftol_abs)
   {

    ones_w= arma::vec(nfolds, arma::fill::ones);
    mses = 0*ones_w;
    pens = 0*ones_w;

    //f1 initialization
    this->f1_w = f1;
    f1vec_w = f1_w*ones_w;

    //must initialize the std::vector of umat_w etc., for the number of folds
    if( (umat.size() != nfolds) | (vmat.size()!=nfolds) | (hmat.size()!=nfolds) ){
      throw std::invalid_argument( "umatc size does not equal nfolds" );
    }
    umat_w.resize(nfolds);
    hmat_w.resize(nfolds);
    vmat_w.resize(nfolds);
    RTheta_w.resize(nfolds); // working values qxq penalty
    MTheta_w.resize(nfolds); // qxq penalty
    RtildeTheta_w.resize(nfolds); // qxq penalty
    BTheta_w.resize(nfolds);      //qxq full kernel inverse
    RThetaTest_w.resize(nfolds);
    MThetaTest_w.resize(nfolds);

    if( graphDistMat != nullptr ){
      dispatch = 1;
    } else {
      dispatch=0;
    }

    //Kmax , maximum K for which we can initialize given umat
    this->KMAX = this->umat[0].n_cols;

    // Set a default KGrid from 1 to KMAX
    default_KGrid = arma::regspace<arma::uvec>(1, this->KMAX);
    this->setKGrid(default_KGrid);

    // Set a default f2Grid containing standard regularizer values
    default_f2Grid = arma::exp10( arma::regspace(-7,0) );
    this->setf2Grid(default_f2Grid);

    KWp = std::make_unique<KWorker>( this );
    F2Wp= std::make_unique<f2Worker>( this );
    
    //to avoid oversubscription with nested workers, 
    //.  set some serial choisces
    if( family == nullptr ){
      //we make the calculations at every grid point (i.e., over folds)
      //  parallel, but make processing the grid serial . 
      outerGridSerial=true;   // K/f2 grid level
      innerFoldsSerial=false;  // SMLFCOnColsListTT / IMLFCOnColsListTT
      estimateCVFnSerial=false;  // optTheta & optf2 
    } else {
      // For family settings, we have parallel row-wise workers within 
      // gFitterOnColumns or igFitterOnColumns. Additionlly for Poisson, 
      // we have parallel computations for hessian. 
      //we leave the inner row-wise workers parallel. This is why 
      // estimateCVFnSerial is also true in this case. 
      // for poisson, additionally, we keep the hessian hpcj serial. 
      outerGridSerial=true;   // K/f2 grid level
      innerFoldsSerial=true;  // SMLFCOnColsListTT / IMLFCOnColsListTT
      estimateCVFnSerial=true;  // optTheta & optf2 
    }
    

  };


  /*
   * Optimization grid settings
   */
  void setKGrid( const arma::uvec& KGrid ){
    this->KGrid= KGrid;
    this->KGrid_aveMses = arma::vec(KGrid.size(), arma::fill::zeros);
    this->KGridset = true;
  }

  void setf2Grid( const arma::vec& f2Grid ){
    this->f2Grid= f2Grid;
    this->f2Grid_aveMses = arma::vec(f2Grid.size(), arma::fill::zeros);
    this->f2Gridset = true;
  }

  /*
   * Optimization helpers
   */
  void setK( const unsigned int& K ){
    if( K > this->KMAX ){
      throw std::invalid_argument( "K to set is higher than the number of columns in initializer umat" );
    }
    this->K_w=K;
    for( int f=0; f<nfolds; f++ ){
      this->umat_w[f]=umat[f].cols(0,K_w-1);
      this->hmat_w[f]=hmat[f].cols(0,K_w-1);
      this->vmat_w[f]=vmat[f].cols(0,K_w-1);
    }
    Kset = true;
  }
  void setf2( const double& f2 ){
    this->f2_w = f2;
    this->f2vec_w=f2*ones_w;
    this->f2set=true;
  }

  void setf3( const double& f3 ){
    this->f3_w= f3;
    this->f3vec_w=f3*ones_w;
    this->f3vec_w_ptr = &(this->f3vec_w); //caller is sensible, so this won't be a nullptr
    f3set=true;
  }

  void setTheta( const arma::vec& theta ){
    this->theta_w = theta;
    if( theta.n_elem != nTheta ){
      throw std::invalid_argument("theta vector does not have the expected length");
    }

    //prepare Mtheta=[S, Rtheta], RtildeTheta=Rtheta[,qids] for each fold
    int st, en, qfld;
    for(int fld=0; fld < nfolds; fld++){
      qfld=qForFolds[fld];
      st=0; en= st+qfld-1;
      RTheta_w[fld] = theta_w[0]*RsCbound[ fld ].cols( st, en );
      RThetaTest_w[fld] = theta_w[0]*RsCboundTest[ fld ].cols( st, en );
      st=en+1;
      for( int t=1; t < nTheta ; t++ ){ //loop over rest of the terms
        en =st+qfld-1;
        RTheta_w[fld] += theta_w[ t ] * RsCbound[ fld ].cols( st, en );
        RThetaTest_w[fld] += theta_w[ t ] * RsCboundTest[ fld ].cols( st, en );
        st=en+1;
      }
      RtildeTheta_w[fld] = RTheta_w[fld].rows( qids1[fld] );
      //the orth kernel
      BTheta_w[fld]=invpsd(SStq[fld] + RtildeTheta_w[fld], 1e-7, false);
    }

    //generate M
    for(int fld=0; fld < nfolds; fld++){
      MTheta_w[fld] = arma::join_horiz( S[fld], RTheta_w[fld] );
      //RtildethetaForFolds_w already ready from previous loop, which is for training data
      MThetaTest_w[fld] = arma::join_horiz( STest[fld], RThetaTest_w[fld] );
      //don't need an RtildethetaForFoldsTest_w for prediction on test
    }

    thetaset=true;
  }


  arma::mat getCVObjectiveTrace() const {
    if (cvObjectiveTrace.empty()) return arma::mat();
    arma::mat out(cvObjectiveTrace.size(), cvObjectiveTrace[0].n_elem);
    for (std::size_t i=0; i<cvObjectiveTrace.size(); ++i)
      out.row(i)=cvObjectiveTrace[i].t();
    return out;
  }

  double estimateCVScore() {
    
    
    mses.fill(FAILED_CV_SCORE);

    if( !( f2set & Kset & thetaset  ) ){
      throw std::invalid_argument( "f2/K/theta have not been set yet. Initialize with setf2(), setf3() and setK()."  );
    }
    if( (graphDistMat != nullptr ) ){
      if(  (!f3set)  ){
        throw std::invalid_argument( "f3 has not been set yet. "  );
      }
    }
    SMLFCOnColsListTT<T> so( Y,
                       columns1,
                       Z,       //Z may or may not be present
                       Zeta,
                       MTheta_w,
                       hmat_w,
                       umat_w,
                       vmat_w,
                       RtildeTheta_w, // qxq penalty
                       qids1, //qids must be a subset of columns1 for each case to avoid leakage into test !
                       BTheta_w,      //qxq full kernel inverse
                       rowOffsets,
                       colOffsets,

                       graphDistMat, // qxq graph distances post semi def at least
                       graphDistOnlyOnPoints, // qxq graph distances post semi def at least

                       family,
                       aphi,

                       columns1Test,
                       ZTest,      //Z may or may not be present
                       MThetaTest_w,
                       rowOffsetsTest,
                       colOffsetsTest,

                       K_w,
                       r,
                       f1vec_w,
                       f2vec_w,
                       f3vec_w_ptr,
                       fitter_maxIter,
                       fitter_ftol_abs,
                       fitter_xtol_abs,
                       fitter_inner_maxIter,
                       fitter_inner_ftol_abs,
                       //const bool& switchFromRtoCindexing, //if set to TRUE, columns and qids are decremented by 1
                       &mses,
                       &pens

    );
    if( this->estimateCVFnSerial ){
      so.executeSerial();
    } else {
      so.execute();  
    }
    
    //return arma::mean(mses);
    const double cvScore = arma::mean(mses);
    {
      const arma::uword oldWidth = 3 + theta_w.n_elem + mses.n_elem;
      arma::vec traceRow(oldWidth + 1 + KMAX, arma::fill::value(arma::datum::nan));
      traceRow(0) = static_cast<double>(++cvObjectiveEvalCounter);
      traceRow(1) = f2_w;
      traceRow.subvec(2, 1 + theta_w.n_elem) = theta_w;
      traceRow.subvec(2 + theta_w.n_elem,
                      1 + theta_w.n_elem + mses.n_elem) = mses;
      traceRow(oldWidth - 1) =
        std::isfinite(cvScore) ? cvScore : FAILED_CV_SCORE;
      traceRow(oldWidth) = 0.0; // ordinary CV objective
      cvObjectiveTrace.push_back(traceRow);
    }
    if (!std::isfinite(cvScore)) {
      return FAILED_CV_SCORE;
    }
    
    return cvScore;

  }


  /*
   * Gu-style reduced-basis initialization for theta/f2.
   *
   */
  arma::vec thetaTraceBalanced() const {
    arma::vec meanTraces(nTheta, arma::fill::zeros);

    for (int fld = 0; fld < nfolds; ++fld) {
      const int qfld = qForFolds[fld];
      int st = 0;
      for (unsigned int beta = 0; beta < nTheta; ++beta) {
        const int en = st + qfld - 1;
        arma::mat Rfold = RsCbound[fld].cols(st, en);
        const arma::mat Qbeta = Rfold.rows(qids1[fld]);
        meanTraces(beta) += arma::trace(Qbeta);
        st = en + 1;
      }
    }

    meanTraces /= static_cast<double>(nfolds);
    arma::vec out(nTheta);
    for (unsigned int beta = 0; beta < nTheta; ++beta) {
      // Same practical protection as the old R0-trace initialization:
      // do not allow a nearly-zero trace to explode the starting theta.
      const double tr = std::max(std::abs(meanTraces(beta)), 1e-3);
      out(beta) = 1.0 / tr;
    }
    return out;
  }

  /*
   * Gu common theta scaling for the reduced-basis single-dataset problem.
   *
   */
  arma::vec thetaCommonScaled(const arma::vec& theta) {
    // Build the fold-specific RTheta matrices for this relative theta.
    setTheta(theta);

    double sScale = 0.0;
    double rScale = 0.0;

    for (int fld = 0; fld < nfolds; ++fld) {
      const arma::uword nnull = S[fld].n_cols;
      const arma::uword nxi = RTheta_w[fld].n_cols;

      if (nxi == 0) {
        throw std::runtime_error(
          "Gu theta.common scaling requires non-empty RTheta."
        );
      }

      // In SVDX the null-space dimension is determined by the formula and is
      // therefore common across folds, so return the relative theta unchanged
      // if S has no columns.
      if (nnull == 0) {
        return theta;
      }

      sScale += arma::accu(arma::square(S[fld])) /
        static_cast<double>(nnull);
      rScale += arma::accu(arma::square(RTheta_w[fld])) /
        static_cast<double>(nxi);
    }

    sScale /= static_cast<double>(nfolds);
    rScale /= static_cast<double>(nfolds);

    if (!(std::isfinite(sScale) && std::isfinite(rScale)) ||
        sScale <= 0.0 || rScale <= 0.0) {
      throw std::runtime_error(
        "Non-finite or non-positive scale in Gu theta.common initialization."
      );
    }

    const double commonScale = std::sqrt(sScale / rScale);
    arma::vec out = commonScale * theta;
    out = arma::clamp(out, LB, UB);
    return out;
  }

  /*
   * Fit every CV training fold once at the CURRENT K/f2/theta and return
   * the fitted h matrices. This deliberately does not alter hmat_w/umat_w/
   * vmat_w. Only an initialization probe.
   */
  std::vector<arma::mat> fitFoldHmatsAtCurrentState() {
    std::vector<arma::mat> fittedHmats(nfolds);
    arma::vec initMses(nfolds, arma::fill::value(FAILED_CV_SCORE));
    arma::vec initPens(nfolds, arma::fill::value(FAILED_CV_SCORE));

    SMLFCOnColsListTT<T> so(
      Y, columns1, Z, Zeta,
      MTheta_w, hmat_w, umat_w, vmat_w,
      RtildeTheta_w, qids1, BTheta_w,
      rowOffsets, colOffsets,
      graphDistMat, graphDistOnlyOnPoints,
      family, aphi,
      columns1Test, ZTest, MThetaTest_w,
      rowOffsetsTest, colOffsetsTest,
      K_w, r,
      f1vec_w, f2vec_w, f3vec_w_ptr,
      fitter_maxIter, fitter_ftol_abs, fitter_xtol_abs,
      fitter_inner_maxIter, fitter_inner_ftol_abs,
      &initMses, &initPens, &fittedHmats
    );

    if (estimateCVFnSerial) so.executeSerial();
    else so.execute();

    return fittedHmats;
  }

  /*
   * SVDX extension of Gu Algorithm 3.3:
   * theta_beta,0 proportional to thetaTilde_beta^2 times the average
   * RKHS penalty contrib across BOTH folds and fitted components.
   */
  arma::vec thetaSignalAdaptive(
      const arma::vec& thetaTilde,
      const std::vector<arma::mat>& fittedHmats) const {

    arma::vec out(nTheta, arma::fill::zeros);

    for (int fld = 0; fld < nfolds; ++fld) {
      const int qfld = qForFolds[fld];
      const arma::mat& H = fittedHmats[fld];
      if (H.n_rows < static_cast<arma::uword>(r + qfld) ||
          H.n_cols < static_cast<arma::uword>(K_w)) {
        throw std::runtime_error("Bad fitted hmat dimensions in Gu theta initialization");
      }

      const arma::mat C = H.rows(r, r + qfld - 1).cols(0, K_w - 1);
      int st = 0;
      for (unsigned int beta = 0; beta < nTheta; ++beta) {
        const int en = st + qfld - 1;
        arma::mat Rfold = RsCbound[fld].cols(st, en);
        const arma::mat Qbeta = Rfold.rows(qids1[fld]);

        double componentEnergy = 0.0;
        for (unsigned int k = 0; k < K_w; ++k) {
          componentEnergy += arma::as_scalar(C.col(k).t() * Qbeta * C.col(k));
        }
        componentEnergy /= static_cast<double>(K_w);
        out(beta) += componentEnergy;
        st = en + 1;
      }
    }

    out /= static_cast<double>(nfolds);
    out %= arma::square(thetaTilde);

    // theta is optimized on log10 scale later, so keep starts positive and
    // inside the same numerical box used by the theta optimizer.
    out = arma::clamp(out, LB, UB);
    return out;
  }

  /*
   * Optimizing engines
   */

  // K optimization over a grid
  void optK() {
    if( !this->KGridset ){
      throw std::invalid_argument( "K grid has not been initialized yet optK is being called. " );
    }
    if(this->outerGridSerial){
      KWp->executeSerial();
      //(*KWp)(0, this->KGrid.size());
    } else {
      KWp->execute(); //parallel
    }

    // Debug trace only. Record K-grid scores after the worker has completed.
    // Existing cvObjectiveTrace columns remain in their original positions;
    // trace_type and K-specific scores are appended.
    {
      const arma::uword oldWidth = 3 + theta_w.n_elem + mses.n_elem;
      arma::vec traceRow(oldWidth + 1 + KMAX, arma::fill::value(arma::datum::nan));
      traceRow(0) = static_cast<double>(++cvObjectiveEvalCounter);
      traceRow(1) = f2_w;
      traceRow.subvec(2, 1 + theta_w.n_elem) = theta_w;
      traceRow(oldWidth) = 1.0; // completed K-grid evaluation

      for (arma::uword i = 0; i < KGrid.n_elem; ++i) {
        const arma::uword kval = KGrid(i);
        if (kval >= 1 && kval <= static_cast<arma::uword>(KMAX)) {
          traceRow(oldWidth + kval) = KGrid_aveMses(i);
        }
      }
      cvObjectiveTrace.push_back(traceRow);
    }

    this->setK(static_cast<unsigned int>(KWp->getMinPoint()));
  }

  // f2 optimization over a grid
  void optf2(){
    if( !this->f2Gridset ){
      throw std::invalid_argument( "f2 grid has not been initialized yet optf2 is being called. " );
    }
    if( this->outerGridSerial ){
      F2Wp->executeSerial();
    } else {
      F2Wp->execute(); 
    }
    // Debug trace only. Record each completed f2-grid candidate here, after
    // f2Worker has finished. Do not print or mutate tracing from operator().
    {
      const arma::uword oldWidth = 3 + theta_w.n_elem + mses.n_elem;
      for (arma::uword i = 0; i < f2Grid.n_elem; ++i) {
        arma::vec traceRow(oldWidth + 1 + KMAX, arma::fill::value(arma::datum::nan));
        traceRow(0) = static_cast<double>(++cvObjectiveEvalCounter);
        traceRow(1) = f2Grid(i);
        traceRow.subvec(2, 1 + theta_w.n_elem) = theta_w;
        // Preserve the existing ordinary-trace layout: mean CV is the last
        // column before the appended trace_type column. Fold scores are NaN.
        traceRow(oldWidth - 1) = f2Grid_aveMses(i);
        traceRow(oldWidth) = 2.0; // completed f2-grid candidate
        cvObjectiveTrace.push_back(traceRow);
      }
    }

    this->setf2(F2Wp->getMinPoint()); //double here, no issue.
  }

  /*
   * theta and f3 univariate opt for SERIAL computations only!
   *    via nlopt with nlopt wrappers.
   *
   *    Did not implement them thread safe, as nlopt works serially on these functions.
   *    If future changes are made, must need to revisit this.
   */
  double thetaObjective(arma::vec& th){

    this->setTheta( th );
    return estimateCVScore();
  }
  double f3Objective( double& f3 ){
    this->setf3(f3);
    return estimateCVScore();
  }

  // NLOPT wrapper for arma::vec based objective
  static double nlopt_log10_thetaWrapper(unsigned n, const double* x, double* grad, void* data) {
    SMLFCOpt<T>* instance = static_cast<SMLFCOpt<T>*>(data);
    // Wrap the raw pointer 'x' into an arma::vec without copying
    arma::vec l10_current_x(const_cast<double*>(x), n, false, true);
    arma::vec current_x = arma::exp10( l10_current_x );

    return instance->thetaObjective(current_x);
  }
  // Wrapper for the double f3 objective
  static double nlopt_log10_f3Wrapper(unsigned n, const double* x, double* grad, void* data) {
    SMLFCOpt<T>* instance = static_cast<SMLFCOpt<T>*>(data);
    double current_f3 = std::pow(10,x[0]);
    return instance->f3Objective(current_f3);
  }


  /*
   * Replace the fold-specific non-family initializers once with generalized
   * fits at the supplied initial hyperparameters.
   *
   * This routine is called only after setK(KInit), setf2(f2Init),
   * setTheta(thetaInit), and (for graph models) setf3(f3Init).  Each fold is
   * fitted independently on its own training data, so no response-derived
   * information is shared across folds.
   *
   * The fit is performed at KMAX so that later K-grid evaluations have
   * generalized initializers available for every component.  Before fitting,
   * umat is made coherent with the initial theta by setting
   *
   *     umat = MTheta_w * hmat.
   *
   * The fitted Zeta/hmat/umat/vmat values replace the source initializer
   * members.  A subsequent setK(KInit) refreshes the working matrices from
   * these updated source values.
   */
  void initializeGeneralizedFolds(){

    if( family == nullptr ){
      return;
    }
    if( !(f2set && thetaset) ){
      throw std::invalid_argument(
        "initializeGeneralizedFolds requires f2 and theta to be set."
      );
    }
    if( graphDistMat != nullptr && !f3set ){
      throw std::invalid_argument(
        "initializeGeneralizedFolds requires f3 for graph models."
      );
    }

    for( int fld=0; fld<nfolds; fld++ ){

      arma::mat Zeta_f = Zeta[fld];
      arma::mat hmat_f = hmat[fld].cols(0, KMAX-1);
      arma::mat vmat_f = vmat[fld].cols(0, KMAX-1);

      // The non-family umat was constructed under a different theta.
      // Reconstruct it deterministically for this fold at thetaInit.
      arma::mat umat_f = MTheta_w[fld] * hmat_f;

      arma::vec aphi_f = (*aphi)[fld];
      if( graphDistMat == nullptr ){
        
        gFitterOnColumns<T> D(
          Y,
          columns1[fld],
          (*family),
          Z[fld],
          Zeta_f,
          MTheta_w[fld],
          hmat_f,
          umat_f,
          vmat_f,
          aphi_f,
          RtildeTheta_w[fld],
          qids1[fld],
          BTheta_w[fld],
          rowOffsets[fld],
          colOffsets[fld],
          KMAX,
          r
        );

        D.estimate(
          f1_w,
          f2_w,
          fitter_maxIter,
          fitter_ftol_abs,
          fitter_xtol_abs,
          fitter_inner_maxIter,
          fitter_inner_ftol_abs
        );

        Zeta[fld] = D.Zeta;
        hmat[fld] = D.hmat;
        umat[fld] = D.umat;
        vmat[fld] = D.vmat;

      } else {

        gFitterOnColumnsGraph<T> D(
          Y,
          columns1[fld],
          (*family),
          Z[fld],
          Zeta_f,
          MTheta_w[fld],
          hmat_f,
          umat_f,
          vmat_f,
          aphi_f,
          RtildeTheta_w[fld],
          (*graphDistMat)[fld],
          (*graphDistOnlyOnPoints)[fld],
          qids1[fld],
          BTheta_w[fld],
          rowOffsets[fld],
          colOffsets[fld],
          KMAX,
          r
        );

        D.estimate(
          f1_w,
          f2_w,
          f3_w,
          fitter_maxIter,
          fitter_ftol_abs,
          fitter_xtol_abs,
          fitter_inner_maxIter,
          fitter_inner_ftol_abs
        );

        Zeta[fld] = D.Zeta;
        hmat[fld] = D.hmat;
        umat[fld] = D.umat;
        vmat[fld] = D.vmat;
      }
    }
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
                ){

    Rcpp::Rcout<< "=======================" << std::endl;
    Rcpp::Rcout << "Init, K: " << KInit << std::endl;
    Rcpp::Rcout <<"Init, f2: "<< f2Init << std::endl;
    Rcpp::Rcout << "Init, theta:" << std::endl ; Rcpp::Rcout << thetaInit << std::endl;
    Rcpp::Rcout<< "=======================" << std::endl;

    //initialize
    cvObjectiveTrace.clear();
    cvObjectiveEvalCounter = 0;
    setK( KInit );
    setf2( f2Init );
    setTheta(thetaInit);
    if( f3Init!=nullptr ){
      setf3(*f3Init);
    }

    // For family models, pay the non-family-to-generalized initialization
    // cost once per fold at the supplied initial hyperparameters.
    if( family != nullptr ){
      Rcpp::Rcout << "Initializing folds with gfitter ..." << KInit << std::endl;
      initializeGeneralizedFolds();
      Rcpp::Rcout << "done ..." << KInit << std::endl;

      // The source hmat/umat/vmat values were replaced by generalized fits.
      // Refresh the existing working state at the originally requested K.
      setK( KInit );
    }

    /*
     * Sanity checks for internal state
     */
    if( optForK ){
      if( !KGridset ){
        this->KGrid = arma::regspace<uvec>(1,10);
        this->setKGrid(  KGrid );
        //throw std::invalid_argument( "K grid has not been set. " );
      }
    } else {
      if( !Kset ){
        throw std::invalid_argument( "K has not been set. " );
      }
    }
    if( optForf2 ){
      if( !f2Gridset ){
        //throw std::invalid_argument( "f2 grid has not been set. " );
        this->f2Grid = arma::exp10(arma::regspace( -7,0))  ;
        this->setf2Grid( f2Grid );
      }
    } else {
      if( !f2set ){
        throw std::invalid_argument( "f2 has not been set. " );
      }
    }
    if( (!optFortheta) & (!thetaset) )
    {
      throw std::invalid_argument( "theta has not been set, and no optimization is asked over it. " );
    }

    if( (graphDistMat!=nullptr) ){
      if( (!optForf3) & (!f3set) )
      {
        throw std::invalid_argument( "f3 has not been set, and no optimization is asked over it. " );
      }
    }


    ///////////////////////////////////////////////////////////////////////////
    /*
     * Gu-style initialization for the single-dataset reduced-basis problem.
     * Only applies when BOTH f2 and theta are being estimated.  K is left at
     * its current initialization (currently KInit == KMAX in svdxC).
     *
     *  1. thetaTilde_beta = 1 / mean_fold trace(Q_beta,fold)
     *  2. optimize f2 once at thetaTilde
     *  3. fit folds once and form signal-adaptive theta0
     *  4. optimize f2 once at theta0; thereafter f2 is frozen
     */
    // SVDX-specific computational adaptation: before the expensive Gu
    // initialization, do one provisional K-grid optimization at the current
    // incoming hyperparameter state. K remains free in the main loop below.
    if (optForK) {
      this->optK();
    }

    bool f2FrozenAfterGuInit = false;
    if (optForf2 && optFortheta) {
      arma::vec thetaTilde = thetaTraceBalanced();
      thetaTilde = thetaCommonScaled(thetaTilde);
      setTheta(thetaTilde);

      optf2();

      const std::vector<arma::mat> preliminaryHmats =
        fitFoldHmatsAtCurrentState();
      arma::vec theta0 = thetaSignalAdaptive(thetaTilde, preliminaryHmats);
      theta0 = thetaCommonScaled(theta0);
      setTheta(theta0);

      optf2();
      f2FrozenAfterGuInit = true;
    }

    ///////////////////////////////////////////////////////////////////////////
    /*
     * Optimize
     */
    ///////////////////////////////////////////////////////////////////////////

    // SETUP
    double cur_K = this->K_w;
    double cur_f2=this->f2_w;
    arma::vec cur_theta = this->theta_w.clamp( LB, UB ); //cur theta is in the usual scale. so use LB, UB
    double cur_f3 = std::max(LB, std::min(this->f3_w, UB));

    unsigned int old_K = cur_K;
    double old_f2 = cur_f2;
    arma::vec old_theta = cur_theta;
    double old_f3 = cur_f3;

    std::vector<double> cur_l10_theta;
    if(optFortheta){
      cur_l10_theta = arma::conv_to<std::vector<double> >::from( arma::log10(cur_theta) );
    }
    nlopt::opt thetaOpter(nlopt::algorithm::LN_BOBYQA, this->nTheta); //define algorithm
    thetaOpter.set_min_objective(SMLFCOpt::nlopt_log10_thetaWrapper, this); // Set the objective function with the generic lambda and the function wrapper
    arma::vec thLengthOnes(nTheta, arma::fill::ones);
    std::vector<double> l10Thetalb = arma::conv_to< std::vector<double> >::from( -7*thLengthOnes );
    std::vector<double> l10Thetaub = arma::conv_to< std::vector<double> >::from( 7*thLengthOnes );
    thetaOpter.set_lower_bounds( l10Thetalb  ); //10^-7 as lower bound for theta
    thetaOpter.set_upper_bounds( l10Thetaub  ); //10^7 as upper bound
    thetaOpter.set_xtol_abs(nlopt_xtol_abs);
    thetaOpter.set_ftol_abs(nlopt_ftol_abs);
    
    //for JSM!
    if (nlopt_maxeval > 0) thetaOpter.set_maxeval(nlopt_maxeval);

    std::vector<double> cur_l10_f3;
    if(optForf3){
      cur_l10_f3 = std::vector<double>(1,std::log10(this->f3_w));
    }
    nlopt::opt f3Opter(nlopt::algorithm::LN_BOBYQA, 1); //define algorithm
    f3Opter.set_min_objective(SMLFCOpt::nlopt_log10_f3Wrapper, this); // Set the objective function with the generic lambda and the function wrapper
    std::vector<double> l10f3lb = {log10LB}; std::vector<double> l10f3ub = {log10UB};
    f3Opter.set_lower_bounds( l10f3lb  ); //10^-7 as lower bound for theta
    f3Opter.set_upper_bounds( l10f3ub  ); //10^-7 as lower bound for theta
    thetaOpter.set_upper_bounds( arma::conv_to< std::vector<double> >::from( 7*thLengthOnes )  ); //10^7 as upper bound
    f3Opter.set_xtol_abs(nlopt_xtol_abs);
    f3Opter.set_ftol_abs(nlopt_ftol_abs);
    if (nlopt_maxeval > 0) f3Opter.set_maxeval(nlopt_maxeval);

    double minFnValue=0, delta=0; 
    int ndf=0,iter=0;
    double eps = optCV_xtol_abs + 1;
    double epsFnValue = optCV_ftol_abs + 1; 
    double oldFnValue = minFnValue + epsFnValue;

    //RUN
    while( (iter<optCV_maxIter) && (eps > optCV_xtol_abs) && ( epsFnValue > optCV_ftol_abs) ){

      iter = iter+1; delta = 0; ndf=0;
      Rcpp::Rcout << "Iteration: " << iter << std::endl;
      old_K = cur_K;
      old_f2 = cur_f2;
      old_theta = cur_theta;
      old_f3 = cur_f3;
      oldFnValue = minFnValue; 

      if(optForK){
        Rcpp::Rcout << " + Optimizing for K .. ";

        this->optK();
        cur_K = this->K_w;
        delta += std::pow(std::log10(cur_K)-std::log10(old_K),2);ndf +=1;

        Rcpp::Rcout << " + + K star: " << cur_K << std::endl;
      }


      if(optForf2 && !f2FrozenAfterGuInit){
        Rcpp::Rcout << " + Optimizing for f2 ... ";

        this->optf2();
        cur_f2 = this->f2_w;
        delta += std::pow( std::log10(cur_f2) - std::log10(old_f2), 2 );ndf +=1;

        Rcpp::Rcout << "+ + f2 star: " << cur_f2 << std::endl;
      }


      if( optFortheta ){
        Rcpp::Rcout << " + Optimizing for theta ... " << std::endl;

        cur_l10_theta = arma::conv_to<std::vector<double> >::from( arma::log10(cur_theta) );
        
        nlopt::result result;
        try {
          result =
            thetaOpter.optimize(cur_l10_theta, minFnValue);
          
          Rcpp::Rcout << "theta result = "
                      << static_cast<int>(result)
                      << "\nminFnValue = "
                      << minFnValue
                      << std::endl;
        }
        catch (const std::exception& e) {
          Rcpp::Rcout << "theta NLopt exception: "
                      << e.what()
                      << std::endl;
          throw;
        }
        
        
        

        cur_theta = arma::exp10( arma::clamp(arma::vec( cur_l10_theta ), log10LB, log10UB) ); 
        this->setTheta( cur_theta  ); // we need this because nlopt need not return the last point evaluated (our class internal theta state would reflect the last point evaluated.)
        delta += sum( arma::pow( arma::log10(cur_theta) - arma::log10(old_theta) , 2 ) )  ;ndf +=nTheta;

        Rcpp::Rcout << "+ + theta star: " << std::endl; Rcpp::Rcout << cur_theta << std::endl;
      }

      if( optForf3 ){
        Rcpp::Rcout << " + Optimizing for f3 ... " << std::endl;

        cur_l10_f3[0] = std::log10(cur_f3);
        nlopt::result result = f3Opter.optimize( cur_l10_f3, minFnValue );

        cur_l10_f3[0] = std::max( log10LB, std::min( cur_l10_f3[0], log10UB ) );
        cur_f3 = std::pow(10, cur_l10_f3[0] ); 
        this->setf3( cur_f3 ); // we need this because nlopt need not return the last point evaluated (our class internal f3 state would reflect the last point evaluated.)
        delta += std::pow( std::log10(cur_f3) - std::log10(old_f3), 2 ); ndf +=1;

        Rcpp::Rcout << "+ + f3 star: " << cur_f3 << std::endl;
      }
      
      //objective read out, nlopt returns it, and we have called nlopt last. 
      Rcpp::Rcout << "Current objective : " << minFnValue << std::endl;
      
      // convergence
      eps = sqrt(delta/std::max(ndf,1));
      if(iter > 1){ //needed because oldFnValue is sensibly defined currently only after a first pass. 
        epsFnValue = std::abs(minFnValue - oldFnValue);
      }
    }

    if( (eps > optCV_xtol_abs) && ( epsFnValue > optCV_ftol_abs)  ){
      std::cout << " OptCV has not converged. Details: " << std::endl;
      std::cout << "eps: " + std::to_string(eps) << std::endl;
      std::cout << "epsFn: " + std::to_string(epsFnValue) << std::endl;
      std::cout << "Increase optCV_maxIter?" << std::endl;
    }

  }

};


template<typename T>
constexpr double SMLFCOpt<T>::FAILED_CV_SCORE;

// ============================================================================
// VIRTUAL DESTRUCTOR INSTANTIATIONS (FIXES LINKER LOADING ERROR)
// ============================================================================
template<typename T>
SMLFCOpt<T>::KWorker::~KWorker() {}

template<typename T>
SMLFCOpt<T>::f2Worker::~f2Worker() {}


// Now your explicit class template instantiations are perfectly safe:
template class SMLFCOpt<arma::mat>;
template class SMLFCOpt<arma::sp_mat>;




#endif
