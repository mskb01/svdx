#ifndef SMLFCONCOLSLIST_H
#define SMLFCONCOLSLIST_H


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

#include "npFitterOnColumnsGraph.h"
#include "gFitterOnColumnsGraph.h"

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
class SMLFCOnColsListTT : public Worker{ //npSMLFC = nonparametric Single Matrix Loss Function Calculator (only the L() in  f1*L() + f2*J(),  not the entire objective) on columns for train and test data


  //All lists below mst be of the same length.
  //mses should have the same length.
  
  static constexpr double FAILED_CV_SCORE = 1e100;
  
  const T& Y;
  const std::vector<arma::uvec>& columns1;
  const std::vector<arma::mat>& Z;       //Z may or may not be present
  const std::vector<arma::mat>& Zeta;
  const std::vector<arma::mat>& M;
  const std::vector<arma::mat>& hmat;
  const std::vector<arma::mat>& umat;
  const std::vector<arma::mat>& vmat;
  const std::vector<arma::mat>& Rtilde; // qxq penalty
  const std::vector<arma::uvec>& qids1;
  const std::vector<arma::mat>& B;      //qxq full kernel inverse
  const std::vector<arma::vec>& rowOffsets;
  const std::vector<arma::vec>& colOffsets;

  const std::vector<arma::uvec>& columns1Test;
  const std::vector<arma::mat>& ZTest;       //Z may or may not be present
  const std::vector<arma::mat>& MTest;
  const std::vector<arma::vec>& rowOffsetsTest;
  const std::vector<arma::vec>& colOffsetsTest;

  const int& K;
  const int& r;
  const arma::vec& f1;
  const arma::vec& f2;
  const int& maxIter;
  const double& eps;
  const double& xtol_abs;
  const int& inner_maxIter;
  const double& inner_ftol_abs;

  arma::vec* mses; //size must equal the length of list columns1Test i.e., samples for test data
  arma::vec* pens; //ditto; stores sum of penalty values for each component


  //graph specific additions
  const std::vector<arma::sp_mat>* graphDistMat = nullptr; // qxq penalty
  const std::vector<arma::uvec>* graphDistOnlyOnPoints = nullptr;
  const arma::vec* f3 = nullptr;

  //family for generalized
  const std::string* family = nullptr;
  const std::vector<arma::vec>* aphi=nullptr;

  int dispatch; // 0 if npfitter, 1 if npfittergraph, 2 if gfitter, 3 if gfittergraph
  std::vector<arma::mat>* fittedHmats = nullptr; // optional fitted hmat capture, one per fold

public:

  SMLFCOnColsListTT( const T& Y,
                     const std::vector<arma::uvec>& columns1,
                     const std::vector<arma::mat>& Z,       //Z may or may not be present
                     const std::vector<arma::mat>& Zeta,
                     const std::vector<arma::mat>& M,
                     const std::vector<arma::mat>& hmat,
                     const std::vector<arma::mat>& umat,
                     const std::vector<arma::mat>& vmat,
                     const std::vector<arma::mat>& Rtilde, // qxq penalty
                     const std::vector<arma::uvec>& qids1, //qids must be a subset of columns1 for each case to avoid leakage into test !
                     const std::vector<arma::mat>& B,      //qxq full kernel inverse
                     const std::vector<arma::vec>& rowOffsets,
                     const std::vector<arma::vec>& colOffsets,

                     const std::vector<arma::sp_mat>* graphDistMat, // qxq graph distances post semi def at least
                     const std::vector<arma::uvec>* graphDistOnlyOnPoints, // qxq graph distances post semi def at least

                     const std::string* family,
                     const std::vector<arma::vec>* aphi,

                     const std::vector<arma::uvec>& columns1Test,
                     const std::vector<arma::mat>& ZTest,      //Z may or may not be present
                     const std::vector<arma::mat>& MTest,
                     const std::vector<arma::vec>& rowOffsetsTest,
                     const std::vector<arma::vec>& colOffsetsTest,

                     const int& K,
                     const int& r,
                     arma::vec& f1,
                     arma::vec& f2,
                     arma::vec* f3,

                     const int& maxIter,
                     const double& eps,
                     const double& xtol_abs,
                     const int& inner_maxIter,
                     const double& inner_ftol_abs,
                     arma::vec* mses,
                     arma::vec* pens,
                     std::vector<arma::mat>* fittedHmats = nullptr

  ): Y(Y), columns1(columns1), Z(Z), Zeta(Zeta), M(M), hmat(hmat), umat(umat),
  vmat(vmat),
  Rtilde(Rtilde), qids1(qids1), B(B),
  rowOffsets(rowOffsets), colOffsets(colOffsets),
  graphDistMat(graphDistMat), graphDistOnlyOnPoints(graphDistOnlyOnPoints),
  family(family), aphi(aphi),
  columns1Test(columns1Test), ZTest(ZTest), MTest(MTest),
  rowOffsetsTest(rowOffsetsTest), colOffsetsTest(colOffsetsTest),
  K(K), r(r),
  f1(f1), f2(f2), f3(f3),
  maxIter(maxIter), eps(eps), xtol_abs(xtol_abs), inner_maxIter(inner_maxIter), inner_ftol_abs(inner_ftol_abs),
  mses(mses), pens(pens), fittedHmats(fittedHmats) {

    if( graphDistMat != nullptr ){
      if( family != nullptr ){
        this->dispatch = 3;
      } else {
        this->dispatch = 1;
      }

    } else {
      if( family !=nullptr ){
        this->dispatch=2;
      } else {
        this->dispatch=0;
      }

    }

    if( K != hmat[0].n_cols ){
      throw std::invalid_argument( "Bad hmat / K: " + std::to_string(hmat[0].n_cols) + " / " + std::to_string( K )  );
    }
    if( K != umat[0].n_cols ){
      throw std::invalid_argument( "Bad umat / K: " + std::to_string(umat[0].n_cols) + " / " + std::to_string( K )  );
    }
    if( K != vmat[0].n_cols ){
      throw std::invalid_argument( "Bad vmat / K: " + std::to_string(vmat[0].n_cols) + " / " + std::to_string( K )  );
    }

  };

  void operator()( std::size_t begin, std::size_t end ) {

    for( std::size_t w=begin; w < end; w++   ){


      //Train
      arma::uvec columnsw = columns1[w];
      arma::uvec qidsw=qids1[w];
      arma::mat Zw = Z[w];
      arma::mat Zetaw = Zeta[w];
      arma::mat Mw = M[w];
      arma::mat hmatw = hmat[w];
      arma::mat umatw = umat[w];
      arma::mat vmatw = vmat[w];
      arma::mat Rtildew = Rtilde[w];
      arma::mat Bw = B[w];
      arma::vec rowOffsetsw = rowOffsets[w];
      arma::vec colOffsetsw = colOffsets[w];
      double f1w=f1(w);
      double f2w=f2(w);

      //Evaluate the mse (not the objective) on Test
      arma::uvec columnswTest = columns1Test[w];
      arma::mat ZwTest = ZTest[w];
      arma::mat MwTest = MTest[w];
      arma::vec rowOffsetswTest = rowOffsetsTest[w];
      arma::vec colOffsetswTest = colOffsetsTest[w];


      //using redundant code of estimate, mses and pens assignment to just avoid vtable lookup on D within operator()
      // see comments below.
      if( dispatch == 0 ){
        npFitterOnColumns<T> D( Y, columnsw,
                                Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                                Rtildew,
                                qidsw,
                                Bw,
                                rowOffsetsw,
                                colOffsetsw,
                                K,
                                r//,
                                //Q
        );
        D.estimate(f1w, f2w, maxIter, eps, xtol_abs);
        if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
        (*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
        //mses(w) = D.objectiveForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;
        (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.

      } else if( dispatch == 1  ) {
        // this way of calling will invoke the copy constructor.
        // arma::sp_mat graphDistMatw = (*graphDistMat)[w];
        // arma::uvec graphDistOnlyOnPointsw = (*graphDistOnlyOnPoints)[w];
        npFitterOnColumnsGraph<T> D( Y, columnsw,
                                     Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                                     Rtildew,
                                     (*graphDistMat)[w], //this way of calling doesn't call the copy constructor, just bids the reference
                                                    (*graphDistOnlyOnPoints)[w],
                                                                            qidsw,
                                                                            Bw,
                                                                            rowOffsetsw,
                                                                            colOffsetsw,
                                                                            K,
                                                                            r//,
                                                                            //Q
        );
        D.estimate(f1w, f2w, (*f3)[w], maxIter, eps, xtol_abs);
        if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
        (*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
        (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.

      } else if(dispatch == 2){
        // gFitterOnColumns<T> D( Y, columnsw,
        //                        (*family) ,
        //                                 Zw, Zetaw, Mw, hmatw, umatw, vmatw,
        //                                 (*aphi)[w],
        //                                        Rtildew,
        //                                        qidsw,
        //                                        Bw,
        //                                        rowOffsetsw,
        //                                        colOffsetsw,
        //                                        K,
        //                                        r//,
        //                                        //Q
        // );
        // D.estimate(f1w, f2w, maxIter, eps, xtol_abs);
        // (*mses)(w) = D.devianceForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
        // //(*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
        // (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.
        // 
        
        try {
          gFitterOnColumns<T> D( Y, columnsw,
                                 (*family) ,
                                 Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                                 (*aphi)[w],
                                        Rtildew,
                                        qidsw,
                                        Bw,
                                        rowOffsetsw,
                                        colOffsetsw,
                                        K,
                                        r//,
                                        //Q
          );
          
          D.estimate(f1w, f2w, maxIter, eps, xtol_abs, inner_maxIter, inner_ftol_abs);
          if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
          
          const double foldScore = D.devianceForTest(
            columnswTest,
            ZwTest,
            MwTest,
            rowOffsetswTest,
            colOffsetswTest
          );
          
          const double foldPenalty = D.getPenalty();
          
          (*mses)(w) =
            std::isfinite(foldScore)
            ? foldScore
          : FAILED_CV_SCORE;
          
          (*pens)(w) =
            std::isfinite(foldPenalty)
            ? foldPenalty
          : FAILED_CV_SCORE;
          
        } catch (const std::exception& e) {
          (*mses)(w) = FAILED_CV_SCORE;
          (*pens)(w) = FAILED_CV_SCORE;
        }
        
        
      } else if( dispatch == 3) {
        
        try {
          gFitterOnColumnsGraph<T> D(
              Y, columnsw,
              (*family),
              Zw, Zetaw, Mw, hmatw, umatw, vmatw,
              (*aphi)[w],
                     Rtildew,
                     (*graphDistMat)[w],
                                    (*graphDistOnlyOnPoints)[w],
                                                            qidsw,
                                                            Bw,
                                                            rowOffsetsw,
                                                            colOffsetsw,
                                                            K,
                                                            r//,
                                                            //Q
          );
          
          D.estimate(f1w, f2w, (*f3)[w], maxIter, eps, xtol_abs, inner_maxIter, inner_ftol_abs);
          if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
          
          const double foldScore = D.devianceForTest(
            columnswTest,
            ZwTest,
            MwTest,
            rowOffsetswTest,
            colOffsetswTest
          );
          
          const double foldPenalty = D.getPenalty();
          
          (*mses)(w) =
            std::isfinite(foldScore)
            ? foldScore
          : FAILED_CV_SCORE;
          
          (*pens)(w) =
            std::isfinite(foldPenalty)
            ? foldPenalty
          : FAILED_CV_SCORE;
          
        } catch (const std::exception& e) {
          (*mses)(w) = FAILED_CV_SCORE;
          (*pens)(w) = FAILED_CV_SCORE;
        }
        
       
      } else {
        throw std::invalid_argument("undefined dispatch specification. ");
      }


     

    }
  }

  void execute(){
    RcppParallel::parallelFor( 0, columns1.size(), *(this)   );
  }
  void executeSerial(){ // We added this because we saw despite controlling the total number of threads, 
                        //.  in tbb & RcppParallel via setNumThreads / global_control / task_arena, 
                        //.    we still saw whenever optf2() or optK() was called, the number of threads shoot up leading to ~1000% CPU usage for 6 threads. 
                        //.   This meant it was not just idle threads. 
                        //.   Both optf2 and optK are nested parallels. They call SMLFCOptWorkerTT in parallel over grid. 
                        //.   So we now call SMLFCOptWorkerTT serially in optf2 and optK , but parallelize over their respective grids. 
    (*this)(0, columns1.size());
  }

};

template class SMLFCOnColsListTT<arma::mat>;
template class SMLFCOnColsListTT<arma::sp_mat>;




#endif
