

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include "IMLFCOnColsListTT.h"

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
IMLFCOnColsListTT::IMLFCOnColsListTT( const dsListC& Y,
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
                                      std::vector<arma::mat>* fittedHmats

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
wd(wd),
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
  if( K != vmat[0][0].n_cols ){
    throw std::invalid_argument( "Bad vmat / K: " + std::to_string(vmat[0][0].n_cols) + " / " + std::to_string( K )  );
  }

}

void IMLFCOnColsListTT::operator()( std::size_t begin, std::size_t end ) {

  for( std::size_t w=begin; w < end; w++   ){

    // We likely don't need copies of all. Just those that are altered.
    //  These are only: Zeta, hmat, umat, vmat,
    //  TODO: check this.


    //Train
    arma::uvec columnsw = columns1[w];
    arma::uvec qidsw=qids1[w];
    std::vector<arma::mat> Zw = Z[w];
    std::vector<arma::mat> Zetaw = Zeta[w];
    arma::mat Mw = M[w];
    arma::mat hmatw = hmat[w];
    arma::mat umatw = umat[w];
    std::vector<arma::mat> vmatw = vmat[w];
    arma::mat Rtildew = Rtilde[w];
    arma::mat Bw = B[w];
    std::vector<arma::vec> rowOffsetsw = rowOffsets[w];
    std::vector<arma::vec> colOffsetsw = colOffsets[w];
    double f1w=f1(w);
    double f2w=f2(w);

    //Evaluate the mse (not the objective) on Test
    arma::uvec columnswTest = columns1Test[w];
    std::vector<arma::mat> ZwTest = ZTest[w];
    arma::mat MwTest = MTest[w];
    std::vector<arma::vec> rowOffsetswTest = rowOffsetsTest[w];
    std::vector<arma::vec> colOffsetswTest = colOffsetsTest[w];


    //using redundant code of estimate, mses and pens assignment to just avoid vtable lookup on D within operator()
    // see comments below.
    if( dispatch == 0 ){
      inpFitterOnColumns D( Y, columnsw,
                            Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                            Rtildew,
                            qidsw,
                            Bw,
                            rowOffsetsw,
                            colOffsetsw,
                            K,
                            r, 
                            wd//,
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
      inpFitterOnColumnsGraph D( Y, columnsw,
                                 Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                                 Rtildew,
                                 (*graphDistMat)[w], //this way of calling doesn't call the copy constructor, just bids the reference
                                                (*graphDistOnlyOnPoints)[w],
                                                                        qidsw,
                                                                        Bw,
                                                                        rowOffsetsw,
                                                                        colOffsetsw,
                                                                        K,
                                                                        r, 
                                                                        wd//,
                                                                        //Q
      );
      D.estimate(f1w, f2w, (*f3)[w], maxIter, eps, xtol_abs);
      if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
      (*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
      (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.

    } else if(dispatch == 2){
#if 0
      // Previous dispatch == 2 implementation retained for reference.
      igFitterOnColumns D( Y, columnsw,
                           (*family) ,
                           Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                           (*aphi)[w],
                                  Rtildew,
                                  qidsw,
                                  Bw,
                                  rowOffsetsw,
                                  colOffsetsw,
                                  K,
                                  r, 
                                  wd//,
                                  //Q
      );
      D.estimate(f1w, f2w, maxIter, eps, xtol_abs);
      if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
      //found the deviance to be unstable, even for gaussian, TODO: probe more
      (*mses)(w) = D.devianceForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
      //(*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
      (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.
#endif
      try {
        igFitterOnColumns D( Y, columnsw,
                             (*family) ,
                             Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                             (*aphi)[w],
                                    Rtildew,
                                    qidsw,
                                    Bw,
                                    rowOffsetsw,
                                    colOffsetsw,
                                    K,
                                    r, 
                                    wd//,
                                    //Q
        );
        D.estimate(f1w, f2w, maxIter, eps, xtol_abs, inner_maxIter, inner_ftol_abs);
        if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();

        const double foldScore =
          D.devianceForTest( columnswTest, ZwTest, MwTest,
                             rowOffsetswTest, colOffsetswTest );
        const double foldPenalty = D.getPenalty();

        (*mses)(w) = std::isfinite(foldScore)
          ? foldScore
          : FAILED_CV_SCORE;
        (*pens)(w) = std::isfinite(foldPenalty)
          ? foldPenalty
          : FAILED_CV_SCORE;
      } catch (const std::exception&) {
        (*mses)(w) = FAILED_CV_SCORE;
        (*pens)(w) = FAILED_CV_SCORE;
      }
    } else if( dispatch == 3) {
#if 0
      // Previous dispatch == 3 implementation retained for reference.
      igFitterOnColumnsGraph D( Y, columnsw,
                                (*family),
                                Zw, Zetaw, Mw, hmatw, umatw, vmatw,
                                (*aphi)[w],
                                       Rtildew,
                                       (*graphDistMat)[w], //this way of calling doesn't call the copy constructor, just bids the reference
                                                      (*graphDistOnlyOnPoints)[w],
                                                                              qidsw,
                                                                              Bw,
                                                                              rowOffsetsw,
                                                                              colOffsetsw,
                                                                              K,
                                                                              r, 
                                                                              wd//,
                                                                              //Q
      );
      D.estimate(f1w, f2w, (*f3)[w], maxIter, eps, xtol_abs);
      if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();
      //found the deviance to be unstable, even for gaussian, TODO: probe more
      (*mses)(w) = D.devianceForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
      //(*mses)(w) = D.mseForTest( columnswTest, ZwTest, MwTest, rowOffsetswTest, colOffsetswTest  ) ;//mse and sum of penalty values
      (*pens)(w) = D.getPenalty(); //sum of penalties for each component built.
#endif
      try {
        igFitterOnColumnsGraph D( Y, columnsw,
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
                                         r, 
                                         wd//,
                                         //Q
        );
        D.estimate(f1w, f2w, (*f3)[w], maxIter, eps, xtol_abs, inner_maxIter, inner_ftol_abs);
        if (fittedHmats != nullptr) (*fittedHmats)[w] = D.get_hmat();

        const double foldScore =
          D.devianceForTest( columnswTest, ZwTest, MwTest,
                             rowOffsetswTest, colOffsetswTest );
        const double foldPenalty = D.getPenalty();

        (*mses)(w) = std::isfinite(foldScore)
          ? foldScore
          : FAILED_CV_SCORE;
        (*pens)(w) = std::isfinite(foldPenalty)
          ? foldPenalty
          : FAILED_CV_SCORE;
      } catch (const std::exception&) {
        (*mses)(w) = FAILED_CV_SCORE;
        (*pens)(w) = FAILED_CV_SCORE;
      }


    } else {
      throw std::invalid_argument("undefined dispatch specification. ");
    }

  }
}

void IMLFCOnColsListTT::execute(){
  RcppParallel::parallelFor( 0, columns1.size(), *(this)   );
}

void IMLFCOnColsListTT::executeSerial(){
  (*this)( 0, columns1.size());
}




