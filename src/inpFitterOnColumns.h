#ifndef INPFITTERONCOLUMNS_H
#define INPFITTERONCOLUMNS_H


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

#include "myutils.h"
#include "dsListC.h"
#include "iparallelWorkersTemplates.h"
#include "estimatePar.h"

//For dense
class inpFitterOnColumns {

  
public:

  const dsListC& Y;
  const arma::uvec& columns; //these are the only columns that will be used for fitting the data. Useful for CV, instead of making copies of Y!

  //these two get updated based on
  //  whatever last f1, f2 pair the
  //  estimate function was called with.
  // This is useful to know because the umat, vmat, hmat etc.,
  // only make sense with these hyperparameter settings.
  double last_f1_estimatedWith;
  double last_f2_estimatedWith;

  const std::vector<arma::mat> Z;       //Z may or may not be present; can be a column vector of 1 simply
  const arma::mat& M;       //M=[S; R] , S is nxr R is nxq entering in Sd + Rc = Mh
  const arma::uvec& qids;         //we assume qids -> qids-1 i.e., R to C indexing conversion has happened before
  const arma::mat& Rtilde; //qxq penalty = Rtheta[qids,]
  const arma::mat& B;      //qxq full kernel inverse
  const arma::mat Q;       //any space to be orthogonal to; not making this as a reference because then it is harder to set up the situation without an initial Q (that constructor will need to initialize Q as well)
  const std::vector<arma::vec> rowOffsets;
  const std::vector<arma::vec> colOffsets;
  std::vector<arma::vec> colOffsets_for_columns; //extracts and stores colOffsets for desired columns

  int D;
  const int K;
  const int r;
  int q;
  int n;
  std::vector<int> pd;
  const arma::vec& wd; 
  int spd;

  std::vector<arma::mat> Zeta;
  arma::mat hmat;    //umat = M hmat
  arma::mat umat;
  std::vector<arma::mat> vmat;

  arma::mat A; //holds M'M
  arma::mat Aex; //expanded form of A matrix that incorporates +f2 c'Rc in the right indices, and the orthogonal constraints.
  arma::mat Aex_pinv;
  arma::mat Aex_pinv_rPlusq_submat; //submat of Aex_pinv
  std::vector<arma::mat> Zt;
  arma::mat Mt;
  std::vector<arma::mat> ZtZ;
  std::vector<arma::mat> Zls;
  std::vector<arma::mat> YZlst;
  std::vector<arma::mat> ZtYtilde; // Z_d' times offset-adjusted Y_d' for cheap objective-change bookkeeping
  arma::vec w_hat;
  std::vector<arma::vec> v_score; // unnormalised scores from the most recent v update, one per dataset
  arma::uvec seqOfComponents;
  arma::uvec qindx;
  arma::uvec rqindx;

  //whether to fit or condition on zeta
  // fixing useful for structure discovery
  const bool fitZeta;

  //lf if needed
  arma::vec lf_swise;

  //some ones for offsets
  arma::vec ones_n; //ones of length coloffsets respectively
  std::vector<arma::vec> ones_pd; //ones of length pd for each d for row offstes

  //usefulf or estimation
  inline void move_Zeta(arma::vec& u_k, std::vector<arma::vec>& v_k);             //for k=0 component, for all d
  inline void move_Zeta(arma::vec& u_k, std::vector<arma::vec>& v_k,              // for k >0 components,f or all d
                        arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers);

  void move_v_k(arma::vec& uk, std::vector<arma::vec>& vk); //for k=0
  void move_v_k(arma::vec& uk, std::vector<arma::vec>& vk,
                arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers); //for k>0


  inline void populate_w_hat_for_k(std::vector<arma::vec>& vk); //for k = 0
  inline void populate_w_hat_for_k(std::vector<arma::vec>& vk,
                                   arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers); //for k > 0

  void append_to_Aex_and_update_Aexpinv( const arma::vec& u );
  void append_to_Aex_and_update_Aexpinv( const arma::mat& Q1 );
  void estimate_k0(int maxIter=25, double eps=1e-4); //for k=0, the first component
  void estimate_k(int k, arma::mat& umatOthers,
                  std::vector<arma::mat>& vmatOthers,
                  int maxIter=25, double eps=1e-4); //for k>0
  inline void move_h_k(arma::vec& hk, arma::vec& uk, std::vector<arma::vec>& vk);
  inline void move_h_k(arma::vec& hk, arma::vec& uk, std::vector<arma::vec>& vk,
                       arma::mat& umatOthers, std::vector<arma::mat>& vmatOthers);


  /*
   * KEY FUNCTIONS
   */


  //Constructors
  explicit inpFitterOnColumns(const dsListC& Y,   //without Q, orthogonal space to avoid
                               const arma::uvec& columns,
                     const std::vector<arma::mat>& Z,
                     const std::vector<arma::mat> Zeta,
                     const arma::mat& M, arma::mat hmat, // our hmat will be a copy, we will use the same memory as output. Otherwise the R side input gets screwed up too! So would umat, and vmat  that we will generate.
                     arma::mat umat, std::vector<arma::mat> vmat,
                     const arma::mat& Rtilde,
                     const arma::uvec& qids,
                     const arma::mat& B,
                     const std::vector<arma::vec> rowOffsets,
                     const std::vector<arma::vec> colOffsets,
                     const int K,
                     const int r,
                     const arma::vec& wd, 
                     const bool fitZeta=true
  );

  explicit inpFitterOnColumns(const dsListC& Y,   //without Q, orthogonal space to avoid
                               const arma::uvec& columns,
                     const std::vector<arma::mat>& Z,
                     const std::vector<arma::mat> Zeta,
                     const arma::mat& M, arma::mat hmat,
                     arma::mat umat, std::vector<arma::mat> vmat,
                     const arma::mat& Rtilde,
                     const arma::uvec& qids,
                     const arma::mat& B,
                     const std::vector<arma::vec>& rowOffsets,
                     const std::vector<arma::vec>& colOffsets,
                     const int K,
                     const int r,
                     const arma::vec& wd, 
                     const arma::mat& Q,
                     const bool fitZeta=true
  );



  //Estimation utilities
  void estimate(double f1, double f2, double maxIter=25, double eps=1e-4, double xtol_abs=.01);
  void estimate(const estimatePar& ep);

  //objective
  double getPenalty();
  double objective(double f1, double f2);
  double mseForTest(const arma::uvec& columns1Test,
                    const std::vector<arma::mat>& ZTest,
                    const arma::mat& MTest,
                    const std::vector<arma::vec>& rowOffsetsTest,
                    const std::vector<arma::vec>& colOffsetsTest
  );
  double objectiveForTest(const arma::uvec& columns1Test,
                    const std::vector<arma::mat>& ZTest,
                    const arma::mat& MTest,
                    const std::vector<arma::vec>& rowOffsetsTest,
                    const std::vector<arma::vec>& colOffsetsTest
  );


  //outputs, althoguh everyhing is public.
  std::vector<arma::mat> get_Zeta() const;
  arma::mat get_hmat() const;
  arma::mat get_umat() const;
  arma::mat get_gmat(); //u = f+g = Sd + Rc
  std::vector<arma::mat> get_vmat() const;

  ~inpFitterOnColumns()=default;

protected:
  void setAexToAandLock(); //same strategy as npFitterOnColumns.h - check that file for details
  void estimateHelper(double f1, double f2, double maxIter=25, double eps=1e-4 );

private:
  bool estimateHelperCalledViaEstimate=false;
};
#endif
