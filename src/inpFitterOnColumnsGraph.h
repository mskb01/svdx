#ifndef INPFITTERONCOLUMNSGRAPH_H
#define INPFITTERONCOLUMNSGRAPH_H

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

#include "inpFitterOnColumns.h"


//For dense
class inpFitterOnColumnsGraph final: public inpFitterOnColumns {



public:

  const arma::sp_mat& graphDistMat; //qxq graph dist matrix, must be pos semi def/pos def
  const arma::uvec& graphDistOnlyOnPoints;    //the graph distances are defined for these points, in the order

  using inpFitterOnColumns::Y;
  using inpFitterOnColumns::columns; //these are the only columns that will be used for fitting the data. Useful for CV, instead of making copies of Y!

  //these two get updated based on
  //  whatever last f1, f2 pair the
  //  estimate function was called with.
  // This is useful to know because the umat, vmat, hmat etc.,
  // only make sense with these hyperparameter settings.
  using inpFitterOnColumns::last_f1_estimatedWith;
  using inpFitterOnColumns::last_f2_estimatedWith;
  double last_f3_estimatedWith;

  using inpFitterOnColumns::Z;       //Z may or may not be present; can be a column vector of 1 simply
  using inpFitterOnColumns::M;       //M=[S; R] , S is nxr R is nxq entering in Sd + Rc = Mh
  using inpFitterOnColumns::qids;         //we assume qids -> qids-1 i.e., R to C indexing conversion has happened before
  using inpFitterOnColumns::Rtilde; //qxq penalty = Rtheta[qids,]

  using inpFitterOnColumns::B;      //qxq full kernel inverse
  using inpFitterOnColumns::Q;       //any space to be orthogonal to; not making this as a reference because then it is harder to set up the situation without an initial Q (that constructor will need to initialize Q as well)
  using inpFitterOnColumns::rowOffsets;
  using inpFitterOnColumns::colOffsets;
  using inpFitterOnColumns::colOffsets_for_columns; //extracts and stores colOffsets for desired columns

  using inpFitterOnColumns::D;
  using inpFitterOnColumns::K;
  using inpFitterOnColumns::r;
  using inpFitterOnColumns::wd;
  using inpFitterOnColumns::q;
  using inpFitterOnColumns::n;
  using inpFitterOnColumns::pd;
  using inpFitterOnColumns::spd;

  using inpFitterOnColumns::Zeta;
  using inpFitterOnColumns::hmat;    //umat = M hmat
  using inpFitterOnColumns::umat;
  using inpFitterOnColumns::vmat;

  using inpFitterOnColumns::A; //holds M'M
  arma::mat G; //holds Mq'graphDistMat Mq
  using inpFitterOnColumns::Aex; //expanded form of A matrix that incorporates +f2 c'Rc in the right indices, and the orthogonal constraints.
  using inpFitterOnColumns::Aex_pinv;
  using inpFitterOnColumns::Aex_pinv_rPlusq_submat; //submat of Aex_pinv
  using inpFitterOnColumns::Zt;
  using inpFitterOnColumns::Mt;
  using inpFitterOnColumns::ZtZ;
  using inpFitterOnColumns::Zls;
  using inpFitterOnColumns::YZlst;
  using inpFitterOnColumns::w_hat;
  using inpFitterOnColumns::seqOfComponents;
  using inpFitterOnColumns::qindx;
  using inpFitterOnColumns::rqindx;

  //whether to fit or condition on zeta
  // fixing useful for structure discovery
  using inpFitterOnColumns::fitZeta;

  //lf if needed
  using inpFitterOnColumns::lf_swise;

  //some ones for offsets
  using inpFitterOnColumns::ones_n; //ones of length coloffsets respectively
  using inpFitterOnColumns::ones_pd; //ones of length pd for each d for row offstes

  //usefulf or estimation
  using inpFitterOnColumns::move_Zeta;             //for k=0 component, for all d and its overloaded forms for k>0 components
  using inpFitterOnColumns::move_v_k;
  using inpFitterOnColumns::populate_w_hat_for_k;

  using inpFitterOnColumns::append_to_Aex_and_update_Aexpinv;
  using inpFitterOnColumns::estimate_k0; //for k=0, the first component
  using inpFitterOnColumns::estimate_k; //for k>0
  using inpFitterOnColumns::move_h_k;

public:

  //Constructors
  explicit inpFitterOnColumnsGraph(const dsListC& Y,   //without Q, orthogonal space to avoid
                               const arma::uvec& columns,
                     const std::vector<arma::mat>& Z,
                     const std::vector<arma::mat> Zeta,
                     const arma::mat& M, arma::mat hmat, // our hmat will be a copy, we will use the same memory as output. Otherwise the R side input gets screwed up too! So would umat, and vmat  that we will generate.
                     arma::mat umat, std::vector<arma::mat> vmat,
                     const arma::mat& Rtilde,
                     const arma::sp_mat& graphDistMat,
                     const arma::uvec& graphDistOnlyOnPoints,
                     const arma::uvec& qids,
                     const arma::mat& B,
                     const std::vector<arma::vec> rowOffsets,
                     const std::vector<arma::vec> colOffsets,
                     const int K,
                     const int r,
                     const arma::vec& wd,
                     const bool fitZeta=true
  );

  explicit inpFitterOnColumnsGraph(const dsListC& Y,   //without Q, orthogonal space to avoid
                               const arma::uvec& columns,
                     const std::vector<arma::mat>& Z,
                     const std::vector<arma::mat> Zeta,
                     const arma::mat& M, arma::mat hmat,
                     arma::mat umat, std::vector<arma::mat> vmat,
                     const arma::mat& Rtilde,
                     const arma::sp_mat& graphDistMat,
                     const arma::uvec& graphDistOnlyOnPoints,
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
  using inpFitterOnColumns::setAexToAandLock;
  using inpFitterOnColumns::estimateHelper;
  void estimate(double f1, double f2, double f3, double maxIter=25, double eps=1e-4, double xtol_abs=.01);
  void estimate(const estimatePar& ep); //not using base class estiamatePar fn as here we need f3 as well
  
  //objective
  double getPenalty();
  double getPenaltyForGraph();
  double objective(double f1, double f2, double f3);
  using inpFitterOnColumns::mseForTest;
  double objectiveForTest(const arma::uvec& columns1Test,
                    const std::vector<arma::mat>& ZTest,
                    const arma::mat& MTest,
                    const std::vector<arma::vec>& rowOffsetsTest,
                    const std::vector<arma::vec>& colOffsetsTest
  );


  //outputs, althoguh everyhing is public.
  using inpFitterOnColumns::get_Zeta;
  std::vector<arma::mat> get_Zeta_R() const;
  using inpFitterOnColumns::get_hmat;
  using inpFitterOnColumns::get_umat;
  using inpFitterOnColumns::get_gmat; //u = f+g = Sd + Rc
  using inpFitterOnColumns::get_vmat;

  ~inpFitterOnColumnsGraph()=default;

};
#endif
