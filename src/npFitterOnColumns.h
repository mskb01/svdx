#ifndef NPFITTERONCOLUMNS_H
#define NPFITTERONCOLUMNS_H

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
#include "parallelWokersTemplates.h"
#include "estimatePar.h"
//#include "abstractNPFitterOnColumns.h"

template<typename T>

class npFitterOnColumns { //same as npFitter but operates on columns specified, w/o a copy. NOTE: ALl other input variables (e.g., qids) must have been subsetted. Only Y[,cols] is passed as Y, cols
  public:
    const T& Y;
    const arma::uvec columns; //these are the only columns that will be used for fitting the data. Useful for CV, instead of making copies of Y!
    const arma::mat& Z;       //Z may or may not be present; can be a column vector of 1 simply
    const arma::mat& M;       //M=[S; R] , S is nxr R is nxq entering in Sd + Rc = Mh
    const arma::uvec qids;    //let's not use a refernece here, we will be subbin 1 too, don't want to the R part to reflect that!!!!
    const arma::mat& Rtilde; //qxq penalty = Rtheta[qids,]
    const arma::mat& B;      //qxq full kernel inverse
    const arma::mat Q;       //any space to be orthogonal to; this is not a reference as declaring it so will require initialization in the initialiezr list, and we have situations where we don't have a Q (we will be overloading the constructor for this one).
    const arma::vec& rowOffsets;
    const arma::vec& colOffsets;
    arma::vec colOffsets_for_columns; //stores colOffests corresponding to desired columns

    const int K;
    const int r;
    const int q;
    const int n;
    const int p;


    //these two get updated based on
    //  whatever last f1, f2 pair the
    //  estimate function was called with.
    // This is useful to know because the umat, vmat, hmat etc.,
    // only make sense with these hyperparameter settings.
    double last_f1_estimatedWith;
    double last_f2_estimatedWith;


    arma::mat Zeta;
    arma::mat hmat;    //umat = M hmat
    arma::mat umat;
    arma::mat vmat;

    arma::mat A; //holds M'M
    arma::mat Aex; //expanded form of A matrix that incorporates +f2 c'Rc in the right indices, and the orthogonal constraints.
    arma::mat Aex_pinv;
    arma::mat Aex_pinv_rPlusq_submat;
    arma::mat Zt;
    arma::mat Mt;
    arma::mat ZtZ;
    arma::mat Zls;
    arma::mat YZlst;
    arma::mat ZtYtilde; // Z' times offset-adjusted Y' ; cached for cheap exact objective-change bookkeeping
    arma::vec w_hat;
    arma::vec v_score; // unnormalised score used in the most recent v update
    arma::uvec seqOfComponents;
    arma::uvec qindx;
    arma::uvec rqindx;

    //whether to fit zeta
    // conditioning useful for struc discover
    const bool fitZeta;

    //lf if needed
    arma::vec lf_swise;

    //some ones for offsets
    arma::vec ones_p, ones_n; //ones of length roffset, coloffsets respectively

    //usefulf or estimation
    inline void move_Zeta(arma::vec& u_k, arma::vec& v_k);             //for k=0 component
    inline void move_Zeta(arma::vec& u_k, arma::vec& v_k,              // for k >0 components
                   arma::mat& umatOthers, arma::mat& vmatOthers);

    inline void move_v_k(arma::vec& uk, arma::vec& vk); //for k=0
    inline void move_v_k(arma::vec& uk, arma::vec& vk,
                  arma::mat& umatOthers, arma::mat& vmatOthers); //for k>0


    inline void populate_w_hat_for_k(arma::vec& vk); //for k = 0
    inline void populate_w_hat_for_k(arma::vec& vk,
                              arma::mat& umatOthers, arma::mat& vmatOthers); //for k > 0
    void append_to_Aex_and_update_Aexpinv( const arma::vec& u );
    void append_to_Aex_and_update_Aexpinv( const arma::mat& Q1 );
    void estimate_k0(int maxIter=25, double eps=1e-4); //for k=0, the first component
    void estimate_k(int k, arma::mat& umatOthers,
                    arma::mat& vmatOthers,
                    int maxIter=25, double eps=1e-4); //for k>0
    inline void move_h_k(arma::vec& hk, arma::vec& uk, arma::vec& vk);
    inline void move_h_k(arma::vec& hk, arma::vec& uk, arma::vec& vk,
                  arma::mat& umatOthers, arma::mat& vmatOthers);


    //Constructors
    explicit npFitterOnColumns(const T& Y,   //without Q, orthogonal space to avoid
                      const arma::uvec& columns1,
                      const arma::mat& Z, arma::mat Zeta,
                      const arma::mat& M,arma::mat hmat,
                      arma::mat umat, arma::mat vmat, //it is very important these umat, vmat, hmat, Zeta etc., are all copies, as we will be writing to these! Otherwise what is passed from R gets messed up, and very hard to track why we get different results when rerunning code!
                      const arma::mat& Rtilde,
                      const arma::uvec& qids1,
                      const arma::mat& B,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const int K,
                      const int r,
                      const bool fitZeta=true
    );

    explicit npFitterOnColumns(const T& Y, //with Q,
                      const arma::uvec& columns1,
                      const arma::mat& Z,arma::mat Zeta,
                      const arma::mat& M,arma::mat hmat,
                      arma::mat umat, arma::mat vmat,
                      const arma::mat& Rtilde,
                      const arma::uvec& qids1,
                      const arma::mat& B,
                      const arma::vec& rowOffsets,
                      const arma::vec& colOffsets,
                      const int K,
                      const int r,
                      const arma::mat& Q,
                      const bool fitZeta=true
               );

    //Estimation utilities
    void estimate(double f1, double f2, double maxIter=25, double eps=1e-4, double xtol_abs=.01);
    
    void estimate(const estimatePar& ep );
    
    double getPenalty();
    double objective(double f1, double f2);
    double mseForTest(const arma::uvec& columns1Test,
                      const arma::mat& ZTest,
                      const arma::mat& MTest,
                      const arma::vec& rowOffsetsTest,
                      const arma::vec& colOffsetsTest
    );
    double objectiveForTest(const arma::uvec& columns1Test,
                      const arma::mat& ZTest,
                      const arma::mat& MTest,
                      const arma::vec& rowOffsetsTest,
                      const arma::vec& colOffsetsTest
    );

    //outputs, althoguh everyhing is public.
    arma::mat get_Zeta();
    arma::mat get_hmat();
    arma::mat get_umat();
    arma::mat get_vmat();

    ~npFitterOnColumns()=default;

protected:
  //Estimation utilities
  /*
   * A critical function that should never be called by itself as it can expand Aex.
   * It should only be called via estimate(). There are additional issues for derived classes as
   * seen in the explanation above the function in the npFitterOnColumns.h file.
   * This is why we also keep track of a boolean if it has been called via estimate().
   */
  void setAexToAandLock();
  void estimateHelper(double f1, double f2, double maxIter=25, double eps=1e-4, double xtol_abs=.01);

private:
  bool estimateHelperCalledViaEstimate=false;
};


template class npFitterOnColumns<arma::mat>;
template class npFitterOnColumns<arma::sp_mat>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * CONSTRUCTORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
 * With Q
 */
template<typename T>
npFitterOnColumns<T>::npFitterOnColumns(
  const T& Y,
  const arma::uvec& columns1, //NOTE!! we assume columns -> columns-1 has already been made before. Similarly, for qids below!
  const arma::mat& Z,       //Z may or may not be present
  arma::mat Zeta,
  const arma::mat& M, arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  const arma::mat& Rtilde, // qxq penalty
  const arma::uvec& qids1,//NOTE!! we assume qids -> qids-1 has already been made before. Similarly, for columns1 above.
  const arma::mat& B,      //qxq full kernel inverse
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const int K,
  const int r,
  const arma::mat& Q,       //nxq
  const bool fitZeta
): Y(Y),columns(columns1), Z(Z),M(M),Rtilde(Rtilde),B(B),rowOffsets(rowOffsets),colOffsets(colOffsets), K(K), r(r), qids(qids1), q(qids1.size()), p(Y.n_rows), n(columns.n_elem), Q(Q), fitZeta(fitZeta){


  //TODO enforce a check for length(qids)==nrow(Rtilde)==ncol(Rtilde)

  this->seqOfComponents=arma::regspace<uvec>(0,K-1);
  this->qindx= arma::regspace<arma::uvec>( r, r+q-1 );
  this->rqindx= arma::regspace<arma::uvec>( 0, r+q-1 );

  this->Zeta = Zeta;
  this->hmat = hmat;
  this->umat = umat;
  this->vmat = vmat;

  this->ones_p = arma::vec( p, arma::fill::ones );
  this->ones_n = arma::vec( n, arma::fill::ones );
  this->colOffsets_for_columns = colOffsets.elem( columns );

  //construct A, Z
  this->Mt = M.t();
  this->A =2* Mt * M;
  // The matrix Aex  is dealt with separately in the estimate() function
  //  as Aex is dependent on f1 and f2.

  this->Zt = Z.t();
  this->ZtZ = Zt * Z;
  this->Zls = invpsd( ZtZ, 1e-6, false ) * Z.t();
  this->YZlst = ((Y.cols(columns) * (Zls.t())).t()).eval(); // (pxn x  nx|Zeta|)' = |Zeta| x p
  this->YZlst = this->YZlst - ((( Zls * colOffsets_for_columns ).eval())*(ones_p.t())) - ((( Zls * ones_n ).eval())*(rowOffsets.t())); //accounting for offsets
  this->ZtYtilde = (Zt * Y.cols(columns).t()).eval() - (Zt * colOffsets_for_columns) * ones_p.t() - (Zt * ones_n) * rowOffsets.t();
  this->w_hat = arma::vec( n );

  this->lf_swise=arma::vec(n, arma::fill::zeros);

}


/*
 * Without Q
 */
template<typename T>
npFitterOnColumns<T>::npFitterOnColumns(
  const T& Y,
  const arma::uvec& columns1,
  const arma::mat& Z,       //Z may or may not be present
  arma::mat Zeta,
  const arma::mat& M,arma::mat hmat,
  arma::mat umat, arma::mat vmat,
  const arma::mat& Rtilde, // qxq penalty
  const arma::uvec& qids1,
  const arma::mat& B,      //qxq full kernel inverse
  const arma::vec& rowOffsets,
  const arma::vec& colOffsets,
  const int K,
  const int r,
  const bool fitZeta
): Y(Y), columns(columns1), Z(Z), M(M), Rtilde(Rtilde), B(B), rowOffsets(rowOffsets), colOffsets(colOffsets),K(K), r(r), qids(qids1), q(qids1.size()),  p(Y.n_rows), n(columns.n_elem), fitZeta(fitZeta) {


  //TODO enforce a check for length(qids)==nrow(Rtilde)==ncol(Rtilde)

  this->seqOfComponents=arma::regspace<uvec>(0,K-1);
  this->qindx= arma::regspace<arma::uvec>( r, r+q-1 );
  this->rqindx= arma::regspace<arma::uvec>( 0, r+q-1 );

  this->hmat = hmat;
  this->umat = umat;
  this->vmat = vmat;
  this->Zeta = Zeta;

  this->ones_p = arma::vec( p, arma::fill::ones );
  this->ones_n = arma::vec( n, arma::fill::ones );
  this->colOffsets_for_columns = colOffsets.elem( columns );

  //construct A, Z
  this->Mt = M.t();
  this->A =2* Mt * M;
  
  this->Zt = Z.t();
  this->ZtZ = Zt * Z;
  this->Zls = invpsd( ZtZ, 1e-6, false ) * Zt;
  this->YZlst = ((Y.cols(columns) * (Zls.t())).t()).eval(); // (pxn x  nx|Zeta|)' = |Zeta| x p
  this->YZlst = this->YZlst - ((( Zls * colOffsets_for_columns ).eval())*(ones_p.t())) - ((( Zls * ones_n ).eval())*(rowOffsets.t())); //accounting for offsets
  this->ZtYtilde = (Zt * Y.cols(columns).t()).eval() - (Zt * colOffsets_for_columns) * ones_p.t() - (Zt * ones_n) * rowOffsets.t();
  this->w_hat = arma::vec( n );

  this->lf_swise=arma::vec(n, arma::fill::zeros);

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT ZETA
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////
/*
 * For the first component (k=0) only, without any prior
 * components fitted.
 */
//////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::move_Zeta(arma::vec& u_k, arma::vec& v_k){
  if(fitZeta){
    Zeta = YZlst - ( (Zls * u_k).eval() * v_k.t() ) ; //already accounted for offsets in YZlst
  }
}


/////////////////////////////////////////////////////////////////////
/*
 * For subsequent components with k>0
 * with k-1 components fitted and stored in umatOthers and vmatOthers.
 */
/////////////////////////////////////////////////////////////////////
//Overloaded forms w/o umatOthers and vmatOthers useful only for k=1.
template<typename T>
void npFitterOnColumns<T>::move_Zeta( arma::vec& u_k, arma::vec& v_k,
                             arma::mat& umatOthers, arma::mat& vmatOthers){

  if(fitZeta){
    // Kxp
    this->Zeta = YZlst - ( (Zls * u_k).eval() * v_k.t() )   - ( (Zls * umatOthers).eval() * vmatOthers.t() ); //already accounted for offsets in YZlst
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT V_K
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
////OVERLOADED FORM WITHOUT umatOthers and vmatOthers
//// useful for example, for moving k=1
////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::move_v_k(arma::vec& uk, arma::vec& vk){
  vk = (Y.cols(columns)*uk) - ( Zeta.t() * (Zt*uk).eval() );
  vk = vk - arma::dot(ones_n, uk)*rowOffsets - arma::dot( colOffsets_for_columns, uk )*ones_p ;  //accounting for offsets
  v_score = vk; // retain the already-computed unnormalised score for objective-change bookkeeping
  double dk = arma::norm(vk);
  if( dk < 1e-7 ){
    vk = 0*vk;
  } else {
    vk = vk/dk;
  }
  //vk = arma::normalise(vk); //we pass variable vk as reference and reuse in outer loops, so cannot ignore normalization.
  //vmat.col(0)=vk; //this update happens after convergence of v0 in the outer loops
}


/////////////////////////////////////
////MOVE REST OF THE COMPONENTS//////
/////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::move_v_k(arma::vec& uk, arma::vec& vk, arma::mat& umatOthers, arma::mat& vmatOthers ){
  vk = (Y.cols(columns)*uk).eval() - vmatOthers*( umatOthers.t() * uk ).eval() - ( Zeta.t() * (Zt*uk).eval() );
  vk = vk - arma::dot(ones_n, uk)*rowOffsets - arma::dot( colOffsets_for_columns, uk )*ones_p ;  //accounting for offsets
  v_score = vk; // retain the already-computed unnormalised score for objective-change bookkeeping
  double dk = arma::norm(vk);
  if( dk < 1e-7 ){
    vk = 0*vk;
  } else {
    vk = vk/dk;
  }
  //vk = arma::normalise(vk);
  //vmat.col(k)=vk; //this update happens after convergence of vk in the outer loops
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * FIT u_K
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// For the first comp, k=0
////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::populate_w_hat_for_k(arma::vec& vk){
  arma::rowvec vtY=(vk.t()*Y.cols(columns)).eval();    // 1xn
  vtY = vtY - arma::dot( vk, ones_p )*colOffsets_for_columns.t() - arma::dot( vk, rowOffsets )*ones_n.t(); //accounting for offsets
  arma::mat Zetav= ((Zeta * vk)).eval();  //|Zeta|xp x px1
  w_hat = vtY.t() - Z * Zetav;            //nx1  - nx|Zeta| x |Zeta| x1

}

template<typename T>
void npFitterOnColumns<T>::move_h_k(arma::vec& hk, arma::vec& uk, arma::vec& vk){ //moves and updates Aex for the next component
  this->populate_w_hat_for_k( vk );          //construct w_hat=((Y - Zb) %*% v) \in Rn
  hk =2* Aex_pinv_rPlusq_submat * ( Mt * w_hat); //solving for h & gamma
  uk = M*hk;
  // hmat.col(k)=hk;
  // umat.col(k)=uk;
}

////////////////////////////////////////////////////////////
// Overloaded forms. For the subsequents, k>0
////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::populate_w_hat_for_k(arma::vec& vk,
                                       arma::mat& umatOthers, arma::mat& vmatOthers
){
  arma::rowvec vtY=(vk.t()*Y.cols(columns)).eval();      // 1xn
  vtY = vtY - arma::dot( vk, ones_p )*colOffsets_for_columns.t() - arma::dot( vk, rowOffsets )*ones_n.t(); //accounting for offsets
  arma::mat Zetav= ((Zeta * vk)).eval(); // |Zeta| x p * px1
  arma::mat vv = (vmatOthers.t() * vk).eval();  //|K|xp * px1
  w_hat = vtY.t() - umatOthers * vv - Z * Zetav; // nx1  - nx|K| * Kx1 - nx|Zeta| x |Zeta|x1
}


template<typename T>
void npFitterOnColumns<T>::move_h_k(arma::vec& hk, arma::vec& uk, arma::vec& vk,
                           arma::mat& umatOthers, arma::mat& vmatOthers){ //moves and updates Aex for the next component

  this->populate_w_hat_for_k( vk, umatOthers, vmatOthers );   //construct w_hat=((Y - Zb) %*% v) \in Rn
  hk= 2* Aex_pinv_rPlusq_submat * ( Mt * w_hat); //solving for just the h part, as we have a system like [h, gamma]'= [ A1 A2 ][ M'what; 0  ]'  = A1 M'What => h=A1rplusqSubblock M'What,which extracts the first rplusq rows corresponding to h
  uk = M*hk;
  
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * ESTIMATORS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::append_to_Aex_and_update_Aexpinv( const arma::vec& u ){

  //Transform u for the Kernel inn Prod
  arma::vec Qq = this->B * u.elem(qids); // B must be of |qids| x |qids|

  //update Aex for next component
  arma::rowvec rowAppendage = arma::rowvec( Aex.n_cols, fill::zeros );
  //rowAppendage( arma::span( 0, r+q-1 ) ) =  Qq.t() * M.rows(qids); //check if we need to do: rowAppendage.elem( <uvec> ) here.
  rowAppendage.elem( rqindx ) = Qq.t() * M.rows(qids);
  arma::mat Aex0 = arma::join_vert(Aex, rowAppendage);
  arma::vec zeroBlock=arma::vec(1,fill::zeros);
  arma::vec colAppendage = arma::join_vert(rowAppendage.t(),  zeroBlock );
  this->Aex = arma::join_horiz( Aex0, colAppendage  );
  this->Aex_pinv=invpsd( Aex, 1e-6, false );
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );


}

template<typename T>
void npFitterOnColumns<T>::append_to_Aex_and_update_Aexpinv( const arma::mat& Q1 ){

  //Transform Q1 for the Kernel inn Prod
  arma::mat Qq = this->B * Q1.rows(qids); // B must be of |qids| x |qids|

  //update Aex for next component
  arma::mat rowAppendage = arma::mat( Q1.n_cols, Aex.n_cols, fill::zeros );
  rowAppendage( arma::span::all, arma::span( 0, r+q-1 ) ) =  Qq.t() * M.rows(qids);
  arma::mat Aex0 = arma::join_vert(Aex, rowAppendage);
  arma::mat zeroBlock = arma::mat(Q1.n_cols,Q1.n_cols,fill::zeros);
  arma::mat colAppendage = join_vert( rowAppendage.t().eval(), zeroBlock );
  this->Aex = join_horiz( Aex0, colAppendage  );
  this->Aex_pinv=invpsd( Aex, 1e-6, false );
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );
}

// Estimate the first component, when there are no
//   umatOthers and vmatOthers to account for.
template<typename T>
void npFitterOnColumns<T>::estimate_k0( int maxIter, double eps){

  int k =0;

  arma::vec hk=hmat.col(k);
  arma::vec uk=umat.col(k);
  arma::vec vk=vmat.col(k);
  bool conv = false;
  int iter = 0;
  const arma::mat Hhalf = 0.5 * Aex.submat(0, 0, r+q-1, r+q-1);
  const arma::mat Hpen = Hhalf - 0.5 * A;
  arma::mat Zeta_old; 
  arma::vec hk_old, dh, vk_old; 
  while( !conv && (iter < maxIter) ){
    iter = iter+1;
    double deltaF = 0.0;

    // Zeta block: exact before/after quadratic change; no exact-pseudoinverse assumption.
    Zeta_old = Zeta;
    move_Zeta(uk, vk);
    if(fitZeta){
      arma::mat dZeta = Zeta - Zeta_old;
      arma::mat ZtR_old = ZtYtilde - (Zt * uk) * vk.t() - ZtZ * Zeta_old;
      deltaF += -2.0 * arma::accu(dZeta % ZtR_old)
                + arma::accu(dZeta % (ZtZ * dZeta));
    }

    // h block: use the already-computed w_hat and the primal quadratic block only.
    hk_old = hk;
    move_h_k(hk, uk, vk);
    dh = hk - hk_old;
    deltaF += -2.0 * arma::dot(dh, Mt * w_hat)
              + arma::dot(vk, vk) * arma::dot(dh, A * hk_old)
              + 0.5 * arma::dot(vk, vk) * arma::dot(dh, A * dh)
              + 2.0 * arma::dot(dh, Hpen * hk_old)
              + arma::dot(dh, Hpen * dh);

    // v block: v_score is the unnormalised score already formed by move_v_k().
    vk_old = vk;
    move_v_k(uk, vk);
    deltaF += arma::dot(uk, uk) * (arma::dot(vk, vk) - arma::dot(vk_old, vk_old))
              - 2.0 * arma::dot(v_score, vk - vk_old);

    if( arma::sign( vk(0) ) <0 ){//rotation identifiabilty
      vk = -vk;
      hk = -hk;
      uk = -uk;
    }

    // eps is on the canonical objective scale; F_solved=(n/f1)F_canonical.
    conv = std::abs(deltaF) < eps;
  }

  if(!conv){
    std::cout << "Component " + std::to_string( k ) + " fails to converge. Increase maxIter?" << std::endl;
  }

  //update the internal datastructures
  hmat.col(k)=hk;
  umat.col(k)=uk;
  vmat.col(k)=vk;

}

template<typename T>
void npFitterOnColumns<T>::estimate_k(int k, arma::mat& umatOthers, arma::mat& vmatOthers, int maxIter, double eps){

  arma::vec hk=hmat.col(k);
  arma::vec uk=umat.col(k);
  arma::vec vk=vmat.col(k);

  bool conv = false;
  int iter = 0;
  const arma::mat Hhalf = 0.5 * Aex.submat(0, 0, r+q-1, r+q-1);
  const arma::mat Hpen = Hhalf - 0.5 * A;
  arma::mat Zeta_old; arma::vec hk_old, dh, vk_old;
  while( !conv && (iter < maxIter) ){
    iter = iter+1;
    double deltaF = 0.0;

    // Zeta block: exact before/after quadratic change; no exact-pseudoinverse assumption.
    Zeta_old = Zeta;
    move_Zeta(uk, vk, umatOthers, vmatOthers);
    if(fitZeta){
      arma::mat dZeta = Zeta - Zeta_old;
      arma::mat ZtR_old = ZtYtilde
        - (Zt * uk) * vk.t()
        - (Zt * umatOthers) * vmatOthers.t()
        - ZtZ * Zeta_old;
      deltaF += -2.0 * arma::accu(dZeta % ZtR_old)
                + arma::accu(dZeta % (ZtZ * dZeta));
    }

    // h block: use the already-computed w_hat and the primal quadratic block only.
    hk_old = hk;
    move_h_k(hk, uk, vk, umatOthers, vmatOthers);
    dh = hk - hk_old;
    deltaF += -2.0 * arma::dot(dh, Mt * w_hat)
              + arma::dot(vk, vk) * arma::dot(dh, A * hk_old)
              + 0.5 * arma::dot(vk, vk) * arma::dot(dh, A * dh)
              + 2.0 * arma::dot(dh, Hpen * hk_old)
              + arma::dot(dh, Hpen * dh);

    // v block: v_score is the unnormalised score already formed by move_v_k().
    vk_old = vk;
    move_v_k(uk, vk, umatOthers, vmatOthers);
    deltaF += arma::dot(uk, uk) * (arma::dot(vk, vk) - arma::dot(vk_old, vk_old))
              - 2.0 * arma::dot(v_score, vk - vk_old);

    if( arma::as_scalar(arma::sign( vk(0) )) <0 ){
      vk = -vk;
      hk = -hk;
      uk = -uk;
    }

    conv = std::abs(deltaF) < eps;
  }

  if(!conv){
    std::cout << "Component " + std::to_string( k ) + " fails to converge. Increase maxIter?" << std::endl;
  }

  hmat.col(k)=hk;
  umat.col(k)=uk;
  vmat.col(k)=vk;

}



////////////////////////////////////////////////////////////////////////////////////////////
////IMPORTANT: this function should not be called in parallel. Two issues;
////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void npFitterOnColumns<T>::estimateHelper(double f1, double f2, double maxIter, double eps, double xtol_abs){

  //Guard for making sure this function is never called by itself without estimate!
  if( !this->estimateHelperCalledViaEstimate ){ // the private boolean is set to true by a protected setAexToAandLock() function.
    throw std::runtime_error( "FATAL: estimateHelper() invoked directly or out-of-order."
                              "Calculations must be initiated via the public estimate() interface." );
  }

  Aex.submat( r, r, r+q-1, r+q-1 ) += 2*n*(f2/f1)*Rtilde; //n arises because of prefactor of 1/n in objective
  this->Aex_pinv=invpsd(Aex, 1e-6, false);
  this->Aex_pinv_rPlusq_submat=Aex_pinv.submat( 0, 0, r+q-1, r+q-1 );
  if( this->Q.n_cols > 0 ){
    this->append_to_Aex_and_update_Aexpinv(Q); //orth with k-1
  }
  
  //for constant function
  arma::vec u0 = arma::vec(n, arma::fill::ones);
  this->append_to_Aex_and_update_Aexpinv(u0); //orth with k-1


  /////////////
  //ESTIMATION
  ////////////

  //First component
  const double solved_eps = (n/f1) * eps; // eps is canonical- f1/n*SSE objective tolerance. Change to n/f1 scaled objecrtive that is being solved here.
  this->estimate_k0(maxIter, solved_eps);

  //Subsequents.
  if( K > 1 ){

    arma::mat umatOthers;
    arma::mat vmatOthers;
    arma::uvec before_k;

    for( int k=1; k < K; k++ ){
      //Append orth with the just constructed k-1 st.
      //  The previous orth are already appended in previous cycles.
      arma::vec ukm1 = umat.col(k-1);
      this->append_to_Aex_and_update_Aexpinv(ukm1); //orth with k-1

      //Fit the k^th while djusting the mean
      //   for the already reconstructed 0 .. k-1 components.
      before_k = find( this->seqOfComponents < k ) ;
      umatOthers = umat.cols( before_k );
      vmatOthers = vmat.cols( before_k );
      this->estimate_k(k, umatOthers, vmatOthers, maxIter, solved_eps);
    }
  }

  last_f1_estimatedWith = f1;
  last_f2_estimatedWith = f2;

}

template<typename T>
void npFitterOnColumns<T>::setAexToAandLock(){
  this->Aex=A;
  this->estimateHelperCalledViaEstimate=true;
}

template<typename T>
void npFitterOnColumns<T>::estimate(double f1, double f2, double maxIter, double eps, double xtol_abs){
  (void)xtol_abs; // reserved for future parameter-change convergence

  if( Y.n_rows==0 ){
    throw std::runtime_error("Y has not been assigned. Has the right constructor been called? ");
  }

  //BUILD LHS mat for (hk, gamma) by adding f2 dependent penalty terms, and optional Q
  //  and create the initial inverses

  //This resets Aex=A; This reset to Aex is crucial to avoid race condition, as Aex is built from M and M is passed via refer.
  // Also prevents Aex from uncontrolled expansion via repeated calls (in any future derived classes), via boolean as outlined above estimateHelper() fun.
  this->setAexToAandLock(); //ressets Aex = A , and locks a private boolean so that we are able to call estimateHelper.
  this->estimateHelper( f1, f2, maxIter, eps, xtol_abs );

}

template<typename T>
void npFitterOnColumns<T>::estimate(const estimatePar& ep){
  this->estimate( ep.f1, ep.f2, ep.maxIter, ep.eps, .01 );
}

// USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
template<typename T>
double npFitterOnColumns<T>::getPenalty(){

  //TODO : Number of threads must be controlled.
  double penalty = 0;
  arma::vec hk(hmat.n_rows);
  arma::vec hk_q(r+q-1);
  for( int k=0; k<K; k++ ){
    hk = hmat.col(k);
    //hk_q=hk.elem( qindx );
    hk_q = hk.subvec(r, r+q-1);
    penalty += arma::dot( hk_q, Rtilde * hk_q  );
  }
  return penalty;
}


// KEY: USES CURRENTLY STORED VALUES OF U AND V and ZETA. If f1 and f2 do not reflect the currently stored values, estimate must be rerun!
template<typename T>
double npFitterOnColumns<T>::objective(double f1, double f2){

  npSMLFCOnColumns<T> LFC( Y, columns, Z, Zeta, umat, vmat, lf_swise, rowOffsets, colOffsets ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty
  RcppParallel::parallelFor( 0, n, LFC );
  double lf = arma::mean( lf_swise );
  double penalty = this->getPenalty();
  return (f1*lf)+(f2*penalty);
}


template<typename T>
double npFitterOnColumns<T>::mseForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  if( colOffsetsTest.n_elem != Y.n_cols  ){
    throw std::invalid_argument( "colOffsets must have the same length as the number of columns in matrix Y " );
  }

  arma::mat umatTest = MTest* this->hmat;
  const int ntest = columns1Test.n_elem;
  arma::vec lf_swiseTest( ntest, arma::fill::zeros );
  npSMLFCOnColumns<T> LFCTest( Y, columns1Test,
                               ZTest, this->Zeta,
                               umatTest, this->vmat,
                               lf_swiseTest,
                               rowOffsetsTest,
                               colOffsetsTest
                               ); //sample-wise loss functions j=1...n in f1* sum( sample wise loss functions ) + f2*penalty

  RcppParallel::parallelFor( 0, ntest, LFCTest );
  double lf = arma::mean( lf_swiseTest ); //mse
  return lf;
}

template<typename T>
double npFitterOnColumns<T>::objectiveForTest(
    const arma::uvec& columns1Test,
    const arma::mat& ZTest,
    const arma::mat& MTest,
    const arma::vec& rowOffsetsTest,
    const arma::vec& colOffsetsTest
){

  double lf = this->mseForTest( columns1Test, ZTest, MTest,
                                rowOffsetsTest, colOffsetsTest); //mse
  double penalty = this->getPenalty();
  double obj= last_f1_estimatedWith* lf + last_f2_estimatedWith* penalty;
  return obj;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * GETTERS
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
arma::mat npFitterOnColumns<T>::get_Zeta(){
  return( Zeta );
}

template<typename T>
arma::mat npFitterOnColumns<T>::get_hmat(){
  return( hmat );
}

template<typename T>
arma::mat npFitterOnColumns<T>::get_umat(){
  return( umat );
}

template<typename T>
arma::mat npFitterOnColumns<T>::get_vmat(){
  return( vmat );
}






#endif
