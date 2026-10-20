#include "myutils.h"

arma::mat invpsd(const arma::mat& psd, double thresh=1e-7, bool force=false) {
  if( force || (!psd.is_zero(10*arma::datum::eps)) ){
    arma::mat umat1;
    arma::mat vmat1;
    arma::vec d;
    arma::svd( umat1, d, vmat1, psd );
    d.clean(thresh); //as psd
    d.clamp( 0, arma::max( d )+1 ); //as psd
    if( d.is_zero(10*arma::datum::eps) ){
      return(0*psd);
    } else {
      arma::uvec nz = find( d > 0 );
      umat1 = umat1.cols( nz );
      vmat1 = vmat1.cols( nz );
      arma::vec dd = d.elem(nz);
      //vmat1.each_row() /= dd.t();
      for (unsigned int i = 0; i < vmat1.n_cols; ++i) {
        vmat1.col(i) /= dd(i);
      }
      return( vmat1 * umat1.t()  );
    }
  } else {
    return( 0*psd );
  }

}


double mse_vec(const arma::vec& vec1, const arma::vec& vec2) {

  if( (vec1.n_elem != vec2.n_elem) ){
    throw std::invalid_argument("Both vectors should have the same length");
  }

  return arma::accu(arma::square(vec1-vec2))/vec1.n_elem;

}

double mse_mats(const arma::mat& mat1, const arma::mat& mat2) {

  if( (mat1.n_rows != mat2.n_rows) || ( mat1.n_cols != mat2.n_cols ) ){
    throw std::invalid_argument("Both matrices should have the same dimension");
  }

  return arma::accu(arma::square(mat1-mat2))/( mat1.n_rows*mat1.n_cols );

}

double mse_VectorOfVecs(const std::vector<arma::vec>& vec1, const std::vector<arma::vec>& vec2) {

  if( vec1.size() != vec2.size() ){
    throw std::invalid_argument( "Both vector of vectors must have the same number of length" );
  }

  const int D = vec1.size();
  std::vector<double> local_mse(D, 0.0);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, D),
                    [&](const tbb::blocked_range<size_t>& ra) {
                      for (size_t d = ra.begin(); d != ra.end(); ++d) {
                        local_mse[d] = mse_vec(vec1[d], vec2[d]);
                      }
                    });

  double total_sum = 0.0;
  for (size_t i = 0; i < D; ++i) {
    total_sum += local_mse[i];
  }
  return total_sum / D;

}

double mse_VectorOfMats(const std::vector<arma::mat>& vecMat1, const std::vector<arma::mat>& vecMat2) {

  if( vecMat1.size() != vecMat2.size() ){
    throw std::invalid_argument( "Both vectors must have the same number of matrices" );
  }

  const int D = vecMat1.size();
  std::vector<double> local_mse(D, 0.0);
  tbb::parallel_for( tbb::blocked_range<size_t>(0,D),
                     [&](const tbb::blocked_range<size_t>& ra) {
                       for(size_t d=ra.begin(); d!=ra.end(); ++d){
                         local_mse[d] = mse_mats( vecMat1[d], vecMat2[d] );
                         }
                       });

  double total_sum = 0.0;
  for (size_t i = 0; i < D; ++i) {
    total_sum += local_mse[i];
  }
  return total_sum / D;

}

std::vector<arma::mat> transposeVecOfMats(const std::vector<arma::mat>& vecOfMats){
  std::vector<arma::mat> Zeta_R(vecOfMats.size());
  for( int d=0; d < vecOfMats.size(); d++ ){
    Zeta_R[d] = vecOfMats[d].t();
  }
  return Zeta_R;
}
