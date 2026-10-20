initalizeHUVZ_np <- function(trainObjsForEachFold, train,
                             D, Y, K,
                             Q=NULL,
                             kfolds=length(trainObjsForEachFold),
                             kseq= seq_len( kfolds ), wd=NULL
                                ){

  umatList <- vector( "list", kfolds )
  hmatList <- vector( "list", kfolds )
  vmatList <- vector( "list", kfolds )
  ZetaList <- vector( "list", kfolds )

  hmatList <- lapply( kseq, function(k)
    matrix(0,
           nrow=ncol( trainObjsForEachFold[[k]]$S  ) + length( trainObjsForEachFold[[k]]$qids ),
           ncol=K
    ) # hmat = [d; c] in Sd+ Rc = M hmat
  ) #already subsetted for sids

  # Zeta, U and V depending on single / multiple datasets
  if( D==1 ){
    ZetaList <- lapply( kseq, function(k){
      #t(limma::lmFit( Y[,train$columns[[k]]], train$ZList[[k]] )$coef) ##ols( Z.d, Y), : t ( pdx|Zterms| ) = |Zterms| xpd
      as.matrix(Matrix::t( tcrossprod( Y[,train$columnsList[[k]],drop=F], train$ZlsList[[k]] ) )) #the as.matrix is absolutely essential here!!! The Matrix:: is useful for t() to choose based on arma::mat or arma::sp_mat, instead of t.dfefault()
    } )
    svssubs <- lapply( kseq, function(k){ #TODO IRLBA ETC
      
      irlba::irlba( Y[,train$columnsList[[k]],drop=F], K, K )
      ##rsvd::rsvd( Y[,train$columns[[k]],drop=F], k=K, nu=K, nv=K )
      
      # --- this is expensive though computationally when n is large to construct the residuaals
      #Yres <- Y - t(train$ZList[[k]] %*% ZetaList[[k]])
      #irlba::irlba(Yres, K, K)
    } )
    umatList <- lapply( kseq, function(k) sweep(svssubs[[k]]$v,2, svssubs[[k]]$d[1:K], "*") ) #already subsetted for sids ; just choosing the first dataset's u. Better initial needed!
    vmatList <- lapply( kseq, function(k) svssubs[[k]]$u )
  } else {
    if(is.null(wd)){
      stop("wd expected in initalizeHUVZ_np")
    }
    seqD <- seq(D)
    ZetaList <- lapply( kseq, function(k){
      lapply( seqD, function(d) {
        #t(limma::lmFit( Y[,train$columns[[k]]], train$ZList[[k]] )$coef) ##ols( Z.d, Y), : t ( pdx|Zterms| ) = |Zterms| xpd
        as.matrix(Matrix::t( tcrossprod( Y[[d]][,train$columnsList[[k]],drop=F], train$ZlsList[[k]][[d]] ) )) #the as.matrix is absolutely essential here!!! The Matrix:: is useful for t() to choose based on arma::mat or arma::sp_mat, instead of t.dfefault()
      })
    } )
    svssubs <- lapply( kseq, function(k){ #TODO IRLBA ETC
      lapply( seqD, function(d){
        
        irlba::irlba( Y[[d]][,train$columnsList[[k]],drop=F], K, K )
        
        # initializing from residualized Y
        # --- both yield similar results
        #Yres <- Y[[d]] - t(train$ZList[[k]][[d]] %*% ZetaList[[k]][[d]])
        #irlba::irlba(Yres, K, K)
      }  )
    } )
    umatList <- lapply( kseq, function(k){
      #collated_u <- do.call( cbind, lapply( svssubs[[k]], function(x) sweep( x$v, 2, x$d[1:K],"*" )  ) )
      collated_u <- do.call( cbind, lapply( seqD, function(d){
        x <- svssubs[[k]][[d]]
        sqrt(wd[d])*sweep( x$v, 2, x$d[1:K],"*" ) 
      }   ) )
      ir <- svd( collated_u, K, K )
      sweep( ir$u , 2, ir$d[1:K], "*"  )
    }  )
    vmatList <- lapply( kseq, function(k){ #over folds
      
      vd_k <- lapply(seqD, function(d){ #over datasets in fold
        vd <- Y[[d]][,train$columnsList[[k]],drop=F] %*% umatList[[k]]
      })
      
      #norm fac for v across datsets
      vd_all_k <- do.call( rbind, lapply(seqD, function(d){
         sqrt(wd[d])*vd_k[[d]]
       })  )
      nm <- sqrt(colSums( vd_all_k^2 ))
      
      #normalize v, transfer scale to u
      nearz <- which( nm < 1e-7  )
      seqK <- seq(K)
      if( (length(nearz) > 0) ){
        for( d in seqD ){ #nearz zero out
          vd_k[[d]][,nearz] <- 0*vd_k[[d]][,nearz]
        }
        if( length(nearz) < K ){ #not all are zeros
          for(d in seqD){
            vd_k[[d]][,-nearz] <- sweep( vd_k[[d]][,-nearz], 2, nm[-nearz], "/" )  
          }
        }
      } else {
        for( d in seqD ){
          vd_k[[d]] <- sweep( vd_k[[d]], 2, nm, "/" )
        }
      }
      
      #transfer scale to u of fold
      umatList[[k]] <- sweep( umatList[[k]], 2, nm, "*" )
      
      #return vd for the fold
      vd_k
    })
  }

  list(
    "ZetaList" = ZetaList,
    "hmatList" = hmatList,
    "umatList" = umatList,
    "vmatList" = vmatList
    )

}
