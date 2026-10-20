constructModAtTheta_np <- function(
    theta=NULL, #there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
    data,
    zdata,
    train.si=seq(nrow(data)),
    formula,
    z.formula,
    Q=NULL,
    type=NULL,
    id.basis=NULL,
    perm=FALSE, perm.cols=NULL, #these allow constructing permutation models
    graphData=NULL,
    graphDistFn=NULL,
    graphDistOnlyOnPoints=train.si, #train.si[id.basis]
    n_id.basis = NULL,
    init=FALSE,
    K = NULL,
    Y = NULL,
    rowOffsets = NULL,
    colOffsets = NULL,
    includeSubsetDataForTrainSi=FALSE, 
    wd = NULL#,
    #...
){

  if( init ){
    if( is.null(K) ){
      stop( "K must be supplied if init=TRUE" )
    }
    if( is.null(Y) ){
      stop( "Y must be supplied if init=TRUE" )
    }
  }

  if( is.list( z.formula ) ){

  }

  # all data, zdata would be subset for train.si
  # for id.basis and graphDistOnlyOnPoints when non-null, will be
  #   matched to train.si. So pass idBasis and graphDistOnlyOnPoints in the train.si scale
  #   For example, if train.si is c(1,9,3,4,5), train.si can be c(1,9,4) , but not c(1,2,4)

  n <-nrow(data)
  N0IDBASIS <- max( 30, ceiling(10* n^(2/9))   )
  if( is.null( n_id.basis ) ){
    n_id.basis <- N0IDBASIS
  }
  if( n_id.basis > n ){
    n_id.basis <- n
  }

  r <- range( train.si )
  if( (r[1] < 1) || r[2] > n ){
    stop("train.si exceeds possible indices" )
  }
  if( is.list( z.formula )  ){
    if( !all( sapply( zdata, nrow ) ==n )  ){
      stop( "list zdata must have the same number of rows as data in all of its entries" )
    }
  } else {
    if( nrow(zdata) != n ){
      stop( "zdata must have the same number of rows as data" )
    }
  }
  if(!is.null(id.basis)  ){
    if( length(setdiff(id.basis, train.si)) > 0 ) {
      stop( "id.basis must be elements in train.si" )
    }
    r <- range( id.basis )
    if( (r[1] < 1) || (r[2] > n) ){
      stop( "id.basis must be in the range of 1 to nrow(data)" )
    }
  }
  if( !is.null(graphData) ){
    if( ncol(graphData)!=n ){
      stop( "graphData should have the same number of columns as rows in data" )
    }
    if( is.null( graphDistFn ) ){
      stop( "graphDistFn is not available but graphData is given" )
    }
    if(!is.null(graphDistOnlyOnPoints)  ){
      if( length(setdiff(graphDistOnlyOnPoints, train.si)) > 0 ) {
        stop( "graphDistOnlyOnPoints not seen in train.si" )
      }
      r <- range( graphDistOnlyOnPoints )
      if( (r[1] < 1) || (r[2] > n) ){
        stop( "graphDistOnlyOnPoints must be in the range of 1 to nrow(data)" )
      }
    }
  }

  #revise n after subsetting for train.si
  data <- data[train.si,,drop=F]
  if( is.list( z.formula )  ){
    zdata <- lapply( zdata, function(x) x[train.si,,drop=F] )
  } else {
    zdata <- zdata[train.si,,drop=F]
  }
  if( !is.null(graphData) ){
    graphData <- graphData[,train.si,drop=F]
  }
  n <- nrow(data)

  #match id.basis and graphDistOnlyOnPoints to train.si
  if( is.null(id.basis)  ){
    id.basis <- sample( n, n_id.basis, replace=F )
  } else {
    n_id.basis <- length(id.basis)
    id.basis <- match( id.basis, train.si )
  }
  if( !is.null(graphData) ){
    if( is.null( graphDistOnlyOnPoints ) ){
      message( "graphDistOnlyOnPoints cannot be NULL when passing graphData and graphDistFn. Using all training data. " )
      graphDistOnlyOnPoints<-seq(n)
    } else {
      graphDistOnlyOnPoints <- match( graphDistOnlyOnPoints, train.si )
    }
  }

  if( perm ){
    if(is.null(perm.cols)){
      stop("Permutation is TRUE, but col labels are not given")
    }
    warning("We assume all perm columns are present in the data frame")
    perm.si <- sample( n, n, replace=F )
    data[ , perm.cols ] <- data[perm.si,perm.cols]
  }

  m <- constructMod( data=data,
                     zdata=zdata,
                     formula=formula,
                     z.formula=z.formula,
                     type=type,
                     Q=Q,
                     id.basis=id.basis,
                     n_id.basis = n_id.basis,
                     graphData = graphData,
                     graphDistFn=graphDistFn,
                     graphDistOnlyOnPoints=graphDistOnlyOnPoints#,
                     #...
                     )

  # theta specific objects
  R0Theta <- R0TilTheta <- MTheta <- R0trace <- B <- NA
  if(!is.null(theta)){
    R0Theta <- with(m, Reduce("+", lapply(seq_along(theta), function(i) theta[i]*Rs[[i]] ) ) )
    R0TilTheta <- R0Theta[ m$qids, ]
    MTheta <- cbind( m$S, R0Theta )
    R0trace <- abs(sum(diag(R0TilTheta)))  #absolute value of tra; it must be pos as mat is posdef

    #B is pseudoinv of full R Ktilde, entering the inner prod
    # f=\sum_t \theta_t^{-1} ( theta_t Rt(), f   )
    orthR <- m$SStq + R0TilTheta
    #print( dim(orthR) )
    B <- svd( orthR )
    B$d[ B$d < .Machine$double.eps ] <- 0
    B$d[ B$d!=0  ] <-  1/B$d[ B$d!=0 ]
    B <- tcrossprod( sweep( B$v, 2, B$d, "*" ), B$u )
    B <- .5*(B+t(B)) #to avoid num errors due to asymmetry; we know this inverse, B, of othR, must be symmetric.
    diag(B) <- pmax( diag(B), 0 )
  }


  #Store and return
  m$RsCbound <- m$R0
  m$MTheta <- MTheta
  m$RtildeTheta <- R0TilTheta
  m$B <- B
  m$qids_oriIndx <- train.si[m$qids]
  m$columns <- train.si


  #initalize, Zeta, umat, vmat, hmat
  m["Zeta"] <- m["umat"] <- m["vmat"] <- m["hmat"] <- list(NULL)
  if(init){

    D <- 1
    if( is.list(m$Z ) )  {
      D <- length(m$Z)
    }
    m$hmat <- matrix(0,
                     nrow=ncol( m$S  ) + length( m$qids ),
                     ncol=K
    ) # hmat = [d; c] in Sd+ Rc = M hmat

    # Zeta, U and V depending on single / multiple datasets
    if( D==1 ){
      #t(limma::lmFit( Y[,train$columns[[k]]], train$ZList[[k]] )$coef) ##ols( Z.d, Y), : t ( pdx|Zterms| ) = |Zterms| xpd
      m$Zeta <- as.matrix(Matrix::t( tcrossprod( Y[,m$columns,drop=F], m$Zls ) )) #the as.matrix is absolutely essential here!!! The Matrix:: is useful for t() to choose based on arma::mat or arma::sp_mat, instead of t.dfefault()
      svssubs <- irlba::irlba( Y[,m$columns,drop=F], K, K )
      m$umat <- sweep(svssubs$v,2, svssubs$d[1:K], "*") #already subsetted for sids ; just choosing the first dataset's u. Better initial needed!
      m$vmat <- svssubs$u
      if( is.null(rowOffsets) ){
        rowOffsets <- rep(0, nrow(Y))
      }
      if(is.null(colOffsets) ){
        colOffsets <- rep(0, ncol(Y))
        # note: colOffsets are of length ncol(Y), not length(train.si)
      }
    } else {
      if(is.null(wd)){
        stop("wd is null in constructModsAtTheta_np")
      }
      seqD <- seq(D)
      m$Zeta <- lapply( seqD, function(d) {
        #t(limma::lmFit( Y[,train$columns[[k]]], train$ZList[[k]] )$coef) ##ols( Z.d, Y), : t ( pdx|Zterms| ) = |Zterms| xpd
        as.matrix(Matrix::t( tcrossprod( Y[[d]][,m$columns,drop=F], m$Zls[[d]] ) )) #the as.matrix is absolutely essential here!!! The Matrix:: is useful for t() to choose based on arma::mat or arma::sp_mat, instead of t.dfefault()
      })
      svssubs <- lapply( seqD, function(d){
        #svd( Y[[d]][,train$columns[[k]],drop=F], K, K )
        irlba::irlba( Y[[d]][,m$columns,drop=F], K, K )
      }  )
      collated_u <- do.call( cbind, lapply( svssubs, function(x) sweep( x$v, 2, x$d[1:K],"*" )  ) )
      ir <- svd( collated_u, K, K )
      m$umat <-  sweep( ir$u , 2, ir$d[1:K], "*"  )

      if( is.null(rowOffsets) ){
        rowOffsets <- lapply(seq_along(Y), function(d) rep(0, nrow(Y[[d]])) )
      }
      if(is.null(colOffsets) ){
        colOffsets <- lapply(seq_along(Y), function(d) rep(0, ncol(Y[[d]])) )
        # note: colOffsets are of length ncol(Y), not length(train.si)
      }

      vd_k <- lapply(seqD, function(d){
        vd <- Y[[d]][,m$columns,drop=F] %*% m$umat
      })
      
      vd_all_k <- do.call( rbind, lapply(seqD, function(d){
        sqrt(wd[d])*vd_k[[d]]
      })  )
      nm <- sqrt(colSums( vd_all_k^2 ))
      
      nearz <- which( nm < 1e-7  )
      if( (length(nearz) > 0) ){
        for( d in seqD ){
          vd_k[[d]][,nearz] <- 0*vd_k[[d]][,nearz]
        }
        if( length(nearz) < K ){
          vd_k[[d]][,-nearz] <- sweep( vd_k[[d]][,-nearz], 2, nm[-nearz], "/" )
        }
      } else {
        for( d in seqD ){
          vd_k[[d]] <- sweep( vd_k[[d]], 2, nm, "/" )
        }
      }
      m$vmat <- vd_k
    }

  }

  m$rowOffsets <- rowOffsets
  m$colOffsets <- colOffsets
  if( includeSubsetDataForTrainSi ){
    m$data <- data
    m$zdata <- zdata
    m$graphData <- graphData #when non null, this function adjusts for train.si and we want to retain that, graphDistOnlyOnPoints is already stored via constructMod
    m$id.basis <- id.basis
  }
  if(!is.null(wd)){
    m$wd <- wd  
  }
  


  m
}
