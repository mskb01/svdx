initalizeHUVZ_g <- function(  trainObjsForEachFold, train,
                              D, Y, K,
                              data,
                              zdata,
                              formula,
                              z.formula,
                              family,
                              type=NULL,
                              Q=NULL,
                              graphData = NULL,
                              graphDistFn=NULL,
                              f1,
                              init_control = list(),
                              numThreads = 2,
                              setNumThreads = FALSE, #when calling svdx again in this fn, false means we don't set numthreads again.
                              kfolds=length(trainObjsForEachFold),
                              kseq= seq_len( kfolds ), 
                              wd = NULL#,
                              #...
){

  .nestedCVObjectiveTrace <- NULL
  .nestedTheta <- NULL
  .nestedF2 <- NULL
  .nestedH <- NULL
  .nestedCVObjectiveTrace <- NULL
  # Internal R interfaces use bundled controls only. Direct callers that do
  # not specify init_control receive the package defaults.
  init_control <- .normalize_init_control(
    init_control,
    .default_fitter_control(),
    .default_optCV_control(),
    .default_nlopt_control()
  )
  init_fitter_maxIter <- init_control$fitter$maxIter
  init_fitter_ftol_abs <- init_control$fitter$ftol_abs
  init_fitter_xtol_abs <- init_control$fitter$xtol_abs
  init_fitter_inner_maxIter <- init_control$fitter$inner_maxIter
  init_fitter_inner_ftol_abs <- init_control$fitter$inner_ftol_abs
  seqD <- seq(D)
  umatList <- vector( "list", kfolds )
  hmatList <- vector( "list", kfolds )
  vmatList <- vector( "list", kfolds )
  ZetaList <- vector( "list", kfolds )
  aphiList <- vector( "list", kfolds )

  if( D==1 ){

    if(family=="poisson"){
      if("dgCMatrix" %in% class(Y)){
        Y@x = log(1+Y@x)
        #offsets need to be in the right scale; example -- log for poisson
      } else {
        Y=log(1+Y) #users need to log transforms
        # Y = sweep( Y, 1, rowOffsets, "-" ) #implement this for sparse case above
        # Y = sweep( Y, 2, colOffsets, "-" )
      }
    }

    gdat <- gdpnts <- .f2 <- .f3 <- .theta <- NULL #f2, f3 used for return
    if( !is.null(graphData) ){
      gdat <- graphData[,train$columnsList[[1]],drop=F]
      gdpnts <- train$graphDistOnlyOnPointsList[[1]]
    }

    #cat( " + + Initializing for family specification via svdx-np ... \n" )
    svs <- svdx( Y=Y[,train$columnsList[[1]],drop=F],
                 K=K,
                 formula=formula,
                 zformula = z.formula,
                 data=data[train$columnsList[[1]],],
                 zdata=zdata[train$columnsList[[1]],],
                 graphData = gdat,
                 graphDistFn = graphDistFn,
                 graphDistOnlyOnPoints = gdpnts,
                 family = NULL,
                 rowOffsets=train$rowOffsetsList[[1]],
                 colOffsets=train$colOffsetsList[[1]][ train$columnsList[[1]] ],
                 fitter_control = init_control$fitter,
                 optCV_control = init_control$optCV,
                 nlopt_control = init_control$nlopt,
                 idBasis = train$qidsList[[1]],
                 modelType = type,
                 #Q=Q,
                 f1=f1,
                 nfolds=kfolds,
                 est_K = FALSE,
                 numThreads=numThreads,
                 setNumThreads = setNumThreads)

    hmatList[[1]] <- svs$fitAtOptCV$hmat
    umatList[[1]] <- svs$fitAtOptCV$umat
    vmatList[[1]] <- svs$fitAtOptCV$vmat
    ZetaList[[1]] <- t(svs$fitAtOptCV$Zeta)
    .theta <- svs$optCV$theta
    .f2 <- svs$optCV$f2
    .nestedTheta <- svs$optCV$theta
    .nestedF2 <- svs$optCV$f2
    .nestedH <- svs$optCV$h
    .nestedCVObjectiveTrace <- svs$optCV$cvObjectiveTrace
    .f3 <- NULL
    if( !is.null(graphData) ){
      .f3 <- svs$optCV$f3
    }

    for( k in kseq[-1] ){
      if( !is.null(graphData) ){
        gdat <- graphData[,train$columnsList[[k]],drop=F]
        gdpnts <- train$graphDistOnlyOnPointsList[[k]]
      }
      adm <- constructModAtTheta_np(
        theta=svs$optCV$theta, #there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
        K=K,
        Y=Y[,train$columnsList[[k]],drop=F], #initializes Zeta, u, v and hmats
        formula=formula,
        z.formula=z.formula,
        data=data[train$columnsList[[k]],,drop=F],
        zdata=zdata[train$columnsList[[k]],,drop=F],
        train.si=seq(length(train$columnsList[[k]])),
        Q=Q,
        type=type,
        id.basis=train$qidsList[[k]],
        perm=FALSE, perm.cols=NULL, #these allow constructing permutation models
        graphData=gdat,
        graphDistFn=graphDistFn,
        graphDistOnlyOnPoints=gdpnts, #train.si[id.basis]
        rowOffsets = train$rowOffsetsList[[k]],
        colOffsets = train$colOffsetsList[[k]][train$columnsList[[k]]],
        n_id.basis = NULL,
        init=TRUE
      )

      sv_k <- svdxAtThetaC( Y=Y[,train$columnsList[[k]],drop=F],
                            f2 = svs$optCV$f2,
                            f3 = .f3,
                            theta = svs$optCV$theta,
                            K=K,
                            allDataMod = adm,
                            fitter_maxIter=init_fitter_maxIter,
                            fitter_ftol_abs=init_fitter_ftol_abs,
                            fitter_xtol_abs=init_fitter_xtol_abs,
                             fitter_inner_maxIter=init_fitter_inner_maxIter,
                             fitter_inner_ftol_abs=init_fitter_inner_ftol_abs,
                            #Q=Q,
                            f1=f1, estK = FALSE
      )
      vmatList[[k]] <- sv_k$vmat
      ZetaList[[k]] <- t(sv_k$Zeta)
      umatList[[k]] <- 0*sv_k$umat#this initialization performed significantly better than using svdx initialization for these two parameters because when solving for M(theta)h=w, h needs to match with M(theta), and that wasn't happening through CV 
      hmatList[[k]] <- 0*sv_k$hmat#this initialization performed significantly better than using svdx initialization for these two parameters because when solving for M(theta)h=w, h needs to match with M(theta), and that wasn't happening through CV 
    }
    #cat( "  ... done. \n")

    #aphi
    if( family=="poisson" ){
      aphiList <- lapply( kseq, function(k) rep( 1, nrow( Y ) ) )
    } else if( family=="gaussian" ){
      aphiList <- lapply( kseq, function(k) mseRowWiseOnColumnsC( Y, train$columnsList[[k]], train$ZList[[k]],
                                                                 ZetaList[[k]], umatList[[k]], vmatList[[k]],
                                                                 train$rowOffsetsList[[k]], train$colOffsetsList[[k]]
      ) )
    }

  }
  else {
    
    if(is.null(wd)){
      stop("wd is null in initalizeHUVZ_g")
    }

    for( d in seqD ){
      if( family[[d]] == "poisson" ){
        if("dgCMatrix" %in% class(Y[[d]])){
          Y[[d]]@x = log(1+Y[[d]]@x)
          #Offsets are assumed to be in the right scale (log for poisson; )
        } else {
          Y[[d]]=log(1+Y[[d]]) #users will log transform on their end if needed.
        }
      }
    }
    gdat <- gdpnts <- .f2 <- .f3 <- NULL
    if( !is.null(graphData) ){
      gdat <- graphData[,train$columnsList[[1]],drop=F]
      gdpnts <- train$graphDistOnlyOnPointsList[[1]]
    }
    svs <- isvdx( Y=lapply(Y, function(x) x[,train$columnsList[[1]],drop=F]),
                       K=K,
                       formula=formula,
                       zformula = z.formula,
                       data=data[train$columnsList[[1]],,drop=F],
                       zdata=lapply(zdata, function(x) x[train$columnsList[[1]],,drop=F]),
                       graphData = gdat,
                       graphDistFn = graphDistFn,
                       graphDistOnlyOnPoints = gdpnts,
                       family = NULL,
                       rowOffsets=train$rowOffsetsList[[1]],
                       colOffsets=lapply( train$colOffsetsList[[1]], function(x) x[ train$columnsList[[1]] ] ),
                       fitter_control = init_control$fitter,
                       optCV_control = init_control$optCV,
                       nlopt_control = init_control$nlopt,
                       idBasis = train$qidsList[[1]],
                       modelType = type,
                       #Q=Q,
                       f1=f1,
                       wd=wd,
                       nfolds=kfolds,
                       est_K = FALSE,
                       numThreads=numThreads,
                       setNumThreads = setNumThreads)
    
    vmatList[[1]] <- svs$fitAtOptCV$vmat
    ZetaList[[1]] <- lapply(svs$fitAtOptCV$Zeta, function(x) t(x) )
    hmatList[[1]] <- 0*svs$fitAtOptCV$hmat
    umatList[[1]] <- 0*svs$fitAtOptCV$umat
    
    .theta <- svs$optCV$theta
    .f2 <- svs$optCV$f2
    .nestedTheta <- svs$optCV$theta
    .nestedF2 <- svs$optCV$f2
    .nestedH <- svs$optCV$h
    .nestedCVObjectiveTrace <- svs$optCV$cvObjectiveTrace
    .f3 <- NULL
    if( !is.null(graphData) ){
      .f3 <- svs$optCV$f3
    }
    for( k in kseq[-1] ){
      if( !is.null(graphData) ){
        gdat <- graphData[,train$columnsList[[k]],drop=F]
        gdpnts <- train$graphDistOnlyOnPointsList[[k]]
      }

      adm <- constructModAtTheta_np(
        theta=svs$optCV$theta, #there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
        K=K,
        Y=lapply(Y, function(x) x[,train$columnsList[[k]],drop=F]), #initializes Zeta, u, v and hmats
        formula=formula,
        z.formula=z.formula,
        data=data[train$columnsList[[k]],,drop=F],
        zdata=lapply(zdata, function(x) x[train$columnsList[[k]],,drop=F]),
        train.si=seq(length(train$columnsList[[k]])),
        Q=Q,
        type=type,
        id.basis=train$qidsList[[k]],
        perm=FALSE, perm.cols=NULL, #these allow constructing permutation models
        graphData=gdat,
        graphDistFn=graphDistFn,
        graphDistOnlyOnPoints=gdpnts, #train.si[id.basis]
        rowOffsets = train$rowOffsetsList[[k]],
        colOffsets = lapply( train$colOffsetsList[[k]], function(x) x[ train$columnsList[[k]] ] ),
        n_id.basis = NULL,
        init=TRUE, 
        wd=wd
        
      )
      
      
      sv_k <- isvdxAtThetaC( Y=lapply(Y, function(x)x[,train$columnsList[[k]],drop=F]),
                            f2 = svs$optCV$f2,
                            f3 = .f3,
                            theta = svs$optCV$theta,
                            K=K,
                            allDataMod = adm,
                            fitter_maxIter=init_fitter_maxIter,
                            fitter_ftol_abs=init_fitter_ftol_abs,
                            fitter_xtol_abs=init_fitter_xtol_abs,
                             fitter_inner_maxIter=init_fitter_inner_maxIter,
                             fitter_inner_ftol_abs=init_fitter_inner_ftol_abs,
                            #Q=Q,
                            f1=f1, estK = FALSE
      )
      
      vmatList[[k]] <- sv_k$vmat
      ZetaList[[k]] <- lapply(sv_k$Zeta, function(x) t(x) )
      umatList[[k]] <- 0*sv_k$umat
      hmatList[[k]] <- 0*sv_k$hmat
    }
    #cat( "... done with svs. ", "\n")

    #aphi
    aphiList <- lapply( kseq, function(k){
      lapply( seq_along(Y), function(d){
        if( family[[d]] =="poisson" ){
          rep( 1, nrow( Y[[d]] ) )
        } else if (family[[d]]=="gaussian") {
          mseRowWiseOnColumnsC( Y[[d]],
                                train$columnsList[[k]],
                                train$ZList[[k]][[d]],
                                ZetaList[[k]][[d]],
                                umatList[[k]],
                                vmatList[[k]][[d]],
                                train$rowOffsetsList[[k]][[d]],
                                train$colOffsetsList[[k]][[d]]
          )
        }
      } )
    } )

  }

  list(
    "ZetaList" = ZetaList,
    "hmatList" = hmatList,
    "umatList" = umatList,
    "vmatList" = vmatList,
    "aphiList" = aphiList,
    "theta" =.theta, #save the optcv pars from first fold for later
    "f2"=.f2,
    "f3"=.f3,  #this will be NULL if graph isn't used
    "family"=family,
    "nestedTheta"=.nestedTheta,
    "nestedF2"=.nestedF2,
    "nestedH"=.nestedH,
    "nestedCVObjectiveTrace"=.nestedCVObjectiveTrace
  )

}
