
#If using for integrative, we expect:
# z.formula to be a list one for each dataset.
# zdata to be a list of dataframes.

#If integrative, Z, ZLS, Zeta, and vmat become vectors for every fold.
# If single, they remain matrices.

constrTrainTestModsForFolds <- function(Y,
                                        K,
                                        data,
                                        zdata,
                                        formula,
                                        z.formula,
                                        family = NULL,
                                        rowOffsets=NULL,
                                        colOffsets=NULL,
                                        type=NULL,
                                        Q=NULL,
                                        id.basis=NULL,
                                        n_id.basis = NULL,
                                        graphData = NULL,
                                        graphDistFn=NULL,
                                        graphDistOnPoints=seq(nrow(data)),
                                        trainFolds = NULL,
                                        testFolds=NULL,
                                        kfolds=NULL,
                                        init_control = list(),
                                        f1 = NULL,
                                        numThreads= 2,
                                        setNumThreads = FALSE, 
                                        wd = NULL
                                        ){

  if (!requireNamespace("Matrix", quietly = TRUE)) stop("Package 'Matrix' is required.")
  if (!requireNamespace("irlba", quietly = TRUE)) stop("Package 'irlba' is required.")

  n <- nrow(data)
  N0IDBASIS <- max( 30, ceiling(10* n^(2/9))   )
  if( !is.null(id.basis) ){
    message( "id.basis given. Only the number of id.basis is used for each folds" )
    n_id.basis<- length(id.basis)
  }
  if( is.null( n_id.basis ) ){
    n_id.basis <- N0IDBASIS
  }
  if( n_id.basis > n ){
    n_id.basis <- n
  }

  ## SANITY CHECKS

  #Number of datasets
  D <- 1
  if( (is.list( z.formula ))  ){
    D  <- length(z.formula)
    if( !is.list( Y ) ){
      stop( "Y must be a list as z.formula is a list"  )
    }
    if( length(Y)!=D ){
      stop( "Y must be of the same length as z.formula" )
    }
    if( !(all( sapply(Y, ncol) == n ) ) ){
      stop( "Y must be a list of matrices with each entry having the same
              number of columns as data" )
    }
  }

  if( !is.null(family) ){
    if( D ==1 ){
      if( length(family) >1 ){
        stop( "only one family is allowed for a single dataset" )
      }
    }
    if( D > 1 ){
      if( length(family) != D ){
        stop( "One family specification per dataset is needed" )
      }
    }
    if( !all(family %in% c("poisson", "gaussian")) ){
      stop( "Only poisson and gaussian family
          specifications are currently supported." )
    }
  }

  if( !is.null(rowOffsets) ){
    if( D > 1  ) {
      if( !is.list( rowOffsets ) ){
        stop( "Row offsets must be a list with row offsets for each matrix in Y list " )
      }
      if( !all( sapply( rowOffsets, length ) == sapply( Y, nrow ) ) ){
        stop( "Row offsets must have the same length as the number of rows in each matrix of Y list" )
      }
    } else {

      # if( length(rowOffsets)!=nrow() ){
      #   print(rowOffsets)
      #   print( class(rowOffsets) )
      #   stop( "Row offsets must be a vector of same number of rows as Y" )
      # }
      if( length( rowOffsets )!= nrow(Y)  ){
        stop( "Row offsets must have the same length as the number of rows in matrix Y" )
      }
    }
  } else {
    if( D > 1 ){
      rowOffsets <- lapply( Y, function(x) rep(0, nrow(x)) )
    } else {
      rowOffsets <- rep(0, nrow(Y))
    }
  }
  if( !is.null(colOffsets) ){
    if( D > 1  ) {
      if( !is.list( colOffsets ) ){
        stop( "Col offsets must be a list with col offsets for each matrix in Y list " )
      }
      if( !all( sapply( colOffsets, length ) == n ) ){
        stop( "Col offsets must have the same length as the number of colmns in each matrix of Y list" )
      }
    } else {
      # if( !is.vector(colOffsets) ){
      #   stop( "Col offsets must be a vector of same number of columns as Y" )
      # }
      if( length( colOffsets )!= n  ){
        stop( "Col offsets must have the same length as the number of columns in matrix Y" )
      }
    }
  } else {
    if( D > 1 ){
      colOffsets <- lapply( Y, function(x) rep(0, n) )
    } else {
      colOffsets <- rep(0, n)
    }
  }

  if( class(formula) != "formula"  ){
    stop( "formula should be of class formula" )
  }
  if( is.list(z.formula) ){
    if( !all( sapply( z.formula, class ) == "formula" ) ){
      stop( "z.formula must only contain entries of class formula" )
    }
    if( !all( sapply( zdata, nrow ) == n ) ){
      stop( "list zdata should have the same number of rows as data in each entry" )
    }
  } else {
    if(class(z.formula) != "formula"){
      stop("z.formula must be of class formula")
    }
    if( nrow( zdata ) != n ){
      stop( "zdata should have the same number of rows as data" )
    }
  }
  if( is.null(id.basis)  ){
    n_id.basis <- max( 30, N0IDBASIS )
    if( n_id.basis > n ){
      n_id.basis <- n
    }
    id.basis <- sample( n, n_id.basis )
  } else {
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
    if( is.null( graphDistOnPoints ) ){
      message( "graphDistOnPoints cannot be NULL when passing graphData and graphDistFn. Using all data. " )
      graphDistOnPoints<-seq(n)
    } else {
      r <- range( graphDistOnPoints )
      if( (r[1] < 1) || (r[2] > n) ){
        stop( "graphDistOnPoints must be in the range of 1 to nrow(data)" )
      }
    }
  }
  if(is.null(kfolds)){
    kfolds <- 5
  }

  #Step 1:Construct folds
  siAvailable <- 1:n
  if( !is.null(trainFolds) ){
    kfolds <- length(trainFolds)
    if( length( trainFolds )!= length(testFolds) ){
      stop( "Both train and test fold lists must have the same length" )
    }
  } else {
    # set.seed(1)
    # message( "Setting seed in create folds! For benchmarking equivalence in single and integrative. " )
    testFolds <- caret_createFolds( siAvailable, k=kfolds  )
    trainFolds <- lapply( testFolds, function(fold) setdiff(siAvailable,fold) )
  }
  trainFolds <- lapply(trainFolds, sort)
  testFolds <- lapply(testFolds, sort)

  ## STEP 2: Assemble base parameters for each fold. Store in a list.
  #Note: We are doing this storing to do nested foreach loops. So memory
  #may or may not be an issue depending on the #folds, and #q oints
  #cat("Assembling base objects ... \n")
  trainObjsForEachFold <- lapply( trainFolds,
                                  function(trainFold_k, ... ){

                                    Qsub <- Q
                                    if( !is.null(Q) ){
                                      Qsub <- Q[ trainFold_k,,drop=F ]
                                    }

                                    zdatasetForFold <- NA
                                    if( D > 1 ){
                                      zdatasetForFold <- lapply(seq(D), function(d) zdata[[d]][ trainFold_k,,drop=F ])
                                    } else {
                                      zdatasetForFold <- zdata[ trainFold_k,,drop=F ]
                                    }

                                    idBasisForFold <- sample( trainFold_k, size=length(id.basis)*length(trainFold_k)/n ,replace=F )
                                    graphPointsForFold <- graphDataForFold <- NULL
                                    if( !is.null(graphData) ){
                                      graphDataForFold <- graphData[,trainFold_k,drop=F]
                                      graphPointsForFold_oriIndx <- sample( trainFold_k, length(graphDistOnPoints)*length(trainFold_k)/n, replace=F)
                                      graphPointsForFold <- match(graphPointsForFold_oriIndx, trainFold_k)
                                    }
                                    constructMod(data=data[trainFold_k,,drop=F], zdata = zdatasetForFold,
                                                 formula=formula, z.formula=z.formula,
                                                 type=type,Q=Qsub,
                                                 id.basis=match(idBasisForFold, trainFold_k),
                                                 graphData = graphDataForFold,
                                                 graphDistFn=graphDistFn,
                                                 graphDistOnlyOnPoints=graphPointsForFold#,
                                                 #...
                                    )


                                  } )

  testObjsForEachFold <-lapply( seq_along(testFolds),
                                function(k, ... ){
                                  data.fold <- data[testFolds[[k]],,drop=F]
                                  obj_k <- trainObjsForEachFold[[k]]
                                  zdatasetForFold <- NA
                                  if( D > 1 ){
                                    zdatasetForFold <- lapply(seq(D), function(d)zdata[[d]][ testFolds[[k]],,drop=F ])
                                  } else {
                                    zdatasetForFold <- zdata[ testFolds[[k]],,drop=F ]
                                  }

                                  constructTestMods( object=obj_k$mod,
                                                     newdata=data.fold,
                                                     newzdata = zdatasetForFold,
                                                     qids=obj_k$qids )
                                } )


  kseq <- seq_along(trainFolds)

  train <- list()

  #model objects are needed for building constraints via buildM, so we add it
  train$modList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$mod )

  train$columnsList <- lapply( trainFolds, sort)
  train$qidsList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$qids)#already subsetted for sids
  train$SStqList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$SStq) #already subsetted for sids
  train$nqList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$nq )
  train$SList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$S )
  train$RsCboundList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$R0 )
  train$RtildesCboundList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$R0.til )
  train$R0traceList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$R0trace )
  train$thetaCard <- dim( trainObjsForEachFold[[1]]$mod$r )
  if( !is.null( graphData) & !is.null(graphDistFn) & !is.null(graphDistOnPoints) ){
    train$graphDistMatList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$graphDistMat )
    train$graphDistOnlyOnPointsList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$graphDistOnlyOnPoints )
  }
  train$rowOffsetsList <- lapply( kseq, function(k) rowOffsets )
  #train$colOffsetsList <- lapply( kseq, function(k) colOffsets[train$columns[[k]]] ) #TODO: unfortunately, current implementation of fitters e.g., npFitter expects colOffsets as full n length,and subset internally. Future releases should remedy this especially when the rest of the objects are passed as subsetted to them.
  train$colOffsetsList <- lapply( kseq, function(k) colOffsets ) #TODO: unfortunately, current implementation of fitters e.g., npFitter expects colOffsets as full n length,and subset internally. Future releases should remedy this especially when the rest of the objects are passed as subsetted to them (e.g., Rtilde, SStq etc.,).

  train$ZList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$Z )
  train$ZlsList <- lapply( kseq, function(k) trainObjsForEachFold[[k]]$Zls )

  # Zeta, U and V depending on single / multiple datasets
  huv <- list()
  if( is.null(family) ){
    huv <- initalizeHUVZ_np(  trainObjsForEachFold=trainObjsForEachFold,
                              train=train,
                              D=D,
                              Y=Y,
                              K=K,
                              Q=Q,
                              kfolds=kfolds,
                              kseq=kseq, 
                              wd=wd
                              ) #no f1/f2/f3 dependence on this one, simply uses pca
  } else {
    huv <- initalizeHUVZ_g(  trainObjsForEachFold=trainObjsForEachFold,
                             train=train,
                             D=D, Y=Y,K=K,
                             data=data,
                             zdata=zdata,
                             formula=formula,
                             z.formula=z.formula,
                             family = family,
                             type=type,
                             Q=Q,
                             graphData = graphData,
                             graphDistFn=graphDistFn,
                             init_control = init_control,
                             setNumThreads = setNumThreads,
                             f1=f1,
                             kfolds=kfolds,
                             kseq=kseq,
                             wd=wd #need wd for intiailizng with isvdx, unlike initializeHUVZ_np
    )

  }
  train$ZetaList <- huv$ZetaList
  train$hmatList <- huv$hmatList
  train$umatList <- huv$umatList
  train$vmatList <- huv$vmatList
  train$theta <- huv$theta #the theta, f2 and f3 will be results of optcv from initializeHUVZ_g if that is called, otherwise NULL;
  train$f2 <- huv$f2
  train$f3 <- huv$f3
  train$nestedTheta <- huv$nestedTheta
  train$nestedF2 <- huv$nestedF2
  train$nestedH <- huv$nestedH
  train$nestedCVObjectiveTrace <- huv$nestedCVObjectiveTrace
  train$nestedCVObjectiveTrace <- huv$nestedCVObjectiveTrace
  if( "aphiList" %in% names( huv) ){
    train$aphiList <- huv$aphiList
    train$family <- family
  }
  #cat(' ... done.\n')


  test <- list()
  test$columnsList <- lapply(testFolds, sort)
  test$ZList <- lapply( kseq, function(k) testObjsForEachFold[[k]]$Ztest )
  test$SList <-  lapply( kseq, function(k) testObjsForEachFold[[k]]$S )
  test$RsCboundList <- lapply( kseq, function(k) testObjsForEachFold[[k]]$R0test )
  test$rowOffsetsList <- lapply( kseq, function(k) rowOffsets )
  #test$colOffsetsList <- lapply( kseq, function(k) colOffsets[folds[[k]]] )
  test$colOffsetsList <- lapply( kseq, function(k) colOffsets ) #TODO: unfortunately, current implementation of fitters e.g., npFitter expects colOffsets as full n length,and subset internally. Future releases should remedy this especially when the rest of the objects are passed as subsetted to them (e.g., Rtilde, SStq etc.,).

  list("train"=train,
       "test"=test,
       "nTheta"=trainObjsForEachFold[[1]]$nTheta, #number of theta parameters, fixed for all folds
       "wd" = wd
  )
}
