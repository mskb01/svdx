#These functions construct train and test models at a given theta.


constructModAtTheta <- function(
    theta=NULL, #there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
    data,
    zdata,
    train.si=seq(nrow(data)),
    formula,
    z.formula,
    family = NULL,
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
    init_control = list(),
    fitter_control = list(),
    f2 = NULL,
    f3=NULL,
    f1,
    numThreads = 2,
    kfolds=5, 
    wd = NULL #,
    #,
    #...
){

  res <- NA
  if( is.null(family) ){
    res <- constructModAtTheta_np(
      theta=theta, #if null, we will run the full svdx, get theta, f2 and f3 if needed for initializing umat, vmat, hmat
      # f2=f2, # if null, ditto as above
      # f3=f3, #if null, ditto as above
      Y=Y,
      formula=formula,
      z.formula=z.formula,
      K=K,
      data=data,
      zdata=zdata,
      train.si=train.si,
      graphData=graphData,
      graphDistFn=graphDistFn,
      graphDistOnlyOnPoints=graphDistOnlyOnPoints,
      rowOffsets=rowOffsets,
      colOffsets=colOffsets,
      type=type,
      id.basis=id.basis,
      n_id.basis = n_id.basis,
      Q=Q,
      perm=perm, perm.cols=perm.cols, #these allow constructing permutation models
      init=init, 
      wd = wd#,
      #...
    )

  } else {
    res <- constructModAtTheta_g(
      theta=theta, #if null, we will run the full svdx, get theta, f2 and f3 if needed for initializing umat, vmat, hmat
      f2=f2, # if null, ditto as above
      f3=f3, #if null, ditto as above
      Y=Y,
      family = family,
      formula=formula,
      z.formula=z.formula,
      K=K,
      data=data,
      zdata=zdata,
      train.si=train.si,
      graphData=graphData,
      graphDistFn=graphDistFn,
      graphDistOnlyOnPoints=graphDistOnlyOnPoints,
      rowOffsets=rowOffsets,
      colOffsets=colOffsets,
      type=type,
      id.basis=id.basis,
      n_id.basis = n_id.basis,
      f1=f1,
      Q=Q,
      perm=perm, perm.cols=perm.cols, #these allow constructing permutation models
      init_control = init_control,
      fitter_control = fitter_control,
      numThreads = numThreads,
      setNumThreads=FALSE,
      kfolds = kfolds, 
      wd = wd#, 
      #...
    )

  }
  res$wd <- wd
  
  res

}


constructTestModAtTheta <- function(theta, object, newdata,newzdata=newdata,
                                    qids=object$qids,
                                    include = c(object$terms$labels,
                                                object$lab.p),
                                    ...
){

  m <- constructTestMods( object=object,
                          newdata=newdata,
                          newzdata=newzdata,
                          qids=qids,
                          include = include,
                          ... )

  if( length(theta) != length( m$Rs ) ){
    stop( "theta must be of the same length as number of RKs" )
  }

  list( "MTheta"=with(m, cbind( Stest, Reduce("+", lapply( seq_along( Rs ), function(k) theta[k]*Rs[[k]] )) ) ),
        "Z"=m$Ztest,
        "Zls"=m$Zlstest
  )

}

