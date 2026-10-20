
#revised
constructModAtTheta_g <- function(
    theta=NULL, #if null, we will run the full svdx, get theta, f2 and f3 if needed for initializing umat, vmat, hmat
    f2=NULL, # if null, ditto as above
    f3=NULL, #if null, ditto as above
    Y,
    family,
    formula,
    z.formula,
    K,
    data,
    zdata,
    train.si=seq(nrow(data)),
    graphData=NULL,
    graphDistFn=NULL,
    graphDistOnlyOnPoints=train.si,
    rowOffsets,
    colOffsets,
    type=NULL,
    id.basis=NULL,
    n_id.basis = NULL,
    f1=1,
    Q=NULL,
    perm=FALSE, perm.cols=NULL, #these allow constructing permutation models
    init_control = list(),
    fitter_control = list(),
    numThreads = 2,
    setNumThreads = FALSE,
    kfolds=5, wd = NULL#,
    #...
){

  # all data, zdata would be subset for train.si
  # for id.basis and graphDistOnlyOnPoints when non-null, will be
  #   matched to train.si. So pass idBasis and graphDistOnlyOnPoints in the train.si scale
  #   For example, if train.si is c(1,9,3,4,5), train.si can be c(1,9,4) , but not c(1,2,4)

  if( is.null(family) ){
    stop( "family must be defined. If not, use the function constructModAtTheta()" )
  } 
  

  # Internal R interfaces use bundled controls only. Direct callers that do
  # not specify init_control receive the package defaults.
  init_control <- .normalize_init_control(
    init_control,
    .default_fitter_control(),
    .default_optCV_control(),
    .default_nlopt_control()
  )
  init_fitter_maxIter <- init_control$fitter$maxIter
  init_fitter_eps <- init_control$fitter$ftol_abs

  fitter_control <- .normalize_fitter_control(
    fitter_control,
    .default_fitter_control()
  )

  if(!is.list(Y)){
    if(family=="poisson"){
      if("dgCMatrix" %in% class(Y)){
        Y@x = log(1+Y@x)
        #offsets need to be in the right scale; example -- log for poisson
      } else {
        Y=log(1+Y) #users need to log transforms
      }
    }
  } else {
    for( d in seq_along(Y) ){
      if( family[[d]] == "poisson" ){
        if("dgCMatrix" %in% class(Y[[d]])){
          Y[[d]]@x = log(1+Y[[d]]@x)
          #Offsets are assumed to be in the right scale (log for poisson; )
        } else {
          Y[[d]]=log(1+Y[[d]]) #users will log transform on their end if needed.
        }
      }
    }
  }
  
  
  # Initialize zeta,h,u,v mats via svdx-np
  adm <- constructModAtTheta_np(
    theta=theta, #there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
    data=data,
    zdata=zdata,
    train.si=train.si,
    formula=formula,
    z.formula=z.formula,
    Q=Q,
    type=type,
    id.basis=id.basis,
    perm=perm, perm.cols=perm.cols, #these allow constructing permutation models
    graphData=graphData,
    graphDistFn=graphDistFn,
    graphDistOnlyOnPoints=graphDistOnlyOnPoints, #train.si[id.basis]
    n_id.basis = NULL,
    rowOffsets = rowOffsets,
    colOffsets = colOffsets,
    init=TRUE, 
    K = K, Y=Y,
    includeSubsetDataForTrainSi = TRUE, #we want to the adjusted graph data, data, zdata,id.basis for trainsi
    wd = wd
  )

  ft0 <- list()
  if( is.null(theta) || is.null(f2) ||
      ( is.null(f3) && (! is.null(graphData))  )
      )  { #indicates we don't have initial conditions for these and umat, vmat, zeta, hmat

    svs <- NA
    if( !is.list(Y) ){
      
      svs <- svdx( Y=Y[,train.si],
                         K=K,
                         formula=formula,
                         zformula = z.formula,
                         data=adm$data,
                         zdata=adm$zdata,
                         idBasis = adm$id.basis,
                         modelType = type,
                         family = NULL,
                         #Q=Q,
                         rowOffsets=adm$rowOffsets,
                         colOffsets=adm$colOffsets,
                         graphData = adm$graphData, #this is adjusted for train.si
                         graphDistFn = graphDistFn,
                         graphDistOnlyOnPoints = adm$graphDistOnlyOnPoints,#this is adjusted for train.si
                         f1=f1,
                         nfolds=kfolds,
                         numThreads=numThreads,
                         fitter_control = init_control$fitter,
                         optCV_control = init_control$optCV,
                         nlopt_control = init_control$nlopt,
                         est_K = FALSE, setNumThreads = setNumThreads)

    } else {
      
      svs <- isvdx( Y=lapply(Y, function(x) x[,train.si,drop=F]),
                         K=K,
                         formula=formula,
                         zformula = z.formula,
                         data=adm$data,
                         zdata=adm$zdata,
                         idBasis = adm$id.basis,
                         modelType = type,
                         family = NULL,
                         #Q=Q,
                         rowOffsets=adm$rowOffsets,
                         colOffsets=adm$colOffsets,
                         graphData = adm$graphData, #this is adjusted for train.si
                         graphDistFn = graphDistFn,
                         graphDistOnlyOnPoints = adm$graphDistOnlyOnPoints,#this is adjusted for train.si
                         f1=f1,
                         nfolds=kfolds,
                         numThreads=numThreads,
                         fitter_control = init_control$fitter,
                         optCV_control = init_control$optCV,
                         nlopt_control = init_control$nlopt,
                         est_K = FALSE, setNumThreads = setNumThreads, wd=wd)

    }


    ft0$hmat <- svs$fitAtOptCV$hmat
    ft0$umat <- svs$fitAtOptCV$umat
    ft0$vmat <- svs$fitAtOptCV$vmat
    ft0$Zeta <- svs$fitAtOptCV$Zeta
    ft0$theta <- svs$optCV$theta
    ft0$f2 <- svs$optCV$f2
    ft0["f3"] <- list(NULL)
    if( !is.null(graphData) ){
      ft0$f3 <- svs$optCV$f3
    }
    
  } else {


    if( !is.list( Y ) ){
      ft0 <- svdxAtThetaC(
        Y=Y,
        f2=f2,
        f3=f3,
        theta = theta,
        K=K,
        allDataMod = adm,
        fitter_maxIter = fitter_control$maxIter,
        fitter_ftol_abs = fitter_control$ftol_abs,
        fitter_xtol_abs = fitter_control$xtol_abs,
        fitter_inner_maxIter = fitter_control$inner_maxIter,
        fitter_inner_ftol_abs = fitter_control$inner_ftol_abs,
        #Q=Q,
        f1=f1,estK = FALSE
      )

    } else {
      ft0 <- isvdxAtThetaC(
        Y=Y,
        f2=f2,
        f3=f3,
        theta = theta,
        K=K,
        allDataMod = adm,
        fitter_maxIter = fitter_control$maxIter,
        fitter_ftol_abs = fitter_control$ftol_abs,
        fitter_xtol_abs = fitter_control$xtol_abs,
        fitter_inner_maxIter = fitter_control$inner_maxIter,
        fitter_inner_ftol_abs = fitter_control$inner_ftol_abs,
        #Q=Q,
        f1=f1,estK = FALSE
      )
    }


    ft0$theta <- theta
    ft0$f2 <- f2
    ft0["f3"] <- list( f3 )

  }



  res <- adm #trainmodel
  res$vmat <- ft0$vmat
  res$umat <- 0*ft0$umat #nasty hessians & gradients occur when we initialize M(desired theta) for exp family, but input hmat from non-family i/svdx that doesn't match the theta; so zeroing it out
  res$hmat <- 0*ft0$hmat #nasty hessians & gradients occur when we initialize M(desired theta) for exp family, but input hmat from non-family i/svdx that doesn't match the theta; so zeroing it out
  
  #here we are not choosing from ft0 because we know the theta that we want to construct
  res$theta <- theta
  res$f2 <- f2
  res$f3 <- f3

  #Zeta and aphi initialization
  if( is.list( Y ) ){
    res$Zeta <- lapply(ft0$Zeta, function(x) t(x) )
    res$aphi <- init_aphi_integ( Y = Y,
                                 Z= res$Z,
                                 Zeta = res$Zeta,
                                 umat = res$umat,
                                 vmat = res$vmat,
                                 family = family,
                                 columns = train.si,
                                 transZeta = FALSE,
                                 rowOffsets=rowOffsets,
                                 colOffsets=colOffsets )


  } else {
    res$Zeta <- t(ft0$Zeta)
    res$aphi <- init_aphi( Y,
                           Z = res$Z, Zeta = res$Zeta,
                           umat = res$umat, vmat = res$vmat,
                           family = family, columns = train.si, transZeta = FALSE,
                           rowOffsets=rowOffsets, colOffsets=colOffsets
    )
  }

  res$family <- family
  
  #debug
  if(!is.null(theta)){
    theta_out <- theta   # explicit requested theta for constructModAtTheta_g()
    Rtheta_out <- Reduce(
      `+`,
      lapply(seq_along(theta_out), function(i) {
        theta_out[i] * adm$mod$r[, , i]
      })
    )
    Mtheta_out <- cbind(adm$mod$s, Rtheta_out)
    res$debug_ft0_umat_minus_requested_Mh <-
      apply(
        abs(ft0$umat - Mtheta_out %*% ft0$hmat),
        2,
        max
      )
    Rtheta_ft0 <- Reduce(
      `+`,
      lapply(seq_along(ft0$theta), function(i) {
        ft0$theta[i] * adm$mod$r[, , i]
      })
    )
    Mtheta_ft0 <- cbind(adm$mod$s, Rtheta_ft0)
    res$debug_ft0_umat_minus_ft0_Mh <-
      apply(
        abs(ft0$umat - Mtheta_ft0 %*% ft0$hmat),
        2,
        max
      )
    
  }
  


  res

}
