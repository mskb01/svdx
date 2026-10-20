#' Fit an iSSANOVA-SVD model for multiple datasets
#'
#' Fits an interpretable integrative matrix factorization model for a features by samples
#' data matrices measured on the same/matched samples (e.g., multi-modal settings) using sample covariates. 
#' The model represents the data as
#' low-rank structure whose sample scores are smooth, interpretable functions of across-datasets shared
#' covariates relevant to a study, supplied through \code{formula} and \code{xdata}, while optional nuisance 
#' covariates (e.g., batches) to adjust individual datasets for, may be supplied through \code{zformula} 
#' and \code{zdata} (each a list, one for each dataset). Predictions of processes, can be made to reconstruct 
#' sample scores for external datasets, without double dipping. 
#'
#' @param Y A named or unnamed list of feature-by-sample matrices. All matrices
#'   must have the same number of columns/samples, but may have different
#'   numbers of rows/features. If unnamed, datasets are named
#'   \code{dataset_1}, \code{dataset_2}, and so on.
#' @param K Maximum number of components to fit.
#' @param formula Formula describing the shared metadata-driven component scores.
#' @param zformula A list of nuisance formulas, one per dataset.
#' @param data Shared sample metadata data frame. Rows correspond to
#'   samples/columns of every matrix in \code{Y}.
#' @param zdata A list of data frames, one per dataset, containing variables used
#'   in \code{zformula}.
#' @param wd A vector of length equal to the number of datasets, weighting each dataset.
#' @param family Optional list of response families, one per dataset. \code{poisson} / \code{gaussian} 
#'    for each entry accepted. If \code{family} = \code{quasipoisson}, a poisson based fit will be returned. 
#'    Note: Inference workflows will still respect \code{quasipoisson}.  
#'    Use \code{NULL} for the default non-family fitting path. 
#' @param graphData Optional across-dataset shared graph data used by graph-aware model components. Must 
#'   have dimensions graphFeatures x samples. This will be used with \code{graphDistFn} 
#'   to calculate graph based distance matrices. So both must be supplied for graph aware fitting. 
#' @param graphDistFn  Optional function for graph distances. Must return a positive
#'   semi-definite / definite matrix. 
#' @param graphDistOnlyOnPoints  Indices on which graph distances are evaluated.
#' @param rowOffsets Optional feature-level offsets.
#' @param colOffsets Optional sample-level offsets.
#' @param modelType Optional model type passed to lower-level fitting routines.
#' @param idBasis Optional sample indices on which the penalized component is approximated with (basis identifiers). Default
#'                chooses a subset for speed. 
#' @param f1 default 1, scales the loss function part of the iSSANOVA-SVD objective (not the penalty).
#' @param nfolds Number of folds used for cross-validation. Default 5. For large datasets, higher nfolds is preferred (e.g., 10). 
#' @param est_K Logical; whether to estimate the number of components.
#' @param fitter_control Named list of fitter controls: `ftol_abs`, `xtol_abs`, `maxIter`, `inner_maxIter`, and `inner_ftol_abs`.
#' @param optCV_control Named list of outer cross-validation coordinate-descent controls: `ftol_abs`, `xtol_abs`, and `maxIter`.
#' @param nlopt_control Named list of NLopt controls: `ftol_abs`, `xtol_abs`, and `maxeval`.
#' @param init_control Optional family-initialization overrides with named sublists `fitter`, `optCV`, and `nlopt`. Missing entries inherit from the corresponding main controls.
#' @param numThreads Number of threads for parallel computation.
#' @param setNumThreads Logical; whether to set the RcppParallel thread count.
#' @param optCV_maxIter (legacy, use bundled \code{*_control} options above ) Maximum iterations for cross-validation optimization.
#' @param optCV_eps (legacy, use bundled \code{*_control} options above ) Convergence tolerance for cross-validation optimization.
#' @param fitter_maxIter (legacy, use bundled \code{*_control} options above ) Maximum iterations for the model fitter.
#' @param fitter_eps (legacy, use bundled \code{*_control} options above ) Convergence tolerance for the model fitter.
#' @param xtol_abs (legacy, use bundled \code{*_control} options above ) Absolute parameter tolerance for numerical optimization.
#' @param ftol_abs (legacy, use bundled \code{*_control} options above ) Absolute objective tolerance for numerical optimization.
#'
#'
#' @seealso \code{\link{runISVDXReport}, \link{runISVDXWorkflow}, \link{svdx}, \link{runSVDXReport}, \link{runSVDXWorkflow} }
#'
#' @return An object of class `"iSSANOVASVD"` containing shared component scores,
#'   dataset-specific loading matrices, nuisance effects, selected tuning
#'   parameters, and training-model objects.
#'
#' @details
#' \code{isvdx()} is the integrated analogue of \code{svdx()}. It is intended for
#' multi-view, multi-modal, multi-omics settings where several datasets are measured on
#' the same/matched samples. The shared score functions are learned from common sample
#' metadata, while each dataset receives its own loading matrix and optional
#' nuisance adjustment terms.
#'
#' @examples
#' \dontrun{
#' Y <- list(
#'   rna = Y_rna,
#'   protein = Y_protein
#' )
#'
#' zformula <- list(
#'   rna = ~ batch,
#'   protein = ~ plate
#' )
#'
#' zdata <- list(
#'   rna = rna_meta,
#'   protein = protein_meta
#' )
#'
#' fit <- isvdx(
#'   Y = Y,
#'   K = 5,
#'   formula = ~ group * time,
#'   zformula = zformula,
#'   data = sample_data,
#'   zdata = zdata, 
#'   modelType = list( "time"=list( "cubic" , range(sample_data$time) ) )
#' )
#'
#' summary(fit, Y = Y)
#' plotVHeatmap(fit, Y = Y$rna, data = sample_data, dataset = "rna")
#' }
#'
#' @export
isvdx <- function( Y, #list of matrices (dense, then sparse matrices in list)
                   K,
                   formula,
                   zformula=lapply( seq_along(Y), function(x) ~1 ),
                   data,
                   zdata=lapply( seq_along(Y), function(x) data ),
                   wd = NULL,
                   family = NULL,
                   graphData = NULL,
                   graphDistFn = NULL,
                   graphDistOnlyOnPoints = seq_len(ncol(Y[[1]])),
                   rowOffsets=lapply(Y, function(x) rep(0, nrow(x)) ),
                   colOffsets=lapply(Y, function(x) rep(0, ncol(x)) ),
                   modelType=NULL,
                   idBasis=NULL,
                   f1=1,
                   nfolds=5,
                   est_K = TRUE, #if K needs to be estimated.
                   fitter_control = list(),
                   optCV_control = list(),
                   nlopt_control = list(),
                   init_control = list(),
                   numThreads=RcppParallel::defaultNumThreads()/2,
                   setNumThreads = TRUE,  #useful when calling recursively to set once and then false
                   
                   # ----- legacy ------#
                   optCV_maxIter=20, #this maxIter is optCV maxIter
                   optCV_eps=.01,     #for optCV eps
                   fitter_maxIter=200,
                   fitter_eps=.001,
                   xtol_abs = .01, #nlopt xtol parameer
                   ftol_abs = .001 #nlopt ftol parameter
                   
){

  legacy_supplied <- list(
    fitter_maxIter = !missing(fitter_maxIter),
    fitter_eps = !missing(fitter_eps),
    optCV_maxIter = !missing(optCV_maxIter),
    optCV_eps = !missing(optCV_eps),
    xtol_abs = !missing(xtol_abs),
    ftol_abs = !missing(ftol_abs)
  )
  controls <- .resolve_optimizer_controls(
    fitter_control = fitter_control,
    optCV_control = optCV_control,
    nlopt_control = nlopt_control,
    init_control = init_control,
    fitter_maxIter = fitter_maxIter,
    fitter_eps = fitter_eps,
    optCV_maxIter = optCV_maxIter,
    optCV_eps = optCV_eps,
    xtol_abs = xtol_abs,
    ftol_abs = ftol_abs,
    legacy_supplied = legacy_supplied
  )

  fitter_maxIter <- controls$fitter$maxIter
  fitter_eps <- controls$fitter$ftol_abs
  optCV_maxIter <- controls$optCV$maxIter
  optCV_eps <- controls$optCV$xtol_abs
  xtol_abs <- controls$nlopt$xtol_abs
  ftol_abs <- controls$nlopt$ftol_abs

  Q <- NULL
  trainFolds <- NULL
  testFolds <- NULL
  
  if (!is.null(trainFolds) || !is.null(testFolds)) {
    stop(
      "Custom trainFolds/testFolds are not currently supported in the public API.",
      call. = FALSE
    )
  }
  
  if( !is.list(Y) ){
    stop(
    "This function performs integrative analysis.
    A list of matrices ( numbers of rows can vary, but same number of columns)
    are needed for input Y.
    Use svdx(), if processing a single dataset.
    " )
  }
  D <- length(Y)
  
  if( any( sapply(Y, dim) <= 0 ) ){
    stop("Data matrices must be non-empty")
  }
  if( is.null(wd) ){
    wd <- 1/sapply(Y, nrow)
  } else {
    if( is.list(wd) ){
      wd <- unlist(wd)
    }
    if((length(wd)!=D) || (any(wd <= 0)) || any(is.infinite(wd)) ){
      stop("wd must be positive, length of wd should equal the number of datasets in Y ")
    }
  }
  wd <- as.numeric(wd)
  wd <- wd/exp(mean(log(wd)))
  
  dataset_names <- names(Y)
  if (is.null(dataset_names) || any(dataset_names == "")) {
    dataset_names <- paste0("dataset_", seq_len(D))
    names(Y) <- dataset_names
  } else {
    if (any(dataset_names == "")) {
      stop(
        "Dataset names cannot contain empty strings.",
        call. = FALSE
      )
    }
    if (anyDuplicated(dataset_names)) {
      stop(
        "Dataset names must be unique.",
        call. = FALSE
      )
    }
  }
  dataset_names <- names(Y)
  
  if( (!is.list(zformula)) ||
      (length(zformula)!=length(Y))
      ){
    stop("zformula must be a list with each entry a formula for the corresponding data matrix in Y ")
  }
  names(zformula) <- dataset_names

  if(!is.null(family)){
    family <- unlist( family )
    if( length(family) != length(Y) ){
      stop( "One family specification per data matrix is needed" )
    }
    if( !all(family %in% c("poisson", "gaussian", "quasipoisson")) ){
      stop( "Only poisson and gaussian family
          specifications are currently supported." )
    }
    family[which(family == "quasipoisson")] <-"poisson"
    names(family) <- dataset_names
  }

  if( is( data, "list" ) ){
    stop( "data must be a single data frame, used for all matrices in Y" )
  }
  if( !is( zdata, "list" ) ){
    stop( "zdata must be a list of data frames, one for each matrix in Y" )
  }
  names(zdata) <- dataset_names

  if( !is.null(trainFolds) ){
    message( "trainfolds are given. passed as SEXP. Much SLOWER code at least in teh case of gsvdx & igsvdx for some reason! FIX TODO!" )
    if( is.null(testFolds) ){
      stop( "Test folds should also be given. " )
    }
    if( length(trainFolds) != length(testFolds) ){
      stop( "Both train and testfold lists must have the same length" )
    }
    nfolds <- length(trainFolds)
  }
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads = numThreads)
    message(sprintf("Using %d threads.",numThreads))
  }

  n <- ncol(Y[[1]])

  #SANITY CHECKS

  if( (nrow(data) != n) ||
      (any(sapply(zdata, nrow )!=n)) ||
      (any(sapply(colOffsets, length)!=n))
  ){
    stop("Number of rows in data and zdata, and length of colOffsets must equal the
         total number of columns in Y")
  }
  if( any( sapply( seq_along(Y), function(d) nrow(Y[[d]])!=length(rowOffsets[[d]]) )  ) ){
    stop( "Length of rowOffsets must equal the total number of rows in Y" )
  }

  if( !is.null(Q)){
    if( n != nrow(Q) ){
      stop("Number of rows in Q must equal trainColumns.
           Q is components matrix to be adjusted for, where the
           components with sample loadings corresponding to
           trainColumns. ")
    }
  }

  # Two models: one collcetion for train test folds, the other is for entire data
  #construct train test folds
  cat(' + Constructing train test folds ... \n')
  ttf <- constrTrainTestModsForFolds(Y=Y, K=K,
                                     data=data,
                                     zdata=zdata,
                                     formula=formula,
                                     z.formula=zformula,
                                     family = family,
                                     rowOffsets=rowOffsets,
                                     colOffsets=colOffsets,
                                     type=modelType,
                                     Q=Q,
                                     id.basis=idBasis,
                                     n_id.basis =NULL,
                                     graphData = graphData,
                                     graphDistFn=graphDistFn,
                                     graphDistOnPoints=graphDistOnlyOnPoints,
                                     trainFolds = trainFolds,
                                     testFolds=testFolds,
                                     kfolds=nfolds,
                                     init_control = controls$init,
                                     f1 = f1, 
                                     wd = wd#,
                                     #setNumThreads = FALSE #already set
                                     )
  cat('... done. ')


  #model for all data
  cat(' + Constructing all data model ...\n')

  adm <- constructModAtTheta(
    #theta=ttf$train$theta, #
    theta=NULL, #for np, there is no f1, f2 dependence here because we are only constructing the theta mod, without runing any other svdx routines, unlike onstructModAtTheta_g.
    f2 = ttf$train$f2,       #for _g, ttf will hold theta, f2, f3 from doing CV on the first fold to get umat, vmat, zeta initial conditions.
    f3 = ttf$train$f3,
    f1=f1,
    data=data,
    zdata=zdata,
    train.si=seq(nrow(data)),
    formula=formula,
    z.formula=zformula,
    family = family,
    Q=Q,
    type=modelType,
    id.basis=idBasis,
    perm=FALSE, perm.cols=NULL, #these allow constructing permutation models
    graphData=graphData,
    graphDistFn=graphDistFn,
    graphDistOnlyOnPoints=graphDistOnlyOnPoints, #train.si[id.basis]
    n_id.basis = NULL,
    init=TRUE, K = K, Y=Y, #initializes Zeta, u, v and hmats
    rowOffsets = rowOffsets,
    colOffsets = colOffsets,
    init_control = controls$init,
    fitter_control = controls$fitter,
    numThreads = numThreads,
    kfolds = nfolds, 
    wd = wd
  )
  cat(' ... done.\n')

  cat( " + Fitting model ...\n" )
  res <- isvdxC(Y=Y,
                     allDataMod=adm,
                     trainTestFoldMods=ttf,
                     optCV_maxIter=controls$optCV$maxIter,
                     optCV_xtol_abs=controls$optCV$xtol_abs,
                     optCV_ftol_abs=controls$optCV$ftol_abs,
                     fitter_maxIter=controls$fitter$maxIter,
                     fitter_ftol_abs=controls$fitter$ftol_abs,
                     fitter_xtol_abs=controls$fitter$xtol_abs,
                     fitter_inner_maxIter=controls$fitter$inner_maxIter,
                     fitter_inner_ftol_abs=controls$fitter$inner_ftol_abs,
                     nlopt_xtol_abs=controls$nlopt$xtol_abs,
                     nlopt_ftol_abs=controls$nlopt$ftol_abs,
                     nlopt_maxeval=controls$nlopt$maxeval,
                     f1=f1,
                     estK = est_K,
                     KMAX=K, 
                numThreads = numThreads
  );
  cat( " done. \n" )
  res$fitAtOptCV$rowOffsets <- rowOffsets
  res$fitAtOptCV$colOffsets <- colOffsets
  res$fitAtOptCV$trainModel <- adm
  res$optCV$TrainTestFolds <- ttf
  res$fitAtOptCV$wd <- wd
  

  rownames(res$fitAtOptCV$umat) <- colnames(Y[[1]])
  colnames(res$fitAtOptCV$umat) <- paste0("IRCF_", seq( ncol(res$fitAtOptCV$umat) ))
  for( d in seq(length(Y)) ){
    colnames(res$fitAtOptCV$vmat[[d]]) <- colnames(res$fitAtOptCV$umat)
    rownames(res$fitAtOptCV$vmat[[d]]) <- rownames(res$fitAtOptCV$Zeta[[d]]) <- rownames(Y[[d]])
  }
  
  res$dataset_names <- dataset_names
  if (!is.null(res$fitAtOptCV$vmat) && is.list(res$fitAtOptCV$vmat)) {
    names(res$fitAtOptCV$vmat) <- dataset_names
  }
  if (!is.null(res$fitAtOptCV$Zeta) && is.list(res$fitAtOptCV$Zeta)) {
    names(res$fitAtOptCV$Zeta) <- dataset_names
  }
  if (!is.null(res$fitAtOptCV$rowOffsets) && is.list(res$fitAtOptCV$rowOffsets)) {
    names(res$fitAtOptCV$rowOffsets) <- dataset_names
  }
  if (!is.null(res$fitAtOptCV$colOffsets) && is.list(res$fitAtOptCV$colOffsets)) {
    names(res$fitAtOptCV$colOffsets) <- dataset_names
  }
  if (!is.null(res$fitAtOptCV$trainModel$Z) && is.list(res$fitAtOptCV$trainModel$Z)) {
    names(res$fitAtOptCV$trainModel$Z) <- dataset_names
  }
  if (!is.null(res$fitAtOptCV$Z) && is.list(res$fitAtOptCV$Z)) {
    names(res$fitAtOptCV$Z) <- dataset_names
  }


  res$fitAtOptCV[c(
    "debugTrace", "hDebugTrace", "hNewtonDebugTrace", "hProjectionDebugTrace"
  )] <- NULL
  res$optCV$cvObjectiveTrace <- NULL
  
  class(res)=c("iSSANOVASVD", "svdxFit", class(res))

  res

}
