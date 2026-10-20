

#' Fit an SSANOVA-SVD model for a single dataset
#'
#' Fits an interpretable factorization model for a features by samples
#' data matrix using sample-level covariates. The model represents the data as
#' low-rank structure whose sample scores are smooth, interpretable functions of
#' covariates relevant to a study, supplied through \code{formula} and \code{xdata}, while optional nuisance 
#' covariates (e.g., batches) to adjust the data for, may be supplied through \code{zformula} and \code{zdata}.
#' Predictions of processes, can be made to reconstruct sample scores for external datasets, without data reuse.
#' Predicted scores can then be used to perform conditional inference for feature participation.  
#'
#' @param Y A feature-by-sample matrix. For example, a log(1+count gene expression matrix) 
#' @param K Maximum number of components to fit.
#' @param formula Formula describing the metadata-driven component scores.
#' @param zformula Optional formula for nuisance or adjustment covariates.
#' @param data Data frame containing variables used in `formula`. Rows correspond
#'   to samples/columns of `Y`.
#' @param zdata Data frame containing variables used in `zformula`. Rows
#'   correspond to samples/columns of `Y`.
#' @param family  Optional response family for the data. \code{gaussian}, 
#'   \code{poisson} are supported. If \code{quasipoisson}, 
#'   a poisson based fit will be returned. Note: Inference workflows will still respect \code{quasipoisson}.  
#' @param graphData  Optional graph data used by graph-aware model components. Must 
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
#' @param f1 default 1, scales the loss function part of the SSANOVA-SVD objective (not the penalty).
#' @param nfolds Number of folds used for cross-validation. Default 5. For large datasets, higher nfolds is preferred (e.g., 10). 
#' @param est_K Logical; whether to estimate the number of components.
#' @param fitter_control Named list of fitter controls: \code{ftol_abs}, \code{xtol_abs}, \code{maxIter}, \code{inner_maxIter}, and \code{inner_ftol_abs}.
#' @param optCV_control Named list of outer cross-validation coordinate-descent controls: \code{ftol_abs}, \code{xtol_abs}, and \code{maxIter}.
#' @param nlopt_control Named list of NLopt controls: \code{ftol_abs}, \code{xtol_abs}, and \code{maxeval}.
#' @param init_control Optional family-initialization overrides with named sublists \code{fitter}, \code{optCV}, and \code{nlopt}. 
#'                     Missing entries inherit from the corresponding main controls.
#' @param numThreads Number of threads for parallel computation.
#' @param setNumThreads Logical; whether to set the RcppParallel thread count.
#' @param optCV_maxIter (legacy, use bundled \code{*_control} options above ) Maximum iterations for cross-validation optimization.
#' @param optCV_eps  (legacy, use bundled \code{*_control} options above ) Convergence tolerance for cross-validation optimization.
#' @param fitter_maxIter (legacy, use bundled \code{*_control} options above ) Maximum iterations for the model fitter.
#' @param fitter_eps (legacy, use bundled \code{*_control} options above ) Convergence tolerance for the model fitter.
#' @param xtol_abs (legacy, use bundled \code{*_control} options above ) Absolute parameter tolerance for numerical optimization.
#' @param ftol_abs (legacy, use bundled \code{*_control} options above ) Absolute objective tolerance for numerical optimization.
#'
#' @return An object of class \code{SSANOVASVD} containing the fitted model,
#'   selected tuning parameters, fitted component scores, feature loadings,
#'   nuisance effects, and training-model objects.
#'   
#' @seealso \code{\link{runSVDXReport}, \link{runSVDXWorkflow}, \link{isvdx}, \link{runISVDXReport}, \link{runISVDXWorkflow}  }
#'
#' @details
#' \code{svdx()} is for a single matrix. For a list of matrices sharing
#' the same samples/multi-modal setting, use \code{isvdx()} instead. 
#' 
#' Exploits model space construction from package \code{gss}. 
#' 
#'
#' @examples
#' \dontrun{
#' fit <- svdx(
#'   Y = Y,
#'   K = 5,
#'   formula = ~ group * time,
#'   zformula = ~ batch,
#'   data = sample_data,
#'   zdata = sample_data, 
#'   modelType = list( "time"=list( "cubic" , range(sample_data$time) ) )
#' )
#'
#' summary(fit, Y = Y)
#' plotU(fit, data = sample_data, component = 1, x = "time", group = "group")
#' predict(fit, newdata=sample_data, include=NULL)
#' predict(fit, newdata=sample_data, include="time")
#' predict(fit, newdata=sample_data, include="group")
#' }
#'
#' @export
svdx <- function( Y,
                  K,
                  formula,
                  zformula=~1,
                  data,
                  zdata,
                  family = NULL,
                  graphData = NULL,
                  graphDistFn = NULL,
                  graphDistOnlyOnPoints = seq(ncol(Y)),
                  rowOffsets=rep(0, nrow(Y)),
                  colOffsets=rep(0, ncol(Y)),
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
                  setNumThreads = TRUE, #useful when calling recursively to set once and then false
                  
                                   #legacy below
                  optCV_maxIter=20, #legacy, this maxIter is optCV maxIter
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
  
  if(!is.null(family)){
    if(family == "quasipoisson"){
      family <- "poisson"
    }
  }
  
  if (!is.null(trainFolds) || !is.null(testFolds)) {
    stop(
      "Custom trainFolds/testFolds are not currently supported in the public API.",
      call. = FALSE
    )
  }

  if(is.list(Y)){
    stop("Y is a list, did you mean to call isvdx()?")
  }

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


  n <- ncol(Y)
  p <- nrow(Y)

  #SANITY CHECKS

  if( (nrow(data) != n) ||
      (nrow(zdata)!=n) ||
      (length(colOffsets)!=n)
  ){
    stop("Number of rows in data and zdata, and length of colOffsets must equal the
         total number of columns in Y")
  }
  if( length(rowOffsets)  != nrow(Y) ){
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
                                     f1 = f1#,
                                     #setNumThreads = FALSE #already set
                                     )
  cat('... done. ')


  #model for all data
  cat(' + Constructing all data model ...\n')

  adm <- constructModAtTheta(
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
    kfolds = nfolds
  )
  cat(' ... done.')

  cat( " + Fitting model ...\n" )
  res <- svdxC(Y=Y,
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

  rownames(res$fitAtOptCV$umat) <- colnames(Y)
  colnames(res$fitAtOptCV$umat) <- colnames(res$fitAtOptCV$vmat) <- paste0("IRCF_", seq( ncol(res$fitAtOptCV$umat) ))
  rownames(res$fitAtOptCV$vmat) <- rownames(res$fitAtOptCV$Zeta) <- rownames(Y)

  res$fitAtOptCV$trainModel <- adm
  res$optCV$TrainTestFolds <- ttf
  
  res$fitAtOptCV[c(
    "debugTrace", "hDebugTrace", "hNewtonDebugTrace", "hProjectionDebugTrace"
  )] <- NULL
  res$optCV$cvObjectiveTrace <- NULL

  class(res)=c("SSANOVASVD", "svdxFit")

  res

}
