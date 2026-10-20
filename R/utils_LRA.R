#utils_LRA

getZforData <- function(object, newdata){

  z.formula <- object$fitAtOptCV$trainModel$mod$z.formula
  if(is(z.formula, "list")){ # //is.list will return TRUE for data frame as wel.
    Z <- lapply( seq_along(z.formula), function(d) model.matrix( z.formula[[d]], data=newdata[[d]] )  )
    #Z <- as.matrix( do.call( Matrix::bdiag, Z ) )
  } else {
    Z <- model.matrix( z.formula, data=newdata )
  }
  Z
}


#' @export
getLRA <- function(object, ... ){
  UseMethod("getLRA")
}

#' When include.Z.effects is TRUE, we assume all data are comparable in the test data (newdata, if given )
#' to the train data. That is if a predictor says "batch1" for newdata, it corresponds to "batch1" effect in train.
#' At the time of training, we need to make sure that all the Z effects are represented in teh training data,
#' and all order of any discrete factor levels are appropriately declared and preserved.
#' @export
getLRA.SSANOVASVD <- function(
    object,
    components=1:2,
    newdata=NULL,
    include.only.terms=NULL,
    include.Z.effects=FALSE, #when available, should the LRA include effects due to Z zeta?
    offsetMat=NULL,
    ...
){


  Z <- NULL
  if(include.Z.effects){
    if( is.null(newdata) ){
      Z <- object$fitAtOptCV$Z
    } else {
      Z <- getZforData( object=object, newdata=newdata )
    }
  }

  #cat("Dimension of Z: ", dim(Z), "\n")
  #cat( is.null(Z) )
  #cat(is.null(newdata), "\n")


  .predict <- FALSE
  if( !is.null( include.only.terms ) ){
    .predict <- TRUE
    .include <- include.only.terms
  } else { #include all terms, by default
    .include <- c(object$fitAtOptCV$trainModel$mod$terms$labels,
                  object$fitAtOptCV$trainModel$mod$lab.p )
  }

  data <- object$fitAtOptCV$trainModel$mod$mf  # metadata used for fitting the object
  if( !is.null( newdata ) ){
    .predict <- TRUE
    data <- newdata #we will need to predict on the given data
  }

  obj <- object$fitAtOptCV #train object

  #Get u: trained, or predicted if newdata/include.terms is not NULL
  if( !.predict ){ #plot what was trained
    u <- obj$u
  }  else {
    u <- predict(object,
                 newdata=data,
                 include=.include,
                 ...
    ) #outputs scaled by svals components as lra = (uv')=(effect1fn+effect2fn+...)v'
  }

  #CONSTRUCT uv' lra.
  u <- u[,components,drop=F]
  lra <- tcrossprod( obj$v[,components,drop=F], u ) # note: u is correctly unnormed here
  dimnames(lra) <- list( rownames(obj$v), rownames(u)  )

  # ADD effects due to Z to lra, if needed, and available
  # all MSE calculations must use this to avoid biases.
  if( include.Z.effects ){
    if( is.null(obj$Zeta) ){
      warning( "No effects due to Z found in the fit object. Skipping inclusion in LRA" )
    } else {
      lra <- lra + tcrossprod( obj$Zeta,  #effects
                               Z )
      }
  }

  if( !is.null(offsetMat) ){
    lra <- lra+offsetMat
  }

  lra
}

