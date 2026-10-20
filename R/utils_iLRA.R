getZforData <- function(object, newdata){
  Z <- model.matrix( object$fitAtOptCV$zformula, data=newdata )
  Z
}

#' @export
get_dth_fit <- function(x, ...){
  UseMethod("get_dth_fit")
}

#' @export
get_dth_fit.iSSANOVASVD0 <-function(object,d=1){

  obj.d <- object
  obj.d$fitAtOptCV$Z <- object$fitAtOptCV$Z[[d]]
  obj.d$fitAtOptCV$Zeta <- object$fitAtOptCV$Zeta[[d]]
  obj.d$fitAtOptCV$vmat <- object$fitAtOptCV$vmat[[d]]
  obj.d$fitAtOptCV$zformula <- object$fitAtOptCV$zformula[[d]]
  obj.d$fitAtOptCV$trainModel$Zls <- object$fitAtOptCV$trainModel$Zls[[d]]
  obj.d$fitAtOptCV$trainModel$Z <- object$fitAtOptCV$trainModel$Z[[d]]
  obj.d$fitAtOptCV$rowOffsets <- object$fitAtOptCV$rowOffsets[[d]]
  obj.d$fitAtOptCV$colOffsets <- object$fitAtOptCV$colOffsets[[d]]
  if( is.list( object$fitAtOptCV$aphi ) ){
    obj.d$fitAtOptCV$aphi <- object$fitAtOptCV$aphi[[d]]
    obj.d$fitAtOptCV$family <- object$fitAtOptCV$family[[d]]
  } else {
    obj.d$fitAtOptCV$aphi <- object$fitAtOptCV$aphi
    obj.d$fitAtOptCV$family <- object$fitAtOptCV$family
  }
  class(obj.d) <- "SSANOVASVD"

  obj.d

}

#' @export
get_dth_fit.iSSANOVASVD <-function(object,d=1){

  obj.d <- get_dth_fit.iSSANOVASVD0( object, d=d )
  class(obj.d) <- "SSANOVASVD"

  obj.d

}

#' @export
get_dth_LRA <- function(x, ...){
  UseMethod("get_dth_LRA")
}

#grabs the dth dataset's lra for issanovasvd0 objects
#' @export
get_dth_LRA.iSSANOVASVD0 <- function(object,
                         d=1, #the dataset for which the LRA is needed.
                         components=1:2,
                         newdata=NULL,
                         newzdata=NULL,
                         include.only.terms=NULL,
                         include.Z.effects=FALSE, #when available, should the LRA include effects due to Z zeta?
                         ...){

  estVonMat=NULL #if v needs to be reest based on u on this mat.

  #Generate an SSANOVASVD0 object
  obj.d <- get_dth_fit.iSSANOVASVD0( object, d=d  )

  if( !is.null( newzdata ) ){
    if( !is.list( newzdata ) ){
      stop( "newzdata must be a list with dataframes (corresponding to each dataset) with which the Z formula design matrix is to be computed. " )
    }
    newdata <- cbind( newdata, newzdata[[d]] )
  }

  getLRA( object=obj.d, components=components,
          newdata=newdata, include.only.terms=include.only.terms,
          include.Z.effects=include.Z.effects,
          ...
          )


}

#grabs the dth dataset's lra for issanovasvd objects
#' @export
get_dth_LRA.iSSANOVASVD <- function(object,
                          d=1, #the dataset for which the LRA is needed.
                          components=1:2,
                          newdata=NULL,
                          newzdata=NULL,
                          include.only.terms=NULL,
                          include.Z.effects=FALSE, #when available, should the LRA include effects due to Z zeta?
                          ...){

  #Generate an SSANOVASVD object
  obj.d <- get_dth_fit.iSSANOVASVD0( object, d=d  )


  if( !is.null( newzdata ) ){
    if( !is.list( newzdata ) ){
      stop( "newzdata must be a list with dataframes (corresponding to each dataset) with which the Z formula design matrix is to be computed. " )
    }
    newdata <- cbind( newdata, newzdata[[d]] )
  }

  getLRA( object=obj.d, components=components,
          newdata=newdata, include.only.terms=include.only.terms,
          include.Z.effects=include.Z.effects,
          ...
  )


}

#' When include.Z.effects is TRUE, we assume all data are comparable in the test data (newdata, if given )
#' to the train data. That is if a predictor says "batch1" for newdata, it corresponds to "batch1" effect in train.
#' At the time of training, we need to make sure that all the Z effects are represented in teh training data,
#' and all order of any discrete factor levels are appropriately declared and preserved.
#' @export
getLRA.iSSANOVASVD0 <- function(
  object,
  datasets=seq( length(object$fitAtOptCV$vmat) ), #datasets for which LRA is to be computed
  components=1:2,
  newdata=NULL,
  newzdata=NULL,
  include.only.terms=NULL,
  include.Z.effects=FALSE, #when available, should the LRA include effects due to Z zeta?
  ...
){


  ndat <- length(object$fitAtOptCV$vmat) #nunber of all datasets processed
  nmes <- names( object$fitAtOptCV$vmat )
  if( is.null( nmes  ) ){
    names( object$fitAtOptCV$vmat ) <- nmes <- paste0( "dataset", seq(ndat) )
  }
  names( datasets ) <- nmes[datasets]


  lapply( datasets, function(d){
    get_dth_LRA.iSSANOVASVD0( object,
                   d=d, #the dataset for which the LRA is needed.
                   components=components,
                   newdata=newdata,
                   newzdata=newzdata,
                   include.only.terms=include.only.terms,
                   include.Z.effects=include.Z.effects, #when available, should the LRA include effects due to Z zeta?
                   ... )
  } )


}

#' @export
getLRA.iSSANOVASVD <- function(
  object,
  datasets=seq( length(object$fitAtOptCV$vmat) ), #datasets for which LRA is to be computed
  components=1:2,
  newdata=NULL,
  newzdata=NULL,
  include.only.terms=NULL,
  include.Z.effects=FALSE, #when available, should the LRA include effects due to Z zeta?
  ...
){


  ndat <- length( object$fitAtOptCV$vmat ) #nunber of all datasets processed
  nmes <- names( object$fitAtOptCV$vmat )
  if( is.null( nmes  ) ){
    names( object$fitAtOptCV$vmat ) <- nmes <- paste0( "dataset", seq(ndat) )
  }
  names( datasets ) <- nmes[datasets]


  lapply( datasets, function(d){
    get_dth_LRA.iSSANOVASVD( object,
                   d=d, #the dataset for which the LRA is needed.
                   components=components,
                   newdata=newdata,
                   newzdata=newzdata,
                   include.only.terms=include.only.terms,
                   include.Z.effects=include.Z.effects, #when available, should the LRA include effects due to Z zeta?
                   ... )
  } )


}

