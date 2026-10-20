#mse.R

#checks for sanity in inputs - dimensions etc.,
.checkInputs_single <- function(Y, Z, Zeta, umat, vmat,
                          rowOffsets=rep(0, nrow(Y)),
                          colOffsets=rep(0, ncol(Y)),
                          columnsR=NULL,
                          transZeta=FALSE
                          ) {



  if( ncol(Y) != length(colOffsets) ){
    stop( "colOffsets should have the same length as number of columns in Y (not length of columnsR)" )
  }

  n <- ncol(Y)
  if( !is.null(columnsR) ){
    n <- length(columnsR) #redefining n this way is OK for the rest of the code; we checked for colOffsets in earlier statement
  }
  p <- nrow(Y)

  if( (n != nrow(Z)) | (n!=nrow(umat))  )
    stop( "Z, and umat should have the same number of columns as Y if columnsR is not given, or
          length of columnsR if columnsR is given" )

  # print( nrow(Y) )
  # print( nrow(Zeta) )
  # print( nrow(vmat) )
  # print( length(rowOffsets) )

  if( transZeta ){
    if( (p != ncol(Zeta)) | (p!=nrow(vmat)) | ( p!= length(rowOffsets) ) )
      stop( "t(Zeta), vmat and rowOffsets should have the same number of rows as Y" )
  } else {
    if( (p != nrow(Zeta)) | (p!=nrow(vmat)) | ( p!= length(rowOffsets) ) )
      stop( "Zeta, vmat and rowOffsets should have the same number of rows as Y" )
  }


  if( !is.null(columnsR) ){
    if( (max(columnsR) > ncol(Y))  | (min(columnsR) < 1)  ){
      stop("columnsR should contain values in 1 to number of columns in Y")
    }
  }

}

# for a single matrix and its parameters, calculates the mse
mse <- function( Y, Z, Zeta, umat, vmat,
                 rowOffsets=rep(0, nrow(Y)),
                 colOffsets=rep(0, ncol(Y)),
                 numThreads = RcppParallel::defaultNumThreads()/2,
                 setNumThreads = TRUE
                 ){ #Zeta is p x |Zeta| here, on the C side we have the |Zeta| x p
  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets)
  if( setNumThreads ) {
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }

 mseC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets ) #our C implementation uses |Zeta| x p instead.

}

# for a single matrix and its parameters, calculates the mse over columnsR
mseOnColumns<- function(Y, columnsR, Z, Zeta, umat, vmat,
                        rowOffsets=rep(0, nrow(Y)),
                        colOffsets=rep(0, ncol(Y)),
                        numThreads = RcppParallel::defaultNumThreads()/2,
                        setNumThreads=TRUE
                        ){

  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets, columnsR)
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }
  mseOnColumnsC( Y, columnsR, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets ) #our C implementation uses |Zeta| x p instead.

}

# for a single matrix and its parameters, calculates the rowwise mses
mseRowWise <- function(Y, Z, Zeta, umat, vmat,
                       rowOffsets=rep(0, nrow(Y)),
                       colOffsets=rep(0, ncol(Y)),
                       numThreads = RcppParallel::defaultNumThreads()/2,
                       setNumThreads = TRUE
                       ){

  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets)
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }
  mseRowWiseC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets ) #our C implementation uses |Zeta| x p instead.

}

# for a single matrix and its parameters, calculates the rowwise mses over columns
mseRowWiseOnColumns <- function(Y, columnsR, Z, Zeta, umat, vmat,
                                rowOffsets=rep(0, nrow(Y)),
                                colOffsets=rep(0, ncol(Y)),
                                numThreads = RcppParallel::defaultNumThreads()/2,
                                setNumThreads=TRUE
                                ){
  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets, columnsR)
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }
  mseRowWiseOnColumnsC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets, columnsR ) #our C implementation uses |Zeta| x p instead.

}

# for a single matrix and its parameters, calculates the rowwise mses
mseColWise <- function(Y, Z, Zeta, umat, vmat,
                       rowOffsets=rep(0, nrow(Y)),
                       colOffsets=rep(0, ncol(Y)),
                       numThreads = RcppParallel::defaultNumThreads()/2,
                       setNumThreads = TRUE
){

  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets)
  if(setNumThreads){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }
  mseColWiseC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets ) #our C implementation uses |Zeta| x p instead.

}

# for a single matrix and its parameters, calculates the rowwise mses over columns
mseColWiseOnColumns <- function(Y, columnsR, Z, Zeta, umat, vmat,
                                rowOffsets=rep(0, nrow(Y)),
                                colOffsets=rep(0, ncol(Y)),
                                numThreads = RcppParallel::defaultNumThreads()/2,
                                setNumThreads = TRUE
){
  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets, columnsR)
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }
  mseColWiseOnColumnsC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets, columnsR ) #our C implementation uses |Zeta| x p instead.

}





init_aphi <- function( Y, Z, Zeta, umat, vmat, family,
                       columns=NULL,
                       transZeta=TRUE, #if passing Zeta as px|Zeta|, this needs to be TRUE
                                      #if passing as |Zetad| x pd, this needs to be false
                       rowOffsets=rep(0, nrow(Y)),
                       colOffsets = rep(0, ncol(Y)),
                       numThreads = RcppParallel::defaultNumThreads()/2,
                       setNumThreads = TRUE
                       ){

  #TODO : incorporate offsets.

  .checkInputs_single(Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets, columns, transZeta = !transZeta)
  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }

  aphi <- NA
  if( family=="poisson" ){
    aphi <- rep( 1, nrow(Y) )
  } else if( family=="gaussian" ){
    if( is.null(columns) ){
      if(transZeta)
        aphi <- mseRowWiseC( Y, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets )
      else
        aphi <- mseRowWiseC( Y, Z, Zeta, umat, vmat, rowOffsets, colOffsets )
    } else {
      if(transZeta)
        aphi <- mseRowWiseOnColumnsC( Y, columns, Z, t(Zeta), umat, vmat, rowOffsets, colOffsets )
      else
        aphi <- mseRowWiseOnColumnsC( Y, columns, Z, Zeta, umat, vmat, rowOffsets, colOffsets )
    }
  } else {
    stop("Family unknown.")
  }

  aphi

}


init_aphi_integ <- function( Y, Z, Zeta, umat, vmat, family,
                             columns=NULL,
                             transZeta=TRUE, #if passing Zeta as as list of pd x|Zetad|, this needs to be TRUE
                                            #if passing as |Zetad| x pd, this needs to be false
                             rowOffsets=lapply(Y, function(d) rep(0, nrow(Y[[d]])) ),
                             colOffsets = lapply(Y, function(d) rep(0, ncol(Y[[d]])) ),
                             numThreads = RcppParallel::defaultNumThreads()/2,
                             setNumThreads = TRUE
                             ){

  if( setNumThreads ){
    RcppParallel::setThreadOptions(numThreads=numThreads)
  }

  D <- length(Y)

  n <- ncol(Y[[1]])
  if(!is.null(columns)){
    n <- length(columns)
  }

  if(n!=nrow(umat)){
    stop( "umat must have the same number of rows as given number of columns , or
          if columns is NULL, same number of columns as all Y" )
  }

  pd <- sapply( Y, nrow)
  aphi <- list()

  for(d in seq(D)){
    if( n != nrow(Z[[d]]) )
      stop( "Z and umat should have the same number of columns as Y" )

    if( !transZeta ){
      if( (pd[d] != ncol(Zeta[[d]])) | (pd[d]!=nrow(vmat[[d]]))  )
        stop( "Each entry in Zeta and vmat should have the same number of rows as corresponding entry in Y" )
    } else {
      if( (pd[d] != nrow(Zeta[[d]])) | (pd[d]!=nrow(vmat[[d]]))  )
        stop( "Each entry in Zeta and vmat should have the same number of rows as corresponding entry in Y" )
    }

    if( family[[d]]=="poisson" ){
      aphi[[d]] <- rep( 1, pd[d] )
    } else if( family[[d]]=="gaussian" ){
      if( is.null(columns) ){
        if( transZeta )
          aphi[[d]] <- mseRowWiseC( Y[[d]], Z[[d]], t(Zeta[[d]]), umat, vmat[[d]], rowOffsets[[d]], colOffsets[[d]] )
        else
          aphi[[d]] <- mseRowWiseC( Y[[d]], Z[[d]], Zeta[[d]], umat, vmat[[d]], rowOffsets[[d]], colOffsets[[d]] )
      } else {
        if(transZeta)
          aphi[[d]] <- mseRowWiseOnColumnsC( Y[[d]], columns, Z[[d]], t(Zeta[[d]]),
                                             umat, vmat[[d]], rowOffsets[[d]], colOffsets[[d]] )
        else
          aphi[[d]] <- mseRowWiseOnColumnsC( Y[[d]], columns, Z[[d]], Zeta[[d]],
                                             umat, vmat[[d]], rowOffsets[[d]], colOffsets[[d]] )
      }
    } else {
      stop("Family unknown.")
    }
  }

  aphi

}
