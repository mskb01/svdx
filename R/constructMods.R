
consZ <- function( z.formula, zdata ){

  if(is(z.formula, "list")){ # 
    Z <- lapply( seq_along(z.formula), function(d) model.matrix( z.formula[[d]], data=zdata[[d]] )  )
    Zls <- lapply(Z, function(z) tcrossprod(.invpsd(crossprod(z)),z) ) #|Z terms| x n#this is unnecessary for non-Gaussian svdXEF.
    
  } else {
    Z <- model.matrix( z.formula, data=zdata )
    Zls <- tcrossprod(.invpsd(crossprod(Z)),Z) #|Z terms| x n#this is unnecessary for non-Gaussian svdXEF.
  }

  list("Z"=Z,
       "Zls"=Zls
       )

}

#Construct Model space
constructMod <- function(
    data,
    zdata=data,
    formula,
    z.formula,
    type=NULL,
    Q=NULL,
    id.basis=NULL,
    n_id.basis = NULL,
    graphData = NULL,
    graphDistFn=NULL,
    graphDistOnlyOnPoints=seq(nrow(data)),
    ...
){
  require(Matrix)

  n <- nrow(data)
  N0IDBASIS <- max( 30, ceiling(10* n^(2/9))   )
  if( is.null( n_id.basis ) ){
    n_id.basis <- N0IDBASIS
  }
  if( n_id.basis > n ){
    n_id.basis <- n
  }

  #sanity checks
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
    id.basis <- sample( n, n_id.basis, replace=F )
  } else {
    n_id.basis <- length(id.basis)
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
    if( is.null( graphDistOnlyOnPoints ) ){
      message( "graphDistOnlyOnPoints cannot be NULL when passing graphData and graphDistFn. Using all data. " )
      graphDistOnlyOnPoints<-seq(n)
    } else {
      r <- range( graphDistOnlyOnPoints )
      if( (r[1] < 1) || (r[2] > n) ){
        stop( "graphDistOnlyOnPoints must be in the range of 1 to nrow(data)" )
      }
    }
  }

  #Z stuff
  zmods <- consZ( z.formula=z.formula, zdata=zdata )

  #U stuff
  mod <-  ssanova.mkterms(
    formula, type=type,
    data=data, id.basis = id.basis,
    ...
  )
  mod$formula <- formula
  mod$z.formula <- z.formula
  qids <- mod$qids <- mod$id.basis
  nq <- length(qids)
  Rs <- NA
  if(length(dim(mod$r))==2){ #just one r ; This unpleasant code can be resolved if we return r in mod with 3 dimensions always.
    Rs <- list( mod$r)
  } else { #mult Rs
    Rs <- lapply(
      seq(dim( mod$r )[3]),
      function(i){
        x <- mod$r[,,i]
        x
      }  )
  }
  Rs.tilde <- lapply(
    Rs,
    function(x){
      x[qids,,drop=F]
    }  )
  R0trace <- abs( sapply( Rs.tilde, function(x) sum(diag(x)) ) ) #absolute value of tra
  nTheta <- length(Rs)

  #STORE
  S <- mod$s
  nr <- ncol(S)
  SStq <- tcrossprod( S[qids,,drop=F] ) #if incorporating 2, remove it in orthR, SSt, B below in the next code block.
  R0 <- do.call(cbind, Rs) #n x sq
  R0.til <- do.call(cbind, Rs.tilde)
  graphDistMat <- NA
  if( !is.null( graphData) & !is.null(graphDistFn) & !is.null(graphDistOnlyOnPoints) ){
    graphDistMat <- graphDistFn( graphData, graphDistOnlyOnPoints  ); #ex: graphDistOnlyOnPoints = fold_indices[id.basis]
    graphDistMat <- as( graphDistMat, "dgCMatrix" )
  }

  list(
    "nr"=nr,
    "nq"=nq,
    "nTheta"=nTheta,
    "qids"=qids,
    "Z"=zmods$Z,
    "Zls"=zmods$Zls,
    "S"=S,
    "SStq"=SStq, #D.S is already defined for qids only
    "Rs"=Rs, #list version : todo remove dependency
    "R0"=R0, #cbound version
    "R0.til"=R0.til,
    "R0trace"=R0trace, 
    "Q"=Q,
    "graphDistMat"=graphDistMat,
    "graphDistOnlyOnPoints" = graphDistOnlyOnPoints,
    "isGraphOn"=!is.null(graphData),
    "mod"=mod
  )
}

constructTestMods <- function(object, newdata,newzdata=newdata,
                                 qids=object$qids,
                                 include = c(object$terms$labels,
                                             object$lab.p),
                                 ...
){



    zmods <- consZ( z.formula = object$z.formula, zdata=newzdata )

    nnew <- nrow(newdata)
    object$id.basis <- qids #SENTHIL
    nbasis <- length(object$id.basis)
    nnull <- ncol(object$s)
    nz <- length(object$b)
    nn <- nbasis + nnull + nz
    labels.p <- object$lab.p
    term <- object$terms
    philist <- rklist <- NULL
    s <- r <- NULL
    nq <- 0
    for (label in include) {
      if (label == "1") {
        philist <- c(philist, term[[label]]$iphi)
        s <- cbind(s, rep(1, len = nnew))
        next
      }
      if (label %in% labels.p)
        next
      if (label == "offset")
        next
      xnew <- newdata[, term[[label]]$vlist]
      x <- object$mf[object$id.basis, term[[label]]$vlist]
      nphi <- term[[label]]$nphi
      nrk <- term[[label]]$nrk
      if (nphi) {
        iphi <- term[[label]]$iphi
        phi <- term[[label]]$phi
        for (i in 1:nphi) {
          philist <- c(philist, iphi + (i - 1))
          s <- cbind(s, phi$fun(xnew, nu = i, env = phi$env))
        }
      }
      if (nrk) {
        irk <- term[[label]]$irk
        rk <- term[[label]]$rk
        for (i in 1:nrk) {
          rklist <- c(rklist, irk + (i - 1))
          nq <- nq + 1
          r <- array(c(r, rk$fun(xnew, x, nu = i, env = rk$env,
                                 out = TRUE)), c(nnew, nbasis, nq))
        }
      }
    }
    if (!is.null(object$partial)) {
      vars.p <- as.character(attr(object$partial$mt, "variables"))[-1]
      facs.p <- attr(object$partial$mt, "factors")
      vlist <- vars.p[as.logical(apply(facs.p, 1, sum))]
      for (lab in labels.p) {
        if (lab %in% include) {
          vlist.wk <- vars.p[as.logical(facs.p[, lab])]
          vlist <- vlist[!(vlist %in% vlist.wk)]
        }
      }
      if (length(vlist)) {
        for (lab in vlist) newdata[[lab]] <- 0
      }
      matx.p <- model.matrix(object$partial$mt, newdata)[,
                                                         -1, drop = FALSE]
      matx.p <- sweep(matx.p, 2, object$partial$center)
      matx.p <- sweep(matx.p, 2, object$partial$scale, "/")
      nu <- nnull - dim(matx.p)[2]
      for (label in labels.p) {
        nu <- nu + 1
        if (label %in% include) {
          philist <- c(philist, nu)
          s <- cbind(s, matx.p[, label])
        }
      }
    }

    Rs <- NA;
    if(nrk > 0){
      Rs <- lapply(
        seq(dim( r )[3]),
        function(i){
          x <- r[,,i]
          x
        }  )
    }


    list( "R0test"=do.call( cbind, Rs ),
          "Stest"=s,
          "Ztest"=zmods$Z,
          "Zlstest"=zmods$Zls,
          "Rs"=Rs #used in ConstructModsatTheta functions, list version of R0test
          )


}




