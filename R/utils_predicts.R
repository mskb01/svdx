# Prof. Gu's gss based utils 2 #
# predict scripts

my.predict.ssanova <- function (object, newdata,
                                qids=object$qids,
                                include = c(object$terms$labels,
                                            object$lab.p),
                                theta = object$RES$theta$min.agn,
                                .c=object$RES$min.agn$.c,
                                h=object$RES$min.agn$h,
                                ...
)
{
  nnew <- nrow(newdata)
  object$id.basis <- qids #SENTHIL
  nbasis <- length(object$id.basis)
  nnull <- nrow(h)
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

  r.wk <- matrix(0, nnew, nbasis)
  nq <- 0
  for (i in rklist) {
    nq <- nq + 1
    #r.wk <- r.wk + 10^object$theta[i] * r[, , nq]
    r.wk <- r.wk + theta[i] * r[, , nq] #SENTHIL
  }
  if (nz) {
    if (is.null(newdata$random))
      z.wk <- matrix(0, nnew, nz)
    else z.wk <- newdata$random
    r.wk <- cbind(r.wk, z.wk)
  }
  nphi <- length(philist)
  #pmean <- as.vector(r.wk %*% c(object$c, object$b))
  pmean <- r.wk %*% .c  #the b terms are for random effects
  if (nphi)
    #pmean <- pmean + as.vector(s %*% object$d[philist])
    pmean <- pmean + s %*% h[philist,,drop=F]
  if (any(include == "offset")) {
    if (is.null(model.offset(object$mf)))
      stop("gss error: no offset in the fit")
    offset <- newdata$offset
    if (is.null(offset))
      offset <- newdata$"(offset)"
    if (is.null(offset))
      stop("gss error: missing offset")
    pmean <- sweep(pmean, 1, offset, "+" )
  }
  #no se.fit block #SENTHIL
  pmean
}


my.predict.ssanova.netR <- function (object, newdata,
                                     qids=object$qids,
                                     include = c(object$terms$labels,
                                                 object$lab.p),
                                     theta = object$RES$theta$min.agn,
                                     .c=object$RES$min.agn$.c,
                                     h=object$RES$min.agn$h,
                                     ...
)
{

  stopifnot( object$netR )
  stopifnot( !is.null(object$netR.scales) )
  stopifnot(length(theta)==1)

  nnew <- nrow(newdata)
  object$id.basis <- qids #SENTHIL
  nbasis <- length(object$id.basis)
  nnull <- nrow(h)
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
        r <- array(c(r, object$netR.scales[[irk + (i - 1)]]*rk$fun(xnew, x, nu = i, env = rk$env,
                                                                   out = TRUE)), c(nnew, nbasis, nq)) #SENTHIL, change also relative to no netR predict function above
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

  r.wk <- matrix(0, nnew, nbasis)
  nq <- 0
  for (i in rklist) {
    nq <- nq + 1
    #r.wk <- r.wk + 10^object$theta[i] * r[, , nq]
    r.wk <- r.wk + theta * r[, , nq] #SENTHIL, change also relative to no netR predict function above
  }
  if (nz) {
    if (is.null(newdata$random))
      z.wk <- matrix(0, nnew, nz)
    else z.wk <- newdata$random
    r.wk <- cbind(r.wk, z.wk)
  }
  nphi <- length(philist)
  #pmean <- as.vector(r.wk %*% c(object$c, object$b))
  pmean <- r.wk %*% .c  #the b terms are for random effects
  if (nphi)
    #pmean <- pmean + as.vector(s %*% object$d[philist])
    pmean <- pmean + s %*% h[philist,,drop=F]
  if (any(include == "offset")) {
    if (is.null(model.offset(object$mf)))
      stop("gss error: no offset in the fit")
    offset <- newdata$offset
    if (is.null(offset))
      offset <- newdata$"(offset)"
    if (is.null(offset))
      stop("gss error: missing offset")
    pmean <- sweep(pmean, 1, offset, "+" )
  }
  #no se.fit block #SENTHIL
  pmean
}

#' Predict method
#' @export
predict.SSANOVASVD <- function(object,#output object of ssanovasvd
                               newdata,
                               ... ){


  fit <- object$fitAtOptCV
  nr <- fit$trainModel$nr
  nq <- fit$trainModel$nq

  dmat <- fit$hmat[1:nr,,drop=F]
  cmat <- fit$hmat[nr+(1:nq),,drop=F]

  th <- object$optCV$theta
  qids <- fit$trainModel$qids

  #predict and revise fit
  
  u.unnormalized <- my.predict.ssanova(object = fit$trainModel$mod,
                                       qids=qids,
                                       newdata = newdata,
                                       theta = th,
                                       .c = cmat,
                                       h = dmat, ...  )
  
  
  dimnames(u.unnormalized) <- list( rownames(newdata), colnames(fit$u) )

  u.unnormalized

}

#' Predict method
#' @export
predict.iSSANOVASVD <- function(object,
                                newdata,
                                d=1, #to predict for the first dataset set d=1
                                ... ){
  predict.SSANOVASVD(object=get_dth_fit(object, d=d), newdata=newdata, ... )
}

