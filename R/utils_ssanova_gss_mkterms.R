#GSS UTILS code 1###
gss.mkterms.force <- function (mf, type#, forcePen2Groups
                               ) #SENTHIL
{
  #library(gss)
  mt <- attr(mf, "terms")
  xvars <- as.character(attr(mt, "variables"))[-1]
  xfacs <- attr(mt, "factors")
  term.labels <- labels(mt)
  if (attr(attr(mf, "terms"), "intercept"))
    term.labels <- c("1", term.labels)
  vlist <- xvars[as.logical(apply(xfacs, 1, sum))]
  if (!is.null(type) && !is.list(type) && (type %in% c("cubic",         # this default gss impl doesn't penalize linear terms
                                                       "cubic.wLinPen", #SENTHIL: added this to allow for penalty on linear terms as well
                                                       "linear", "tp"))) {
    type.wk <- type
    type <- NULL
    for (xlab in vlist) type[[xlab]] <- type.wk
  }
  var.type <- NULL
  for (xlab in vlist) {
    x <- mf[, xlab]
    if (!is.null(type[[xlab]])) {
      type.wk <- type[[xlab]][[1]]
      if (!(type.wk %in% c("ordinal", "nominal", "cubic",
                           "cubic.wLinPen",                              #SENTHIL
                           "linear", "per", "trig", "cubic.per", "linear.per",
                           "tp", "sphere", "custom")))
        stop("gss error in mkterm: unknown type")
      if (type.wk %in% c("ordinal", "nominal")) {
        par.wk <- NULL
        if (!is.factor(x))
          stop("gss error in mkterm: wrong type")
      }
      if (type.wk %in% c("cubic",
                         "cubic.wLinPen", #SENTHIL
                         "linear")) {
        if (length(type[[xlab]]) == 1) {
          mn <- min(x)
          mx <- max(x)
          par.wk <- c(mn, mx) + c(-1, 1) * 0.05 * (mx -
                                                     mn)
        }
        else par.wk <- type[[xlab]][[2]]
        if (is.factor(x) | !is.vector(x))
          stop("gss error in mkterm: wrong type")
      }
      if (type.wk %in% c("per", "cubic.per", "linear.per",
                         "trig")) {
        if (type.wk == "per")
          type.wk <- "cubic.per"
        if (length(type[[xlab]]) == 1)
          stop("gss error in mkterm: missing domain of periodicity")
        else par.wk <- type[[xlab]][[2]]
        if (is.factor(x) | !is.vector(x))
          stop("gss error in mkterm: wrong type")
      }
      if (type.wk == "tp") {
        if (length(type[[xlab]]) == 1)
          par.wk <- list(order = 2, mesh = x, weight = 1)
        else {
          par.wk <- par.wk1 <- type[[xlab]][[2]]
          if (length(par.wk1) == 1)
            par.wk <- list(order = par.wk1, mesh = x,
                           weight = 1)
          if (is.null(par.wk$mesh))
            par.wk$mesh <- x
          if (is.null(par.wk$weight))
            par.wk$weight <- 1
        }
        if (dim(as.matrix(x))[2] != dim(as.matrix(par.wk$mesh))[2])
          stop("gss error in mkterm: wrong dimension in normalizing mesh")
      }
      if (type.wk == "sphere") {
        if (length(type[[xlab]]) == 1)
          par.wk <- 2
        else par.wk <- type[[xlab]][[2]]
        if (!(par.wk %in% (2:4)))
          stop("gss error in mkterm: spherical order not implemented")
      }
      if (type.wk == "custom")
        par.wk <- type[[xlab]][[2]]
    }
    else {
      if (is.factor(x)) {
        if (is.ordered(x))
          type.wk <- "ordinal"
        else type.wk <- "nominal"
        par.wk <- NULL
      }
      else {
        if (is.vector(x)) {
          type.wk <- "cubic"
          mn <- min(x)
          mx <- max(x)
          par.wk <- c(mn, mx) + c(-1, 1) * 0.05 * (mx -
                                                     mn)
        }
        else {
          type.wk <- "tp"
          par.wk <- list(order = 2, mesh = x, weight = 1)
        }
      }
    }
    var.type[[xlab]] <- list(type.wk, par.wk)
  }
  term <- list(labels = term.labels)
  iphi.wk <- 1
  irk.wk <- 1
  for (label in term.labels) {
    iphi <- irk <- phi <- rk <- NULL
    if (label == "1") {
      iphi <- iphi.wk
      iphi.wk <- iphi.wk + 1
      term[[label]] <- list(iphi = iphi, nphi = 1, nrk = 0)
      next
    }
    vlist <- xvars[as.logical(xfacs[, label])]
    x <- mf[, vlist]
    dm <- length(vlist)

    if (dm == 1) {
      # SENTHIL: this block in original code for
      #  single label terms (i.e., non-interaction terms)

      type.wk <- var.type[[vlist]][[1]]
      if (type.wk %in% c("nominal", "ordinal")) {
        if (type.wk == "nominal")
          fun.env <- mkrk.nominal(levels(x))
        else fun.env <- mkrk.ordinal(levels(x))
        if(nlevels(x)>2){ #added by senthil, default Gss behavior
          nphi <- 0
          rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
            env$fun(x, y, env$env, outer.prod)
          }
          nrk <- 1
          irk <- irk.wk
          irk.wk <- irk.wk + nrk
          rk <- list(fun = rk.fun, env = fun.env)
        } else if ( (nlevels(x) >= 2) #& (label %in% forcePen2Groups)
                    ) { #added BY SENTHIL to allow inequality, when forced
          nphi <- 0
          rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
            env$fun(x, y, env$env, outer.prod)
          }
          nrk <- 1
          irk <- irk.wk
          irk.wk <- irk.wk + nrk
          rk <- list(fun = rk.fun, env = fun.env)
        }
        else {
          phi.fun <- function(x, nu = 1, env) {
            wk <- as.factor(names(env$env$code)[1])
            env$fun(x, wk, env$env)
          }
          nphi <- 1
          iphi <- iphi.wk
          iphi.wk <- iphi.wk + nphi
          phi <- list(fun = phi.fun, env = fun.env)
          nrk <- 0
        }
      }
      if (type.wk == "cubic.wLinPen") {


        #Original code here being replaced by
        # one below to penalize linear as well.

        # If linear contribution isn't to be penalized,
        # use "cubic" as the type.wk, in which case,
        # we use the standard gss implementation

        # range <- var.type[[vlist]][[2]]
        # phi.env <-  mkphi.cubic(range)
        # phi.fun <- function(x, nu = 1, env) env$fun(x,
        #                                             nu, env$env)
        # nphi <- 1
        # iphi <- iphi.wk
        # iphi.wk <- iphi.wk + nphi
        # phi <- list(fun = phi.fun, env = phi.env)
        # rk.env <-  mkrk.cubic(range)
        # rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
        #   env$fun(x, y, env$env, outer.prod)
        # }
        # nrk <- 1
        # irk <- irk.wk
        # irk.wk <- irk.wk + nrk
        # rk <- list(fun = rk.fun, env = rk.env)

        #SENTHIL below, penalizing linear as well
        range <- var.type[[vlist]][[2]]
        nphi <- 0
        rk.env <- mkrk.cubic.linearAlsoPenalized(range)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)
      }

      if (type.wk == "cubic") { #this is the original gss block for "cubic", which doesn't penalize linear component


        range <- var.type[[vlist]][[2]]
        phi.env <-  mkphi.cubic(range)
        phi.fun <- function(x, nu = 1, env) env$fun(x,
                                                    nu, env$env)
        nphi <- 1
        iphi <- iphi.wk
        iphi.wk <- iphi.wk + nphi
        phi <- list(fun = phi.fun, env = phi.env)
        rk.env <-  mkrk.cubic(range)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)


      }

      if (type.wk %in% c("cubic.per", "linear", "linear.per",
                         "sphere")) {
        range <- var.type[[vlist]][[2]]
        nphi <- 0
        if (type.wk == "cubic.per")
          rk.env <-  mkrk.cubic.per(range)
        if (type.wk == "linear")
          rk.env <-  mkrk.linear(range)
        if (type.wk == "linear.per")
          rk.env <-  mkrk.linear.per(range)
        if (type.wk == "sphere")
          rk.env <-  mkrk.sphere(range)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)
      }
      if (type.wk == "trig") {
        range <- var.type[[vlist]][[2]]
        phi.env <-  mkphi.trig(range)
        phi.fun <- function(x, nu = 1, env) env$fun(x,
                                                    nu, env$env)
        nphi <- 2
        iphi <- iphi.wk
        iphi.wk <- iphi.wk + nphi
        phi <- list(fun = phi.fun, env = phi.env)
        rk.env <-  mkrk.trig(range)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)
      }
      if (type.wk == "tp") {
        par <- var.type[[vlist]][[2]]
        order <- par$order
        mesh <- par$mesh
        weight <- par$weight
        if (is.vector(x))
          xdim <- 1
        else xdim <- dim(x)[2]
        phi.env <-  mkphi.tp(xdim, order, mesh, weight)
        phi.fun <- function(x, nu = 1, env) {
          env$fun(x, nu, env$env)
        }
        nphi <- choose(xdim + order - 1, xdim) - 1
        iphi <- iphi.wk
        iphi.wk <- iphi.wk + nphi
        phi <- list(fun = phi.fun, env = phi.env)
        rk.env <-  mkrk.tp(xdim, order, mesh, weight)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)
      }
      if (type.wk == "custom") {
        par <- var.type[[vlist]][[2]]
        nphi <- par$nphi
        if (nphi > 0) {
          phi.env <- par$mkphi(par$env)
          phi.fun <- function(x, nu = 1, env) {
            env$fun(x, nu, env$env)
          }
          iphi <- iphi.wk
          iphi.wk <- iphi.wk + nphi
          phi <- list(fun = phi.fun, env = phi.env)
        }
        rk.env <- par$mkrk(par$env)
        rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
          env$fun(x, y, env$env, outer.prod)
        }
        nrk <- 1
        irk <- irk.wk
        irk.wk <- irk.wk + nrk
        rk <- list(fun = rk.fun, env = rk.env)
      }
    }
    else {

      #SENTHIL : this block in original code is for interaction terms.

      bin.fac <- n.phi <- phi.list <- rk.list <- NULL
      for (i in 1:dm) {
        type.wk <- var.type[[vlist[i]]][[1]]
        if (type.wk %in% c("nominal", "ordinal")) {
          if (type.wk == "nominal")
            rk.wk <-  mkrk.nominal(levels(x[[i]]))
          else rk.wk <-  mkrk.ordinal(levels(x[[i]]))
          phi.wk <- rk.wk
          n.phi <- c(n.phi, 0)
          #bin.fac <- c(bin.fac, !(nlevels(x[[i]]) > 2))
          bin.fac <- c(bin.fac, !(nlevels(x[[i]]) >= 2)) #edited by SENTHIL to force penalty for 2 groups as well
        }
        if (type.wk == "cubic.wLinPen") {

          #SENTHIL original code block below,
          # replcing for linear also penalized cubics
          # no phi, rk has an extra term for linear term penalty

          # range <- var.type[[vlist[i]]][[2]]
          # phi.wk <-  mkphi.cubic(range)
          # n.phi <- c(n.phi, 1)
          # rk.wk <- mkrk.cubic(range)
          # bin.fac <- c(bin.fac, 0)

          range <- var.type[[vlist[i]]][[2]]
          n.phi <- c(n.phi, 0)
          rk.wk <- mkrk.cubic.linearAlsoPenalized(range)
          bin.fac <- c(bin.fac, 0)

        }

        if (type.wk == "cubic") { #SENTHIL: this is the original block for cubic in gss, which doesn't penalize lin terms

          range <- var.type[[vlist[i]]][[2]]
          phi.wk <-  mkphi.cubic(range)
          n.phi <- c(n.phi, 1)
          rk.wk <- mkrk.cubic(range)
          bin.fac <- c(bin.fac, 0)

        }

        if (type.wk %in% c("cubic.per", "linear", "linear.per",
                           "sphere")) {
          range <- var.type[[vlist[i]]][[2]]
          n.phi <- c(n.phi, 0)
          phi.wk <- NULL
          if (type.wk == "cubic.per")
            rk.wk <-  mkrk.cubic.per(range)
          if (type.wk == "linear")
            rk.wk <-  mkrk.linear(range)
          if (type.wk == "linear.per")
            rk.wk <-  mkrk.linear.per(range)
          if (type.wk == "sphere")
            rk.wk <-  mkrk.sphere(range)
          bin.fac <- c(bin.fac, 0)
        }
        if (type.wk == "trig") {
          range <- var.type[[vlist[i]]][[2]]
          phi.wk <-  mkphi.trig(range)
          n.phi <- c(n.phi, 2)
          rk.wk <-  mkrk.trig(range)
          bin.fac <- c(bin.fac, 0)
        }
        if (type.wk == "tp") {
          par <- var.type[[vlist[i]]][[2]]
          order <- par$order
          mesh <- par$mesh
          weight <- par$weight
          if (is.vector(x[[i]]))
            xdim <- 1
          else xdim <- dim(x[[i]])[2]
          phi.wk <-  mkphi.tp(xdim, order, mesh, weight)
          n.phi <- c(n.phi, choose(xdim + order - 1,
                                   xdim) - 1)
          rk.wk <-  mkrk.tp(xdim, order, mesh, weight)
          bin.fac <- c(bin.fac, 0)
        }
        if (type.wk == "custom") {
          par <- var.type[[vlist[i]]][[2]]
          n.phi <- c(n.phi, par$nphi)
          if (par$nphi > 0)
            phi.wk <- par$mkphi(par$env)
          else phi.wk <- NULL
          rk.wk <- par$mkrk(par$env)
          bin.fac <- c(bin.fac, 0)
        }
        phi.list <- c(phi.list, list(phi.wk))
        rk.list <- c(rk.list, list(rk.wk))
      }
      if (!all(as.logical(n.phi + bin.fac)))
        nphi <- 0
      else {
        phi.env <- list(dim = dm, phi = phi.list, n.phi = n.phi,
                        bin.fac = bin.fac)
        phi.fun <- function(x, nu = 1, env) {
          ind <- nu - 1
          z <- 1
          for (i in 1:env$dim) {
            if (env$bin.fac[i]) {
              wk <- as.factor(names(env$phi[[i]]$env$code)[1])
              z <- z * env$phi[[i]]$fun(x[[i]], wk, env$phi[[i]]$env)
            }
            else {
              code <- ind%%env$n.phi[i] + 1
              ind <- ind%/%env$n.phi[i]
              z <- z * env$phi[[i]]$fun(x[[i]], code,
                                        env$phi[[i]]$env)
            }
          }
          z
        }
        nphi <- prod(n.phi + bin.fac)
        iphi <- iphi.wk
        iphi.wk <- iphi.wk + nphi
        phi <- list(fun = phi.fun, env = phi.env)
      }
      rk.env <- list(dim = dm, n.phi = n.phi, nphi = nphi,
                     phi = phi.list, rk = rk.list)
      rk.fun <- function(x, y, nu = 1, env, outer.prod = FALSE) {
        n.rk <- ifelse(env$n.phi, 2, 1)
        ind <- nu - !env$nphi
        z <- 1
        for (i in 1:env$dim) {
          code <- ind%%n.rk[i] + 1
          ind <- ind%/%n.rk[i]
          if (code == n.rk[i]) {
            z <- z * env$rk[[i]]$fun(x[[i]], y[[i]],
                                     env$rk[[i]]$env, outer.prod)
          }
          else {
            z.wk <- 0
            for (j in 1:env$n.phi[i]) {
              phix <- env$phi[[i]]$fun(x[[i]], j, env$phi[[i]]$env)
              phiy <- env$phi[[i]]$fun(y[[i]], j, env$phi[[i]]$env)
              if (outer.prod)
                z.wk <- z.wk + outer(phix, phiy)
              else z.wk <- z.wk + phix * phiy
            }
            z <- z * z.wk
          }
        }
        z
      }
      n.rk <- ifelse(n.phi, 2, 1)
      nrk <- prod(n.rk) - as.logical(nphi)
      irk <- irk.wk
      irk.wk <- irk.wk + nrk
      rk <- list(fun = rk.fun, env = rk.env)
    }
    term[[label]] <- list(vlist = vlist, iphi = iphi, nphi = nphi,
                          phi = phi, irk = irk, nrk = nrk, rk = rk)
  }
  term
}


ssanova0.mkterms <- function (formula, type = NULL, data = list(), weights, subset,
                              offset, na.action = na.omit, partial = NULL, method = "v",
                              varht = 1, prec = 1e-07, maxiter = 30,
                              forcePen2Groups=NULL  #SENTHIL
)
{
  library(gss)
  mf <- match.call()
  #print(mf)
  mf$type <- mf$method <- mf$varht <- mf$partial <- NULL
  mf$prec <- mf$maxiter <- NULL
  #mf$forcePen2Groups <- NULL
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  term <- gss.mkterms.force(mf, type)#, forcePen2Groups)
  nobs <- dim(mf)[1]
  s <- q <- NULL
  nq <- 0
  for (label in term$labels) {
    if (label == "1") {
      s <- cbind(s, rep(1, len = nobs))
      next
    }
    x <- mf[, term[[label]]$vlist]
    nphi <- term[[label]]$nphi
    nrk <- term[[label]]$nrk
    if (nphi) {
      phi <- term[[label]]$phi
      for (i in 1:nphi) s <- cbind(s, phi$fun(x, nu = i,
                                              env = phi$env))
    }
    if (nrk) {
      rk <- term[[label]]$rk
      for (i in 1:nrk) {
        nq <- nq + 1
        q <- array(c(q, rk$fun(x, x, nu = i, env = rk$env,
                               out = TRUE)), c(nobs, nobs, nq))
      }
    }
  }

  #This liss the variables for partial splines model.
  # The partial splines model have the model : Y ~ M\beta + \eta(x)
  #The way partial models are implemented is simple: only the S in Sd+Rc needs to be alterned relative to the standard splines only model. This is done by : cbind( S for \eta(x) with intercept, M from the partial part without itnercept  ) !
  # All related variables have a .p suffix
  if (!is.null(partial)) {
    mf.p <- model.frame(partial, data)
    for (lab in colnames(mf.p)) mf[, lab] <- mf.p[, lab]
    mt.p <- attr(mf.p, "terms")
    lab.p <- labels(mt.p)
    matx.p <- model.matrix(mt.p, data)[, -1, drop = FALSE] #-1 for ignoring the intercept, as intercept is aready included in the part of S matrix from \eta(x) .
    if (dim(matx.p)[1] != dim(mf)[1])
      stop("gss error in ssanova: partial data are of wrong size")
    matx.p <- scale(matx.p)
    center.p <- attr(matx.p, "scaled:center")
    scale.p <- attr(matx.p, "scaled:scale")
    s <- cbind(s, matx.p)
    part <- list(mt = mt.p, center = center.p, scale = scale.p)
  }
  else part <- lab.p <- NULL
  y <- model.response(mf, "numeric")
  w <- model.weights(mf)
  offset <- model.offset(mf)
  if (!is.null(offset)) {
    term$labels <- c(term$labels, "offset")
    term$offset <- list(nphi = 0, nrk = 0)
    y <- y - offset
  }
  if (!is.null(w)) {
    w <- sqrt(w)
    y <- w * y
    s <- w * s
    for (i in 1:nq) q[, , i] <- w * t(w * q[, , i])
  }
  if (qr(s)$rank < dim(s)[2])
    stop("gss error in ssanova0: unpenalized terms are linearly dependent")
  if (!nq)
    stop("gss error in ssanova0: use lm for models with only unpenalized terms")


  if(FALSE){ #SENTHIL: not solving; this is for solving single smoothing paramete.
    if (nq == 1) {
      q <- q[, , 1]
      z <- sspreg0(s, q, y, method, varht)
    }
    else z <- mspreg0(s, q, y, method, varht, prec, maxiter)
  }


  # desc is a variable that lists in each row, the number of unpenalized nd penalized terms
  #   for each variable
  desc <- NULL
  for (label in term$labels) desc <- rbind(desc, as.numeric(c(term[[label]][c("nphi",
                                                                              "nrk")])))
  if (!is.null(partial)) {
    desc <- rbind(desc, matrix(c(1, 0), length(lab.p), 2,
                               byrow = TRUE))
  }
  desc <- rbind(desc, apply(desc, 2, sum)) #catalogs total # penalized and unpenalized terms
  if (is.null(partial))
    rownames(desc) <- c(term$labels, "total")
  else rownames(desc) <- c(term$labels, lab.p, "total")
  colnames(desc) <- c("Unpenalized", "Penalized")
  obj <- c(list(call = match.call(), mf = mf, terms = term,
                partial = part, lab.p = lab.p, desc = desc,
                id.basis = seq(nrow(mf)), #SENTHIl
                s=s, #SENTHIL
                w=w,#SENTHIL
                #q=q #SENTHIL,
                r=q #making the return vars similar to gss.ssanova.terms
  )
  #, z #SENTHIL NOT RETURNING z, which I think  is soln for solving single smoothing paramete.
  )
  class(obj) <- c("ssanova0", "ssanova")
  obj
}

ssanova.mkterms <- function (formula, type = NULL, data = list(), weights, subset,
                             offset, na.action = na.omit, partial = NULL, method = "v",
                             alpha = 1.4, varht = 1, id.basis = NULL, nbasis = NULL, seed = NULL,
                             random = NULL, skip.iter = FALSE,
                             forcePen2Groups=NULL  #SENTHIL
)
{
  library(gss)
  mf <- match.call()
  mf$type <- mf$method <- mf$varht <- mf$partial <- NULL
  mf$alpha <- mf$id.basis <- mf$nbasis <- mf$seed <- NULL
  mf$random <- mf$skip.iter <- NULL
  #mf$forcePen2Groups <- NULL
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())

  wt <- model.weights(mf)
  nobs <- dim(mf)[1]
  if (is.null(id.basis)) {
    if (is.null(nbasis))
      nbasis <- max(30, ceiling(10 * nobs^(2/9)))
    if (nbasis >= nobs)
      nbasis <- nobs
    if (!is.null(seed))
      set.seed(seed)
    id.basis <- sample(nobs, nbasis, prob = wt)
  }
  else {
    if (max(id.basis) > nobs | min(id.basis) < 1)
      stop("gss error in ssanova: id.basis out of range")
    nbasis <- length(id.basis)
  }

  #term <- mkterm(mf, type)
  term <- gss.mkterms.force(mf, type)#, forcePen2Groups)

  if (!is.null(random)) {
    if (class(random) == "formula")
      random <- mkran(random, data)
  }
  s <- r <- NULL
  nq <- 0
  for (label in term$labels) {
    if (label == "1") {
      s <- cbind(s, rep(1, len = nobs))
      next
    }
    x <- mf[, term[[label]]$vlist]
    x.basis <- mf[id.basis, term[[label]]$vlist]
    nphi <- term[[label]]$nphi
    nrk <- term[[label]]$nrk
    if (nphi) {
      phi <- term[[label]]$phi
      for (i in 1:nphi) s <- cbind(s, phi$fun(x, nu = i,
                                              env = phi$env))
    }
    if (nrk) {
      rk <- term[[label]]$rk
      for (i in 1:nrk) {
        nq <- nq + 1
        r <- array(c(r, rk$fun(x, x.basis, nu = i, env = rk$env,
                               out = TRUE)), c(nobs, nbasis, nq))
      }
    }
  }
  if (is.null(r))
    stop("gss error in ssanova: use lm for models with only unpenalized terms")
  if (!is.null(partial)) {
    mf.p <- model.frame(partial, data)
    for (lab in colnames(mf.p)) mf[, lab] <- mf.p[, lab]
    mt.p <- attr(mf.p, "terms")
    lab.p <- labels(mt.p)
    matx.p <- model.matrix(mt.p, data)[, -1, drop = FALSE]
    if (dim(matx.p)[1] != dim(mf)[1])
      stop("gss error in ssanova: partial data are of wrong size")
    matx.p <- scale(matx.p)
    center.p <- attr(matx.p, "scaled:center")
    scale.p <- attr(matx.p, "scaled:scale")
    s <- cbind(s, matx.p)
    part <- list(mt = mt.p, center = center.p, scale = scale.p)
  }
  else part <- lab.p <- NULL
  if (qr(s)$rank < dim(s)[2])
    stop("gss error in ssanova: unpenalized terms are linearly dependent")
  y <- model.response(mf, "numeric")
  offset <- model.offset(mf)
  if (!is.null(offset)) {
    term$labels <- c(term$labels, "offset")
    term$offset <- list(nphi = 0, nrk = 0)
    y <- y - offset
  }
  if (!is.null(wt))
    wt <- sqrt(wt)
  if (nq == 1) {
    r <- r[, , 1]
    #z <- sspreg1(s, r, r[id.basis, ], y, wt, method, alpha,
    #    varht, random)
  }
  #else z <- mspreg1(s, r, id.basis, y, wt, method, alpha, varht,
  #    random, skip.iter)
  desc <- NULL
  for (label in term$labels) desc <- rbind(desc, as.numeric(c(term[[label]][c("nphi",
                                                                              "nrk")])))
  if (!is.null(partial)) {
    desc <- rbind(desc, matrix(c(1, 0), length(lab.p), 2,
                               byrow = TRUE))
  }
  desc <- rbind(desc, apply(desc, 2, sum))
  if (is.null(partial))
    rownames(desc) <- c(term$labels, "total")
  else rownames(desc) <- c(term$labels, lab.p, "total")
  colnames(desc) <- c("Unpenalized", "Penalized")
  obj <- c(list(call = match.call(), mf = mf, terms = term,
                desc = desc, alpha = alpha, id.basis = id.basis, partial = part,
                lab.p = lab.p, random = random, skip.iter = skip.iter,
                s=s, #SENTHIL
                wt=wt,#SENTHIL
                #q=q, #SENTHIL
                r=r
  )

  #, z #
  )


  class(obj) <- c("ssanova")
  obj
}
