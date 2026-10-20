
.invpsd <- function(psd, .inv.thresh=1e-10){

  if(any(psd > .Machine$double.eps) ){
    svs <- svd( psd )
    svs$d[ svs$d < .inv.thresh ] <- 0
    nz <- which( svs$d > 0 )
    if( length(nz) == 0 ){
      0*psd
    } else {
      tcrossprod( sweep( svs$v[,nz,drop=F], 2, 1/svs$d[nz], "*" ), svs$u[,nz,drop=F] )
    }
  } else {
    0*psd
  }

}



.initSvsZeta <- function(Y, K,
                    zformula=NULL, zdata=NULL
                    ){

  
  Zeta <- Zls <- NA
  if( !is.null(zformula) ){
    zres <- consZ( zformula, zdata )
    Z <- zres$Z
    Zls <- zres$Zls
    Zeta <- tcrossprod(Y, Zls )
  }

  svs <- list()
  svs <- tryCatch({
    irlba::irlba( Y, K, K )
  }, error=function(e){
    print(e)
    print("Trying svdr")
    tryCatch({
      z <- irlba::svdr( Y, K) #this returns a vector , not a column as other
      z
    },error=function(e2){
      print(e2)
      print("Trying standard svd")
      svd( Y, K, K )
    })
  })
  u <- sweep(svs$v,2,svs$d,"*")
  v <- svs$u


  list( "umat"=u,
        "vmat"=v,
        "Zls" = Zls,
        "Zeta"=Zeta
  )
}



