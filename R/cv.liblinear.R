cv.liblinear <-function (x, y, weights,  lambda = NULL, 
                         type.measure = c("mse", "deviance", "class", "auc", "mae"), nfolds = 10, foldid, 
                         grouped = TRUE, keep = FALSE, parallel = FALSE, type=0,...) 
{
  # warning liblinear uses type which matches type.measure if not explicitly in cv.liblinear forma arguments list
  if (missing(type.measure)) 
    type.measure = "default"
  else type.measure = match.arg(type.measure)
  if (!is.null(lambda) && length(lambda) < 2) 
    stop("Need more than one value of lambda for cv.liblinear")
  N = nrow(x)
  if (missing(weights)) 
    weights = rep(1, N)
  else weights = as.double(weights)
  y = drop(y)
  liblinear.call = match.call(expand.dots = TRUE)
  which = match(c("type.measure", "nfolds", "foldid", 
                  "keep"), names(liblinear.call), F)
  if (any(which)) 
    liblinear.call = liblinear.call[-which]
  liblinear.call[[1]] = as.name("LiblineaR")
  liblinear.object = LiblineaR(x, y, sample_weights = weights, 
                               lambda = lambda, ...)
  liblinear.object$call = liblinear.call
  # TODO
  #nz = sapply(predict(liblinear.object, type = "nonzero"), 
  #            length)
  nz=c()
  
  if (missing(foldid)) 
    foldid = sample(rep(seq(nfolds), length = N))
  else nfolds = max(foldid)
  if (nfolds < 3) 
    stop("nfolds must be bigger than 3; nfolds=10 recommended")
  outlist = as.list(seq(nfolds))
  if (parallel) {
    outlist = foreach(i = seq(nfolds), .packages = c("LiblineaR")) %dopar% 
    {
      which = foldid == i
      if (is.matrix(y)) 
        y_sub = y[!which, ]
      else y_sub = y[!which]
      LiblineaR(x[!which, , drop = FALSE], y_sub, sample_weights = weights[!which],
                lambda = lambda, ...)
    }
  }
  else {
    for (i in seq(nfolds)) {
      which = foldid == i
      if (is.matrix(y)) 
        y_sub = y[!which, ]
      else y_sub = y[!which]
      
      outlist[[i]] = LiblineaR(x[!which, , drop = FALSE], 
                               y_sub, sample_weights = weights[!which], lambda = lambda, 
                               ...)
    }
  }
  fun = paste("cv.liblinearnet")
  lambda = liblinear.object$lambda
  cvstuff = do.call(fun, list(outlist, lambda, x, y, weights, 
                              foldid, type.measure, type, keep))
  cvm = cvstuff$cvm
  cvsd = cvstuff$cvsd
  nas = is.na(cvsd)
  if (any(nas)) {
    lambda = lambda[!nas]
    cvm = cvm[!nas]
    cvsd = cvsd[!nas]
    #nz = nz[!nas]
  }
  cvname = cvstuff$name
  out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, cvup = cvm + 
               cvsd, cvlo = cvm - cvsd, nzero = nz, name = cvname, liblinear.fit = liblinear.object)
  if (keep) 
    out = c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
  lamin = if (cvname == "AUC") 
    glmnet::getmin(lambda, -cvm, cvsd)
  else glmnet::getmin(lambda, cvm, cvsd)
  obj = c(out, as.list(lamin))
  class(obj) = "cv.liblinear"
  obj
}
  
