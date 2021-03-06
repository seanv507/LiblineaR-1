cv.liblinearnet <-
  function (outlist, lambda, x, y, weights, foldid, type.measure, type=0,
             keep = FALSE)
{
  typenames = c(mse = "Mean-Squared Error", mae = "Mean Absolute Error", 
    deviance = "Binomial Deviance", auc = "AUC", class = "Misclassification Error")
  if (type.measure == "default") 
    type.measure = "deviance"
  if (!match(type.measure, c("mse", "mae", "deviance", "auc", 
                             "class"), FALSE)) {
    warning("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used")
    type.measure = "deviance"
  }
  prob_min = 1e-05
  prob_max = 1 - prob_min
  nc = dim(y)
  if (is.null(nc)) {
    y = as.factor(y)
    ntab = table(y)
    nc = as.integer(length(ntab))
    y = diag(nc)[as.numeric(y), ]
  }
  N = nrow(y)
  nfolds = max(foldid)
  if ((N/nfolds < 10) && type.measure == "auc") {
    warning("Too few (< 10) observations per fold for type.measure='auc' in cv.lognet; changed to type.measure='deviance'. Alternatively, use smaller value for nfolds", 
            call. = FALSE)
    type.measure = "deviance"
  }
  
  mlami=max(sapply(outlist,function(obj)min(obj$lambda)))
  which_lam=lambda >= mlami

  predmat = matrix(NA, nrow(y), length(lambda))
  nlams = double(nfolds)
  for (i in seq(nfolds)) {
    which = foldid == i
    fitobj = outlist[[i]]
    preds = predict(fitobj,x[which, , drop = FALSE], lambda=lambda[which_lam],  
      proba = T)
    nlami = sum(which_lam)
    predmat[which, seq(nlami)] = preds$probabilities
    nlams[i] = nlami
  }
  if (type.measure == "auc") {
    cvraw = matrix(NA, nfolds, length(lambda))
    good = matrix(0, nfolds, length(lambda))
    for (i in seq(nfolds)) {
      good[i, seq(nlams[i])] = 1
      which = foldid == i
      for (j in seq(nlams[i])) {
        cvraw[i, j] = auc.mat(y[which, ], predmat[which, 
               j], weights[which])
      }
    }
    N = apply(good, 2, sum)
    weights = tapply(weights, foldid, sum)
  }
  else {
    ywt = apply(y, 1, sum)
    y = y/ywt
    weights = weights * ywt
    N = nrow(y) - apply(is.na(predmat), 2, sum)
    cvraw = switch(type.measure, mse = (y[, 1] - (1 - predmat))^2 + 
      (y[, 2] - predmat)^2, mae = abs(y[, 1] - (1 - predmat)) + 
      abs(y[, 2] - predmat), deviance = {
        predmat = pmin(pmax(predmat, prob_min), prob_max)
        lp = y[, 1] * log(1 - predmat) + y[, 2] * log(predmat)
        ly = log(y)
        ly[y == 0] = 0
        ly = drop((y * ly) %*% c(1, 1))
        2 * (ly - lp)
      }, class = y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 
           0.5))
    
    cvob = cvcompute(cvraw, weights, foldid, nlams)
    cvraw = cvob$cvraw
    weights = cvob$weights
    N = cvob$N
  }
  cvm = apply(cvraw, 2, weighted.mean, w = weights, na.rm = TRUE)
  cvsd = sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean, 
    w = weights, na.rm = TRUE)/(N - 1))
  out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
  if (keep) 
    out$fit.preval = predmat
  out
}
