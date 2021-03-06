### Documentation ####
#' Predictions with LiblineaR model
#' 
#' The function applies a model (classification or regression) produced by the \code{LiblineaR} function to every row of a
#' data matrix and returns the model predictions.
#' 
#' @param object Object of class \code{"LiblineaR"}, created by
#'   \code{LiblineaR}.
#' @param newx An n x p matrix containing the new input data. A vector will be
#'   transformed to a n x 1 matrix. A sparse matrix (from Matrix package) will
#'   also work.
#' @param proba Logical indicating whether class probabilities should be
#'   computed and returned. Only possible if the model was fitted with
#'   \code{type}=0, \code{type}=6 or \code{type}=7, i.e. a Logistic Regression.
#'   Default is \code{FALSE}.
#' @param type Character 'response' return probabilities for logistic regression. Not tested for other model types.
#'   \code{type}=0, \code{type}=6 or \code{type}=7, i.e. a Logistic Regression.
#'   Default is \code{FALSE}.

#' @param decisionValues Logical indicating whether model decision values should
#'   be computed and returned. Only possible for classification models
#'   (\code{type}<10). Default is \code{FALSE}.
#' @param ... Currently not used
#' 
#' @return 	By default, the returned value is a list with a single entry:
#' \item{predictions}{A vector of predicted labels (or values for regression).}
#' If \code{proba} is set to \code{TRUE}, and the model is a logistic
#' regression, an additional entry is returned:
#' \item{probabilities}{An n x k matrix (k number of classes) of the class
#'   probabilities. The columns of this matrix are named after class labels.}
#' If \code{decisionValues} is set to \code{TRUE}, and the model is not a
#' regression model, an additional entry is returned:
#' \item{decisionValues}{An n x k matrix (k number of classes) of the model
#'   decision values. The columns of this matrix are named after class labels.}
#' 
#' @references
#' 	\itemize{
#' \item 
#' For more information on 'LIBLINEAR' itself, refer to:\cr
#' R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.\cr
#' \emph{LIBLINEAR: A Library for Large Linear Classification,}\cr
#' Journal of Machine Learning Research 9(2008), 1871-1874.\cr
#' \url{http://www.csie.ntu.edu.tw/~cjlin/liblinear}
#' }
#' 
#' @author Thibault Helleputte \email{thibault.helleputte@@dnalytics.com} and\cr
#'   Pierre Gramme \email{pierre.gramme@@dnalytics.com} and\cr
#'   Jerome Paul \email{jerome.paul@@dnalytics.com}.\cr 
#'   Based on C/C++-code by Chih-Chung Chang and Chih-Jen Lin
#' 
#' @note If the data on which the model has been fitted have been centered
#'   and/or scaled, it is very important to apply the same process on the
#'   \code{newx} data as well, with the scale and center values of the training
#'   data.
#' 
#' @seealso \code{\link{LiblineaR}}
#' 
#' @keywords classif regression multivariate models optimize classes
#' 
#' @useDynLib LiblineaR predictLinear
#' @export

### Implementation ####
predict.LiblineaR<-function(object, newx, type=NULL, proba=FALSE, decisionValues=FALSE, lambda=NULL,...){
    
    # <Arg preparation>
    
    error=c()
    
    if(sparse <- inherits(newx, "sparseMatrix")){
        newx <- as(newx,"RsparseMatrix")
        #TODO if(requireNamespace("SparseM",quietly=TRUE)){
            # trying to handle the sparse matrix case
            #newx = SparseM::t(SparseM::t(newx)) # make sure column index are sorted
        n = newx@Dim[1]
        p = newx@Dim[2]
    } else {
        # Nb samples
        n=dim(newx)[1]
        # Nb features
        p=dim(newx)[2]
    }
    
    # restrict features of newx to those used in model's W
    fNames = colnames(object$W)[colnames(object$W) != "Bias"]
    if(is.null(colnames(newx))) { # also handles SparseM (matrix.csr) case which always returns NULL for colnames
        if (p != length(fNames))
            stop("dims of 'test' and 'train' differ")
    } else {
        # Check presence of all features and drop irrelevant features (allow reordering)
        if (!all(fNames %in% colnames(newx)))
            stop("columns of 'test' and 'train' differ")
        if(!identical(fNames,colnames(newx)))
            newx <- newx[,fNames,drop=FALSE]
    }
    
    # Type 
    if(!object$Type %in% c(0:7,11:13)){
        stop("Invalid model object: Wrong value for 'type'. Must be an integer between 0 and 7  or between 11 and 13 included.\n")
    }
    isRegression = object$Type>=11
    
    # Bias
    b = object$Bias
    
    if (!is.null(type) && (type=='response') && (object$Type %in% c(0,6,7))) proba=T

    # Returned probabilities default storage preparation
    Probabilities=matrix(data=-1)
    
    # Proba allowed?
    if(proba){
        if(!object$Type %in% c(0,6,7)){
            warning("Computing probabilities is only supported for Logistic Regressions (LiblineaR 'type' 0, 6 or 7).\n",
                    "Accordingly, 'proba' was set to FALSE.")
            proba=FALSE
        }
        else{
            # Returned probabilities storage preparation 
            Probabilities=matrix(ncol=n*length(object$ClassNames),nrow=1,data=0)
        }
    }
    
    # Returned labels storage preparation
    Y=matrix(ncol=n,nrow=1,data=0)
    
    # Returned decision values default storage preparation
    DecisionValues=matrix(data=-1)
    
    # Returned decision values storage preparation
    if(decisionValues) {
        if(isRegression){
            warning("Computing decision values is only supported for classification (LiblineaR 'type' between 0 and 7).\n",
                    "Accordingly, 'decisionValues' was set to FALSE.")
            decisionValues=FALSE
        }
        else{
            DecisionValues=matrix(ncol=n*length(object$ClassNames),nrow=1,data=0)
        }
    }
    
    # Codebook for labels
    cn=c(1:length(object$ClassNames))
    
    #
    # </Arg preparation>
    
    # as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
    if (is.null(lambda)){
      wts_index <-seq(length(object$cost))
    }else{
      wts_index <- match(lambda,object$lambda)
      if (length(wts_index)!=length(lambda)) warning('not all lambdas found')
    }
    predictions <- list()
    probabilities <- list()
    decisionval <- list()
    
    for (i in seq(length(wts_index))){
      wti <- wts_index[i]
      if (object$Type %in% c(0,6,7))
        n_class_wt <- 1
      else 
        n_class_wt <- object$NbClass
      
      W=object$W[((wti-1)*n_class_wt+1):(wti*n_class_wt), ]
      ret <- .C(
        "predictLinear",
        as.double(Y),
        as.double(if(sparse) newx@x else t(newx)),
        as.double(W),
        as.integer(decisionValues),
        as.double(t(DecisionValues)),
        as.integer(proba),
        as.double(t(Probabilities)),
        as.integer(object$NbClass),
        as.integer(p),
        as.integer(n),
        # sparse index info
        as.integer(sparse),
        as.integer(if(sparse) newx@p+1 else 0),
        as.integer(if(sparse) newx@j+1 else 0),
        
        as.double(b),
        as.integer(cn),
        as.integer(object$Type),
        PACKAGE="LiblineaR"
      )
      if(isRegression)
        predictions[[i]]=ret[[1]]
      else
        predictions[[i]]=object$ClassNames[as.integer(ret[[1]])]
      
      if(proba){
        if (n_class_wt ==1){
          if(is.logical(object$ClassNames)) pred_case_col <- object$ClassNames==TRUE 
          else pred_case_col <- 1 # we take first class presented (in Liblinear) as positive class
          probabilities[[i]] =  matrix(ncol=length(object$ClassNames),nrow=n,data=ret[[7]],byrow=TRUE)[,pred_case_col]
        }else{
          probabilities[[i]] =  matrix(ncol=length(object$ClassNames),nrow=n,data=ret[[7]],byrow=TRUE)
          colnames(probabilities[[i]])=object$ClassNames
        }
        
      }
      decisionval[[i]]=matrix(ncol=length(object$ClassNames),nrow=n,data=ret[[5]],byrow=TRUE)
      colnames(decisionval[[i]])=object$ClassNames
    }
    
    if (!is.null(type) && type=='response'){
      if (proba ) result = do.call(cbind,probabilities)
      else result=do.call(cbind,predictions)
    }else{
      result=list()
      result$predictions=do.call(cbind,predictions)
      if(proba){
        result$probabilities=do.call(cbind,probabilities)
      }
    
      if(decisionValues){
        result$decisionValues=do.call(cbind,decisionval)
      }
      
    }
    return(result)
    
}
