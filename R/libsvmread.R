
### Documentation ####
#' Read libsvm format data from file
#' 
#' The function \code{libsvmread} reads a libSVM format file
#' returning a list with vector y and sparse matrix x (with column names if available).
#' 
#' @param filename char vector
#' 
#' @return the returned value is a list with 2 elements:
#' \item{y}{A vector of values.}
#' \item{x}{A sparse matrix. The columns of this matrix are named using the feature labels (if noninteger).}
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
#' @export

libsvmread <- function(filename ){
  lins <- readLines(filename)
  nrows <- length(lins)
  
  
  labels <- list(seq(nrows)) 
  vals <- list(seq(nrows)) 
  i_list <- list(seq(nrows))
  y_feat <- strsplit(lins, split=' ')
  prob_y <- sapply(y_feat, function(lin) as.double(lin[1]))
  for (i in seq(nrows)){
    # split line
    
    #if len(line) == 1: line += [''] what to do about this?
    
    n_feat_row <- length(y_feat[[i]])-1
    label_row <- character(n_feat_row)
    val_row <- numeric(n_feat_row)
    i_list[[i]] <- rep(i, n_feat_row)
    for (i_feat in seq(n_feat_row)){
      feat <- y_feat[[i]][i_feat + 1]
      label_value <- strsplit(feat,split=':')
      label_row[i_feat] <- label_value[[1]][1]
      val_row[i_feat] <- as.double(label_value[[1]][2])
    }
    labels[[i]] <- label_row
    vals[[i]] <- val_row
    
  }
  labels <- unlist(labels)
  labs <-  unique(labels)
  if (all(!is.na(as.integer(labs)))){
    j <- as.integer(labels)
    colnams <-  NULL
  }else{
    labels <- factor(labels)
    j <- as.integer(labels)
    colnams <- levels(j)
  }
  vals <- unlist(vals)
  i_list <- unlist(i_list)
  x <- sparseMatrix(i=i_list, j=j, x=vals, dimnames = list(NULL,colnams))
  list(y=prob_y, x=x)
}
