\name{LiblineaR}
\Rdversion{1.1}
\alias{LiblineaR}
\title{Linear Predictive Models Estimation Based On The Liblinear C/C++ Library.}
\description{
	\code{LiblineaR} allows the estimation of predictive linear models such as L1- or L2-regularized logistic regression, L1- or L2-regularized L2-loss support vector classification, L2-regularized L1-loss support vector classification and multi-class support vector classification. The estimation of the models is particularly fast as compared to other libraries. The implementation is based on the liblinear C/C++ library for machine learning. Multiclass classification is supported. 
}
\usage{
	LiblineaR(data, labels, type=0, cost=1, epsilon = 0.01,

bias = TRUE, wi = NULL, cross = 0, verbose = FALSE)

}
\arguments{
  \item{data}{
	a nxp data matrix. Each row stands for an example (sample, point) and each column stands for a dimension (feature, variable).
	A sparse matrix (from SparseM package) will also work. 
}
  \item{labels}{
	a response vector for classification tasks with one label for each of the n rows of \code{data}. Can be a 1xn matrix, a simple vector or a factor.
}
  \item{type}{\code{LiblineaR} can produce 7 types of (generalized) linear models, by combining several types of loss functions and regularization schemes. The regularization can be L1 or L2, and the losses can be the regular L2-loss for SVM (hinge loss), L1-loss for SVM, or the logistic loss for logistic regression. The default value for \code{type} is 0. See details below. Valid options are:
		\itemize{
	        \item 0 -- L2-regularized logistic regression
	        \item 1 -- L2-regularized L2-loss support vector classification (dual)
	        \item 2 -- L2-regularized L2-loss support vector classification (primal)
	        \item 3 -- L2-regularized L1-loss support vector classification (dual)
	        \item 4 -- multi-class support vector classification by Crammer and Singer
	        \item 5 -- L1-regularized L2-loss support vector classification
	        \item 6 -- L1-regularized logistic regression
		\item 7 -- L2-regularized logistic regression (dual)
		}
}
  \item{cost}{
	cost of constraints violation (default: 1). Rules the trade-off between regularization and correct classification on \code{data}. It can be seen as the inverse of a regularization constant. See information on the 'C' constant in details below. A usually good baseline heuristics to tune this constant is provided by the \code{heuristicC} function of this package.
}
  \item{epsilon}{
	set tolerance of termination criterion for optimization. Default to 0.01. If \code{NULL}, the liblinear defaults are used, which are:
		\describe{
	        \item{if \code{type} is 0, 2, 5 or 6}{\code{epsilon}=0.01}
	        \item{if \code{type} is 1, 3, 4 or 7}{\code{epsilon}=0.1}
		}
	The meaning of \code{epsilon} is as follows:
		\describe{
	        \item{if \code{type} is 0 or 2:}{\eqn{|f'(w)|_{2} \le \code{epsilon}*\min(pos,neg)/l*|f'(w_{0})|_{2}}{|f'(w)|_2 <= \code{epsilon}*min(pos,neg)/l*|f'(w0)|_2},\cr
	                where f is the primal function and pos/neg are # of positive/negative data.}
	        \item{if \code{type} is 1, 3, 4 or 7:}{Dual maximal violation \eqn{\le}{<=} \code{epsilon}}
	        \item{if \code{type} is 5 or 6:}{\eqn{|f'(w)|_\infty \le \code{epsilon}*\min(pos,neg)/l*|f'(w_{0})|_\infty}{|f'(w)|_inf <= \code{epsilon}*min(pos,neg)/l*|f'(w0)|_inf},\cr
	        		where f is the primal function}
		}
}
  \item{bias}{
	if \code{bias} is \code{TRUE} (default), instances of \code{data} becomes [\code{data}; 1].
}
  \item{wi}{
	a named vector of weights for the different classes, used for asymetric class sizes. Not all factor levels have to be supplied (default weight: 1). All components have to be named according to the corresponding class label.
}
  \item{cross}{
	if an integer value k>0 is specified, a k-fold cross validation on \code{data} is performed to assess the quality of the model via a measure of the accuracy. Note that this metric might not be appropriate if classes are largely unbalanced. Default is 0.
}
  \item{verbose}{
	if \code{TRUE}, information are printed. Default is \code{FALSE}.
}
}
\details{
	For details for the implementation of liblinear, see the README file of the original c/c++ liblinear library at \url{http://www.csie.ntu.edu.tw/~cjlin/liblinear}.
}
\value{
	If \code{cross}>0, the average accuracy computed over \code{cross} runs of cross-validation is returned.\cr\cr
	Otherwise, an object of class \code{"LiblineaR"} containing the fitted model is returned, including:
	\item{TypeDetail}{A string decsribing the type of model fitted, as determined by \code{type}.}
	\item{Type}{An integer corresponding to \code{type}.}
	\item{W}{A matrix with the model weights. If \code{bias} is TRUE, \code{W} contains p+1 columns, the last being the bias term. The columns are named according to the names of \code{data}, if provided, or \code{"Wx"} where \code{"x"} ranges from 1 to the number of dimensions. The bias term is named \code{"Bias"}.If the number of classes is 2, the matrix only has one row. If the number of classes is k>2, it has k rows. Each row i corresponds then to a linear model discriminating between class i and all the other classes. If there are more than 2 classes, rows are named according to the class i which is opposed to the other classes.}
	\item{Bias}{TRUE or FALSE, according to the value of \code{bias}}
	\item{ClassNames}{A vector containing the class names}
}
\references{
	\itemize{
		\item 
			For more information on liblinear itself, refer to:\cr
			R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.\cr
			\emph{LIBLINEAR: A Library for Large Linear Classification,}\cr
			Journal of Machine Learning Research 9(2008), 1871-1874.\cr
			\url{http://www.csie.ntu.edu.tw/~cjlin/liblinear}
	}
}
\author{
	Thibault Helleputte (based on C/C++-code by Chih-Chung Chang and Chih-Jen Lin)\cr
	\email{thelleputte@gmail.com}
}
\note{
	Classification models usually perform better if each dimension of the data is first centered and scaled.
}

\seealso{
	\code{\link{predict.LiblineaR}},\code{\link{heuristicC}}
}
\examples{
data(iris)
attach(iris)

x=iris[,1:4]
y=factor(iris[,5])
train=sample(1:dim(iris)[1],100)

xTrain=x[train,]
xTest=x[-train,]
yTrain=y[train]
yTest=y[-train]

# Center and scale data
s=scale(xTrain,center=TRUE,scale=TRUE)

# Logistic Regression
t=0

# Find the best model with the best cost parameter via 10-fold cross-validations
tryTypes=c(0:7)
tryCosts=c(1000,100,10,1,0.1,0.01,0.001)
bestCost=NA
bestAcc=0
bestType=NA

for(ty in tryTypes){
for(co in tryCosts){
	acc=LiblineaR(data=s,labels=yTrain,type=ty,cost=co,bias=TRUE,cross=10,verbose=FALSE)
	cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
	if(acc>bestAcc){
		bestCost=co
		bestAcc=acc
		bestType=ty
	}
}
}

cat("Best model type is:",bestType,"\n")
cat("Best cost is:",bestCost,"\n")
cat("Best accuracy is:",bestAcc,"\n")

# Re-train best model with best cost value.
m=LiblineaR(data=s,labels=yTrain,type=bestType,cost=bestCost,bias=TRUE,verbose=FALSE)

# Scale the test data
s2=scale(xTest,attr(s,"scaled:center"),attr(s,"scaled:scale"))

# Make prediction
pr=FALSE
if(bestType==0 | bestType==7) pr=TRUE

p=predict(m,s2,proba=pr,decisionValues=TRUE)

# Display confusion matrix
res=table(p$predictions,yTest)
print(res)

# Compute Balanced Classification Rate
BCR=mean(c(res[1,1]/sum(res[,1]),res[2,2]/sum(res[,2]),res[3,3]/sum(res[,3])))
print(BCR)

# Example of the use of a sparse matrix:

library(SparseM)

# Sparsifying the iris dataset:
iS=apply(iris[,1:4],2,function(a){a[a<quantile(a,probs=c(0.25))]=0;return(a)})
irisSparse<-as.matrix.csr(iS)

# Applying a similar methodology as above:

xTrain=irisSparse[train,]
xTest=irisSparse[-train,]

# Re-train best model with best cost value.
m=LiblineaR(data=xTrain,labels=yTrain,type=bestType,cost=bestCost,bias=TRUE,verbose=FALSE)

# Make prediction
p=predict(m,xTest,proba=pr,decisionValues=TRUE)

# Display confusion matrix
res=table(p$predictions,yTest)
print(res)

}

\keyword{ classif }
\keyword{ multivariate }
\keyword{ models }
\keyword{ optimize }
\keyword{ classes }
