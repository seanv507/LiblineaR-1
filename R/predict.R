predict.LiblineaR<-function(object,newx,proba=FALSE,decisionValues=FALSE,...){ # Si on met decisionValues=TRUE comme default, les exemples passent le check. Le prob vient donc de DecisonValues=matrix()
	
	# <Arg preparation>
	
	error=c()
	
	# Nb samples
	n=dim(newx)[1]
	# Nb features
	p=dim(newx)[2]
	
	# Bias
	if(object$Bias){
		b=1
	}
	else{
		b=-1
	}
	
	# Returned probabilities default storage preparation
	Probabilities=matrix(data=-1)

	# Proba allowed?
	if(proba){
		if(!(object$Type==0 | object$Type==6 | object$Type==7)){
			cat("Probabilities only supported for Logistic Regressions, either L2-regularized primal or dual (LiblineaR 'type' 0 or 7), or L1-regularized (LiblineaR 'type' 6).\n")
			cat("Accordingly, 'proba' is set to FALSE.\n")
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
	if(decisionValues)
		DecisionValues=matrix(ncol=n*length(object$ClassNames),nrow=1,data=0)


	# Type 
	if(object$Type<0 || object$Type>7){
		cat("Invalid model object: Wrong value for 'type'. Must be an integer between 0 and 7 included.\n")
		return(-1)
	}
	
	# Codebook for labels
	cn=c(1:length(object$ClassNames))
	
	#
	# </Arg preparation>
	
	# as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
	
	ret <- .C(
		"predictLinear",
		as.double(Y),
		as.double(t(newx)),
		as.double(t(object$W)),
		as.integer(decisionValues),
		as.double(t(DecisionValues)),
		as.integer(proba),
		as.double(t(Probabilities)),
		as.integer(object$NbClass),
		as.integer(p),
		as.integer(n),
		as.double(b),
		as.integer(cn),
		as.integer(object$Type)
		)
	
	result=list()
	result$predictions=object$ClassNames[ret[[1]]]

	if(proba){
		result$probabilities=matrix(ncol=length(object$ClassNames),nrow=n,data=ret[[7]],byrow=TRUE)
		colnames(result$probabilities)=object$ClassNames
	}
	
	if(decisionValues){
		result$decisionValues=matrix(ncol=length(object$ClassNames),nrow=n,data=ret[[5]],byrow=TRUE)
		colnames(result$decisionValues)=object$ClassNames
	}

	return(result)

}
