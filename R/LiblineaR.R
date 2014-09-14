LiblineaR<-function(data,labels,type=0,cost=1,epsilon=0.01,bias=TRUE,wi=NULL,cross=0,verbose=FALSE){
	# <Arg preparation>
		if(sparse <- inherits(data, "matrix.csr")){
				# trying to handle the sparse martix case
				library("SparseM")
				data = SparseM::t(SparseM::t(data)) # make sure column index are sorted
				n = data@dimension[1]
				p = data@dimension[2]
		} else {
				# Nb samples
				n=dim(data)[1]
				# Nb features
				p=dim(data)[2]
		}

	# Bias
	if(bias){
		b=1
	}
	else{
		b=-1
	}
	
	# Type 
	if(type<0 || type>7){
		cat("Wrong value for 'type'. Must be an integer between 0 and 7 included.\n")
		return(-1)
	}
	
	# Epsilon
	if(is.null(epsilon) || epsilon<0){
		# Will use liblinear default value for epsilon
		epsilon = -1
	}
	
	# Different class penalties?
	y=as.vector(labels)
	if(length(y)!=n){
		cat("Number of labels elements disagrees with number of data instances.\n")
		return(-1)
	}
	yLev=unique(y)
	nbClass=length(yLev)
	yLevC=c(1:nbClass)
	yC=y
	for(i in 1:nbClass){
		ind=which(y==yLev[i])
		yC[ind]=yLevC[i]
	}
	# Default
	defaultWi=rep(1,times=nbClass)
	names(defaultWi)=as.character(yLev)
	nrWi=nbClass
	WiLabels=yLevC
	
	if(!is.null(wi)){
		if(!is.null(names(wi))){
			if(as.integer(length(intersect(as.character(names(wi)),as.character(yLev))))<length(names(wi))){
				cat("Mismatch between provided names for 'wi' and class labels.\n")
				return(-1)
			}
			else{
				Wi=defaultWi
				for(i in 1:length(wi)){
					Wi[as.character(names(wi)[i])]=wi[i]
				}
			}
		}
		else{
			cat("wi has to be a named vector!\n")
			return(-1)
		}
	}
	else{
		Wi=defaultWi
	}
	
	# Cross-validation?
	if(cross<0){
		cat("Cross-validation argument 'cross' cannot be negative!\n")
		return(-1)
	}
	else if(cross>n){
		cat("Cross-validation argument 'cross' cannot be larger than the number of samples (",n,").\n",sep="")
		return(-1)
	}
	
	# Return storage preparation
	if(nbClass==2){
		if(bias){
			W=matrix(ncol=p+1,nrow=1,data=0)
		}
		else{
			W=matrix(ncol=p,nrow=1,data=0)
		}
	}
	else if(nbClass>2){
		if(bias){
			W=matrix(ncol=(p+1)*nbClass,nrow=1,data=0)
		}
		else{
			W=matrix(ncol=p*nbClass,nrow=1,data=0)
		}
	}
	else{
		cat("Wrong number of classes ( < 2 ).\n")
		return(-1)
	}
	
	#
	# </Arg preparation>
	
	# as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
	ret <- .C("trainLinear",
			as.double(W),
			as.double(if(sparse) data@ra else t(data)),
			as.double(yC),
			as.integer(n),
			as.integer(p),
			# sparse index info
			as.integer(sparse),
			as.integer(if(sparse) data@ia else 0),
			as.integer(if(sparse) data@ja else 0),

			as.double(b),
			as.integer(type),
			as.double(cost),
			as.double(epsilon),
			as.integer(nrWi),
			as.double(Wi),
			as.integer(WiLabels),
			as.integer(cross),
			as.integer(verbose),
			PACKAGE="LiblineaR"
			)
			
	if(cross==0){
		if(nbClass==2){
			w=matrix(ncol=dim(W)[2],nrow=1,data=ret[[1]])
		}
		else{
			w=matrix(ncol=dim(W)[2]/nbClass,nrow=nbClass,data=ret[[1]],byrow=FALSE)
		}
		if(!is.null(colnames(data))){
			if(bias){
				colnames(w)=c(colnames(data),"Bias")
			}
			else{
				colnames(w)=colnames(data)
			}
		}
		else{
			if(bias){
				colnames(w)=c(paste("W",c(1:dim(data)[2]),sep=""),"Bias")
			}
			else{
				colnames(w)=c(paste("W",c(1:dim(data)[2]),sep=""))
			}
		}
	
		types=c("L2-regularized logistic regression (L2R_LR)", "L2-regularized L2-loss support vector classification dual (L2R_L2LOSS_SVC_DUAL)", "L2-regularized L2-loss support vector classification primal (L2R_L2LOSS_SVC)", "L2-regularized L1-loss support vector classification dual (L2R_L1LOSS_SVC_DUAL)", "multi-class support vector classification by Crammer and Singer (MCSVM_CS)", "L1-regularized L2-loss support vector classification (L1R_L2LOSS_SVC)", "L1-regularized logistic regression (L1R_LR)","L2-regularized logistic regression dual (L2R_LR_DUAL)")
		m=list()
		class(m)="LiblineaR"
		m$TypeDetail=types[type+1]
		m$Type=type
		m$W=w
		m$Bias=bias
		m$ClassNames=yLev
		m$NbClass=nbClass
		return(m)
	}
	else{
		return(ret[[1]][1])
	}
}
