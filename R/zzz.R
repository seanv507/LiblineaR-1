.onLoad<-function(libname,pkgname){
	library.dynam(chname="trainLinear",package=pkgname,lib.loc=libname,verbose=FALSE)
	library.dynam(chname="predictLinear",package=pkgname,lib.loc=libname,verbose=FALSE)
}