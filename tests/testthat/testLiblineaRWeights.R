library(LiblineaR)
library(glmnet)
library(Matrix)
source('../gen_logreg_test_data.R')


# ./train -W heart_scale.wgt  heart_scale

#.......................****.**
# optimization finished, #iter = 244
# Objective value = -131.819531
# nSV = 199
heart <- libsvmread('../heart_scale')
heart_weight <- read.delim('../heart_scale.wgt', header=F)
heart_model <- LiblineaR(data=heart$x, target=heart$y, sample_weights=heart_weight, bias=-1, type=1, epsilon=0.1, verbose = T)

test_libsvmread <- function(){
  expect_equal(dim(x_y$x),c(270,13))
  x_y <- libsvmread('../heart_scale')
  expect_true(abs(mean(x_y$y) - -0.1111111)<1e-6)
  expect_true(abs(sum(x_y$x) - -666.4009)<1e-4)
}


dat <- gen_logreg_test_data()


lambda=c(100,10,1,0.1,0.01)



test_sparse_Matrix <- function(){
  # create sparse matrix from Matrix package and test preserve column names
  ll_mod <- LiblineaR(data=dat$x_matrix_ll, target=dat$y_vector_ll, sample_weights = dat$w_vector_ll, 
                      type = 0, bias =1000,lambda = lambda, epsilon = 1e-10)
  x_matrix_ll_M <- sparse.model.matrix(~ . -1,data=data.frame(dat$x_matrix_ll))
  ll_mod_M <- LiblineaR(data=x_matrix_ll_M, target=dat$y_vector_ll, sample_weights = dat$w_vector_ll, 
                        type = 0, bias =1000,lambda = lambda, epsilon = 1e-10)
  print(ll_mod_M$W)
  expect_equal(colnames(ll_mod_M$W)[1:ncol(x_matrix_ll_M)],colnames(x_matrix_ll_M))
  expect_equal(ll_mod_M$W , ll_mod$W)
}

test_counts <- function(){
  ll_mod <- LiblineaR(data=dat$x_matrix_ll, target=dat$y_vector_ll, sample_weights = dat$w_vector_ll, 
                      type = 0, bias =1000,lambda = lambda, epsilon = 0.00001)
  ll_mod_count <- LiblineaR(data=dat$x_matrix, target=dat$y_matrix, sample_weights = NULL, 
                      type = 0, bias =1000,lambda = lambda, epsilon = 0.00001)
  # Liblinear takes first occurence as positive class
  expect_equal(ll_mod$W, ll_mod_count$W)
}

test_response <- function(){
  ll_mod <- LiblineaR(data=dat$x_matrix_ll, target=dat$y_vector_ll, sample_weights = dat$w_vector_ll, 
                      type = 0, bias =1000,lambda = lambda, epsilon = 0.00001)
  z_ll <- predict(ll_mod,newx = dat$x_matrix,proba=T)
  z_ll_lambdas <- predict(ll_mod,newx = dat$x_matrix,proba=T, lambda=lambda)
  expect_equal(z_ll$probabilities,z_ll_lambdas$probabilities)
  z_ll_response <- predict(ll_mod,newx = dat$x_matrix,type='response')
  expect_equal(z_ll$probabilities,z_ll_response)
  
}

test_glmnet_liblinear <- function(){
  glm_mod <- glmnet( y= dat$y_matrix, x= dat$x_matrix, family = "binomial",alpha = 0, lambda=lambda, standardize=FALSE)
  
  
  bias <- 1000
  ll_mod <- LiblineaR(data=dat$x_matrix_ll, target=dat$y_vector_ll, sample_weights = dat$w_vector_ll, 
                      type = 0, bias =bias,lambda = lambda, epsilon = 1e-10)
  expect_lt(max(abs((glm_mod$a0 - ll_mod$W[,'Bias']*bias)/glm_mod$a0)) , .01) #1% relative error
  expect_lt(max(abs((glm_mod$beta - t(ll_mod$W[,colnames(dat$x_matrix)]))/(glm_mod$beta+1e-3))), .02) # 2% rel error for large enough numbers
  z<-predict(glm_mod,newx = dat$x_matrix,type='response')
  z_ll<-predict(ll_mod,newx = dat$x_matrix,type='response')
  expect_lt(max(abs((z - z_ll)/(z))), .001) # 0.1% rel error
  
  # test get correct (same)  response for single lambda
  
  z_ll_2 <- predict(ll_mod,newx = dat$x_matrix,type='response', lambda=ll_mod$lambda[2])
  expect_equal(z_ll[,2], drop(z_ll_2))
}



#ll_mod <- LiblineaR(data=x_matrix_ll, target=y_vector_ll, sample_weights = w_vector_ll, type = 0, bias =1000,lambda = glm_mod$lambda, epsilon = 0.0001)

# do crossvalidation.  we need to specify the folds because for the liblinear case 
# all the positve cases are at the beginning and all the negative cases (of each group) at end
# nfolds <- 10
# foldid <- sample(rep(seq(nfolds), length = nrow(x_matrix)))
# foldid_ll <- c(foldid, foldid)

# glmnet using matrix of y' (count of positive cases, count of negative)
# cvmod <- cv.glmnet( y=y_matrix , x= x_matrix, lambda=lambda, foldid=foldid, family = "binomial",alpha=0, standardize=FALSE, keep=T)
# glmnet using  count of positive cases, then count of negative (ie same as liblinear)
# cvmod_w <- cv.glmnet( y=y_vector_ll , x= x_matrix_ll, weights = w_vector_ll, lambda=lambda,  foldid=foldid_ll, family = "binomial",alpha=0, standardize=FALSE, keep=T)
# cvmod_ll <- cv.liblinear(x_matrix_ll, y=y_vector_ll, weights = w_vector_ll, lambda = lambda, foldid = foldid_ll,type = 0, bias =1000, epsilon = 0.0001,
#                          keep = T, parallel = FALSE) 





print("There is a constant offset between deviance metric using counts and 1/0- known effect")
# cvmod$cvm-cvmod$cvm[1]
# cvmod_w$cvm-cvmod_w$cvm[1]
# cvmod_ll$cvm-cvmod_ll$cvm[1]