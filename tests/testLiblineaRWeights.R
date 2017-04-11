require(LiblineaR)
require(tidyr)
require(dplyr)
require(glmnet)

# ./train -W heart_scale.wgt  heart_scale

#.......................****.**
# optimization finished, #iter = 244
# Objective value = -131.819531
# nSV = 199
logistic <- function(x)  1/(1+exp(-x))

set.seed(1234)
n_samples <- 1e6
df <- data.frame(
  x1 = sample(10, n_samples, TRUE),
  x2 = sample(20, n_samples, TRUE),
  x3 = sample(5, n_samples, TRUE),
  x4 = sample(15, n_samples, TRUE))
df$y = sapply(.3*df$x1, function(x)  rbinom(1, 1, logistic(x)))


dfsmry <- df %>%
  group_by(x1, x2, x3, x4) %>%
  summarise(y = sum(y), ny = n() - y)

dfsmry_split <- dfsmry %>% gather(conv, cnt, y:ny) 
# %>% transmute(conv = conv=='y') didn't work!?
dfsmry_split$conv = dfsmry_split$conv =='y'
     



# to make life easier: see https://github.com/hong-revo/glmnetUtils

# base model
y_matrix <- as.matrix(cbind(dfsmry$ny, dfsmry$y))
x_matrix <- as.matrix(cbind(dfsmry$x1,  dfsmry$x2, dfsmry$x3, dfsmry$x4))
x_matrix_ll <- as.matrix(dfsmry_split[c('x1','x2','x3','x4')])
w_vector_ll <- dfsmry_split$cnt
y_vector_ll <- dfsmry_split$conv


lambda=c(100,10,1,0.1,0.01)
glm_mod <- glmnet( y= y_matrix, x= x_matrix, family = "binomial",alpha = 0, lambda=lambda, standardize=FALSE)
ll_mod <- LiblineaR(data=x_matrix_ll, target=y_vector_ll, sample_weights = w_vector_ll, type = 0, bias =1000,lambda = lambda, epsilon = 0.0001)

z<-predict(glm_mod,newx = x_matrix,type='response')
z_ll <- predict(ll_mod,newx = x_matrix,proba=T)
z_ll_a <- predict(ll_mod,newx = x_matrix,proba=T, lambda=lambda)

head(z)
head(z_ll$probabilities)

max(abs(z_ll$probabilities-z))
sum(abs(z_ll$probabilities == z_ll_a$probabilities))

all(z_ll$probabilities==z_ll_a$probabilities)

z_2<-predict(glm_mod,newx = x_matrix,s=lambda[2], type='response')
z_ll_2 <- predict(ll_mod,newx = x_matrix,proba=T, lambda=ll_mod$lambda[2])

sum(abs(z_2 - z_ll_2$probabilities))

#ll_mod <- LiblineaR(data=x_matrix_ll, target=y_vector_ll, sample_weights = w_vector_ll, type = 0, bias =1000,lambda = glm_mod$lambda, epsilon = 0.0001)

# do crossvalidation.  we need to specify the folds because for the liblinear case 
# all the positve cases are at the beginning and all the negative cases (of each group) at end
nfolds <- 10
foldid <- sample(rep(seq(nfolds), length = nrow(x_matrix)))
foldid_ll <- c(foldid, foldid)

# glmnet using matrix of y' (count of positive cases, count of negative)
cvmod <- cv.glmnet( y=y_matrix , x= x_matrix, lambda=lambda, foldid=foldid, family = "binomial",alpha=0, standardize=FALSE, keep=T)
# glmnet using  count of positive cases, then count of negative (ie same as liblinear)
cvmod_w <- cv.glmnet( y=y_vector_ll , x= x_matrix_ll, weights = w_vector_ll, lambda=lambda,  foldid=foldid_ll, family = "binomial",alpha=0, standardize=FALSE, keep=T)
cvmod_ll <- cv.liblinear(x_matrix_ll, y=y_vector_ll, weights = w_vector_ll, lambda = lambda, foldid = foldid_ll,type = 0, bias =1000, epsilon = 0.0001,
                          keep = T, parallel = FALSE) 

all(x_matrix==x_matrix_ll[1:15000,])
all(x_matrix==x_matrix_ll[15001:30000,])


cvmod$cvm
cvmod_w$cvm
cvmod_ll$cvm

print("There is a constant error between ")
