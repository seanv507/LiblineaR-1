require(LiblineaR)


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
df$y = lapply(.3*df$x1, function(x)  rbinom(1, 1, logistic(x)))

require(dplyr)
dfsmry <- df %>%
  group_by(x1, x2, x3, x4) %>%
  summarise(y = sum(y), ny = n() - y)

dfsmry_split <- dfsmry %>% gather(conv, cnt, y:ny) 
# %>% transmute(conv = conv=='y') didn't work!?
dfsmry_split$conv = dfsmry_split$conv =='y'
     

require(glmnet)
require(LiblineaR)
# to make life easier: see https://github.com/hong-revo/glmnetUtils

# base model
y_matrix <- as.matrix(cbind(dfsmry$ny, dfsmry$y))
x_matrix <- as.matrix(cbind(dfsmry$x1,  dfsmry$x2, dfsmry$x3, dfsmry$x4))
x_matrix_ll <- as.matrix(dfsmry_split[c('x1','x2','x3','x4')])
w_vector_ll <- dfsmry_split$cnt
y_vector_ll <- dfsmry_split$conv

glm_mod <- glmnet( y= y_matrix, x= x_matrix, family = "binomial",alpha = 0)
z2<-predict(glm_mod,newx = x_matrix,s=1,type='response')
ll_mod = LiblineaR(data=x_matrix_ll, target=y_vector_ll, sample_weights = w_vector_ll, type = 0, bias =1000)

# do crossvalidation
cvmod <- cv.glmnet( y=y_matrix , x= x_matrix, family = "binomial",keep=T)
