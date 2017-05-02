library(tidyr)
library(dplyr)

gen_logreg_test_data <- function(){
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
  colnames(x_matrix) <- c('x1','x2','x3','x4')
  x_matrix_ll <- as.matrix(dfsmry_split[c('x1','x2','x3','x4')])
  w_vector_ll <- dfsmry_split$cnt
  y_vector_ll <- dfsmry_split$conv
  
  list(df=df, dfsmry=dfsmry, dfsmry_split=dfsmry_split, 
       y_matrix=y_matrix, x_matrix=x_matrix, x_matrix_ll=x_matrix_ll, 
       y_vector_ll=y_vector_ll, w_vector_ll=w_vector_ll)
}



