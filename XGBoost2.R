df<-read.csv("/home/ashraydimri/Personal/News_Paper/googleplusData.csv",stringsAsFactors = F)
df$X<-NULL
df$GooglePlus<-df$FBPopMean
df$FBPopMean<-NULL
df$Title<-NULL
df$Headline<-NULL
df$Source<-NULL
df$Topic<-NULL


#################################
hist(df$Facebook, breaks=5)
df$Facebook[df$Facebook>=-1 & df$Facebook<=0]<-0
df$Facebook[df$Facebook>0 & df$Facebook<=33]<-1
df$Facebook[df$Facebook>33]<-2
#df$Facebook[df$Facebook>175]<-3
#df$Facebook[df$Facebook>0 & df$Facebook<=10]<-1
#df$Facebook[df$Facebook>10 & df$Facebook<=54]<-2
#df$Facebook[df$Facebook>54 & df$Facebook<=92]<-3
#df$Facebook[df$Facebook>92 & df$Facebook<=175]<-4
#df$Facebook[df$Facebook>175 & df$Facebook<=452]<-5
#df$Facebook[df$Facebook>452]<-6
write.csv(df,"/home/ashraydimri/Personal/News_Paper/facebookcatv1.1.csv",row.names = F)
####################################### XGBOOST

library(data.table)
library(mlr)

dft<-as.data.table(df)
set.seed(101)
sample <- sample.split(df$GooglePlus , SplitRatio = 0.75)
sample_train <- subset(dft , sample == TRUE)
sample_test <- subset(dft , sample == FALSE)

setDT(sample_train)
setDT(sample_test)

lapply(sample_train, class)
library(xgboost)

labels<-sample_train$GooglePlus
ts_labels<-sample_test$GooglePlus


dtrain <- xgb.DMatrix(data = as.matrix(sample_train[,-1]), label= labels)
dtest <- xgb.DMatrix(data =  as.matrix(sample_test[,-1]), label= ts_labels)

params <- list(
  booster="gbtree", objective="multi:softprob", eval_metric="mlogloss",  num_class=2, eta = .02, gamma = 1, max_depth = 4, min_child_weight = 1, subsample = 0.7, colsample_bytree = 0.5
)


xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 100
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print.every.n = 10
                ,early.stop.round = 20
                ,maximize = F
)


xgb2 <- xgb.train(data = dtrain, params = params,watchlist = list(val=dtest,train=dtrain), nrounds = 1000, print_every_n = 10,maximize = F)
xgbpred <- predict(xgb2,dtest)
probs <- t(matrix(xgbpred, nrow=2, ncol=length(xgbpred)/2))
colnames(probs)<-seq(0,1,1)
res<-colnames(probs)[max.col(probs,ties.method="last")]
res<-as.numeric(res)


library(caret)
confusionMatrix(res, ts_labels)

impmat<-xgb.importance(colnames(dft[,-1]),model=xgb2)

impmat<-xgb.importance(c("HeadlineSentiment", "TitleSentiment"  ,  "TimeOfDay"        , "DayOfWeek"   , "TopicLabels" ),model=xgb2)
xgb.plot.importance(impmat)


#######################################

install.packages("caTools")
install.packages("MASS")
install.packages("lars")
install.packages("neuralnet")
install.packages("NeuralNetTools")

library("caTools")
library("MASS")


print("Dividing the indivisual group datasets into training and testing")

set.seed(101)
sample <- sample.split(df$Facebook , SplitRatio = 0.75)


sample_train <- subset(df , sample == TRUE)
sample_test <- subset(df , sample == FALSE)

print("Train")
print(summary(sample_train))
print("Test ")
print(summary(sample_test))

print("Performing Ridge Regression on 1")
#####################################################
rsquare <- function(true, predicted) {
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- 1 - sse / sst
  
  # For this post, impose floor...
  if (rsq < 0) rsq <- 0
  
  return (rsq)
}


getridge<-function(x_train_out,y_train_in,x_test_out,y_test_in)
{
  #y<-x_train_out
  #x<-as.matrix(y_train_in)
  
  y<-sample_train$LinkedIn
  x<-as.matrix(sample_train[,-c(1,2,3)])
  lambda_vals <- 10^seq(3, -2, by = -.1)  # Lambda values to search
  cv_glm_fit  <- cv.glmnet(x,y , alpha = 0, lambda = lambda_vals, nfolds = 10)
  opt_lambda  <- cv_glm_fit$lambda.min  # Optimal Lambda
  glm_fit     <- cv_glm_fit$glmnet.fit
  
  
  #newxmat<-as.matrix(y_test_in)
  newxmat<-as.matrix(sample_test[,-c(1,2,3)])
  glm_test_yhat <- predict(glm_fit, s = opt_lambda, newx = newxmat)
  glm_test_rsq  <- rsquare(x_test_out, glm_test_yhat)
  glm_test_rsq
}



getridge(sample_train$GooglePlus,sample_train[,-c(1,2,3)],sample_test$GooglePlus,sample_test[,-c(1,2,3)])
getridge(sample_train$Facebook,sample_train[,-c(1,2,3)],sample_test$Facebook,sample_test[,-c(1,2,3)])
getridge(sample_train$LinkedIn,sample_train[,-c(1,2,3)],sample_test$LinkedIn,sample_test[,-c(1,2,3)])
#####################################################

google_ridge <- lm.ridge(GooglePlus ~ . - GooglePlus - Facebook - LinkedIn, lambda = seq(0 , 10 , 0.001) , data = sample_train)

google_ridge_predict<-scale(sample_test[,-c(1,2,3)],center = F,scale = google_ridge$scales)%*%google_ridge$coef[,which.min(google_ridge$GCV)] + google_ridge$ym

group_1_ridge_predict <- scale(group_1_test[, !(names(group_2_test) %in% c("GROUP" , "GENSINI" , "SYNTAX"))] , center = F , scale = group_1_ridge$scales)%*%group_1_ridge$coef[,which.min(group_1_ridge$GCV)] + group_1_ridge$ym
print("RIDGE-REGRESSION RMSE VALUES - GROUP 1")
print(summary(sqrt((google_ridge_predict - sample_test$GooglePlus)^2)))



facebook_ridge <- lm.ridge(Facebook ~ . - GooglePlus - Facebook - LinkedIn , lambda = seq(0 , 10 , 0.001) , data = sample_train)
linkedin_ridge <- lm.ridge(LinkedIn ~ . - GooglePlus - Facebook - LinkedIn, lambda = seq(0 , 10 , 0.001) , data = sample_train)

#################################
library(tidyverse)
library(broom)
library(glmnet)


data(mtcars)
y <- mtcars$hp
x <- mtcars %>% select(mpg, wt, drat) %>% data.matrix()
lambdas <- 10^seq(3, -2, by = -.1)

fit <- glmnet(x, y, alpha = 0, lambda = lambdas)
summary(fit)


lambda_vals <- 10^seq(3, -2, by = -.1)  # Lambda values to search
cv_glm_fit  <- cv.glmnet(as.matrix(train[,-1]), train$y, alpha = 0, lambda = lambda_vals, nfolds = 5)
opt_lambda  <- cv_glm_fit$lambda.min  # Optimal Lambda
glm_fit     <- cv_glm_fit$glmnet.fit