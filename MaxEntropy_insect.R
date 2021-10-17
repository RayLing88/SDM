################################################## Part1 Max Entropy ###################################################
#' @title Building max entropy model
#' @author Lei Ling
#' @keywords Max Entropy
#' @import maxnet
#' @import glmnet
#' @import reportROC
#' @email linglei4368@gmail.com


# Feature Matrix Preparation ----------------------------------------------
setwd('G:/Master/Research/Ninghang/DataAnalysis/insect_data/')

Positive_sample <- read.table('insect_positive.csv',header = T,
                            as.is = T,sep = ',')
Negative_sample <- read.csv('insect_negative.csv',header = T,
                            as.is = T,sep = ',')

Negative_sample$BIO13 <- as.numeric(Negative_sample$BIO13)
na_index <- which(is.na(Negative_sample$BIO13))
Negative_sample <- Negative_sample[-na_index,]


### assign label 
#### for 1 positive sample and 0 for negative sample
label <- c(rep(1,nrow(Positive_sample)),
           rep(0,nrow(Negative_sample)))
FeatureMatrix <- rbind(Positive_sample,Negative_sample)
#FeatureMatrix <- cbind(FeatureMatrix,label)


# using LASSO regression to select non-linear feature --------------------------------

fit1 <- glmnet::cv.glmnet(y = label,x = as.matrix(FeatureMatrix),
                  family = "binomial",alpha = 1)
plot(fit1)

fit1$lambda.1se
fit1_coef <- coef(fit1,s = fit1$lambda.1se)
#sort(abs(fit1_coef),decreasing = T)
tmp = fit1_coef[order(abs(fit1_coef),decreasing = T),]
# (Intercept)         BIO2         BIO8        BIO15         BIO5 
# 30.471368041 -0.505529274  0.381271111 -0.371519576 -0.350310321 
# BIO1        BIO14        BIO19        BIO13        BIO16 
# 0.224479517 -0.145047458 -0.123383975  0.082632764 -0.024808807 
# BIO20        BIO18         BIO3         BIO4         BIO6 
# 0.002895312 -0.002697076  0.000000000  0.000000000  0.000000000 
# BIO7         BIO9        BIO10        BIO11        BIO12 
# 0.000000000  0.000000000  0.000000000  0.000000000  0.000000000 
# BIO17 
# 0.000000000 
#### top 5 fea based on LASSO regression coefficients
top_five_fea = c('BIO2','BIO8','BIO15','BIO5','BIO1')
glmnet_coef <- data.frame('top_five_fea' = top_five_fea,
                          'coef' = tmp[2:6])
which(fit1_coef[,1] !=0)

### select 11 features coef !=0
select_feat_index <- which(fit1_coef[,1] !=0) -1
select_feat_index <- select_feat_index[-1]
select_feat_name <- colnames(FeatureMatrix)[select_feat_index]
sub_FM <- FeatureMatrix[,select_feat_index]


library(reportROC)

library(plyr)
CVgroup <- function(k,datasize,seed){
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    
  temp <- sample(n,datasize)  
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x,function(x) dataseq[temp==x])  
  return(cvlist)
}

DoMaxEntropy <- function(y = y,X = X,k = 10,
                         seed = 123,classes = "logistic",regmult = 1){
  tmp <- CVgroup(k = k,datasize = nrow(X),seed = seed)
  result = c()
  if (!is.data.frame(X)){
    X = data.frame(X)
  }
  for(i in 1:k){
    X_train <- X[-tmp[[i]],]
    X_test <- X[tmp[[i]],]
    y_train <- y[-tmp[[i]]]
    y_test <- y[tmp[[i]]]
    ### require maxnet package
    if (!require(maxnet)){
      install.packages('maxnet')
    }
    mod <- maxnet(y_train, X_train,
                  maxnet.formula(y_train,X_train,
                                 classes = classes ),regmult = regmult)
    y_pre_train = predict(mod,newdata = X_train)
    AUC_train <- reportROC::reportROC(gold = y_train,predictor = y_pre_train,
                                      plot = F)[1,2]
    AUC_train <- as.numeric(AUC_train)
    #result[i,1] = AUC_train
    y_pre_test <- predict(mod,newdata = X_test)
    AUC_test <- reportROC::reportROC(gold = y_test,predictor = y_pre_test,
                                     plot = F)[1,2]
    AUC_test <- as.numeric(AUC_test)
    #result[i,2] = AUC_test
    result <- rbind(result,c(AUC_train,AUC_test))
    
    
  }
  return(result)
}

t1 <- Sys.time()
regmult = seq(0.5,4,0.5)
Result_all_fea <- lapply(1:length(regmult),function(idx){DoMaxEntropy(
   y = label,X = FeatureMatrix,
   k = 10,seed = 123,regmult = regmult[idx]
 )})
Result_all_fea = do.call(cbind,Result_all_fea)

t2 <- Sys.time()
print(t2 - t1)
colnames(Result_all_fea) = rep(paste0("Regmult=",regmult),each = 2)
### boxplot to visualize Cross validation result
library(RColorBrewer)
display.brewer.all(type="qual")
col = rep(brewer.pal(4,name = "Set2"),each = 2)
#boxplot(Result_all_fea,col = col,ylab = "AUC in 10 fold cross validation")

save(list = ls(),file = "01_MaxEntropy_Result.Rdata")

#colnames(Result_PC) = rep(type,each = 2)
#boxplot(Result_PC,col = col,ylab = "AUC in 10 fold cross validation")

### run with 11 non-zero features
t1 <- Sys.time()
seed = c(1,12,123,21,22,23,3,43,44,45)
Result_11_feat <- lapply(1:length(seed),function(idx){DoMaxEntropy(
  y = label,
  X = sub_FM,classes = 'exponential',
  k = 10,seed = seed[idx]
)})
Result_11_feat = do.call(cbind,Result_11_feat)
t2 <- Sys.time()
print(t2 - t1)

#### run with top five features
t1 <- Sys.time()
regmult = 1
Result_top_feat <- lapply(1:length(regmult),function(idx){DoMaxEntropy(
  y = label,
  X = FeatureMatrix[,top_five_fea],classes = 'exponential',
  k = 10,seed = 123,regmult = regmult[idx]
)})
Result_top_feat = do.call(cbind,Result_top_feat)
t2 <- Sys.time()
print(t2 - t1)

mod_five <- maxnet(label,FeatureMatrix[,top_five_fea],
                   classes = 'logistic')
library(maxnet)
plot(mod_five,type = 'logistic')






