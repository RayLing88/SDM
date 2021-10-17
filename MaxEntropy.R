################################################## Part1 Max Entropy ###################################################
#' @title Building max entropy model
#' @author Lei Ling
#' @keywords Max Entropy
#' @import maxnet
#' @import glmnet
#' @import reportROC
#' @email linglei4368@gmail.com


# Feature Matrix Preparation ----------------------------------------------
setwd('G:/Master/Research/Ninghang/DataAnalysis')

Positive_sample <- read.table('Positive_sample.csv',header = T,
                            as.is = T,sep = ',')
Negative_sample <- read.csv('negative_sample.csv',header = T,
                            as.is = T,sep = ',')
Negative_sample$Precipitation.of.Wettest.Month <- as.numeric(Negative_sample$Precipitation.of.Wettest.Month)
NA_index <- which(is.na(Negative_sample$Precipitation.of.Wettest.Month))
Negative_sample <- Negative_sample[-NA_index,]
Negative_sample <- Negative_sample[-c(100:150),]

### assign label 
#### for 1 positive sample and 0 for negative sample
label <- c(rep(1,nrow(Positive_sample)),
           rep(0,nrow(Negative_sample)))
FeatureMatrix <- rbind(Positive_sample,Negative_sample)
#FeatureMatrix <- cbind(FeatureMatrix,label)
#### Finish FeatureMatrix construting
### do bartlett test to check co-linearlity of 20 features

# library(psych)
# bartlett_test <- psych::cortest.bartlett(cor(FeatureMatrix),
#                                          n = nrow(FeatureMatrix) )
# ### p.value  < 2e-16
# # $chisq
# # [1] 83488.04 # $p.value 0 $df 190
# psych::KMO(cor(FeatureMatrix))
#### K = 0.91
# Kaiser-Meyer-Olkin factor adequacy
# Call: psych::KMO(r = cor(FeatureMatrix))
# Overall MSA =  0.91

# using LASSO regression to select non-linear feature --------------------------------

fit1 <- glmnet::cv.glmnet(y = label,x = as.matrix(FeatureMatrix),
                  family = "binomial",alpha = 1)
plot(fit1)

fit1$lambda.1se
fit1_coef <- coef(fit1,s = fit1$lambda.1se)
#sort(abs(fit1_coef),decreasing = T)
tmp = fit1_coef[order(abs(fit1_coef),decreasing = T),]
#### top 5 fea
top_five_fea = c('Precipitation.Seasonality',
                 'Precipitation.of.Driest.Month',
                 'Max.Temperature.of.Warmest.Month',
                 'Mean.Monthly.Temperature.Range',
                 'Min.Temperature.of.Coldest.Month')
glmnet_coef <- data.frame('top_five_fea' = top_five_fea,
                          'coef' = tmp[2:6])
which(fit1_coef[,1] !=0)
### select 11 features coef !=0
select_feat_index <- which(fit1_coef[,1] !=0) -1
select_feat_index <- select_feat[-1]
select_feat_name <- colnames(FeatureMatrix)[select_feat]
sub_FM <- FeatureMatrix[,select_feat_index]


mod <- maxnet(label, FeatureMatrix,
              maxnet.formula(label,FeatureMatrix,classes = "link"))
y_pre <- predict(mod,newdata = FeatureMatrix)
library(reportROC)
q = reportROC::reportROC(gold = label,predictor = y_pre,plot = F)

library(plyr)
CVgroup <- function(k,datasize,seed){
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    #将数据分成K份，并生成的完成数据集n
  temp <- sample(n,datasize)   #把n打乱
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x,function(x) dataseq[temp==x])  #dataseq中随机生成k个随机有序数据列
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
#colnames(Result_all_fea) = rep(type,each = 2)

#test1 <- DoMaxEntropy(y = label,X = FeatureMatrix,
                     # k = 10,seed = 123,classes = "link")
t2 <- Sys.time()
print(t2 - t1)
colnames(Result_all_fea) = rep(paste0("Regmult=",regmult),each = 2)
### boxplot to visualize Cross validation result
library(RColorBrewer)
display.brewer.all(type="qual")
col = rep(brewer.pal(4,name = "Set2"),each = 2)
#boxplot(Result_all_fea,col = col,ylab = "AUC in 10 fold cross validation")

save(list = ls(),file = "01_MaxEntropy_Result.Rdata")

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
seed = c(1,12,123,21,22,23,3,43,44,45)
Result_top_feat <- lapply(1:length(seed),function(idx){DoMaxEntropy(
  y = label,
  X = FeatureMatrix[,top_five_fea],classes = 'exponential',
  k = 10,seed = seed[idx]
)})
Result_top_feat = do.call(cbind,Result_top_feat)
t2 <- Sys.time()
print(t2 - t1)

mod_five <- maxnet(label,FeatureMatrix[,top_five_fea],
                   classes = 'logistic')
library(maxnet)
plot(mod_five,type = 'logistic')






