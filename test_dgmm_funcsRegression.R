
library(deepgmm)
library(T4cluster)
library(caTools)
library(caret)
library(Thresher)
source("D:/dgmm_vanilla_version/dgmm_funcs.R")

args = commandArgs(trailingOnly=TRUE)
file<-"hhh"
file_test<-"D:/dgmm_vanilla_version/test_data.csv"
result_train<-"D:/dgmm_vanilla_version/result_train.csv"
result_test<-"D:/dgmm_vanilla_version/result_test.csv"
label_train<-"D:/dgmm_vanilla_version/train_label.csv"
label_test<-"D:/dgmm_vanilla_version/test_label.csv"
classes =5
codenum = "45"
if (length(args)>0){
file<-args[1]
file_test<-args[2]
label_train<-args[3]
label_test<-args[4]
classes <-as.integer(args[5])
codenum<- args[6]
}else{file<- "D:/dgmm_vanilla_version/train_data.csv"}

result_filename =  paste("D:/dgmm_vanilla_version/results_self_regr/output_", codenum,".txt")
sink(result_filename )
print(file)
print(classes)
data3 <-  read.csv2(file)
data4 <-  read.csv2(file_test)
res3 <-  read.csv2(label_train)
res4 <-  read.csv2(label_test)

regr_idx<-as.integer(classes/2)
layers <- 2
k <- c(5,4)
r <-c(3,2)
it <- 200 
eps<-0.001
init <- 'random'
init_est <- 'factanals'
seed <- NULL
set.seed(seed)

arr<-matrix(as.numeric(unlist(data3)), ncol=length(data3))
arr2<-matrix(as.numeric(unlist(data4)), ncol=length(data4))

nans <- which(is.na(arr),arr.ind=TRUE)
nans <-nans[,1]

if(length(nans) >0){
arr<-arr[-nans,]
arr<-as.matrix(arr)
}

res3<-matrix(as.numeric(unlist(res3)), ncol=length(res3))
if(length(nans) >0){
  res3<-res3[-nans,]
  res3<-as.matrix(res3)
}

nans <- which(is.na(res3),arr.ind=TRUE)
nans <-nans[,1]

if(length(nans) >0){
  res3<-res3[-nans,]
  res3<-as.matrix(res3)
  arr<-arr[-nans,]
  arr<-as.matrix(arr)
}


nans <- which(is.na(arr2),arr.ind=TRUE)
nans <-nans[,1]

if(length(nans) >0){
  arr2<-arr2[-nans,]
  arr2<-as.matrix(arr2)
}

res4<-matrix(as.numeric(unlist(res4)), ncol=length(res4))
if(length(nans) >0){
  res4<-res4[-nans,]
  res4<-as.matrix(res4)
}

nans <- which(is.na(res4),arr.ind=TRUE)
nans <-nans[,1]

if(length(nans) >0){
  res4<-res4[-nans,]
  res4<-as.matrix(res4)
  arr2<-arr2[-nans,]
  arr2<-as.matrix(arr2)
}

print("REGRESSION")
print("==DGMM==")
full_dgmm_res <-dgmm_labels_full(cbind(arr,res3),cbind(arr,res3), classes, it, eps, init, init_est, seed)
gmm_k=0

print("============L1============")
tain_cl_dgmm<-full_dgmm_res$l1

if(!is.numeric(full_dgmm_res$l1)){
  
reg<-SelfSupRegrDGMM(arr, arr2,  res3,  type1 = 'DGMMr', k=full_dgmm_res$k1, r=full_dgmm_res$r1, init=init, it=it)
  
print("RMSE  pure dgmm")
y_tr=reg$y_tr
gmm_k=full_dgmm_res$k1
res3e=res3
if(length(reg$nans) >0){
  res3e<-res3e[-reg$nans,]
  
}
esimates = estimateRMSE(y_tr,res3e)

print("RMSE  dgmm+svm")
classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='SVM')
esimates = estimateRMSE(classifRes, res4)

print("RMSE  dgmm+xgb")

classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='xgb')
esimates = estimateRMSE(classifRes, res4)
#print(reg$Y)
}
print("============L2============")
tain_cl_dgmm<-full_dgmm_res$l2

if(!is.numeric(full_dgmm_res$l2)){
  reg<-SelfSupRegrDGMM(arr, arr2,  res3,  type1 = 'DGMMr', k=full_dgmm_res$k2, r=full_dgmm_res$r2, init=init, it=it)
  gmm_k=full_dgmm_res$k2
    print("RMSE  pure dgmm")
  res3e=res3
    if(length(reg$nans) >0){
      res3e<-res3e[-reg$nans,]
      
    }  
  esimates = estimateRMSE(reg$y_tr,res3e)
  
  print("RMSE  dgmm+svm")
  classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='SVM')
  esimates = estimateRMSE(classifRes, res4)
  
  print("RMSE  dgmm+xgb")
  
  classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='xgb')
  esimates = estimateRMSE(classifRes, res4)
}

print("============L3============")
tain_cl_dgmm<-full_dgmm_res$l3

if(!is.numeric (tain_cl_dgmm)){
  reg<-SelfSupRegrDGMM(arr, arr2,  res3,  type1 = 'DGMMr', k=full_dgmm_res$k3, r=full_dgmm_res$r3, init=init, it=it)
  gmm_k=full_dgmm_res$k3 
    print("RMSE  pure dgmm")
    res3e=res3
    if(length(reg$nans) >0){
      res3e<-res3e[-reg$nans,]
      
    }  
  esimates = estimateRMSE(reg$y_tr,res3e)
  
  print("RMSE  dgmm+svm")
  classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='SVM')
  esimates = estimateRMSE(classifRes, res4)
  
  print("RMSE  dgmm+xgb")
  
  classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='xgb')
  esimates = estimateRMSE(classifRes, res4)
}

gmm_k = classes

print("==GMM==")

reg<-SelfSupRegrDGMM(arr, arr2,  res3,  type1 = 'GMMr', k=gmm_k, r=full_dgmm_res$r3, init=init, it=it)
res3e=res3
if(length(reg$nans) >0){
  res3e<-res3[-reg$nans,]
  
}
print("RMSE  pure gmm")
esimates = estimateRMSE(reg$y_tr,res3e)

print("RMSE  gmm+svm")
classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='SVM')
esimates = estimateRMSE(classifRes, res4)

print("RMSE  gmm+xgb")

classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='xgb')
esimates = estimateRMSE(classifRes, res4)




 
 print("pure linear regression:")
 reg<-SelfSupRegrDGMM(arr, arr2,  res3,  type1 = 'linr', k=gmm_k, r=full_dgmm_res$r3, init=init, it=it)
 
 print("RMSE  pure lin")
 esimates = estimateRMSE(reg$y_tr,res3)
 
 print("RMSE  lin+svm")
 classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='SVM')
 esimates = estimateRMSE(classifRes, res4)
 
 print("RMSE  lin+xgb")
 
 classifRes <-regressionSVM_XGB(reg$X, reg$Y, arr2, type2 ='xgb')
 esimates = estimateRMSE(classifRes, res4)
sink()





