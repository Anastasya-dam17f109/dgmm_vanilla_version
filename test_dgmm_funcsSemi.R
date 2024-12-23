
library(deepgmm)
library(T4cluster)
library(caTools)
library(caret)
library(Thresher)
#library(OneR)
#library(pROC)
#library(wireless)
source("D:/dgmm_vanilla_version/dgmm_funcs.R")
delta=0.7
args = commandArgs(trailingOnly=TRUE)
file<-"hhh"
file_test<-"D:/dgmm_vanilla_version/test_data.csv"
result_train<-"D:/dgmm_vanilla_version/result_train.csv"
result_test<-"D:/dgmm_vanilla_version/result_test.csv"
label_train<-"D:/dgmm_vanilla_version/train_label.csv"
label_test<-"D:/dgmm_vanilla_version/test_label.csv"
classes =3
codenum = "109"
if (length(args)>0){
file<-args[1]
file_test<-args[2]
label_train<-args[3]
label_test<-args[4]
classes <-as.integer(args[5])
codenum<- args[6]
}else{file<- "D:/dgmm_vanilla_version/train_data.csv"}
result_filename =  paste("D:/dgmm_vanilla_version/results_class_semi/output", codenum,"_", toString(delta),".txt")
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
init <- 'mclust'
print(init)
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
arr_semi=arr[1:as.integer(delta*nrow(arr)),]
print("CLASSIFICATION")
print("==DGMM==")

# res<-first_stats(data3,1)
# types = list()
# if (as.numeric(length(res$transform_cols))>0){
#   types <- rep(list("num"),as.numeric(length(res$transform_cols)))
# }
# new_res<-prepare(data3, "none",res$transform_cols,types, norm='none')
# arr<-matrix(as.numeric(unlist(new_res)), ncol=length(data3))
# 
# 
# 
# res<-first_stats(data4,1)
# types = list()
# if (as.numeric(length(res$transform_cols))>0){
#   types <- rep(list("num"),as.numeric(length(res$transform_cols)))
# }
# new_res<-prepare(data4, "none",res$transform_cols,types,norm='none')
# arr2<-matrix(as.numeric(unlist(new_res)), ncol=length(data4))

#if(length(nans) >0){
res<-first_stats(res3,1)
#}
types = list()
if (as.numeric(length(res$transform_cols))>0){
  types <- rep(list("num"),as.numeric(length(res$transform_cols)))
}
new_res<-prepare(res3, "none",res$transform_cols,types,norm='none')
res3<-matrix(as.numeric(unlist(new_res)), ncol=length(new_res))
if(length(nans) >0){
  res3<-res3[-nans,]
  res3<-as.matrix(res3)
}
beg =as.integer(delta*nrow(arr))+1
res3_semi=res3[beg:nrow(arr),]

res3=res3[1:as.integer(delta*nrow(arr)),]
nans <- which(is.na(arr2),arr.ind=TRUE)
nans <-nans[,1]

if(length(nans) >0){
  arr2<-arr2[-nans,]
  arr2<-as.matrix(arr2)
}

res3_f = factor(res3, levels = seq(1, classes, by=1))

res<-first_stats(res4,1)
types = list()
if (as.numeric(length(res$transform_cols))>0){
  types <- rep(list("num"),as.numeric(length(res$transform_cols)))
}
new_res<-prepare(res4, "none",res$transform_cols,types,norm='none')
res4<-matrix(as.numeric(unlist(new_res)), ncol=length(new_res))

if(length(nans) >0){
  res4<-res4[-nans,]
  res4<-as.matrix(res4)
}
res4_f = factor(res4, levels = seq(1, classes, length.out=classes))
full_dgmm_res<-dgmm_labels_full(arr_semi,arr2, classes, it, eps, init, init_est, seed)
print("============L1============")
tain_cl_dgmm<-full_dgmm_res$l1
print("F1  pure dgmm")

clust2 = factor(matrix(as.numeric(tain_cl_dgmm$label), ncol=length(tain_cl_dgmm$label)),levels = seq(1, classes, by=1))
res3_semi=matrix(as.numeric(unlist(res3_semi)), nrow=length(res3_semi ))
R <- remap(res3_f,clust2)
p1 = as.numeric(R)
esimates = estimateF1 (as.factor(p1) , res3_f)
dgmm_label = matrix(as.numeric(unlist(p1 )), nrow=length(p1 ))
dgmm_label =rbind(dgmm_label, res3_semi)
print("F1  dgmm+svm")
classifRes <-SelfSupClass(arr, arr2,dgmm_label , res4_f,'svm',  TRUE,classes )
print("F1  dgmm+xgb")

classifRes <-SelfSupClass(arr, arr2, dgmm_label, res4_f,'xgb',  TRUE, classes)

print("============L2============")
tain_cl_dgmm<-full_dgmm_res$l2
print("F1  pure dgmm")
if(!is.numeric(full_dgmm_res$l2)){
clust2 = factor(matrix(as.numeric(tain_cl_dgmm$label), ncol=length(tain_cl_dgmm$label)),levels = seq(1, classes, by=1))

R <- remap(res3_f,clust2)
p1 = as.numeric(R)
esimates = estimateF1 (as.factor(p1) , res3_f)
dgmm_label = matrix(as.numeric(unlist(p1 )), nrow=length(p1 ))
dgmm_label =rbind(dgmm_label, res3_semi)
print("F1  dgmm+svm")
classifRes <-SelfSupClass(arr, arr2,dgmm_label , res4_f,'svm',  TRUE,classes )
print("F1  dgmm+xgb")

classifRes <-SelfSupClass(arr, arr2, dgmm_label, res4_f,'xgb',  TRUE, classes)
}

print("============L3============")
tain_cl_dgmm<-full_dgmm_res$l3
if(!is.numeric (tain_cl_dgmm)){
print("F1  pure dgmm")

clust2 = factor(matrix(as.numeric(tain_cl_dgmm$label), ncol=length(tain_cl_dgmm$label)),levels = seq(1, classes, by=1))

R <- remap(res3_f,clust2)
p1 = as.numeric(R)
esimates = estimateF1 (as.factor(p1) , res3_f)
dgmm_label = matrix(as.numeric(unlist(p1 )), nrow=length(p1 ))
dgmm_label =rbind(dgmm_label, res3_semi)
print("F1  dgmm+svm")
classifRes <-SelfSupClass(arr, arr2,dgmm_label , res4_f,'svm',  TRUE,classes )
print("F1  dgmm+xgb")

classifRes <-SelfSupClass(arr, arr2, dgmm_label, res4_f,'xgb',  TRUE, classes)
}



print("==GMM==")
tain_cl_gmm<-gmm_labels(arr_semi,arr2, classes, it, eps, init, init_est, seed)

print("F1 pure gmm")
clust2 = factor(matrix(as.numeric(tain_cl_gmm$label), ncol=length(tain_cl_gmm$label)),levels = seq(1, classes, by=1))
R <- remap(res3_f,clust2)
p1 = as.numeric(R)
esimates = estimateF1 (as.factor(p1) , res3_f)
gmm_label = matrix(as.numeric(unlist(p1 )), nrow=length(p1 ))
gmm_label =rbind(gmm_label, res3_semi)
print("F1  gmm+svm")
classifRes <-SelfSupClass(arr, arr2, gmm_label, res4_f,'svm',  TRUE,classes )
print("F1  gmm+xgb")
classifRes <-SelfSupClass(arr, arr2, gmm_label, res4_f,'xgb',  TRUE, classes)

print("==KNN==")
tain_cl<-knn_labels(arr_semi,arr2, classes, it, eps, init, init_est, seed)
knn_labels = tain_cl

print("F1 pure knn")


clust2 = factor(matrix(as.numeric(knn_labels), ncol=length(knn_labels)),levels = seq(1, classes, by=1))
R <- remap(res3_f,clust2)
p1 = as.numeric(R)
esimates = estimateF1 (as.factor(p1) , res3_f)
Knn_label =  matrix(as.numeric(unlist(p1 )), nrow=length(p1 ))
Knn_label =rbind(Knn_label, res3_semi)
print("ari  knn+svm")
classifRes <-SelfSupClass(arr, arr2, Knn_label, res4_f,'svm',  TRUE,classes )
print("ari  knn+xgb")
classifRes <-SelfSupClass(arr, arr2, Knn_label, res4_f,'xgb',  TRUE, classes)

# print("REGRESSION")
# 
# print("dgmm regression")
# dgmmr <- DGMMregr( arr,  arr, regr_idx, tain_cl_dgmm$k, tain_cl_dgmm$all_w, tain_cl_dgmm$all_mu, tain_cl_dgmm$all_var)
# print("pure dgmm regression:")
# test_rmse  <- rmse(arr[,regr_idx], dgmmr$regr)
# print(test_rmse)
# print("dgmm+svm regression:")
# regr_dgmm<-SelfSupRegrDGMM(arr, arr2, dgmmr,  regr_idx,  'SVM', TRUE)
# print("dgmm+xgb regression:")
# regr_dgmm<-SelfSupRegrDGMM(arr, arr2, dgmmr,  regr_idx,  'xgb', TRUE)
# 
# 
# dgmmr <- DGMMregr( arr,  arr, regr_idx, tain_cl_gmm$k, tain_cl_gmm$all_w, tain_cl_gmm$all_mu, tain_cl_gmm$all_var)
# print("pure gmm regression:")
# test_rmse  <- rmse(arr[,regr_idx], dgmmr$regr)
# print(test_rmse)
# print("gmm+svm regression:")
# regr_dgmm<-SelfSupRegrDGMM(arr, arr2, dgmmr,  regr_idx,  'SVM', TRUE)
# print("gmm+xgb regression:")
# regr_dgmm<-SelfSupRegrDGMM(arr, arr2, dgmmr,  regr_idx,  'xgb', TRUE)
# 
# print("pure linear regression:")
# regressor = lm(formula = arr[,regr_idx] ~ .,
#                data = as.data.frame(arr))
# 
# # Predicting the Test set results
# y_pred = predict(regressor, newdata = as.data.frame(arr2))
# test_rmse  <- rmse(arr[,regr_idx], y_pred)
# print(test_rmse)
#"C:/Users/ADostovalova/Desktop/work/deepgmm/my_data.csv"
#write.csv2(convert_to_labels(res), result_train)
#write.csv2(convert_to_labels(res2), result_test)
sink()



