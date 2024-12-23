
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
mainDir = "D:/dgmm_vanilla_version/regr_data_outsource"
classes =3
codenum = "53"
delta=0.7
if (length(args)>0){
file<-args[1]
file_test<-args[2]
label_train<-args[3]
label_test<-args[4]
classes <-as.integer(args[5])
codenum<- args[6]
delta<- as.numeric(args[7])
mainDir <- args[8]
}else{file<- "D:/dgmm_vanilla_version/train_data.csv"}

print(mainDir)
print(delta)
subDir = paste(mainDir,"/",codenum,"/", delta, sep = "")
ifelse(!dir.exists(file.path(mainDir, paste(mainDir,"/",codenum, sep = ""))), dir.create(file.path(mainDir, paste(codenum))), FALSE)
ifelse(!dir.exists(file.path(mainDir, subDir)), dir.create(file.path(mainDir, paste(codenum,"/", delta, sep = ""))), FALSE)
#"D:\dgmm_vanilla_version\regr_data_outsource\109"

result_filename =  paste(subDir,"/output_log_",paste(codenum),"_",delta,".txt",sep = "")
print(result_filename)
sink(result_filename )
print(file)
print(classes)

layers <- 2
gmm_k=0
k <- c(5,4)
r <-c(3,2)
it <- 200 
eps<-0.001
init_mass = c('kmeans', 'hclass', 'mclust', 'random')
init_est <- 'factanals'
seed <- NULL
set.seed(seed)

data = semi_regression_preparation(file, file_test, label_train, label_test, delta)

arr = data[0]
arr2 = data[1]
res3 = data[2]
res4 = data[3]
arr_semi = data[4]
res3_semi_tail = data[5]
arr_semi_tail = data[6]
data3 <-  read.csv2(file)
data4 <-  read.csv2(file_test)
res3 <-  read.csv2(label_train)
res4 <-  read.csv2(label_test)
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

arr_semi=arr[1:as.integer(delta*nrow(arr)),]
beg =as.integer(delta*nrow(arr))+1
res3_semi_tail=res3[beg:nrow(arr),]
arr_semi_tail=arr[beg:nrow(arr),]
res3=res3[1:as.integer(delta*nrow(arr)),]
res3_semi_tail=matrix(as.numeric(unlist(res3_semi_tail)), nrow=length(res3_semi_tail ))


write.csv2(arr2, paste(subDir, "/train_data_gmm_", codenum,".csv",sep = ""))
write.csv2(res4, paste(subDir, "/train_target_gmm_", codenum,".csv",sep = ""))

print("REGRESSION")

gmm_k = classes

print("==GMM==")
print(arr_semi)
print(arr2)
print(res3)
reg<-SelfSupRegrDGMM(arr_semi, arr2,  res3,  type1 = 'GMMr', k=gmm_k, r=gmm_k, init=init, it=it)
res3e=res3
if(length(reg$nans) >0){
  res3e<-res3e[-reg$nans]
}
print("RMSE  pure gmm")
esimates = estimateRMSE(reg$y_tr,res3e)

write.csv2(rbind(reg$X, arr_semi_tail), paste(subDir,"/train_data_gmm_",delta,"_", codenum,".csv",sep = ""))
write.csv2(rbind(reg$Y, res3_semi_tail), paste(subDir,"/train_target_gmm_",delta,"_", codenum,".csv",sep = ""))

 print("pure linear regression:")
 reg<-SelfSupRegrDGMM(arr_semi, arr2,  res3,  type1 = 'linr', k=gmm_k, r=full_dgmm_res$r3, init=init, it=it)
 
 print("RMSE  pure lin")
 esimates = estimateRMSE(reg$y_tr,res3)
 
 
 write.csv2(rbind(reg$X, arr_semi_tail), paste(subDir,"/train_data_lin_",delta,"_", codenum,".csv",sep = ""))
 write.csv2(rbind(matrix(as.numeric(unlist(reg$Y)), nrow=length(reg$Y)), res3_semi_tail), paste(subDir,"/train_target_lin_",delta,"_", codenum,".csv",sep = ""))
 
 print("==DGMM==")
 for (init in init_mass) {
   
   result <- tryCatch({ 
     full_dgmm_res <-dgmm_labels_full(cbind(arr_semi,res3),cbind(arr_semi,res3), classes, it, eps, init, init_est, seed)
 
      k_mass = c(full_dgmm_res$k1, full_dgmm_res$k2, full_dgmm_res$k3)
      r_mass = c(full_dgmm_res$r1, full_dgmm_res$r2, full_dgmm_res$r3)
      l_mass = c(full_dgmm_res$l1, full_dgmm_res$l2, full_dgmm_res$l3)
 
 
      for (i in 1:3) {
        print(paste("============L",i,"============"))
        tain_cl_dgmm<-l_mass[i]
   
        if(!is.numeric(l_mass[i])){
     
          reg<-SelfSupRegrDGMM(arr_semi, arr2,  res3,  type1 = 'DGMMr', k=k_mass[i], r=r_mass[i], init=init, it=it)
          y_tr=reg$y_tr
          res3e=res3
          if(length(reg$nans) >0){
            res3e<-res3e[-reg$nans]
            reg$Y<-matrix(as.numeric(unlist(reg$Y)), ncol=length(1))
          }
          print("RMSE  pure dgmm")
          esimates = estimateRMSE(y_tr,res3e)
          print(esimates)
          
          write.csv2(rbind(reg$X, arr_semi_tail), paste(subDir, "/train_data_dgmm_",init, "_",delta,"_", codenum,".csv",sep = ""))
          write.csv2(rbind(reg$Y, res3_semi_tail), paste(subDir, "/train_target_dgmm_", init, "_", delta,"_", codenum,".csv",sep = ""))
          
        }
      }
      
      
      
   }, error = function(err) {
     # error handler picks up where error was generated
     print(paste("MY_ERROR:  ",err))})
 } 
sink()







