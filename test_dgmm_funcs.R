
library(deepgmm)
library(T4cluster)
source("C:/Users/ADostovalova/Desktop/work/deepgmm/tests/dgmm_funcs.R")

args = commandArgs(trailingOnly=TRUE)
file<-"hhh"
file_test<-"hhh"
result_train<-"hhh"
result_test<-"hhh"
classes =0
if (length(args)>0){
file<-args[1]
file_test<-args[2]
result_train<-args[3]
result_test<-args[4]
classes <-as.integer(args[5])
}else{file<- "C:\\Users\\ADostovalova\\PycharmProjects\\test2\\train_data.csv"}

print(file)
print(classes)
data3 <-  read.csv2(file)
data4 <-  read.csv2(file_test)
res<-first_stats(data3,1)


layers <- 2
k <- c(5,4)
r <-c(3,2)
it <- 250 
eps<-0.001
init <- 'kmeans'
init_est <- 'factanal'
seed <- 1
set.seed(seed)


arr<-matrix(as.numeric(unlist(data3)), ncol=length(data3))
arr2<-matrix(as.numeric(unlist(data4)), ncol=length(data4))


res<-dgmm_grid_search(arr, classes, it, eps, init, init_est, seed)
bics<-c(as.numeric(res$bic[1]),as.numeric(res$bic[3]), as.numeric(res$bic[5]))

layers<-which(bics %in% min(unlist(bics)))#as.integer(which.min(unlist(bics)))
print(res$bic[2*layers])
params<-sapply(strsplit((res$bic[2*layers]), split=", "), function(x) (as.numeric(x))) 

print(params)
if (layers == 1)
{ 
k <- c(params[2])
r <-c(params[1])
}
if (layers==2)
{ 
k <- c(params[3],params[4])
r <-c(params[1], params[2])
}
if (layers==3)
{ k <- c(params[4],params[5], params[6])
r <-c(params[1], params[2],params[3])
}


params<- dgmm_param_extraction (arr, layers, k, r,it,eps,seed,init, init_est, classes)
print(params)
res <-dgmm_inference (arr, params[[4]], params[[1]],params[[2]],params[[3]], classes)
res2 <-dgmm_inference (arr2,params[[4]], params[[1]],params[[2]],params[[3]],classes)

#"C:/Users/ADostovalova/Desktop/work/deepgmm/my_data.csv"
write.csv2(convert_to_labels(res), result_train)
write.csv2(convert_to_labels(res2), result_test)
