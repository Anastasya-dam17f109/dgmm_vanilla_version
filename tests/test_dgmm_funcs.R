
library(deepgmm)
library(T4cluster)
source("C:/Users/ADostovalova/Desktop/work/deepgmm/tests/dgmm_funcs.R")

args = commandArgs(trailingOnly=TRUE)
file<-"hhh"
if (length(args)>0){ file<-args[1]
}else{file<- "C:\\Users\\ADostovalova\\PycharmProjects\\test2\\train_data.csv"}

print(file)
data3 <-  read.csv2(file)
print(data3)
res<-first_stats(data3,1)
print("result")
print(res)

layers <- 2
k <- c(5,4)
r <-c(3,2)
it <- 1450
eps <- 0.001
seed <- 1
init <- "random"
set.seed(seed)


arr<-matrix(as.numeric(unlist(data3)), ncol=length(data3))#, byrow=TRUE)

#model <- deepgmm::deepgmm(y = arr, layers = layers, k = k, r = r,
                       #   it = it, eps = eps, init = init,  scale = FALSE)

res<-dgmm_grid_search(arr, 4)
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
print(k)
print(r)
model <- deepgmm::deepgmm(y = arr, layers = layers, k = k, r = r,seed=1)
                          #   it = it, eps = eps, init = init,  scale = FALSE)
write.csv2(model$s, "C:/Users/ADostovalova/Desktop/work/deepgmm/my_data.csv")
