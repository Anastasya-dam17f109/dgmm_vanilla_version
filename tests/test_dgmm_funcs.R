
library(deepgmm)
library(T4cluster)
source("C:/Users/ADostovalova/Desktop/work/deepgmm/tests/dgmm_funcs.R")

args = commandArgs(trailingOnly=TRUE)
file<-"hhh"
if (length(args)>0){ file<-args[1]
}else{file<- "C:/Users/ADostovalova/Desktop/work/deepgmm/my_data_smiley.csv"}

s10 = T4cluster::genSMILEY(200, sd=0.35)
write.csv2(s10$data, file)
data3 <-  read.csv2(file)
res<-first_stats(data3,1)
print("result")
print(res)

data3 <-  read.csv2(args[1])
#s25 = genSMILEY(200, sd=0.25)
#s50 = genSMILEY(200, sd=0.5)

z <-rep(1.3,100)
w <-rep(1.3,100)
#cat(data3[400])
#cat(data3)

## Visualize

y2 <- mtcars
#cat(s10$data)
layers <- 2
k <- c(4,2)
r <-c(2,1)
it <- 1450
eps <- 0.001
seed <- 1
init <- "random"
cat(length(k))
#print(y)
set.seed(seed)

#3d for sata or more
arr <- cbind(data3, c(z,w))
arr1 <- cbind(s10$data, c(z,w))

#print(arr)
#print(arr1)
#arr<- slice(arr, 1:3)
arr<-matrix(as.numeric(unlist(arr)), ncol=length(arr))#, byrow=TRUE)
#arr<-as.numeric(unlist(arr))
#getAnywhere(deepgmm)
model <- deepgmm::deepgmm(y = arr[1:200,2:4], layers = layers, k = k, r = r,
                          it = it, eps = eps, init = init,  scale = FALSE)


#cat(model$s)
#write.table(model$s, file = "C:/Users/ADostovalova/Desktop/work/deepgmmmy_data.txt", sep = ";")
write.csv2(model$s, "C:/Users/ADostovalova/Desktop/work/deepgmm/my_data.csv")
#write.table(model$s, file = "my_data.txt", sep = ";")


opar <- par(no.readonly=TRUE)
par(mfrow=c(1,3), pty="s")
plot(arr[1:200,2:3], col=s10$label, pch=19, main="sd=0.10")
plot(arr[1:200,2:3], col=model$s, pch=19, main="sd=0.25 dgmm")