library(tidyr)
library(deepgmm)
library(mclust)
library(xgboost)
library(corpcor)
library(pracma)
library(einsum)
library(kernlab)
library(Metrics)
library(plyr)
library(data.table)
library(caret)
# Loading package
library(ClusterR)
library(cluster)


# ----------------------------------------------------------------------------------

first_stats <- function(data, cl){
  ##############################################################################
  #
  #        Simple stats on entry data      
  #
  ##############################################################################
  # число групп
  k <- length(unique(data[[cl]])) 
  cat('k =', k,'\n')
  
  # пропуски 
  cat('Пропуски: ', sum(is.na(data)) , ' (',sum(is.na(data))/nrow(data)*100, '%)','\n', sep='')
  
  # группы
  ccc <- count(data, cl)
  #bp <- barplot(ccc$freq, names.arg=ccc$macro.area)
  #text(bp, 0, ccc$freq,cex=1,pos=3) 
  cat('См. график с распределением групп', cl ,'\n')
  
  char_cols <- names(Filter(is.character, data))
  factor_cols <- names(Filter(is.factor, data))
  
  transform_cols <- c(char_cols, factor_cols)
  cat('Текстовые столбцы или факторы: ')
  if(length(transform_cols)>0){
    for (i in 1:length(transform_cols)){
      cat(transform_cols[i],'(',length(unique(data[[ transform_cols[i]]])),')   ', sep='')
    }
  }
  cat("done")
  return(list(char_cols=char_cols, factor_cols=factor_cols, transform_cols=transform_cols))
}

semi_regression_preparation<-function(file, file_test, label_train, label_test, delta){
  data3 <-  read.csv2(file)
  data4 <-  read.csv2(file_test)
  res3  <-  read.csv2(label_train)
  res4  <-  read.csv2(label_test)
  
  arr<-matrix(as.numeric(unlist(data3)), ncol=length(data3))
  arr2<-matrix(as.numeric(unlist(data4)), ncol=length(data4))
  
  nans <- which(is.na(arr),arr.ind=TRUE)
  nans <-nans[,1]
  
  if(length(nans) >0){
    arr<-as.matrix(arr[-nans,])
  }
  
  res3<-matrix(as.numeric(unlist(res3)), ncol=length(res3))
  if(length(nans) >0){
    res3<-as.matrix(res3[-nans,])
  }
  
  nans <- which(is.na(res3),arr.ind=TRUE)[,1]
  
  if(length(nans) >0){
    res3<-as.matrix(res3[-nans,])
    arr<-as.matrix(arr[-nans,])
  }
  
  nans <- which(is.na(arr2),arr.ind=TRUE)[,1]
  
  if(length(nans) >0){
    arr2<-as.matrix(arr2[-nans,])
  }
  
  res4<-matrix(as.numeric(unlist(res4)), ncol=length(res4))
  if(length(nans) >0){
    res4<-as.matrix(res4[-nans,])
  }
  
  nans <- which(is.na(res4),arr.ind=TRUE)[,1]
 
  if(length(nans) >0){
    res4<-as.matrix(res4[-nans,])
    arr2<-as.matrix(arr2[-nans,])
  }
  
  beg = as.integer(delta * nrow(arr)) + 1
  arr_semi = arr[1 : as.integer(delta * nrow(arr)),]
  
  res3_semi_tail = res3[beg:nrow(arr),]
  arr_semi_tail = arr[beg:nrow(arr),]
  res3 = res3[1:as.integer(delta*nrow(arr)),]
  res3_semi_tail = matrix(as.numeric(unlist(res3_semi_tail)), nrow=length(res3_semi_tail))
  return(list(arr, arr2, res3, res4, arr_semi, res3_semi_tail,arr_semi_tail))
}


prepare <- function(data, cl, encode_cols,encode_types, nan='omit', norm='scale'){
  ##############################################################################
  #
  #        Prepare data (Encode character columns/deal with NaNs/Scale)    
  #
  ##############################################################################
  new <- copy(data)
  nan <- tolower(nan)
  norm <- tolower(norm)
  encode_types<- tolower(encode_types)
  
  if (length(encode_cols)!=length(encode_types)){
    stop('Длины encode_cols и encode_types должны совпадать!')
  }
  
  if (!is.null(encode_cols)){
    for (i in 1:length(encode_types)){
      if (any(tolower(encode_types[i]) %in% c('num','as.num','asnum','as.numeric','asnumeric','tonum','to.num'))){
        new[[encode_cols[i]]] <- as.numeric(factor(new[[encode_cols[i]]])) 
      }
      
      if (any(tolower(encode_types[i]) %in% c('one.hot','onehot','one','hot','one-hot'))){
        if (encode_cols[i] == cl){
          stop('Не используйте one-hot для целевого столбца!')
        }
        new <- new %>% mutate(value = 1)  %>% spread(encode_cols[i], value,  fill = 0 ) 
      }
    }
  }
  
  a <- names(Filter(is.character, new))
  b <- names(Filter(is.factor, new))
  d <- c(a,b)
  
  if (!(is.null(d) | identical(d, character(0))) ){
    warning("Остались нечисловые столбцы: ", d, '!')
    return(new)
  }
  
  
  if (any(tolower(nan) %in% c('omit','delete','throw','no'))){
    nan <- 'omit'
    new <- na.omit(new)
  }
  if (any(tolower(nan) %in% c('mean','average'))){
    nan <- 'mean'
    
    for(i in 1:ncol(new)){
      new[is.na(new[,i]), i] <- mean(new[,i], na.rm = TRUE)
    }
  }
  if (any(tolower(nan) %in% c('median'))){
    
    for(i in 1:ncol(new)){
      new[is.na(new[,i]), i] <- median(new[,i], na.rm = TRUE)
    }
  }
  
  if (sum(is.na(data)) != 0){
    warning("Остались пропуски в данных (", sum(is.na(data)),')!')
    return(new)
  }
  
  if (any(tolower(norm) %in% c('scale'))){
    new <- scale(new[names(new) != cl])
  }
  if (any(tolower(norm) %in% c('01','norm01', '0to1', 'scale01'))){
    
    new[names(new) != cl] <- apply(new[names(new) != cl], MARGIN = 2, FUN = function(X) if (diff(range(X))>0){(X-min(X))/(diff(range(X))+0.000001)}else{(X)/(diff(range(X))+0.000001)})
    
  }
  return(new)
}


dgmm_grid_search <- function(data0, k, it, eps,init, init_est,seed = 1,type='AIC'){
  ##############################################################################
  #
  #                Full Grid
  #                    Search
  #
  ##############################################################################
  type <- tolower(type)
  if (type %in% c('aic&bic','aicbic','bicaic','aic_bic', 'bic_aic','aic/bic', 'bic/aic'))
    type <- 'bic&aic'
  
  r <- dim(data0)[2]
  
  bic1 <- c(Inf,0,NaN)
  aic1 <- c(Inf,0,NaN) 
  bic2 <- c(Inf,0,NaN)
  aic2 <- c(Inf,0,NaN)
  bic3 <- c(Inf,0,NaN)
  aic3 <- c(Inf,0,NaN)
  
  if (r==4){
    p1const <- 3
    p2const <- 3#2
    p3const <- 3
  }
  else{
    p1const <- round(r/2) 
    p2const <- min(round(r/3),p1const-1) 
    p3const <- max(1,min(round(r/4),p2const-1))
  }
  
  r1 <- cbind(r1 = 1:p1const)
  r1r2k2 <- crossing(r1 = 1:p1const, r2 = 1:p2const, k2 = 1:(k+1))
  r1r2k2 <- as.matrix(r1r2k2[(r1r2k2[, 1]) > (r1r2k2[, 2]), ])
  
  r1r2r3k2k3 <- crossing(r1 = 1:p1const, r2 = 1:p2const, r3 = 1:p3const, k2 = 1:(k+1), k3 = 1:(k+1))
  r1r2r3k2k3 <- as.matrix(r1r2r3k2k3[((r1r2r3k2k3[, 1]) > (r1r2r3k2k3[, 2])) & ((r1r2r3k2k3[, 2]) > (r1r2r3k2k3[, 3])), ])
  
  # print(c('!!!!',nrow(r1)))
  for (i in 1:nrow(r1)){
    # print(i)
    p1 <- r1[i,1]
    dgmm1 <- deepgmm(data0, layers=1, k=k, r=p1, seed = seed,it=it,eps=eps,init=init, init_est=init_est, scale=FALSE)
    
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm1$bic < bic1[1]) bic1 <- c(dgmm1$bic, toString(c(p1,k))) 
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm1$aic < aic1[1]) aic1 <- c(dgmm1$aic, toString(c(p1,k))) 
    # print(c(seed, p1, toString(c(p1,k))))
  }
  cat('1 слой -- done',nrow(r1) ,'\n')
  dgmm2 <-0
  for (i in 1:nrow(r1r2k2)){
    
    p1p2 <- c(r1r2k2[i,1],r1r2k2[i,2])
    k1k2 <- c(k,r1r2k2[i,3])
    dgmm2 <- deepgmm(data0, layers=2, k=k1k2, r=p1p2, seed=seed, scale=FALSE)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm2$bic < bic2[1]) bic2 <- c(dgmm2$bic, toString(c(p1p2,k1k2)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm2$aic < aic2[1]) aic2 <- c(dgmm2$aic, toString(c(p1p2,k1k2)))
    
  }
  cat('2 слоя -- done',nrow(r1r2k2) ,'\n')
  print(nrow(r1r2r3k2k3))
  if(nrow(r1r2r3k2k3)>0){
  for (i in 1:nrow(r1r2r3k2k3)){
    
    p1p2p3 <- c(r1r2r3k2k3[i,1],r1r2r3k2k3[i,2],r1r2r3k2k3[i,3])
    k1k2k3 <- c(k,r1r2r3k2k3[i,4],r1r2r3k2k3[i,5])
    dgmm3 <- deepgmm(data0, layers=3, k=k1k2k3, r=p1p2p3, seed=seed, scale=FALSE)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm3$bic < bic3[1]) bic3 <- c(dgmm3$bic, toString(c(p1p2p3,k1k2k3)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm3$aic < aic3[1]) aic3 <- c(dgmm3$aic, toString(c(p1p2p3,k1k2k3)))
    
  }
  cat('3 слоя -- done',nrow(r1r2r3k2k3) ,'\n')
  } else{
    aic3 <- aic2
    
  }
  
  if (type == 'bic' | type == 'bic&aic')
    print(cbind(bic1,bic2,bic3))
  if (type == 'aic' | type == 'bic&aic')
    print(cbind(aic1,aic2,aic3)) 
  return(list("bic" = cbind(bic1,bic2,bic3), "aic" = cbind(aic1,aic2,aic3)))
  
}

# ----------------------------------------------------------------------------------

dgmm_rand_search <- function(data0, k, it, eps,init, init_est,seed = 1,type='AIC',ratio = 0.2){
  ##############################################################################
  #
  #                Random  Search
  #
  ##############################################################################
  type <- tolower(type)
  if (type %in% c('aic&bic','aicbic','bicaic','aic_bic', 'bic_aic','aic/bic', 'bic/aic'))
    type <- 'bic&aic'
  
  p <- ncol(data0)-1#dim(data0)[2]-1
  
  bic1 <- c(Inf,0,NaN)
  aic1 <- c(Inf,0,NaN) 
  bic2 <- c(Inf,0,NaN)
  aic2 <- c(Inf,0,NaN)
  bic3 <- c(Inf,0,NaN)
  aic3 <- c(Inf,0,NaN)
  print("p: ")
  print(p)
  if (p==4){
    p1const <- 3#
    p2const <- 3#2
    p3const <- 3
  }
  else{
    if (p==2){
      p1const <- 2#
      p2const <- 2#2
      p3const <- 2
    }
    else{
    p1const <- round(p-1) 
    p2const <- min(round(p1const/2),p1const-1) 
    p3const <- max(1,min(round(p1const/4),p2const-1))
    }
  }
  
  r1 <- cbind(r1 = 1:p1const)
  
  r1r2k2 <- crossing(r1 = 1:p1const, r2 = 1:p2const, k2 = 1:(k+1))
  r1r2k2 <- as.matrix(r1r2k2[(r1r2k2[, 1]) > (r1r2k2[, 2]), ])
  
  r1r2r3k2k3 <- crossing(r1 = 1:p1const, r2 = 1:p2const, r3 = 1:p3const, k2 = 1:(k+1), k3 = 1:(k+1))
  r1r2r3k2k3 <- as.matrix(r1r2r3k2k3[((r1r2r3k2k3[, 1]) > (r1r2r3k2k3[, 2])) & ((r1r2r3k2k3[, 2]) > (r1r2r3k2k3[, 3])), ])
  
  if (floor(ratio*nrow(r1)) > 1){
    sample1 <- sample.int(n = nrow(r1), size = floor(ratio*nrow(r1)), replace = F)
    r1 <- cbind(r1[sample1, ])
  }
  
  if (floor(ratio*nrow(r1r2k2)) > 1){
    sample2 <- sample.int(n = nrow(r1r2k2), size = floor(ratio*nrow(r1r2k2)), replace = F)
    r1r2k2 <- r1r2k2[sample2, ]
  }
  
  if (floor(ratio*nrow(r1r2r3k2k3)) > 1){
    sample3 <- sample.int(n = nrow(r1r2r3k2k3), size = floor(ratio*nrow(r1r2r3k2k3)), replace = F)
    r1r2r3k2k3 <- r1r2r3k2k3[sample3, ]
  }
  
  for (i in 1:nrow(r1)){
    
    p1 <- r1[i,1]
    print(p1)
    dgmm1 <- deepgmm(data0, layers=1, k=k, r=p1, seed = seed,init=init, scale = FALSE)
    if (type == 'bic' | type == 'bic&aic')
      if (as.numeric(dgmm1$bic) < as.numeric(bic1[1]) ) bic1 <- c(toString(dgmm1$bic), toString(c(p1,k))) 
    if (type == 'aic' | type == 'bic&aic')
      if (as.numeric(dgmm1$aic) < as.numeric(aic1[1]) & as.numeric(dgmm1$aic)>0 ) aic1 <- c(toString(dgmm1$aic), toString(c(p1,k))) 
    # print(c(seed, p1, toString(c(p1,k))))
  }
  cat('1 слой -- done',nrow(r1) ,'\n')
  dgmm2 <- 0
  sciters=0
  if(nrow(r1r2k2)>1000){
    sciters = 1000
  }
  else{
    sciters = r1r2k2
  }
 
  for (i in 1:sciters){
    
    p1p2 <- c(r1r2k2[i,1],r1r2k2[i,2])
    k1k2 <- c(k,r1r2k2[i,3])
    #print(i)
    #print(p1p2)
    #print(k1k2)
    dgmm2 <- deepgmm(data0, layers=2, k=k1k2, r=p1p2, seed=seed, init=init, scale = FALSE)
    if (type == 'bic' | type == 'bic&aic')
      if (as.numeric(dgmm2$bic) < as.numeric(bic2[1])) bic2 <- c(toString(dgmm2$bic), toString(c(p1p2,k1k2)))
    if (type == 'aic' | type == 'bic&aic')
      if (as.numeric(dgmm2$aic) < as.numeric(aic2[1])& as.numeric(dgmm2$aic)>0 ) aic2 <- c(toString(dgmm2$aic), toString(c(p1p2,k1k2)))
    
  }
  cat('2 слоя -- done',nrow(r1r2k2) ,'\n')
  cat('3 слоя ',nrow(r1r2r3k2k3) ,'\n')
  a=-1
  if(nrow(r1r2r3k2k3)>0){
    thiters=0
    if(nrow(r1r2r3k2k3)>1000){
      thiters = 1000
    }
    else{
      thiters = r1r2r3k2k3
    }
  for (i in 1:thiters){
    
    p1p2p3 <- c(r1r2r3k2k3[i,1],r1r2r3k2k3[i,2],r1r2r3k2k3[i,3])
    k1k2k3 <- c(k,r1r2r3k2k3[i,4],r1r2r3k2k3[i,5])
    dgmm3 <- deepgmm(data0, layers=3, k=k1k2k3, r=p1p2p3, seed=seed,init=init, scale = FALSE)
    if (type == 'bic' | type == 'bic&aic')
      if (as.numeric(dgmm3$bic )< as.numeric(bic3[1]) ) bic3 <- c(toString(dgmm3$bic), toString(c(p1p2p3,k1k2k3)))
    if (type == 'aic' | type == 'bic&aic')
      if (as.numeric(dgmm3$aic) < as.numeric(aic3[1])& as.numeric(dgmm3$aic)>0  ) aic3 <- c(toString(dgmm3$aic), toString(c(p1p2p3,k1k2k3)))
    
  }
  cat('3 слоя -- done',nrow(r1r2r3k2k3) ,'\n')
  } else{
    aic3 <- aic2
    
  }
  if (type == 'bic' | type == 'bic&aic')
    print(cbind(bic1,bic2,bic3))
  if (type == 'aic' | type == 'bic&aic')
    print(cbind(aic1,aic2,aic3)) 
  return(list("bic" = cbind(bic1,bic2,bic3), "aic" = cbind(aic1,aic2,aic3)))
  
}

# ----------------------------------------------------------------------------------

regr <- function(data, x, target_id , k, w, mu, sigma){  
  ##############################################################################
  #
  #                Regression with Given Mixture Components
  #
  ##############################################################################
  print("regr params")
  #print(k)
  #print(w)
  #print(mu)
  #print(sigma)
  p <- dim(data)[2]  
  n <- dim(x)[1] 
  some_small_value <- 2.220446049250313e-16
  
  indices <- 1:p 
  indep_id <- indices[-target_id]
  
  
  
  X <- x[,indep_id ] 
  Y <- data[,target_id]
  
  regression_coeffs <- array(0, c(k, length(target_id), length(indep_id)))
  coeffs <- array(0,k)
  exps <- array(0, c(n, k))
  
  cov12  <- array(0, c(k, length(target_id), length(indep_id) ))   
  cov11  <- array(0, c(k, length(indep_id), length(indep_id) ))   
  XX <- array(0, c(n,k, length(indep_id) )) 
  for (i in 1:k){    
    cov12[i,,] <- sigma[i,indep_id,target_id]
    cov11[i,,] <- sigma[i,indep_id,indep_id]  
    regression_coeffs[i,,] <- t(array(cov12[i,,],c(p-1,1)))%*% inv(array(cov11[i,,],c(p-1,p-1)))
    
    mean <- mu[indep_id,i] 
    covariance <- sigma[i,indep_id,indep_id] 
    
    coefficient <- array(0,k)
    exponent <- array(0, c(n, k))
    
    C <- t(chol(covariance)) 
    C_det <- max(det(C), some_small_value) 
    
    x_centered <- X - t(array(mean,c(p-1,n)))
    x_norm <- t(forwardsolve(C, t(x_centered)))
    
    coeffs[i]  <- 1/(2*pi)**(0.5*length(indep_id))/C_det
    exps[,i] <- -0.5*rowSums(x_norm**2)
    
    XX[,i,] <- array(X, c(n,1,p-1))  
  }
  
  wx <- exp(exps)
  coeffs <- w*coeffs 
  
  for (i in 1:n){
    wx[i,] = wx[i,]*array(coeffs, c(1,k))
    wx[i,] = wx[i,]/sum(wx[i,])
  }
  wx <- array(wx, c(n,1,k))
  
  muX<-array(0, c(n,k,p-1))
  muY<-array(0, c(n,1,k))
  for (i in 1:n){
    muX[i,,] <- t(mu[indep_id,])
    muY[i,,] <- array(mu[target_id,], c(1,1,k)) 
  }
  
  mx <- muY + einsum("ijk,lik->lji", regression_coeffs, XX - muX)
  
  result <- rowSums(wx*mx, dims = 2) 
  
  return (result)
}

# ----------------------------------------------------------------------------------

DGMMr <- function(model = 'DGMMr', data,  x, target_id, k, p, it=250, init="kmeans", eps=0.001, cl = FALSE){  
  ##############################################################################
  #
  #    Gaussian Mixture (Classical or Deep Models) + Regression
  #
  ##############################################################################
  
  if (any(tolower(model) %in% c('dgmmr', 'deepgmmr', 'dgmm-r')))
    model <- 'dgmmr'
  
  if (any(tolower(model) %in% c('gmmr', 'gmm-r')))
    model <- 'gmmr'
  
  res_cl <- NaN
  
  if (model == "dgmmr") {
      if (length(k) == length(p)){
        layers <- length(k)
        l <- length(k)
        dgmm <- deepgmm(data, layers=l, k=k, r=p,
                        it = it, eps = eps, init = init, init_est = "factanal",
                         scale = FALSE)
        if (cl){
          res_cl <- dgmm$s[,1]
        }
      }
      else{
        stop("Length of k and p must be the same")
      }
    
     
    
    p <- dim(data)[2]
    n <- dim(data)[1]
    
    w = dgmm$w
    mu = dgmm$mu
    Lambda = dgmm$H
    Psi = dgmm$psi
    
    layers <- l
    k<- k
    numobs <-n
    py <- matrix(0, numobs)
    tot.k <- prod(k)
    py.s <- matrix(0, numobs, tot.k)
    pys <- matrix(0, numobs, tot.k)
    
    k.comb <- apply(t(k), 2, function(x) 1 : x)
    if (is.list(k.comb)) {
      k.comb <- expand.grid(k.comb)
      
    }
    if (is.matrix(k.comb)) {
      k.comb <- expand.grid(split(t(k.comb), 1 : ncol(k.comb)))
    }
    if (prod(k) == 1) {
      k.comb <- matrix(k.comb, nrow =1)
    }
    
    all_w = array(0, tot.k)
    all_mu =  array(0, c( p, tot.k))
    all_var = array(0, c(tot.k, p, p))
    
    for (i in 1 : tot.k)  {
      
      mu.tot <- mu[[1]][, k.comb[i, 1]]
      var.tot <- Psi[[1]][k.comb[i, 1],,]
      w.tot <- w[[1]][k.comb[i, 1]]
      w.tot
      if (layers > 1) {
        for (l in 2 : layers) {
          
          tot.H <- diag(p)
          for (m in 1 : (l -1)) {
            tot.H <- tot.H %*% Lambda[[m]][k.comb[i, m],, ]
          }
          
          mu.tot <- mu.tot + tot.H %*% mu[[l]][, k.comb[i, l]]
          
          var.tot <- var.tot + tot.H %*% (Lambda[[l]][k.comb[i, l],, ] %*%
                                            t(Lambda[[l]][k.comb[i, l],, ]) +
                                            Psi[[l]][k.comb[i, l],, ]) %*% t(tot.H)
          w.tot <- w.tot * w[[l]][k.comb[i, l]]
        }
      }
      
      
      if (!is.positive.definite(var.tot)) {
        var.tot <- make.positive.definite(var.tot)
      }
      
      if (w.tot == 0) {
        w.tot <-  10^(-320)
      }
      
      all_w[i] <- w.tot
      
      all_mu[,i] <- mu.tot
      all_var[i,,] <- var.tot 
    }
    all_w = all_w/sum(all_w )
    
    res_regr <- regr(data=data, x=x, target_id=target_id, k=tot.k, w=all_w, mu=all_mu, sigma=all_var)
    res_regrTrue <- regr(data=data, x=data, target_id=target_id, k=tot.k, w=all_w, mu=all_mu, sigma=all_var)
    res_mu <- all_mu
    res_sigma <- all_var
    
  } else {
    if (model != "gmmr")
      stop("Model has to be DGMMr or GMMr")
    print(k)
    gmm <- Mclust(data, control = emControl(itmax = it, eps = eps), G=k,verbose=0) 
    
    if (cl){
      res_cl <- gmm$classification
    }
    
    w <- gmm$parameters$pro
    mu <- gmm$parameters$mean
    res_mu <- mu
    sigma <- aperm(gmm$parameters$variance$sigma, c(3,1,2))
    res_sigma <- sigma
    res_regr <- regr(data=data, x=x, target_id=target_id, k=k, w=w, mu=mu, sigma=sigma)
    res_regrTrue <- regr(data=data, x=data, target_id=target_id, k=k, w=w, mu=mu, sigma=sigma)
  }
  
  return (list(regr=res_regr , cl=res_cl, mu=res_mu, sigma=res_sigma, res_True=res_regrTrue))
}

# ----------------------------------------------------------------------------------

SemiSupClassDGMM <- function(X_train, X_test=NaN, y_train = NaN, y_test = NaN, par, unsup = 'DGMM', sup ='SVM', print_res = TRUE){
  ##############################################################################
  #
  #                Semi Supervised Method for Classification
  #                using combination of 
  #                Gaussian Mixtures / Deep Gauusian Mixture Models(DGMM) + SVM/XGB
  #
  ##############################################################################
  
  if (any(tolower(unsup) %in% c('dgmm', 'deepgmm')))
    unsup <- 'dgmm'
  if (any(tolower(unsup) %in% c('gmm')))
    unsup <- 'gmm'
  if (any(tolower(sup) %in% c('svm', 'ksvm')))
    sup <- 'svm'
  if (any(tolower(sup) %in% c('xgb', 'xgboost')))
    sup <- 'xgb'
  
  if (unsup == 'dgmm'){
    k <- par$k
    p <- par$p
    
    l <- length(k)
    dgmm <- deepgmm(X_train, layers=l, k=k, r=p,
                    it = 250, eps = 0.001, init = "kmeans", init_est = "factanal",
                    seed = NULL, scale = FALSE)
    train_cl <- dgmm$s[,1]
  } else {
    if (unsup != 'gmm')
    {
      stop("Unsupervised model has to be DGMM or GMM")
    }
    
    k <- par$k
    
    if (length(k) >1){
      stop("For GMM k must be scalar (k=par$k)")
    }
    
    gmm <- Mclust(X_train, control = emControl(itmax = 250, eps = 0.001), G=k)
    train_cl <- gmm$classification
  }
  
  train_cl <- dgmm$s[,1]-min(dgmm$s[,1])
  
  if (sup =='svm'){
    svmmodel <- ksvm(as.matrix(X_train), train_cl, type ='C-svc',kernel = 'rbfdot', C = 10) 
    model <- svmmodel
    if (!sum(is.nan(X_test))){
      test_cl <- predict(svmmodel, X_test)
    }
  }
  else {
    if (sup != 'xgb')
    {
      stop("Supervised model has to be SVM or XGB")
    }
    
    params = list(
      booster="gbtree",   
      objective="multi:softprob", 
      num_class=k[1]
    )
    
    xgb.train = xgb.DMatrix(data = as.matrix(X_train), label = train_cl)
    
    # xgb.test = xgb.DMatrix(data = as.matrix(X_test), label = y_test)
    
    xgb.fit=xgb.train(
      params=params,
      data=xgb.train,
      nrounds=50, 
      verbose=0
    )
    model <- xgb.fit
    
    if (!sum(is.nan(X_test))){
      xgb.pred = predict(xgb.fit, as.matrix(X_test),reshape=T)
      xgb.pred = as.data.frame(xgb.pred)
      
      test_cl <- max.col(xgb.pred, 'first')
    }
    
  }
  
  train_ari <- NaN
  test_ari <- NaN
  
  if (!sum(is.nan(y_train))){
    train_ari  <- adjustedRandIndex(train_cl, y_train)
  }
  if (!sum(is.nan(y_test))){
    test_ari  <- adjustedRandIndex(test_cl, y_test)
  }
  
  if(print_res == TRUE){
    if (!sum(is.nan(y_train))){
      print('ARI for train:')
      print(train_ari)
    }
    
    if (!sum(is.nan(y_test))){
      print('ARI for test:')
      print(test_ari)
    }
  }
  
  output <- list(model = model, train_cl=train_cl, test_cl=test_cl, train_ari=train_ari, test_ari=test_ari)
  return(output)
}
# ----------------------------------------------------------------------------------

SemiSupRegrDGMM <- function(X_train, X_test, X_val = NaN, y_train, y_test = NaN, y_val = NaN, par, type1 = 'DGMMr', type2 ='SVM', print_res = TRUE){
  ##############################################################################
  #
  #        Semi Supervised Method for Regression
  #        using combination of 
  #        Gaussian Mixtures (GMM)/Deep Gauusian Mixture Models(DGMM) + SVM/XGB        
  #
  ##############################################################################
  
  if (any(tolower(type1) %in% c('dgmm', 'deepgmm', 'deepgmmr', 'dgmmr')))
    type1 <- 'dgmmr'
  if (any(tolower(type1) %in% c('gmm','gmmr')))
    type1 <- 'gmmr'
  if (any(tolower(type2) %in% c('svm', 'ksvm')))
    type2 <- 'svm'
  if (any(tolower(type2) %in% c('xgb', 'xgboost')))
    type2 <- 'xgb'
  
  if (type1 == 'dgmmr'){
    k <- par$k[1]
    p <- par$p
    
    l <- length(k)
  } else {
    if (type1 != 'gmmr')
    {
      stop("Unsupervised model has to be DGMM or GMM")
    }
    
    k <- par$k
    
    if (length(k) >1){
      stop("For GMM k must be scalar (k=par)")
    }
  }
  
  train <- cbind(X_train, y_train)
  
  dgmmr <- DGMMr(model = type1, train, params=par, X_test, dim(train)[2], k, cl = TRUE)
  test_r <- dgmmr$regr
  test_cl <- dgmmr$cl
  
  # train_cl <- dgmm$s[,1]-min(dgmm$s[,1])
  merged_train_x <- rbind(X_train, X_test)
  merged_train_y <- c(y_train, test_r)
  
  if (type2 =='svm'){
    svmmodel <- ksvm(merged_train_x, merged_train_y, kernel = 'rbfdot', C = 10) 
    model <- svmmodel
    # if (!sum(is.nan(X_val))){
    #   val_r <- predict(svmmodel, X_val)
    # }
  } else {
    if (type2 != 'xgb')
    {
      stop("Supervised model has to be SVM or XGB")
    }
    
    xgb.train = xgb.DMatrix(data = merged_train_x, label = merged_train_y)
    # xgb.test = xgb.DMatrix(data = as.matrix(X_var), label = y_var)
    
    xgbr = xgboost(data = xgb.train, max.depth = 2, nrounds = 50)
    model <- xgbr
    
    # if (!sum(is.nan(X_val))){
    #   val_r <- predict(xgbr, X_val)
    # }
    
  }
  
  if (!sum(is.nan(X_val))){
    val_r <- predict(model, X_val)
  }
  
  test_rmse <- NaN
  val_rmse <- NaN
  
  if (!sum(is.nan(y_test))){
    test_rmse  <- rmse(y_test, test_r)
  }
  
  if (!sum(is.nan(y_val))){
    val_rmse  <- rmse(y_val, val_r)
  }
  
  if(print_res == TRUE){
    if (!sum(is.nan(y_test))){
      print('RMSE for test:')
      print(test_rmse)
    }
    
    if (!sum(is.nan(y_val))){
      print('RMSE for val:')
      print(val_rmse)
    }
  }
  
  output <- list(model = model, test_cl=test_cl, 
                 test_r=test_r, val_r=val_r, test_rmse=test_rmse, val_rmse=val_rmse)
  return(output)
}

dgmm_param_extraction <- function(data, layers, k, r,it,eps,seed,init, init_est,target_id){ 
    print(any(r >= ncol(data)))
    print(r)
    dgmm <- deepgmm(data, layers=layers, k=k, r=r,
                         eps = 0.001, init = init, 
                        seed = seed, scale = FALSE)
    
    p <- dim(data)[2]
    n <- dim(data)[1]
    
    w = dgmm$w
    mu = dgmm$mu
    Lambda = dgmm$H
    Psi = dgmm$psi
    
    layers <- layers
    k<- k
    numobs <-n
    py <- matrix(0, numobs)
    tot.k <- prod(k)
    py.s <- matrix(0, numobs, tot.k)
    pys <- matrix(0, numobs, tot.k)
    
    k.comb <- apply(t(k), 2, function(x) 1 : x)
    if (is.list(k.comb)) {
      k.comb <- expand.grid(k.comb)
      
    }
    if (is.matrix(k.comb)) {
      k.comb <- expand.grid(split(t(k.comb), 1 : ncol(k.comb)))
    }
    if (prod(k) == 1) {
      k.comb <- matrix(k.comb, nrow =1)
    }
    
    all_w = array(0, target_id)
    all_mu =  array(0, c( p, target_id))
    all_var = array(0, c(target_id, p, p))
    
    for (i in 1 : target_id)  {
      
      mu.tot <- mu[[1]][, k.comb[i, 1]]
      var.tot <- Psi[[1]][k.comb[i, 1],,]
      w.tot <- w[[1]][k.comb[i, 1]]
      w.tot
      if (layers > 1) {
        for (l in 2 : layers) {
          
          tot.H <- diag(p)
          for (m in 1 : (l - 1)) {
            tot.H <- tot.H %*% Lambda[[m]][k.comb[i, m],, ]
          }
          
          mu.tot <- mu.tot + tot.H %*% mu[[l]][, k.comb[i, l]]
          
          var.tot <- var.tot + tot.H %*% (Lambda[[l]][k.comb[i, l],, ] %*%
                                            t(Lambda[[l]][k.comb[i, l],, ]) +
                                            Psi[[l]][k.comb[i, l],, ]) %*% t(tot.H)
          w.tot <- w.tot * w[[l]][k.comb[i, l]]
        }
      }
      
      
      if (!is.positive.definite(var.tot)) {
        var.tot <- make.positive.definite(var.tot)
      }
      
      if (w.tot == 0) {
        w.tot <-  10^(-320)
      }
      
      all_w[i] <- w.tot
      all_mu[,i] <- mu.tot
      all_var[i,,] <- var.tot 
    }
    
    
    return(list(all_w,all_mu,all_var, target_id))
}



dgmm_inference <- function(data,  k, all_w,all_mu,all_var, target_id){
  some_small_value <- 2.220446049250313e-16
  p <- dim(data)[2]
  n <- dim(data)[1]
  indices <- 1:p 
  indep_id <- indices#[-target_id]
  
  X <- data[,indep_id ] 
  
  coeffs <- array(1, k)
  exps <- array(1, c(n, k))
  
  #cov12  <- array(1, c(k, length(target_id), length(indep_id) ))   
  #cov11  <- array(1, c(k, length(indep_id), length(indep_id) ))   
  
  for (i in 1:k){    
    #cov12[i,,] <- all_var[i,indep_id,target_id]
    #cov11[i,,] <- all_var[i,indep_id,indep_id]  
    
    
    mean <- all_mu[indep_id,i] 
    covariance <- all_var[i,indep_id,indep_id] 
    
    coefficient <- array(1,k)
    exponent <- array(1, c(n, k))
    
    C <- t(chol(covariance)) 
    C_det <- max(det(C), some_small_value) 
    
    x_centered <- X - t(array(mean,c(p,n)))
    x_norm <- t(forwardsolve(C, t(x_centered)))
    
    coeffs[i]  <- 1/(2*pi)**(0.5*length(indep_id))/C_det
    exps[,i] <- -0.5*rowSums(x_norm**2)
    
  }
  
  wx <- exp(exps)
  coeffs <- all_w*coeffs 
  for (i in 1:n){ 
    wx[i,]<-wx[i,]*coeffs
  }
  return(wx)
}

convert_to_labels <- function(res){ 
  n <- dim(res)[1]
  res_labels <- array(0, c(n, 1))
  for (i in 1:n){ 
    res_labels[i]<-which(res[i,]  %in% max(res[i,]))
  }
  return(res_labels)
}


SelfSupClass <- function(X_train, X_test=NaN, y_train = NaN, y_test = NaN, sup ='SVM', print_res = TRUE,classes=1){
  ##############################################################################
  #
  #                Semi Supervised Method for Classification
  #                using combination of 
  #                Gaussian Mixtures / Deep Gauusian Mixture Models(DGMM) + SVM/XGB
  #
  ##############################################################################

  
  train_cl <- y_train
  
  if (sup =='svm'){
    svmmodel <- ksvm(as.matrix(X_train), train_cl, type ='C-svc',kernel = 'rbfdot')#, C = 10) 
    model <- svmmodel
    if (!sum(is.nan(X_test))){
      test_cl <- predict(svmmodel, X_test)
    }
  }
  else {
    if (sup != 'xgb')
    {
      stop("Supervised model has to be SVM or XGB")
    }
    
    params = list(
      booster="gbtree",   
      objective="multi:softprob", 
      num_class=classes
    )
    
    xgb.train = xgb.DMatrix(data = as.matrix(X_train), label = train_cl-1)
    
    # xgb.test = xgb.DMatrix(data = as.matrix(X_test), label = y_test)
    
    xgb.fit=xgb.train(
      params=params,
      data=xgb.train,
      nrounds=70, 
      verbose=0
    )
    model <- xgb.fit
    
    if (!sum(is.nan(X_test))){
      xgb.pred = predict(xgb.fit, as.matrix(X_test),reshape=T,verbose=0
                         
      )
      xgb.pred = as.data.frame(xgb.pred)
      
      test_cl <- max.col(xgb.pred, 'first')
    }
    
  }
  
  #train_ari <- NaN
  #test_ari <- NaN
  
  #if (!sum(is.nan(y_train))){
    #train_ari  <- adjustedRandIndex(train_cl, y_train)
 # }
  #if (!sum(is.nan(y_test))){
  #  test_ari  <- adjustedRandIndex(test_cl, y_test)
 # }
  
  if(print_res == TRUE){
    #if (!sum(is.nan(y_train))){
#      print('ARI for train:')
#      print(train_ari)
#    }
    
    if (!sum(is.nan(y_test))){
      print('F1  for test:')
      p1 = factor(test_cl, levels = seq(1, classes, length.out=classes))
      r1 = estimateF1(p1, as.factor(y_test))
      #print(test_ari)
    }
  }
  #print(test_cl)
  output <- list(model = model, train_cl=train_cl, test_cl=test_cl)
  return(output)
}



dgmm_labels <- function(arr,arr2, classes, it, eps, init, init_est, seed){
  res<-dgmm_rand_search(arr, classes, it, eps, init, init_est, seed)
  print("finded best aic values: ")
  print(res$aic)
  bics<-c(as.numeric(res$aic[1]),as.numeric(res$aic[3]), as.numeric(res$aic[5]))
  layers<-which(bics %in% min(unlist(bics)))#as.integer(which.min(unlist(bics)))
  print("layer value: ")
  print(layers)
  layers = layers[1]
  params<-sapply(strsplit((res$aic[2*layers]), split=", "), function(x) (as.numeric(x))) 
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
  
  res <-dgmm_inference(arr, params[[4]], params[[1]],params[[2]],params[[3]], classes)
  res2 <-dgmm_inference (arr2,params[[4]], params[[1]],params[[2]],params[[3]],classes)
  res = as.data.frame(res)
  
  tain_cl <- max.col(res, 'first')
  dgmm <- deepgmm(arr, layers=layers, k=k, r=r,
                  it = it, eps = 0.001, init = init, init_est = init_est,
                  seed = seed, scale = FALSE)
  return(list(label=dgmm$s[,1], k=params[[4]], all_w=params[[1]],all_mu=params[[2]],all_var=params[[3]]))
}



dgmm_labels_full <- function(arr,arr2, classes, it, eps, init, init_est, seed){
  res_search<-dgmm_rand_search(arr, classes, it, eps, init, init_est, seed)
  print("finded best aic values: ")
  print( res_search$aic)
  bics<-c(as.numeric( res_search$aic[1]),as.numeric( res_search$aic[4]), as.numeric( res_search$aic[7]))
  layers<-which(bics %in% min(unlist(bics)))#as.integer(which.min(unlist(bics)))
  print("layer value: ")
  print(layers)
  layers = 1
  print(res_search$aic[1])
  print(res_search$aic[3])
  print(res_search$aic[5])
  
  if(length(res_search$aic)==6){
    idx=c(1,3,5)
    scale=2
  }
  else{
    idx=c(1,4,7)
    scale=3
  }
  if (layers == 1 & res_search$aic[idx[1]] != "Inf")
  { 
    params<-sapply(strsplit(( res_search$aic[idx[1]+1]), split=", "), function(x) (as.numeric(x))) 
    k <- c(params[2])
    r <-c(params[1])
  
  print("param_extract")
  print(k)
  print(r)
  print(ncol(arr))
  params<- dgmm_param_extraction (arr, layers, k, r,it,eps,seed,init, init_est, classes)
  k1=k
  r1=r
  
  dgmm <- deepgmm(arr, layers=layers, k=k, r=r, eps = 0.001, init = init, seed = seed, scale = FALSE)
  l1=list(label=dgmm$s[,1],  k=params[[4]], all_w=params[[1]],all_mu=params[[2]],all_var=params[[3]])
  }
  else{
    l1=0
    k1=0
    r1=0
  }
  
  
  layers = 2
  
  
  if (layers==2 & res_search$aic[idx[2]] != "Inf")
  { 
    params<-sapply(strsplit(( res_search$aic[idx[2]+1]), split=", "), function(x) (as.numeric(x))) 
    k <- c(params[3],params[4])
    r <-c(params[1], params[2])
    k2=k
    r2=r
 
  params<- dgmm_param_extraction (arr, layers, k, r,it,eps,seed,init, init_est, classes)
  
  
  dgmm <- deepgmm(arr, layers=layers, k=k, r=r,
                   eps = 0.001, init = init, 
                  seed = seed, scale = FALSE)
  l2=list(label=dgmm$s[,1], k=params[[4]], all_w=params[[1]],all_mu=params[[2]],all_var=params[[3]])
  }
  else{
    l2=0
    k2=0
    r2=0
  }
  
  
  layers = 3
  
  
  if (layers==3& res_search$aic[idx[3]] != "Inf")
  { 
    params<-sapply(strsplit(( res_search$aic[idx[3]+1]), split=", "), function(x) (as.numeric(x)))
    print(length(params))
    if(length(params) == 6){
        k <- c(params[4],params[5], params[6])
        r <-c(params[1], params[2],params[3])
        k3=k
        r3=r
        params<- dgmm_param_extraction (arr, layers, k, r,it,eps,seed,init, init_est, classes)
  
 
        dgmm <- deepgmm(arr, layers=layers, k=k, r=r,
                   eps = eps, init = init, 
                  seed = seed, scale = FALSE)
        l3=list(label=dgmm$s[,1], k=params[[4]], all_w=params[[1]],all_mu=params[[2]],all_var=params[[3]])
    }
    else{
      l3=0
      k3=0
      r3=0
    }
  }
  else{
    l3=0
    k3=0
    r3=0
  }
  return(list(l1 = l1, l2=l2, l3=l3, k1=k1, r1=r1, k2=k2, r2=r2,  k3=k3, r3=r3))
}


gmm_labels <- function(arr, arr2, classes, it, eps, init, init_est, seed){
gmm <- Mclust(arr, control = emControl(itmax = it, eps = eps), G=classes,verbose=0)
train_cl <- gmm$classification
w <- gmm$parameters$pro
mu <- gmm$parameters$mean
sigma <- aperm(gmm$parameters$variance$sigma, c(3,1,2))

return(list (labels = train_cl, k=classes, all_w=w,all_mu=mu,all_var=sigma))
}


DGMMregr <- function( data, x, target_id, all_k, all_w, all_mu, all_var){  
  ##############################################################################
  #
  #    Gaussian Mixture (Classical or Deep Models) + Regression
  #
  ##############################################################################
  
    res_regr <- regr(data=data, x=x, target_id=target_id, k=all_k, w=all_w, mu=all_mu, sigma=all_var)
    res_mu <- all_mu
    res_sigma <- all_var
    
    
  return (list(regr=res_regr ,  mu=res_mu, sigma=res_sigma))
}


knn_labels <- function(arr, arr2, classes, it, eps, init, init_est, seed){
set.seed(seed) # Setting seed
res<-kmeans(arr, centers = classes, nstart = 30, iter.max = it, algorithm = "Hartigan-Wong")

train_cl <-res$cluster

return(train_cl)
}



SelfSupRegrDGMM <- function(X_train, X_test,  y_train,  type1 = 'DGMMr',  k, r, init, it){
  ##############################################################################
  #
  #        Self Supervised Method for Regression
  #        using combination of 
  #        Gaussian Mixtures (GMM)/Deep Gauusian Mixture Models(DGMM) + SVM/XGB        
  #
  ##############################################################################
  
  if (any(tolower(type1) %in% c('dgmm', 'deepgmm', 'deepgmmr', 'dgmmr')))
    type1 <- 'dgmmr'
  if (any(tolower(type1) %in% c('gmm','gmmr')))
    type1 <- 'gmmr'
  if (any(tolower(type1) %in% c('lin','linr')))
    type1 <- 'linr'
  
  
  if (type1 == 'dgmmr'){
    
    p <- r
    
    l <- length(k)
  } else {
    
    
    k <- k
    
   
  }
  if (type1 == 'dgmmr'| type1 == 'gmmr'){
  train <- cbind(X_train, y_train)
  
  dgmmr <- DGMMr(model = type1, train, X_test, dim(train)[2], k=k, p=r, init=init, it=it, cl = TRUE)
  test_r <- dgmmr$regr
  train_r <- dgmmr$res_True
  nans <- which(is.na(train_r),arr.ind=TRUE)
  nans <-nans[,1]
  
  if(length(nans) >0){
    train_r<-train_r[-nans,]
    X_train<-X_train[-nans,]
  }
  
  }
  else  {
    regressor = lm(formula = y_train ~ .,
                                   data = as.data.frame(X_train))
                   # 
                   # # Predicting the Test set results
    test_r = predict(regressor, newdata = as.data.frame(X_test))
    train_r <-predict(regressor, newdata = as.data.frame(X_train))
  }
  
  
 
  merged_train_x <- X_train#rbind(X_train, X_test)
  merged_train_y <- train_r#rbind(train_r, test_r)
  
  
  
  output <- list(X=merged_train_x, Y=merged_train_y, y_tr = train_r, nans = nans)
  return(output)
}

regressionSVM_XGB<-function(merged_train_x, merged_train_y, X_test, type2 ='SVM') {
 
  if (any(tolower(type2) %in% c('svm', 'ksvm')))
    type2 <- 'svm'
  if (any(tolower(type2) %in% c('xgb', 'xgboost')))
    type2 <- 'xgb'
  
  if (type2 =='svm'){
    svmmodel <- ksvm(merged_train_x, merged_train_y, kernel = 'rbfdot', C = 10) 
    model <- svmmodel
    
  } else {
    if (type2 != 'xgb')
    {
      stop("Supervised model has to be SVM or XGB")
    }
    
    xgb.train = xgb.DMatrix(data = merged_train_x, label = merged_train_y)
    # xgb.test = xgb.DMatrix(data = as.matrix(X_var), label = y_var)
    
    xgbr = xgboost(data = xgb.train,  nrounds = 50,verbose=0)
    model <- xgbr
  }
  
  test_predictions =predict(model, X_test) 
  return (test_predictions)
}


calc_SVM_XGB <- function(X, Y, arr2,res4, type, verb=TRUE, type2 ='SVM'){
  
  classifRes <-regressionSVM_XGB(X, Y, arr2, type2)
  esimates1 = estimateRMSE(classifRes, res4)
  if(verb==TRUE){
    print(paste("RMSE  ",type, "+", type2))
    print("RMSE:")  
    print(esimates1)  
  }
  
  
  return (esimates1)
}  


estimateRMSE<- function(result, actual){
  
  val_rmse  <- rmse(result, actual)
  
  return(val_rmse)
}



estimateF1 <- function(result, actual){
  cm <- as.matrix(confusionMatrix(result, actual))
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  rowsums = apply(cm, 1, sum)# number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  diag = diag(cm)  # number of correctly classified instances per class 
  
  precision = diag / (rowsums +0.0000001)
  recall = diag / (colsums +0.0000001) 
  f1 = 2 * precision * recall / (precision + recall+0.0000001) 
  
  #print(" ************ Confusion Matrix ************")
  #print(cm)
  #print(" ************ Diag ************")
  #print(diag)
  print(" ************ Precision/Recall/F1 per class************")
  print(data.frame(precision, recall, f1)) 
  #print(rowsums)
  #print(colsums)
  #print(rowsums==0)
  if(length(which(colsums==0)>0)) precision=precision[-which( colsums==0)]
  if(length(which(colsums==0)>0)) recall=recall[-which( colsums==0)]
  if(length(which(colsums==0)>0)) f1=f1[-which(colsums==0)]
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  
  print(" ************ Macro Precision/Recall/F1 ************")
  print(data.frame(macroPrecision, macroRecall, macroF1)) 
  return (list(mPr=macroPrecision, mR = macroRecall, mF = macroF1))
}
