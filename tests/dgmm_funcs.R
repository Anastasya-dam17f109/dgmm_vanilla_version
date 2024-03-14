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
  bp <- barplot(ccc$freq, names.arg=ccc$macro.area)
  text(bp, 0, ccc$freq,cex=1,pos=3) 
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
    new[names(new) != cl] <- apply(new[names(new) != cl], MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
  }
  return(new)
}


dgmm_grid_search <- function(data0, k, type='BIC', seed = 1){
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
    p2const <- 2
    p3const <- 1
  }
  else{
    p1const <- round(r/2) 
    p2const <- min(round(r/3),p1const-1) 
    p3const <- max(1,min(round(r/4),p2const-1))
  }
  
  r1 <- cbind(r1 = 1:p1const)
  # print(r1)
  
  r1r2k2 <- crossing(r1 = 1:p1const, r2 = 1:p2const, k2 = 1:(k+1))
  r1r2k2 <- as.matrix(r1r2k2[(r1r2k2[, 1]) > (r1r2k2[, 2]), ])
  
  r1r2r3k2k3 <- crossing(r1 = 1:p1const, r2 = 1:p2const, r3 = 1:p3const, k2 = 1:(k+1), k3 = 1:(k+1))
  r1r2r3k2k3 <- as.matrix(r1r2r3k2k3[((r1r2r3k2k3[, 1]) > (r1r2r3k2k3[, 2])) & ((r1r2r3k2k3[, 2]) > (r1r2r3k2k3[, 3])), ])
  
  # print(c('!!!!',nrow(r1)))
  for (i in 1:nrow(r1)){
    # print(i)
    p1 <- r1[i,1]
    dgmm1 <- deepgmm(data0, layers=1, k=k, r=p1, seed = seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm1$bic < bic1[1]) bic1 <- c(dgmm1$bic, toString(c(p1,k))) 
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm1$aic < aic1[1]) aic1 <- c(dgmm1$aic, toString(c(p1,k))) 
    # print(c(seed, p1, toString(c(p1,k))))
  }
  cat('1 слой -- done',nrow(r1) ,'\n')
  
  for (i in 1:nrow(r1r2k2)){
    
    p1p2 <- c(r1r2k2[i,1],r1r2k2[i,2])
    k1k2 <- c(k,r1r2k2[i,3])
    dgmm2 <- deepgmm(data0, layers=2, k=k1k2, r=p1p2, seed=seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm2$bic < bic2[1]) bic2 <- c(dgmm2$bic, toString(c(p1p2,k1k2)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm2$aic < aic2[1]) aic2 <- c(dgmm2$aic, toString(c(p1p2,k1k2)))
    
  }
  cat('2 слоя -- done',nrow(r1r2k2) ,'\n')
  
  for (i in 1:nrow(r1r2r3k2k3)){
    
    p1p2p3 <- c(r1r2r3k2k3[i,1],r1r2r3k2k3[i,2],r1r2r3k2k3[i,3])
    k1k2k3 <- c(k,r1r2r3k2k3[i,4],r1r2r3k2k3[i,5])
    dgmm3 <- deepgmm(data0, layers=3, k=k1k2k3, r=p1p2p3, seed=seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm3$bic < bic3[1]) bic3 <- c(dgmm3$bic, toString(c(p1p2p3,k1k2k3)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm3$aic < aic3[1]) aic3 <- c(dgmm3$aic, toString(c(p1p2p3,k1k2k3)))
    
  }
  cat('3 слоя -- done',nrow(r1r2r3k2k3) ,'\n')
  
  if (type == 'bic' | type == 'bic&aic')
    print(cbind(bic1,bic2,bic3))
  if (type == 'aic' | type == 'bic&aic')
    print(cbind(aic1,aic2,aic3)) 
  return(list("bic" = cbind(bic1,bic2,bic3), "aic" = cbind(aic1,aic2,aic3)))
  
}

# ----------------------------------------------------------------------------------

dgmm_rand_search <- function(data0, k, ratio = 0.5, type='bic&aic', seed = 1){
  ##############################################################################
  #
  #                Random  Search
  #
  ##############################################################################
  type <- tolower(type)
  if (type %in% c('aic&bic','aicbic','bicaic','aic_bic', 'bic_aic','aic/bic', 'bic/aic'))
    type <- 'bic&aic'
  
  p <- dim(data0)[2]
  
  bic1 <- c(Inf,0,NaN)
  aic1 <- c(Inf,0,NaN) 
  bic2 <- c(Inf,0,NaN)
  aic2 <- c(Inf,0,NaN)
  bic3 <- c(Inf,0,NaN)
  aic3 <- c(Inf,0,NaN)
  
  if (p==4){
    p1const <- 3
    p2const <- 2
    p3const <- 1
  }
  else{
    p1const <- round(p/2) 
    p2const <- min(round(p/3),p1const-1) 
    p3const <- max(1,min(round(p/4),p2const-1))
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
    dgmm1 <- deepgmm(data0, layers=1, k=k, r=p1, seed = seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm1$bic < bic1[1]) bic1 <- c(dgmm1$bic, toString(c(p1,k))) 
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm1$aic < aic1[1]) aic1 <- c(dgmm1$aic, toString(c(p1,k))) 
    # print(c(seed, p1, toString(c(p1,k))))
  }
  cat('1 слой -- done',nrow(r1) ,'\n')
  
  for (i in 1:nrow(r1r2k2)){
    
    p1p2 <- c(r1r2k2[i,1],r1r2k2[i,2])
    k1k2 <- c(k,r1r2k2[i,3])
    dgmm2 <- deepgmm(data0, layers=2, k=k1k2, r=p1p2, seed=seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm2$bic < bic2[1]) bic2 <- c(dgmm2$bic, toString(c(p1p2,k1k2)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm2$aic < aic2[1]) aic2 <- c(dgmm2$aic, toString(c(p1p2,k1k2)))
    
  }
  cat('2 слоя -- done',nrow(r1r2k2) ,'\n')
  
  for (i in 1:nrow(r1r2r3k2k3)){
    
    p1p2p3 <- c(r1r2r3k2k3[i,1],r1r2r3k2k3[i,2],r1r2r3k2k3[i,3])
    k1k2k3 <- c(k,r1r2r3k2k3[i,4],r1r2r3k2k3[i,5])
    dgmm3 <- deepgmm(data0, layers=3, k=k1k2k3, r=p1p2p3, seed=seed)
    if (type == 'bic' | type == 'bic&aic')
      if (dgmm3$bic < bic3[1]) bic3 <- c(dgmm3$bic, toString(c(p1p2p3,k1k2k3)))
    if (type == 'aic' | type == 'bic&aic')
      if (dgmm3$aic < aic3[1]) aic3 <- c(dgmm3$aic, toString(c(p1p2p3,k1k2k3)))
    
  }
  cat('3 слоя -- done',nrow(r1r2r3k2k3) ,'\n')
  
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
  p <- dim(data)[2]  
  n <- dim(x)[1] 
  some_small_value <- 2.220446049250313e-16
  
  indices <- 1:p 
  indep_id <- indices[-target_id]
  indep_id
  target_id
  X <- data[,indep_id ] 
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
    
    x_centered <- x - t(array(mean,c(p-1,n)))
    x_norm <- t(forwardsolve(C, t(x_centered)))
    
    coeffs[i]  <- 1/(2*pi)**(0.5*length(indep_id))/C_det
    exps[,i] <- -0.5*rowSums(x_norm**2)
    
    XX[,i,] <- array(x, c(n,1,p-1))  
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

DGMMr <- function(model = 'DGMMr', data, params, x, target_id, k, cl = FALSE){  
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
    if (!is.null(params)) {
      if (length(params$k) == length(params$p)){
        l <- length(params$k)
        dgmm <- deepgmm(data, layers=l, k=params$k, r=params$p,
                        it = 250, eps = 0.001, init = "kmeans", init_est = "factanal",
                        seed = NULL, scale = FALSE)
        if (cl){
          res_cl <- dgmm$s[,1]
        }
      }
      else{
        stop("Length of k and p must be the same")
      }
    }
    else{
      "Use grid or random search to choose optimal parameters"
    } 
    
    p <- dim(data)[2]
    n <- dim(data)[1]
    
    w = dgmm$w
    mu = dgmm$mu
    Lambda = dgmm$H
    Psi = dgmm$psi
    
    layers <- l
    k<- params$k
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
    
    res_regr <- regr(data=data, x=x, target_id=target_id, k=tot.k, w=all_w, mu=all_mu, sigma=all_var)
    res_mu <- all_mu
    res_sigma <- all_var
    
  } else {
    if (model != "gmmr")
      stop("Model has to be DGMMr or GMMr")
    
    gmm <- Mclust(data, control = emControl(itmax = 250, eps = 0.001), G=k) 
    
    if (cl){
      res_cl <- gmm$classification
    }
    
    w <- gmm$parameters$pro
    mu <- gmm$parameters$mean
    res_mu <- mu
    sigma <- aperm(gmm$parameters$variance$sigma, c(3,1,2))
    res_sigma <- sigma
    res_regr <- regr(data=data, x=x, target_id=target_id, k=k, w=w, mu=mu, sigma=sigma)
  }
  
  return (list(regr=res_regr , cl=res_cl, mu=res_mu, sigma=res_sigma))
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

