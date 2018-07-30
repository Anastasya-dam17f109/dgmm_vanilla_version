#rm(list=ls())
require(testthat)
library(deepgmm)
y <- mtcars
layers <- 2
k <- c(3, 4)
r <- c(3, 2)
it <- 50
eps <- 0.001
seed <- 1
init <- "random"

set.seed(seed)
y <- scale(y)
model <- deepgmm(y = y, layers = layers, k = k, r = r,
                  it = it, eps = eps, init = init)

expect_that(model, is_a("dgmm"))
#expect_that(model, is_a("emmix"))
expect_named(model, c("H", "w", "mu", "psi", "lik", "bic",
	                    "aic", "clc", "s", "icl.bic", "h",
                      "k", "r", "numobs", "layers",  "call"))

n <- nrow(y)
p <- ncol(y)

expect_that(layers, equals(model$layers))
expect_that(n, equals(model$numobs))
expect_that(k, equals(model$k))
expect_that(r, equals(model$r))

rp <- c(p, r)

for (j in 1 : layers) {
  expect_length(model$s[, j], n)
  expect_length(model$w[[j]], k[j])

  expect_equal(ncol(model$mu[[j]]), k[j])
  expect_equal(nrow(model$mu[[j]]), rp[j])

  for (i in 1 : model$k[j]) {
    expect_equal(nrow(cbind(model$H[[j]][i,,, drop = TRUE])), rp[j])
    expect_equal(ncol(cbind(model$H[[j]][i,,, drop = TRUE])), rp[j + 1])

    expect_equal(nrow(cbind(model$psi[[j]][i,,, drop = TRUE])), rp[j])
    expect_equal(ncol(cbind(model$psi[[j]][i,,, drop = TRUE])), rp[j])
  }
}

test_that('data types correct', {
    expect_is(model$lik,'numeric')
    expect_is(model$bic,'numeric')
    expect_is(model$aic,'numeric')
    expect_is(model$clc,'numeric')
})