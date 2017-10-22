function [C,Cinv] = RidgeRegression_CovarianceFunction(lambda,dim);

C =  eye(prod(dim));
Cinv = eye(prod(dim));