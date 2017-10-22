function dCinv = Gradient_RidgeRegression_CovarianceFunction(lambda,dims);

dCinv = zeros(prod(dims)); % ridge parameter is taken care of in scaling parameter rho