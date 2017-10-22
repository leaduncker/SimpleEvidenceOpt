function [C,Cinv] = TRD_CovarianceFunction(hprs,dims)
% hprs -- vector of hyperparameters 
% nkt  -- number of temporal strf coefs
% nkx  -- number of spatial strf coefs 
%               [nkx1 nkx2] for 2D [nkx1 1] or nkx1 for 1D
%

% ---- build covariance for spatial RF -------
% check that dimensionality is correct
assert(length(hprs) == 2) % correct number of hyperparameters
assert(any(dims == 1) || length(dims) == 1) % correct number of dimensions, 1D only

% rescale time
tt = flipud((1:max(dims))');
ttwarp = max(dims)./log(1+exp(hprs(1))*max(dims))*log(1 + exp(hprs(1))*tt);
% build covariance matrix
C = exp(-1/(2*hprs(2)^2)*bsxfun(@minus,ttwarp,ttwarp').^2);

% add small jitter for numerical stability
C = C + 1e-05*eye(prod(dims));

% exploit toeplitz structure to do inversion efficiently
% -------> TO DO: write a function that does this efficiently exploiting matrix
% toeplitz structure cf JPC's code for GPFA (mex function avail)
Cinv = inv(C);
