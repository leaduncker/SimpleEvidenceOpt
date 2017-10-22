function dCinv = Gradient_TRD_CovarianceFunction(hprs,dims)
% gradient of prior inverse covariance matrix

% build prior covariance matrix
C = TRD_CovarianceFunction(hprs,dims); % C = rho*kron(Cspace,Ctime); 
dCinv = zeros(prod(dims),prod(dims),length(hprs));

% rescale time
tt = flipud((1:max(dims))');
ttwarp = max(dims)./log(1+exp(hprs(1))*max(dims))*log(1 + exp(hprs(1))*tt);

% derivative of warping function
dttwarp = max(dims)*tt*exp(hprs(1))./((tt*exp(hprs(1)) + 1)*log(1+exp(hprs(1))*max(dims))) ...
    - max(dims)^2*exp(hprs(1))*log(1 + exp(hprs(1)).*tt)...
    ./((max(dims)*exp(hprs(1)) + 1).*(log(max(dims)*exp(hprs(1)) + 1))^2);

% gradients of inverse covariance wrt hyperparameters
dC1 = -1/(hprs(2)^2) *bsxfun(@minus,ttwarp,ttwarp').*bsxfun(@minus,dttwarp,dttwarp').*C; % warping coefficient
dC2 = 1/hprs(2)^3*bsxfun(@minus,ttwarp,ttwarp').^2.*C; % lengthscale

dCinv(:,:,1) = - C\(dC1/C);
dCinv(:,:,2) = - C\(dC2/C);


