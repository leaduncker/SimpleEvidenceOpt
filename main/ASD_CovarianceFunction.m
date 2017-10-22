function [C,Cinv] = ASD_CovarianceFunction(hprs,dims)
% hprs -- vector of hyperparameters 
% nkt  -- number of temporal strf coefs
% nkx  -- number of spatial strf coefs 
%               [nkx1 nkx2] for 2D [nkx1 1] or nkx1 for 1D
%

% ---- build covariance for spatial RF -------
if any(dims == 1) || (length(dims) == 1) % 1D space 
    % check that dimensionality is correct
    assert(length(hprs) == 1)
    
    % build covariance matrix
    C = exp(-1/(2*hprs(1)^2)*bsxfun(@minus,(1:max(dims))',(1:max(dims))).^2);
    
else % 2D space
    
    % build precision term of 2D kernel function
    if length(tauSpace) == 1 % isotropic
        Minv = 1/hprs(1)^2*eye(2);
    else % anisotropic
        
        if length(tauSpace) == 2 % axis-aligned
            Minv = [1/hprs(1)^2, 0; 0,1/hprs(2)^2];
        else % not axis aligned
            % make sure correlation coefficient is valid
            assert(abs(hprs(3)) < 1)
            Minv = 1/(1-hprs(3)^2)*...
                [1/hprs(1)^2, -hprs(3)/(hprs(1)*hprs(2)); ...
                -hprs(3)/(hprs(1)*hprs(2)),1/hprs(2)^2];
        end
    end
    % grid of spatial indices
    [xx1, xx2] = ndgrid(1:dims(1), 1:dims(2));
    xx = [xx1(:) xx2(:)];
    
    % pairwise differences weighted by entires in Minv matrix
    % (x_i-x_j)'Minv(x_i-x_j)
    sqdiffs = Minv(1,1)*bsxfun(@minus,xx(:,1),xx(:,1)').^2 + Minv(2,2)*bsxfun(@minus,xx(:,2),xx(:,2)').^2 ...
        - 2*Minv(2,1)*bsxfun(@minus,xx(:,1),xx(:,1)').*bsxfun(@minus,xx(:,2),xx(:,2)');
    
    % spatial covariance
    C = exp(-0.5*sqdiffs);
    
end

% add small jitter for numerical stability 
C = C + 1e-05*eye(prod(dims));

% exploit toeplitz structure to do inversion efficiently
% -------> TO DO: write a function that does this efficiently exploiting matrix
% toeplitz structure cf JPC's code for GPFA (mex function avail)
Cinv = inv(C);
