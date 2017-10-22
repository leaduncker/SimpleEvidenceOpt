function dCinv = Gradient_ASD_CovarianceFunction(hprs,dims)
% gradient of prior inverse covariance matrix

% build prior covariance matrix
C = ASD_CovarianceFunction(hprs,dims); % C = rho*kron(Cspace,Ctime); 
dCinv = zeros(prod(dims),prod(dims),length(hprs));

% dCinv/dtau

if any(dims == 1) || (length(dims) == 1) % 1D space 
    % check that dimensionality is correct
    assert(length(hprs) == 1)
    % gradient
    dC = 1/hprs^3*bsxfun(@minus,(1:max(dims))',(1:max(dims))).^2.*C;
    dCinv(:,:,1) = - C\(dC/C);
    
else % 2D space
    
    % get grid of spatial indices
    [xx1, xx2] = ndgrid(1:dims(1), 1:dims(2));
    xx = [xx1(:) xx2(:)];
    
    % build precision term of 2D kernel function
    if length(hprs) == 1 % isotropic
        
        dC = 1/(hprs^3)*(bsxfun(@minus,xx(:,1),xx(:,1)').^2 + bsxfun(@minus,xx(:,2),xx(:,2)').^2).*C;
        dCinv(:,:,1) = -C\(dC/C);
        
    else % anisotropic
        
        if length(hprs) == 2 % axis-aligned
            
            dC1 = 1/(hprs(1)^3)*bsxfun(@minus,xx(:,1),xx(:,1)').^2.*C;
            dC2 = 1/(hprs(2)^3)*bsxfun(@minus,xx(:,2),xx(:,2)').^2.*C;
            
            dCinv(:,:,1) = -C\(dC1/C);
            dCinv(:,:,2) = -C\(dC2/C);
            
        else % not axis aligned
            
            % make sure correlation coefficient is valid
            assert(abs(hprs(3)) < 1)
            
            % lengthscale terms
            dC1 = 0.5*(-2/(hprs(1)^3*(1-hprs(3)^2))*bsxfun(@minus,xx(:,1),xx(:,1)').^2 ...
               + 2*(hprs(3)/((1-hprs(3)^2)*hprs(2)*hprs(1)^2))*bsxfun(@minus,xx(:,1),xx(:,1)').*bsxfun(@minus,xx(:,2),xx(:,2)')).*C;
            
            dC2 = 0.5*(-2/(hprs(2)^3*(1-hprs(3)^2))*bsxfun(@minus,xx(:,2),xx(:,2)').^2 ...
               + 2*(hprs(3)/((1-hprs(3)^2)*hprs(1)*hprs(2)^2))*bsxfun(@minus,xx(:,1),xx(:,1)').*bsxfun(@minus,xx(:,2),xx(:,2)')).*C;
            
            % correlation term
            dC3 = 0.5*((2*hprs(3)/((1-hprs(3)^2))^2*hprs(1)^2)*bsxfun(@minus,xx(:,1),xx(:,1)').^2 ...
                + (2*hprs(3)/((1-hprs(3)^2))^2*hprs(2)^2)*bsxfun(@minus,xx(:,2),xx(:,2)').^2 ...
                - 2*((1+hprs(3)^2)/(1-hprs(3)^2)^2*hprs(1)*hprs(2))*bsxfun(@minus,xx(:,1),xx(:,1)').*bsxfun(@minus,xx(:,2),xx(:,2)')).*C;
            
            dCinv(:,:,1) = -C\(dC1/C);
            dCinv(:,:,2) = -C\(dC2/C);
            dCinv(:,:,3) = -C\(dC3/C);

        end
    end
end
