function dCinv = Gradient_ALD_CovarianceFunction(hprs,dims);

% build prior covariance matrix
C = ALD_CovarianceFunction(hprs,dims);

% dCinv/dhprs

if any(dims == 1) || (length(dims) == 1) % isotropic in both Spat and Freq domains
    assert(length(hprs) == 4)
    % extract parameters
    
    % check that dimensionality is correct, extract parameters
    taux  = hprs(1); % is spatial domain width of spatial kernel
    meanx = hprs(2); % is spatial domain localisation of spatial kernel
    tauf  = hprs(3); % is frequency domain width of spatial kernel
    meanf = hprs(4); % is frequency domain localisation of spatial kernel
    
    % build covariance matrix
    xx = 1:max(dims); xx = xx'; % column vector of temporal indices
    ncirc = max(dims); % just use input dimensions
    [Uf,freq] = realfftbasis(max(dims),ncirc); % DFT basis and associated frequencies
    
    % construct spatial domain based covariance matrix
    CxSqrt = diag(exp(-0.25/taux^2*(xx - meanx).^2)); %  diagonal spatial covariance element
    
    % construct frequency domain based covariance matrix
    Cf = Uf'*diag(exp(-0.5*(abs(tauf*freq)-meanf).^2))*Uf;
    
    % gradient
    % CxSqrtTime/dtauxTime
    dCxSqrt1 = diag(0.5/taux^3*(xx - meanx).^2).*CxSqrt;
    
    % CxSqrtTime/dmeanxTime
    dCxSqrt2 = diag(0.5/taux^2*(xx - meanx)).*CxSqrt;
    
    % construct frequency domain based covariance matrix
    dCf3 = Uf'*diag(-(abs(tauf*freq)-meanf).*sign(tauf*freq).*freq)*diag(exp(-0.5*(abs(tauf*freq)-meanf).^2))*Uf;
    dCf4 = Uf'*diag((abs(tauf*freq)-meanf))*diag(exp(-0.5*(abs(tauf*freq)-meanf).^2))*Uf;
    
    % compute gradients of temporal covariance matrix for each temporal parameter
    dC1 = 2*dCxSqrt1*Cf*CxSqrt;
    dC2 = 2*dCxSqrt2*Cf*CxSqrt;
    dC3 = CxSqrt*dCf3*CxSqrt;
    dC4 = CxSqrt*dCf4*CxSqrt;
        
    % dCinv/dtaux
    dCinv(:,:,1) = - C\(dC1/C);
    % dCinv/dmeanx
    dCinv(:,:,2) = - C\(dC2/C);
    % dCinv/dtauf
    dCinv(:,:,3) = - C\(dC3/C);
    % dCinv/dmeanf
    dCinv(:,:,4) = - C\(dC4/C);
    
else % 2D space
    error('2D space not implemented yet')
    % get grid of spatial indices
    [xx1, xx2] = ndgrid(1:dims(1), 1:dims(2));
    xx = [xx1(:) xx2(:)];
    
    % build precision term of 2D kernel function
    if length(tauSpace) == 1 % isotropic
        
        dC = 1/(hprs^3)*(bsxfun(@minus,xx(:,1),xx(:,1)').^2 + bsxfun(@minus,xx(:,2),xx(:,2)').^2).*C;
        dCinv(:,:,1) = -C\(dC/C);
        
    else % anisotropic
        
        if length(tauSpace) == 2 % axis-aligned
            
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

            dCinv(:,:,3) = -C\(dC1/C);
            dCinv(:,:,4) = -C\(dC2/C);
            dCinv(:,:,5) = -C\(dC3/C);
        end
    end
end
