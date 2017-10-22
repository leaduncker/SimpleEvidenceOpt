function [C,Cinv] = ALD_CovarianceFunction(hprs,dims)
% hprs -- vector of hyperparameters 
% dims -- receptive field dimensions

% ---- build covariance matrix -------
if any(dims == 1) || (length(dims) == 1) % 1D space 
    % check that dimensionality is correct, extract parameters
    assert(length(hprs) == 4)
    taux  = hprs(1); % is spatial domain width of spatial kernel
    meanx = hprs(2); % is spatial domain localisation of spatial kernel
    tauf  = hprs(3); % is frequency domain width of spatial kernel
    mean = hprs(4); % is frequency domain localisation of spatial kernel
    
    % build covariance matrix
    xx = 1:max(dims); xx = xx'; % column vector of temporal indices
    ncirc = max(dims); % just use input dimensions
    [Uf,freq] = realfftbasis(max(dims),ncirc); % DFT basis and associated frequencies
    
    % construct spatial domain based covariance matrix
    CxSqrt = diag(exp(-0.25/taux^2*(xx - meanx).^2)); %  diagonal spatial covariance element
    
    % construct frequency domain based covariance matrix
    Cf = Uf'*diag(exp(-0.5*(abs(tauf*freq)-mean).^2))*Uf;
    
    % construct full spatial RF covariance with Cx^1/2 * Cf * Cx^1/2 ALDsf structure
    C = CxSqrt*Cf*CxSqrt;
    
else % 2D space
    
    % build precision term of 2D kernel function
    if length(hprs) == 6 % isotropic in both Spat and Freq domains
        % extract parameters
        taux  = hprs(1);   % spatial domain width of spatial kernel
        meanx = hprs(2:3); % spatial domain localisation of spatial kernel
        tauf  = hprs(4);   % frequency domain width of spatial kernel
        mean = hprs(5:6); % frequency domain localisation of spatial kernel
        
        Mx = 1/taux^2*eye(2);
        Mf = tauf*eye(2);
        
    elseif length(hprs) == 8 % anisotropic, axis-aligned in both Spat and Freq domains
        % extract parameters
        taux  = hprs(1:2); % spatial domain width of spatial kernel
        meanx = hprs(3:4); % spatial domain localisation of spatial kernel
        tauf  = hprs(5:6); % frequency domain width of spatial kernel
        mean = hprs(7:8); % frequency domain localisation of spatial kernel
        
        Mx = [1/taux(1)^2, 0; 0,1/taux(2)^2];
        Mf = [tauf(1), 0; 0, tauf(2)];
        
    else % anisotropic, not axis aligned in both Spat and Freq domains
        assert(length(hprs) == 10)
        
        % extract parameters
        taux  = hprs(1:3); % spatial domain width of spatial kernel
        meanx = hprs(4:5); % spatial domain localisation of spatial kernel
        tauf  = hprs(6:8); % frequency domain width of spatial kernel
        mean =  hprs(9:10); % frequency domain localisation of spatial kernel
        
        Mx = 1/(1-taux(3)^2)*...
            [1/taux(1)^2, -taux(3)/(taux(1)*taux(2)); ...
            -taux(3)/(taux(1)*taux(2)),1/taux(2)^2];
        
        Mf = [tauf(1), tauf(3); tauf(3), tauf(2)];
        
        
    end
    
    % 2D vectors of spatial indices
    [xx1, xx2] = ndgrid(1:dims(1), 1:dims(2));
    xx = [xx1(:) xx2(:)];
    
    % build spatial covariance matrix
    ncirc1 = dims(1); % just use input dimensions
    ncirc2 = dims(2); % just use input dimensions
    
    [Uf1,freq1] = realfftbasis(nkx(1),ncirc1); % DFT basis and associated frequencies for first dim
    [Uf2,freq2] = realfftbasis(nkx(2),ncirc2); % DFT basis and associated frequencies for second dim
    Uf = kron(Uf1,Uf2); % full DFT basis
    % grid of spatial frequencies
    [ff1,ff2] = ndgrid(freq1,freq2);
    freq = [ff1(:) ff2(:)];
    
    % construct spatial domain based covariance matrix
    diffx = bsxfun(@minus,xx,meanx');
    CxSqrt = diag(exp(-0.25*sum((Mx*diffx').^2,1))); %  diagonal spatial covariance element
    
    % construct frequency domain based covariance matrix
    Cf = Uf'*diag(exp(-0.5*sum(bsxfun(@minus,abs(Mf*freq'),mean).^2,1)))*Uf;
    
    % construct full spatial RF covariance with Cx^1/2 * Cf * Cx^1/2 ALDsf structure
    C = CxSqrt*Cf*CxSqrt;
    
end


% add small jitter for numerical stability
C = C + 1e-05*eye(prod(dims));

% inverses we need
Cinv = inv(C);