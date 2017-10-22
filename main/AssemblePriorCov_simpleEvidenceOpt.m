function [C,Cinv,dCinv] = AssemblePriorCov_simpleEvidenceOpt(hprs,m);

%  ------ extract parameters -----

rho = hprs(1);
if length(hprs) > 1
    hprsTime = hprs(2:m.nhprsTime+1);
    hprsSpace =  hprs(m.nhprsTime + 2:end);
else
    hprsTime = [];
    hprsSpace = [];
end
%  ------ assemble covariances -----

% get temporal covariance and derivatives of inverse
[Ctime,invCtime] = m.PriorCovTime(hprsTime,m.nkt);
dCinvTime = m.dInvPriorCovTime(hprsTime,m.nkt);

% get spatial covariance and derivatives of inverse
[Cspace,invCspace] = m.PriorCovSpace(hprsSpace,m.nkx);
dCinvSpace = m.dInvPriorCovSpace(hprsSpace,m.nkx);

% overall prior covariance
C = rho*kron(Cspace,Ctime);
% overall prior inverse covariance
Cinv = 1/rho*kron(invCspace,invCtime);

%  ------ assemble gradients ------
if nargout > 2
    dCinv = zeros(m.nkt*prod(m.nkx),m.nkt*prod(m.nkx),length(hprs));
    
    % dCinv/drho
    dCinv(:,:,1) = - 1/rho*Cinv;
    
    if length(hprs) > 1 % not ridge regression
        % dCinv/dhprsTime
        idxTime = 2:m.nhprsTime+1;
        for ii = 1:length(hprsTime)
            dCinv(:,:,idxTime(ii)) = 1/rho*kron(invCspace,dCinvTime(:,:,ii));
        end
        
        % dCinv/dhprsSpace
        idxSpace = m.nhprsTime + 2:length(hprs);
        for jj = 1:length(hprsSpace)
            dCinv(:,:,idxSpace(jj)) = 1/rho*kron(dCinvSpace(:,:,jj),invCtime);
        end
    end
end