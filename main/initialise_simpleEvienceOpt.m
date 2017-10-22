function m = initialise_simpleEvienceOpt(X,Y,nkt,nkx,priorTime,priorSpace,rho0,hprsTime0,hprsSpace0,sigSq0,varargin)
% initialise_simpleEvienceOpt -- function to initialise model structure
% passed into simpleEvidenceOpt
%
% INPUT
% ------
% X           (NxD)      -- stimulus matrix
% Y           (Nx1)      -- response vector
% nkt         scalar     -- number of time lags to include
% nkx         vector     -- spatial dims in x and y 
% rho0        scalar     -- initial weighting/scaling of prior compared to
%                           likihood
% priorTime   'string'   -- {'Ridge','ASD','ALD','TRD'} temporal RF prior
% priorSpace  'string'   -- {'Ridge','ASD','ALD'} spatial RF prior
% hprsTime0   (nprsTx1)  -- vector with initial hyperparameters for priorTime
% hprsSpace0  (nprsTx1)  -- vector with initial hyperparameters for priorSpace
% sigSq0      scalar     -- initial parameter for sigma^2 noise variance
% opts        struct     -- optional structure containing option values with 
%                          fields: .maxiter
%                                  .stimcov = {'full','reduced'}
%                                  .verbose = {0,1}
%                                  .solver  = {'direct','iterative'}
%
%
% hyperparameters should be supplied in the following order:
%       RR :    hprs0 = []
%       ASD:    hprs0 = [kern_len]
%       ALD:    hprs0 = [kernX_len; meanx_len; kernF_len; meanF_len]
%       TRD:    hprs0 = [time_warp; kern_len]
%
% overall scaling parameter is treated extra
%
% OUTPUT
% ------
% m         --      model structure
%
%
% See also simpleEvidenceOpt
%
% Duncker, 2017
%
%

% save some input
hprs0 = [rho0;hprsTime0;hprsSpace0];
m.nhprsTime = length(hprsTime0);
m.nhprsSpace = length(hprsSpace0);
m.N = size(Y,1); % number of samples
m.nkt = nkt; % time lags
m.nkx = nkx; % vector with spatial dims
m.hprs0 = hprs0; % initial hyperparameter vector
m.sigSq0 = sigSq0; % initial noise variance
if nargin < 11
    m.options = setOptionValues_SimpleEvidenceOpt([]); % pass options onto strucutre
else
     m.options = setOptionValues_SimpleEvidenceOpt(varargin{1});
end
% compute sufficient statistics of data
switch m.options.stimcov
    case 'full'
        [XY,Xmu,XX,Ymu,YY] = CalculateMoments_Full(Y,X,nkt);
       
    case 'reduced'
        [XY,Xmu,XX,Ymu,YY] = CalculateMoments_Reduced(Y,X,nkt);
end

% reshape into needed format
XY = XY(:);
Xmu = Xmu(:);
XX = reshape(permute(XX,[1 3 2 4]),nkx*nkt,nkx*nkt);

% save into model structure
m.SufficientStats.XY = XY;
m.SufficientStats.Xmu = Xmu;
m.SufficientStats.XX = XX;
m.SufficientStats.Ymu = Ymu;
m.SufficientStats.YY = YY;

% create function handle to prior covariance taking hprs0 vector as input
ll = [zeros(size(hprs0));0];
uu = [Inf(size(hprs0));Inf];

switch priorTime
    
    case 'Ridge' % ridge regression

        m.PriorCovTime = @RidgeRegression_CovarianceFunction;
        m.dInvPriorCovTime = @Gradient_RidgeRegression_CovarianceFunction;
        
    case 'ASD' % automatic smoothness determination
        
        m.PriorCovTime = @ASD_CovarianceFunction;
        m.dInvPriorCovTime = @Gradient_ASD_CovarianceFunction;

    case 'ALD' % automatic locality determination
        
        m.PriorCovTime =  @ALD_CovarianceFunction;
        m.dInvPriorCovTime = @Gradient_ALD_CovarianceFunction;
                
        % adjust lower and upper limits for optimisation
        ll = [zeros(size(hprs0));0];
        uu = [Inf(size(hprs0));Inf];
        ll(2) = 0; uu(2) = Inf;
        ll(3) = 1; uu(3) = nkt;
        ll(4) = 0; uu(4) = Inf;
        ll(5) = -1; uu(4) = 0.5*nkt + 1;
        
    case 'TRD'
        m.PriorCovTime =  @TRD_CovarianceFunction;
        m.dInvPriorCovTime = @Gradient_TRD_CovarianceFunction;
        
        % adjust lower and upper limits for optimisation
        ll(2) = -Inf; uu(2) = Inf;
        ll(3) = 0; uu(3) = Inf;
end

switch priorSpace
    
    case 'Ridge' % ridge regression
        
        m.PriorCovSpace = @RidgeRegression_CovarianceFunction;
        m.dInvPriorCovSpace = @Gradient_RidgeRegression_CovarianceFunction;
       
        
    case 'ASD' % automatic smoothness determination
        
        m.PriorCovSpace = @ASD_CovarianceFunction;
        m.dInvPriorCovSpace = @Gradient_ASD_CovarianceFunction;


    case 'ALD' % automatic locality determination
        
        m.PriorCovSpace =  @ALD_CovarianceFunction;
        m.dInvPriorCovSpace = @Gradient_ALD_CovarianceFunction;
        idxSpace = m.nhprsTime+2:length(hprs0);
        
        if any(nkx == 1) || (length(nkx) == 1) % 1D space 
            ll(idxSpace(1)) = 0; uu(idxSpace(1)) = Inf;
            ll(idxSpace(2)) = 1; uu(idxSpace(2)) = max(nkx);
            ll(idxSpace(3)) = 0; uu(idxSpace(3)) = Inf;
            ll(idxSpace(4)) = -1;uu(idxSpace(4)) = 0.5*max(nkx) + 1;
        else % 2D space
            if length(hprsSpace0) == 6 % isotropic
                ll(idxSpace(1)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(2)) = 1; uu(idxSpace(2)) = nkx(1);
                ll(idxSpace(3)) = 1; uu(idxSpace(3)) = nkx(2);
                ll(idxSpace(4)) = 0; uu(idxSpace(4)) = Inf;
                ll(idxSpace(5)) = -1; uu(idxSpace(5)) = 0.5*nkx(2) + 1;
                ll(idxSpace(6)) = -1; uu(idxSpace(6)) = 0.5*nkx(2) + 1;
            elseif length(hprsSpace0) == 8 % anisotropic, axis aligned
                ll(idxSpace(1)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(2)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(3)) = 1; uu(idxSpace(1)) = nkx(1);
                ll(idxSpace(4)) = 1; uu(idxSpace(1)) = nkx(2);
                ll(idxSpace(5)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(6)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(7)) = -1; uu(idxSpace(1)) = 0.5*nkx(1) + 1;
                ll(idxSpace(8)) = -1; uu(idxSpace(1)) = 0.5*nkx(2) + 1;
            else % anisotropic, not axis aligned
                ll(idxSpace(1)) = 0; uu(idxSpace(1)) = Inf;
                ll(idxSpace(2)) = 0; uu(idxSpace(2)) = Inf;
                ll(idxSpace(3)) = -1; uu(idxSpace(3)) = 1;
                ll(idxSpace(4)) = 1; uu(idxSpace(4)) = nkx(1);
                ll(idxSpace(5)) = 1; uu(idxSpace(5)) = nkx(2);
                ll(idxSpace(6)) = 0; uu(idxSpace(6)) = Inf;
                ll(idxSpace(7)) = 0; uu(idxSpace(7)) = Inf;
                ll(idxSpace(8)) = 0; uu(idxSpace(8)) = Inf;
                ll(idxSpace(9)) = -1; uu(idxSpace(9)) = 0.5*nkx(1) + 1;
                ll(idxSpace(10)) = -1; uu(idxSpace(10)) = 0.5*nkx(2) + 1;
            end
        end
end
        
% set bound constraints 
m.constraints.ll = ll;
m.constraints.uu = uu;