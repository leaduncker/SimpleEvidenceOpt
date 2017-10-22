function m = simpleEvidenceOpt(m)
% SIMPLEEVIDENCEOPT -- function for evidence optimisation in hierarchical
% linear regression models for spatiotemporal receptive fields
%
% Nx1 response vector Y linearly depends on NxD design matrix X and Dx1
% weights k, distributed as
%
% k ~ Normal( 0, C(theta) )
%
% theta are hyperparameters which this function learns automatically
%
% Y ~ Normal( X k , sigma^2 * eye(D))
%
% sigma^2 is the noise variance
%
% m is a model structure initialised using initialise_simpleEvienceOpt
%
% See also initialise_simpleEvienceOpt
%
% Duncker, 2017
%
%
t_start = tic;

if m.options.verbose
    optimOpts = optimoptions('fmincon', 'display', 'iter', 'maxiter', m.options.maxiter,'Gradobj','on'); 
else
    optimOpts = optimoptions('fmincon', 'display', 'none', 'maxiter', m.options.maxiter,'Gradobj','on'); 
end

% function handle for optimisation
fun = @(prs) fminconWrapper_simpleEvidenceOpt(prs,m);

% extract constraints
ll = m.constraints.ll; % lower bound
uu = m.constraints.uu; % upper bound
prs0 = [m.hprs0;m.sigSq0]; % initial values

% run optimisation
fprintf('running evidence optimisation ...\n')
prs = fmincon(fun,prs0,[],[],[],[],ll,uu,[],optimOpts);

% extract estimated parameters
m.hprs = prs(1:end-1);
m.sigSq = prs(end);

% final MAP estimate
fprintf('computing final MAP estimate....')

[~,Cinv] = AssemblePriorCov_simpleEvidenceOpt(m.hprs,m);

switch m.options.solver 
    case 'direct' % solve linear system directly using matlab's linsolve
        kMAP = (m.SufficientStats.XX + Cinv)\m.SufficientStats.XY;
    case 'iterative' % solve linear system iteratively using matlab's pgc
        kMAP = pcg(m.SufficientStats.XX + Cinv,m.SufficientStats.XY);
end
% save estimate in model structure
m.kMAP = kMAP;
fprintf('Done!\n')

% save and report elapsed time
m.runTime = toc(t_start);
toc(t_start);