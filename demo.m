%% script for running rank 2 simulated example

%% generate example filters

%%%%%%% set filter sizes, bin width, etc. %%%%%%%%%%%%%%%%%%%%

dtStim = 1/60;  % Bin size for stimulus. 
dtSp = 1/60;  % Bin size for simulating model & computing likelihood (must evenly divide dtStim);

tmax = 30*dtSp; % maximum extent in time
nkt = tmax(1)/dtStim; % number of spatial elements
nkx = 40;  % number of spatial elements

tt = -tmax(1):dtSp:-dtSp;

%%%%%%%%% make temporal filter %%%%%%%%%%%%%%%%%%%%
 
% Temporal filter 1
kt1 = gampdf(-tt,4,.025)'; kt1 = kt1/norm(kt1); % 1st temporal filter
kt2 = gampdf(-tt,6,.03)'; kt2 = -kt2./norm(kt2); % 2nd temporal filter

kt1 = kt1./norm(kt1);
kt2 = kt2./norm(kt2);

kt = [kt1 kt2]; % concatenate RF into matrix

% set offset
dc = 0; 

% concatenate RF into matrix
kt = [kt1 kt2];

%%%%%%%%% make spatial filter %%%%%%%%%%%%%%%%%%%%

xx = linspace(-2,2,nkx)';
% Gabor RFs
kx1 = cos(2*pi*xx/2 + pi/5).*exp(-1/(2*0.35^2)*xx.^2);
kx2 = sin(2*pi*xx/2 + pi/5).*exp(-1/(2*0.35^2)*xx.^2);
kx1 = kx1./norm(kx1);
kx2 = kx2./norm(kx2);
% concatenate
kx = [kx1 kx2];

% plot rank-2 STRF example
k = kt*kx';

% orthogonalise filters
[uu,ss,vv] = svd(k);
ktOrth = uu(:,1:2)*sqrt(ss(1:2,1:2));
kxOrth = vv(:,1:2)*sqrt(ss(1:2,1:2));
%%
clims = [min((k(:))) max(abs(k(:)))];
figure;
subplot(2,4,1);imagesc(k,clims);colormap(hot);title('true STRF');
xlabel('space');ylabel('time'); set(gca, 'XTick', [],'YTick', []);
%% generate data
N = 5000; % sample size
signse = 1; % noise stdev 
Sigma = toeplitz(exp(-(0:nkx-1)/(nkx/6))); % correlated (AR1) stim covariance

mu = zeros(nkx,1); % stimulus mean
Stim = mvnrnd(mu,Sigma,N); % generate white noise stimulus
filterResp = sameconv(Stim,k) + dc;
Y = (filterResp) + signse*randn(N,1);
%% compute STA
rnk = 2; % rank 2
kSTA = simpleRevcorr(Stim,Y,nkt);
kSTA = kSTA./norm(kSTA); % normalize
[uu,ss,vv] = svd(kSTA);
ktEstOrthSTA = uu(:,1:rnk)*sqrt(ss(1:rnk,1:rnk));
kxEstOrthSTA = vv(:,1:rnk)*sqrt(ss(1:rnk,1:rnk));
subplot(2,4,2);imagesc(kSTA,clims);colormap(hot);title('STA');
set(gca, 'XTick', [],'YTick', []);

%% compute MAP estimates
optsMAP.stimcov = 'full';
optsMAP.maxiter = 100;
optsMAP.verbose = 1;
sigSq0 = 1;
%% run Ridge Regression
rho0 = 1;
priorTime = 'Ridge';
priorSpace = 'Ridge';
mRR = initialise_simpleEvienceOpt(Stim,Y,nkt,nkx,priorTime,priorSpace,rho0,[],[],sigSq0,optsMAP);
mRR = simpleEvidenceOpt(mRR);
kRR = reshape(mRR.kMAP,nkt,nkx);
subplot(2,4,3);imagesc(kRR,clims);title('RR');set(gca, 'XTick', [],'YTick', []);

%% run ASD in space, ASD in time
rho0 = 1;
hprs0Space = 1; 
hprs0Time = 1;
priorTime = 'ASD';
priorSpace = 'ASD';
mASD = initialise_simpleEvienceOpt(Stim,Y,nkt,nkx,priorTime,priorSpace,rho0,hprs0Time,hprs0Space,sigSq0,optsMAP);
mASD = simpleEvidenceOpt(mASD);
kASD = reshape(mASD.kMAP,nkt,nkx);
subplot(2,4,4);imagesc(kASD,clims);title('AS-SD');set(gca, 'XTick', [],'YTick', []);

%% run ALD in space, ALD in time
rho0 = 1;
hprs0Space = [1;20;1;0]; 
hprs0Time = [1;25;1;0];
priorTime = 'ALD';
priorSpace = 'ALD';
mALD = initialise_simpleEvienceOpt(Stim,Y,nkt,nkx,priorTime,priorSpace,rho0,hprs0Time,hprs0Space,sigSq0,optsMAP);
mALD = simpleEvidenceOpt(mALD);
kALD = reshape(mALD.kMAP,nkt,nkx);
subplot(2,4,6);imagesc(kALD,clims);title('AL-LD');set(gca, 'XTick', [],'YTick', []);
%% run ALD in space, ASD in time
rho0 = 1;
hprs0Space = [1;20;1;0]; 
hprs0Time = 1;
priorTime = 'ASD';
priorSpace = 'ALD';
mASALD = initialise_simpleEvienceOpt(Stim,Y,nkt,nkx,priorTime,priorSpace,rho0,hprs0Time,hprs0Space,sigSq0,optsMAP);
mASALD = simpleEvidenceOpt(mASALD);
kASALD = reshape(mASALD.kMAP,nkt,nkx);
subplot(2,4,7);imagesc(kASALD,clims);title('AS-ALD');set(gca, 'XTick', [],'YTick', []);
%% run ALD in space, TRD in time 
rho0 = 1;
hprs0Space = [1;20;1;0]; 
hprs0Time = [1;1];
priorTime = 'TRD';
priorSpace = 'ALD';
mTRALD = initialise_simpleEvienceOpt(Stim,Y,nkt,nkx,priorTime,priorSpace,rho0,hprs0Time,hprs0Space,sigSq0,optsMAP);
mTRALD = simpleEvidenceOpt(mTRALD);
kTRALD = reshape(mTRALD.kMAP,nkt,nkx);
subplot(2,4,8);imagesc(kTRALD,clims);title('TR-ALD');set(gca, 'XTick', [],'YTick', []);

