function [logEv,grad] = fminconWrapper_simpleEvidenceOpt(prs,m);

hprs = prs(1:end-1);
sigSq = prs(end);

% make cost function
[C,Cinv,dCinv] = AssemblePriorCov_simpleEvidenceOpt(hprs,m);

% calculate posterior mean and covariance 
switch m.options.solver 
    case 'direct' % solve linear system directly using matlab's linsolve
        mm = (m.SufficientStats.XX + Cinv)\m.SufficientStats.XY; 
        SS = inv(1/sigSq * m.SufficientStats.XX + Cinv); 
    case 'iterative' % solve linear system iteratively using matlab's pgc
        mm = pcg(m.SufficientStats.XX + Cinv,m.SufficientStats.XY);
        SS = block_pcg(1/sigSq * m.SufficientStats.XX + Cinv,speye(size(C)));
end

% ------ compute the log evidence after marginalising over strf ----
logEv = 0.5*logdet(C) + 0.5*m.N*logdet(sigSq) - 0.5*logdet(SS)...
    + 0.5/sigSq*m.SufficientStats.YY ...
    - 0.5/sigSq^2 * m.SufficientStats.XY'*SS*m.SufficientStats.XY;

% ------ compute the gradient of the log evidence -----
grad = zeros(size(prs));

% hyperparameter gradients
grad(1:end-1,:) = (-1/2*vec(C - SS - mm*mm')'*reshape(dCinv,[],length(hprs)))';

% gradient wrt sigma
grad(end,:) = -0.5/sigSq*(-m.N + sum(diag(eye(size(SS))- SS/C)) + ...
    1/sigSq * (m.SufficientStats.YY - 2*mm'*m.SufficientStats.XY ...
    + mm'*m.SufficientStats.XX*mm));