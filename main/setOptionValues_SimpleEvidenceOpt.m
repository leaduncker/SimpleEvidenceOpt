function options = setOptionValues_SimpleEvidenceOpt(opts);

% set default values if options aren't supplied

% --- calculation of stimulus covariance moments ---
if ~isfield(opts,'stimcov')
    options.stimcov = 'full';
else
    options.stimcov = opts.stimcov;
end

% --- maximum number of iterations of gradient descent ---
if ~isfield(opts,'maxiter')
    options.maxiter = 100;
else
    options.maxiter = opts.maxiter;
end

% --- flag to show whether to print output ---
if ~isfield(opts,'verbose')
    options.verbose = 1;
else
    options.verbose = opts.verbose;
end

% --- maximum number of iterations of gradient descent ---
if ~isfield(opts,'solver')
    options.solver = 'direct';
else
    options.solver = opts.solver;
end



