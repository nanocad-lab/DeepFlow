function [C, R, w, info, conf] = rbf_rr_2(X, y, conf)
%
% Ridge regression using radial basis functions.
%
% Solves a regression problem with inputs X and outputs y using ridge
% regression (weight decay) with one re-estimated regularisation
% parameter. Returns the hidden unit centres C, their radii R, the
% hidden-to-output weights w, some additional information info and
% a fully instantiated configuration structure conf.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
%
% For descriptions of the algorithm see:
%
%  'Introduction to RBF Networks', 1996.
%  'Further Advances in RBF Networks', 1999.
%

% Program name (for error messages).
prog = 'rbf_rr_2';

% Configuration specification.
spec(1) = struct( ...
  'comment', 'Centres', ...
  'name', 'cen', ...
  'type', 'matrix', ...
  'options', [], ...
  'default', []);
spec(2) = struct( ...
  'comment', 'Radii', ...
  'name', 'rad', ...
  'type', 'matrix', ...
  'options', [], ...
  'default', []);
spec(3) = struct( ...
  'comment', 'Bias unit', ...
  'name', 'bias', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(4) = struct( ...
  'comment', 'Model selection criterion', ...
  'name', 'msc', ...
  'type', 'string', ...
  'options', {{'uev', 'fpe', 'gcv', 'bic', 'mml'}}, ...
  'default', 'bic');
spec(5) = struct( ...
  'comment', 'Termination threshold', ...
  'name', 'thresh', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 10000);
spec(6) = struct( ...
  'comment', 'Hard limit on re-estimation iterations', ...
  'name', 'hard', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 100);
spec(7) = struct( ...
  'comment', 'Anti-cycling heuristic', ...
  'name', 'cyc', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 1);
spec(8) = struct( ...
  'comment', 'Basis function type', ...
  'name', 'type', ...
  'type', 'string', ...
  'options', [], ...
  'default', 'g');
spec(9) = struct( ...
  'comment', 'Basis function sharpness', ...
  'name', 'exp', ...
  'type', 'number', ...
  'options', [], ...
  'default', 2);
spec(10) = struct( ...
  'comment', 'Basis function scales', ...
  'name', 'scales', ...
  'type', {{'vector', 'positive'}}, ...
  'options', [], ...
  'default', 1);
spec(11) = struct( ...
  'comment', 'Initial regularisation parameter(s)', ...
  'name', 'lambdas', ...
  'type', {{'vector', 'positive'}}, ...
  'options', [], ...
  'default', 1);
spec(12) = struct( ...
  'comment', 'Reestimate regularisation parameter', ...
  'name', 'reest', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 1);
spec(13) = struct( ...
  'comment', 'Verbose output', ...
  'name', 'verb', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(14) = struct( ...
  'comment', 'Display graphical timer', ...
  'name', 'timer', ...
  'type', 'string', ...
  'options', [], ...
  'default', '');
spec(15) = struct( ...
  'comment', 'Type of MML re-estimation', ...
  'name', 'edm', ...
  'type', 'string', ...
  'options', {{'em', 'dm'}}, ...
  'default', 'dm');

% Check number of arguments. Take special action if only one.
switch nargin
case 1
  if isstring(X)
    switch X
    case 'conf'
      conf_print(prog, spec)
      return
    case 'demo'
      [mydemo, myclean] = eval(['demo_' prog]);
      run_demo(mydemo, myclean)
      return
    otherwise
      error([prog ': ''' X ''' unrecognised for single string argument'])
    end
  else
    error([prog ': unrecognised type for single argument'])
  end
case 2
  if isstring(X) & isstring(y)
    switch X
    case 'conf'
      conf_print(prog, spec, y)
      return
    otherwise
      error([prog ': ''' X ''' unrecognised for double string argument'])
    end
  else
    conf = [];
  end
case 3
otherwise
  error([prog ': illegal number of arguments'])
end

% Check type of input arguments.
if ~isnumeric(X) | ndims(X) ~= 2
  error([prog ': first argument (X) should be a matrix'])
end
if ~isnumeric(y) | ndims(y) ~= 2 | size(y,2) ~= 1
  error([prog ': second argument (y) should be a column vector'])
end

% Check for consistent size between X and y.
[d,p] = size(X);
if size(y,1) ~= p
  error([prog ': number of samples inconsistent between X and y'])
end

% Check the configuration is okay and set initial defaults (if required).
conf = conf_check(conf, spec, prog);

% Check or set defaults for centres.
if isempty(conf.cen)
  % The inputs are the centres, if the user doesn't say otherwise.
  conf.cen = X;
else
  % The centre dimension must be equal to the input dimension.
  if size(conf.cen,1) ~= d
    error([prog ': inconsistent dimensions between X and conf.cen'])
  end
end
C = conf.cen;
m = size(C,2);

% Check or set defaults for radii.
if isempty(conf.rad)
  % Default: scaled in each dimension, same for each centre.
  conf.rad = (max(X,[],2) - min(X,[],2));
  conf.rad = conf.rad(:,ones(m,1));
else
  % Expand conf.rad using the same rules as for rbf_dm.
  if max(size(conf.rad)) == 1
    % It's a scalar - same size for each centre and in each dimension.
    conf.rad = conf.rad(ones(d,1),ones(m,1));
  elseif size(conf.rad,1) == 1 & size(conf.rad,2) == m
    % A row vector - same size in each dimension.
    conf.rad = conf.rad(ones(d,1),:);
  elseif size(conf.rad,1) == d & size(conf.rad,2) == 1
    % A column vector - same size for each centre.
    conf.rad = conf.rad(:,ones(m,1));
  elseif size(conf.rad,1) ~= d
    error([prog ': conf.rad has wrong input dimension'])
  elseif size(conf.rad,2) ~= m
    error([prog ': conf.cen and conf.rad are inconsistent'])
  end
end

% Initialisation for flops and time.
%stats.comps = flops;
stats.ticks = clock;

% Feedback header.
if conf.verb
  fprintf('\n')
  fprintf('  scale      lam       %s   ', conf.msc)
  if conf.reest
    fprintf('    lam   \n')
  else
    fprintf('\n')
  end
  fprintf('-----------------------------')
  if conf.reest
    fprintf('----------')
  end
  fprintf('\n')
end

% Set up a timer for scales (if requested).
if ~strcmp(conf.timer,'')
  tmr1 = get_tmr(struct('name', [prog ' scales ' conf.timer], 'n', length(conf.scales)));
end

% Loop over scales.
best.err = Inf;
for s = 1:length(conf.scales)
  
  % What scale are we using.
  sca = conf.scales(s);
  
  % So what radius does that imply.
  R = sca * conf.rad;
  
  % Next, what's the design matrix (the bias, if there is one, goes in column 1).
  H = rbf_dm(X, C, R, struct('type', conf.type, 'exp', conf.exp, 'bias', conf.bias));
  
  % Get the eigen decomposition.
  [U,S,V] = svd(H);
  mu = diag(S).^2;
  if p > m
    mu = [mu; zeros(p-m,1)];
  end
  yu = (U' * y).^2;

  % Set up a timer for lambdas (if requested).
  if ~strcmp(conf.timer,'')
    tmr2 = get_tmr(struct('name', [prog ' lambdas ' conf.timer], 'n', length(conf.lambdas)));
  end

  % Loop over lambdas.
  for l = 1:length(conf.lambdas)
    
    % What initial value of lambda.
    lam = conf.lambdas(l);
    
    % Are we re-estimating or guessing?
    if conf.reest
      % Re-estimating.
      switch conf.msc
      case 'mml'
        % MML, which is optimised over sig and var (not lambda), handled separately.
        [err, errs, sig, sigs, var, vars] = ree_mml(mu, yu, lam, m, conf.edm, conf.hard, conf.thresh);
        lam = sig / var;
      otherwise
        % All other MSCs which are optimised over just lambda.
        [err, errs, lam, lams] = ree_msc(mu, yu, lam, conf.msc, conf.hard, conf.thresh, conf.cyc);
      end
    else
      % Guessing.
      switch conf.msc
      case 'mml'
        % MML handled separately.
        err = ree_mml(mu, yu, lam);
      otherwise
        % All other MSCs.
        err = ree_msc(mu, yu, lam, conf.msc);
      end
    end
    
    % Get gamma.
    gam = sum(mu ./ (lam + mu));

    % Test for best.
    if err <= best.err
      best.sca = sca;
      best.lam = lam;
      best.gam = gam;
      best.err = err;
      best.mu = mu;
      best.yu = yu;
      best.H = H;
      best.R = R;
      if conf.reest
        best.errs = errs;
        switch conf.msc
        case 'mml'
          best.sig = sig;
          best.sigs = sigs;
          best.var = var;
          best.vars = vars;
        otherwise
          best.lams = lams;
        end
      end
    end
    
    % Feedback.
    if conf.verb
      fprintf('%9.2e %9.2e %9.2e', sca, conf.lambdas(l), err)
      if conf.reest
        fprintf(' %9.2e', lam)
      end
      fprintf('\n');
    end
    
    % Increment the lambdas timer.
    if ~strcmp(conf.timer,'')
      inc_tmr(tmr2)
    end
  
  end
  
  % Close the lambdas timer.
  if ~strcmp(conf.timer,'')
    close(tmr2)
  end

  % Increment the scales timer.
  if ~strcmp(conf.timer,'')
    inc_tmr(tmr1)
  end
  
end

% Close the scales timer.
if ~strcmp(conf.timer,'')
  close(tmr1)
end

% Record the computations and time taken.
%stats.comps = flops - stats.comps;
stats.ticks = etime(clock,stats.ticks);

% Get the best R, H and lambda.
R = best.R;
H = best.H;
lam = best.lam;

% Work out the variance and weight vector.
A = inv(H' * H + lam * eye(size(H,2)));
w = A * (H' * y);

% Results other than C, R and w go in 'info'.
info.A = A;
info.H = H;
info.stats = stats;
info.lam = lam;
info.gam = best.gam;
info.mu = best.mu;
info.yu = best.yu;
info.err = best.err;
info.scale = best.sca;
info.bias = conf.bias;
if conf.reest
  info.errs = best.errs;
  switch conf.msc
  case 'mml'
    info.sig = best.sig;
    info.sigs = best.sigs;
    info.var = best.var;
    info.vars = best.vars;
  otherwise
    info.lams = best.lams;
  end
end

% Configuration for rbf_dm also goes into info.
info.dmc = struct('type', conf.type, 'exp', conf.exp, 'bias', conf.bias);


% Re-estimate lambda and MSC (or sometimes just calculate MSC).
function [err, errs, lam, lams] = ree_msc(mu, yu, lam, msc, hard, thresh, cyc)

% Initialise.
p = length(yu);
err = msc_val(p, mu, yu, lam, msc);

% If only one output, got MSC (err) so exit.
if nargout == 1
  return
end

% Re-estimate.
count = 1;
errs = err;
lams = lam;
converged = 0;
while ~converged
  
  % One step re-estimation of lambda.
  nlam = msc_lam(p, mu, yu, lam, msc);
  
  % Anti-cycling heuristic.
  if cyc & count > 5 & abs(nlam - lams(count)) > abs(nlam - lams(count-1))
    nlam = sqrt(lams(count) * lams(count-1));
  end
  
  % Recalculate MSC.
  nerr = msc_val(p, mu, yu, nlam, msc);
  
  % Check for convergence.
  if err == nerr
    converged = 1;
  elseif abs(err / (err - nerr)) > thresh
    converged = 1;
  elseif count > hard
    converged = 1;
  end
  
  % Get ready for next iteration.
  count = count + 1;
  err = nerr;
  errs = [errs err];
  lam = nlam;
  lams = [lams lam];
    
end


% Get the MSC value.
function err = msc_val(p, mu, yu, lam, msc)

% Setup.
lmu = lam + mu;
sse = lam^2 * sum(yu ./ lmu.^2);
gam = sum(mu ./ lmu);

% Patch to deal with very small lambda.
if abs(p - gam) < eps
  err = Inf;
  return;
end

% Go.
switch msc
case 'uev'
  err = sse / (p - gam);
case 'fpe'
  err = sse * (p + gam) / (p * (p - gam));
case 'gcv'
  err = sse * p / (p - gam)^2;
case 'bic'
  err = sse * (p + (log(p) - 1) * gam) / (p * (p - gam));
end


% Get a new lambda value (one step re-estimation).
function lam = msc_lam(p, mu, yu, lam, msc)

% Setup.
lmu = lam + mu;
sse = lam^2 * sum(yu ./ lmu.^2);
waw = sum(mu .* yu ./ lmu.^3);
taa = sum(mu ./ lmu.^2);
gam = sum(mu ./ lmu);

% Patch to deal with very small lambda.
if abs(p - gam) < eps
  return;
end

% Go.
switch msc
case 'uev'
  lam = (sse * taa / waw) / (2 * (p - gam));
case 'fpe'
  lam = (sse * taa / waw) * p / ((p - gam) * (p + gam));
case 'gcv'
  lam = (sse * taa / waw) / (p - gam);
case 'bic'
  lam = (sse * taa / waw) * p * log(p) / (2 * (p - gam) * (p + (log(p) - 1) * gam));
end


% Re-estimate sig and var for MML (or sometimes just calculate MML).
function [err, errs, sig, sigs, var, vars] = ree_mml(mu, yu, lam, m, edm, hard, thresh)

% Initialise.
p = length(yu);

% Change lambda to sig and var.
sig = msc_val(p, mu, yu, lam, 'uev');
var = sig / lam;

% Initial score.
err = mml_val(p, mu, yu, sig, var);

% If only one output, got MML (err) so exit.
if nargout == 1
  return
end

% Re-estimate.
count = 1;
sigs = sig;
vars = var;
errs = err;
converged = 0;
while ~converged
  
  % One step re-estimation of sig and var.
  [nsig, nvar] = eval(['mml_' edm '(p, m, mu, yu, sig, var)']);
  
  % Recalculate MML.
  nerr = mml_val(p, mu, yu, nsig, nvar);
  
  % Check for convergence.
  if err == nerr
    converged = 1;
  elseif abs(err / (err - nerr)) > thresh
    converged = 1;
  elseif count > hard
    converged = 1;
  end
  
  % Get ready for next iteration.
  count = count + 1;
  err = nerr;
  errs = [errs err];
  sig = nsig;
  sigs = [sigs sig];
  var = nvar;
  vars = [vars var];
  
end


% Get the MML value.
function err = mml_val(p, mu, yu, sig, var)

% Setup and go.
vms = var * mu + sig;
err = sum(log(vms)) + sum(yu ./ vms);


% One step MML re-estimation, DM (David MacKay) version.
function [sig, var] = mml_dm(p, m, mu, yu, sig, var)

% Setup.
vms = var * mu + sig;
gam = var * sum(mu ./ vms);
ete = sig^2 * sum(yu ./ vms.^2);
wtw = var^2 * sum(mu .* yu ./ vms.^2);

% Go.
sig = ete / (p - gam);
var = wtw / gam;


% One step MML re-estimation, EM (Expectation-Maximisation) version.
function [sig, var] = mml_em(p, m, mu, yu, sig, var)

% Setup.
vms = var * mu + sig;
gam = var * sum(mu ./ vms);
ete = sig^2 * sum(yu ./ vms.^2);
wtw = var^2 * sum(mu .* yu ./ vms.^2);

% Go.
sig = (ete + gam * sig) / p;
var = (wtw + (m - gam) * var) / m;
