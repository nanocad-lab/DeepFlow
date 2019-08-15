function [C, R, w, info, conf] = rbf_fs_2(X, y, conf)
%
% Regularised forward selection using radial basis functions.
%
% Solves a regression problem with inputs X and outputs y using forward
% selection of radial basis functions and (optionally) regularisation.
% Returns the hidden unit centres C, their radii R, the hidden-to-output
% weights w, some additional information info and a fully instantiated
% configuration structure conf.
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
prog = 'rbf_fs_2';

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
  'options', {{'uev', 'fpe', 'gcv', 'bic'}}, ...
  'default', 'gcv');
spec(5) = struct( ...
  'comment', 'Function to minimise', ...
  'name', 'fmin', ...
  'type', 'string', ...
  'options', {{'cost', 'sse'}}, ...
  'default', 'sse');
spec(6) = struct( ...
  'comment', 'Termination threshold', ...
  'name', 'thresh', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 10000);
spec(7) = struct( ...
  'comment', 'Basis function type', ...
  'name', 'type', ...
  'type', 'string', ...
  'options', [], ...
  'default', 'g');
spec(8) = struct( ...
  'comment', 'Basis function sharpness', ...
  'name', 'exp', ...
  'type', 'number', ...
  'options', [], ...
  'default', 2);
spec(9) = struct( ...
  'comment', 'Initial regularisation parameter', ...
  'name', 'lam', ...
  'type', {{'number', 'nonnegative'}}, ...
  'options', [], ...
  'default', 1e-9);
spec(10) = struct( ...
  'comment', 'Reestimate regularisation parameter', ...
  'name', 'reest', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(11) = struct( ...
  'comment', 'Number of iterations to confirm minimum MSC reached', ...
  'name', 'wait', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 2);
spec(12) = struct( ...
  'comment', 'Verbose output', ...
  'name', 'verb', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(13) = struct( ...
  'comment', 'Basis function scales', ...
  'name', 'scales', ...
  'type', {{'vector', 'positive'}}, ...
  'options', [], ...
  'default', 1);
spec(14) = struct( ...
  'comment', 'Display graphical timer', ...
  'name', 'timer', ...
  'type', 'string', ...
  'options', [], ...
  'default', '');

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
FC = conf.cen;
M = size(FC,2);

% Check or set defaults for radii.
if isempty(conf.rad)
  % Default: scaled in each dimension, same for each centre.
  conf.rad = (max(X,[],2) - min(X,[],2));
  conf.rad = conf.rad(:,ones(M,1));
else
  % Expand conf.rad using the same rules as for rbf_dm.
  if max(size(conf.rad)) == 1
    % It's a scalar - same size for each centre and in each dimension.
    conf.rad = conf.rad(ones(d,1),ones(M,1));
  elseif size(conf.rad,1) == 1 & size(conf.rad,2) == M
    % A row vector - same size in each dimension.
    conf.rad = conf.rad(ones(d,1),:);
  elseif size(conf.rad,1) == d & size(conf.rad,2) == 1
    % A column vector - same size for each centre.
    conf.rad = conf.rad(:,ones(M,1));
  elseif size(conf.rad,1) ~= d
    error([prog ': conf.rad has wrong input dimension'])
  elseif size(conf.rad,2) ~= M
    error([prog ': conf.cen and conf.rad are inconsistent'])
  end
end
FR = conf.rad;

% Initialisation for flops and time.
% stats.comps = flops;
stats.ticks = clock;

% Set up a timer (if requested).
if ~strcmp(conf.timer,'')
  tmr = get_tmr(struct('name', [prog ' scales ' conf.timer], 'n', length(conf.scales)));
end

% Loop over scales.
best.err = Inf;
for s = 1:length(conf.scales)
  
  % What scale are we using.
  scale = conf.scales(s);
  
  % Select with this scale.
  [CC, RR, ww, iinfo] = forwardSelect(X, y, FC, FR, scale, conf);
  
  % Is this the best so far?
  if iinfo.err < best.err
    C = CC;
    R = RR;
    w = ww;
    info = iinfo;
    best.err = info.err;
    best.scale = scale;
  end
  
  % Increment the timer.
  if ~strcmp(conf.timer,'')
    inc_tmr(tmr)
  end
  
end

% Close the timer.
if ~strcmp(conf.timer,'')
  close(tmr)
end

% Record the computations and time taken.
%stats.comps = flops - stats.comps;
stats.ticks = etime(clock,stats.ticks);
info.stats = stats;

% Best scale.
info.scale = best.scale;

% Configuration for rbf_dm.
info.dmc = struct('type', conf.type, 'exp', conf.exp, 'bias', info.bias);

% Feedback.
if conf.verb
  fprintf('\n-----------------------\n')
  fprintf('best scale is %.3f\n', info.scale)
  fprintf('%d centres selected\n', length(info.subset))
  if conf.bias
    if info.bias
      fprintf('one bias unit (%d)\n', info.bias)
    else
      fprintf('no bias unit\n')
    end
  end
  if conf.reest
    fprintf('final lambda = %9.2e\n', info.lam)
  end
  fprintf('%s = %9.2e\n', conf.msc, info.err)
  fprintf('-----------------------\n')
end


function [C, R, w, info] = forwardSelect(X, y, C, R, scale, conf)

% Initialise.
[d,p] = size(X);
M = size(C,2);
R = R * scale;

% Get full design matrix. Errors in specifying conf.type and conf.exp caught by rbf_dm.
conf_dm.type = conf.type;
conf_dm.exp = conf.exp;
F = rbf_dm(X, C, R, conf_dm);

% Add a bias unit in the last column, if conf.bias is on.
if conf.bias
  F = [F ones(p,1)];
  M = M + 1;
end

% Feedback header.
if conf.verb
  fprintf('\n')
  fprintf('scale = %.3f\n', scale)
  fprintf(' m   j   sse     %s   ', conf.msc)
  if conf.reest
    fprintf('    lam   \n')
  else
    fprintf('\n')
  end
  fprintf('-----------------------')
  if conf.reest
    fprintf('----------\n')
  else
    fprintf('\n')
  end
end

% Initialise.
m = 0;
finished = 0;
yy = y' * y;
sse = yy;
subset = [];
lam = conf.lam;
lams = [];
gams = [];
mscs = [];

% Search for the most significant regressors.
while ~finished

  % Increment the number of regressors.
  m = m + 1;
  
  % Different action depending on whether this is the first regressor.
  if m == 1
    
    % The change in cost or sum squared error due to the first regressor.
    FF = sum(F.*F,1)';
    switch conf.fmin
    case 'cost'
      err = (F'*y).^2 ./ (lam + FF);
    case 'sse'
      err = ((F'*y).^2 .* (2 * lam + FF)) ./ (lam + FF).^2;
    end

    % Select the maximum.
    [merr,j] = max(err);
    subset = [subset j];

    % Initialise Hm, the current orthogonalised design matrix,
    % and Hn, the same but with normalised columns.
    f = F(:,j);
    ff = f' * f;
    fy = f' * y;
    Hm = f;
    Hn = f / ff;
    Hy = fy;
    hh = ff;

    % Initialise Fm ready for second iteration.
    Fm = F - Hn * (Hm' * F);
    FmFm = sum(Fm.*Fm,1)';

    % initialise upper triangular matrix U
    U = 1;

  else

    % Prepare to find most significant remaining regressor.
    switch conf.fmin
    case 'cost'
      numerator = (Fm' * y).^2;
      denominator = lam + FmFm;
    case 'sse'
      numerator = (Fm' * y).^2 .* (2 * lam + FmFm);
      denominator = (lam + FmFm).^2;
    end
    
    % Avoid division by zero.
    denominator(subset) = ones(m-1,1);
    err = numerator ./ denominator;
    
    % Avoid selecting the same regressor twice.
    err(subset) = zeros(m-1,1);

    % Select the maximum change.
    [merr,j] = max(err);
    subset = [subset j];

    % Collect next columns for Hm and Hn.
    f = Fm(:,j);
    ff = f' * f;
    fy = f' * y;
    Hm(:,m) = f;
    Hn(:,m) = f / ff;
    Hy = [Hy; fy];
    hh = [hh; ff];

    % Recompute Fm ready for the next iteration.
    Fm = Fm - Hn(:,m) * (Hm(:,m)' * Fm);
    FmFm = sum(Fm.*Fm,1)';

    % Update U.
    U = [U Hn(:,1:m-1)' * F(:,j); zeros(1,m-1) 1];

  end
  
  % Update the sum of squared errors (and optionally reestimate lambda).
  if conf.reest
    sse = yy - sum((Hy.^2) .* (2*lam + hh) ./ ((lam + hh).^2));
    lam = reest(sse, p, hh, lam, Hy, conf.msc);
    lams = [lams lam];
    sse = yy - sum((Hy.^2) .* (2*lam + hh) ./ ((lam + hh).^2));
  else
    if lam == 0
      sse = sse - fy^2 / ff;
    else
      sse = sse - fy^2 * (2*lam + ff) / (lam + ff)^2;
    end
  end

  % Calculate current MSC.
  [msc, gam] = get_msc(sse, p, hh, lam, conf.msc);
  mscs = [mscs msc];
  gams = [gams gam];

  % Feedback.
  if conf.verb
    % fprintf('%3d %3d %5.3f %9.2e', m, j, (1 - sse/yy), msc)
    fprintf('%3d %3d %5.3f %9.2e', m, j, sse/yy, msc)
    if conf.reest
      fprintf(' %9.2e', lam)
    end
    fprintf('\n')
  end

  % Are we ready to terminate yet.
  if m == M
    % Run out of candidates.
    finished = 1;
    if msc < min_msc
      age = 0;
    else
      age = age + 1;
    end
  elseif sse == 0
    % Reduced error to zero.
    finished = 1;
    age = 0;
  elseif sse < 0
    % Reduced error beyond zero (numerical error).
    finished = 1;
    age = 1;
  elseif max(FmFm) < eps
    % No more significant regressors left.
    finished = 1;
    age = 0;
  else
    if m == 1
      % First value, must be minimum so far.
      min_msc = msc;
      age = 0;
    else
      if msc < min_msc
        % Is it sufficiently smaller than the old minimum to really count?
        if msc < min_msc * (conf.thresh - 1) / conf.thresh
          % Yes it's small enough - count as new minimum.
          min_msc = msc;
          age = 0;
        else
          % No it's not small enough - count as ageing minimum.
          age = age + 1;
        end
      else
        % Age old minimum.
        age = age + 1;
      end
    end
    if age >= conf.wait
      finished = 1;
    end
  end
    
end

% Don't include last few regressors which aged the minimum.
m = m - age;
subset = subset(1:m);
  
% Also truncate recursive OLS productions.
U = U(1:m, 1:m);
hh = hh(1:m);
Hy = Hy(1:m);

% Get lambda at time of minimum (if reestimating).
if conf.reest
  lams = lams(1:m);
  lam = lams(m);
end

% Also, get msc and gamma at time of minimum.
msc = mscs(m);
gam = gams(m);

% Design matrix.
H = F(:,subset);

% Did we choose the bias unit?
if conf.bias
  bias = find(subset == M);
  if isempty(bias)
    bias = 0;
  else
    subset = subset([1:(bias-1) (bias+1):end]);
  end
else
  bias = 0;
end

% Centres and radii.
C = C(:,subset);
R = R(:,subset);
  
% Variance matrix.
Ui = inv(U);
A = Ui * diag(1 ./ (lam + hh)) * Ui';
  
% Weight vector.
w = Ui * (Hy ./ (lam + hh));

% Other potentially useful results go in 'info'.
info.F = F;
info.H = H;
info.A = A;
info.U = U;
info.lam = lam;
info.gam = gam;
info.err = msc;
info.bias = bias;
info.subset = subset;


% Get the current model selection criterion.
function [msc, gam] = get_msc(sse, p, hh, lam, type)

% Effective number of free parameters.
if lam == 0
  gam = length(hh);
else
  gam = sum(hh ./ (lam + hh));
end

% Value of model selection criterion.
switch type
case 'uev'
  msc = sse / (p - gam);
case 'fpe'
  msc = (p + gam) * sse / (p * (p - gam));
case 'gcv'
  msc = p * sse / (p - gam)^2;
case 'bic'
  msc = (p + (log(p) - 1) * gam) * sse / (p * (p - gam));
end


% Reestimate the regularisation parameter.
function lam = reest(sse, p, hh, lam, Hy, type)

% Some things we'll need.
lhh = lam + hh;
tra = sum(1 ./ lhh) - lam * sum(1 ./ (lhh.^2));
waw = sum((Hy.^2) ./ (lhh.^3));

% Effective number of free parameters.
gam = sum(hh ./ (lam + hh));

% A scaling factor which depend on the type of MSC.
switch type
case 'uev'
  sca = 1 / (2 * (p - gam));
case 'fpe'
  sca = p / ((p - gam) * (p + gam));
case 'gcv'
  sca = 1 / (p - gam);
case 'bic'
  sca = p * log(p) / (2 * (p - gam) * (p + (log(p) - 1) * gam));
end

% We're ready for the re-estimation now.
lam = sca * sse * tra / waw;
