function [C, R, w, info, conf] = rbf_rt_2(X, y, conf)
%
% Hybrid radial basis function network and regression tree.
%
% Solves a regression problem with inputs X and outputs y using a
% regression tree and an RBF network selected using standard
% forward subset selection. Returns the hidden unit centres C,
% their radii R, the hidden-to-output weights w, some additional
% information info and a fully instantiated configuration
% structure conf.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
%
% For a description of the algorithm see:
%
%  'Further Advances in RBF Networks', 1999.
%

% Program name (for error messages).
prog = 'rbf_rt_2';

% Configuration specification.
spec(1) = struct( ...
  'comment', 'Scale length trial values', ...
  'name',    'scales', ...
  'type',    {{'vector', 'positive'}}, ...
  'options', [], ...
  'default', [1 2]);
spec(2) = struct( ...
  'comment', 'Minimum membership trial values', ...
  'name', 'minmem', ...
  'type', {{'vector', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 5);
spec(3) = struct( ...
  'comment', 'Placement of centres', ...
  'name', 'edge', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
  'default', 0);
spec(4) = struct( ...
  'comment', 'Which cells get translated into RBFs', ...
  'name', 'trans', ...
  'type', 'string', ...
  'options', {{'all', 'leaves'}}, ...
  'default', 'all');
spec(5) = struct( ...
  'comment', 'Basis function type', ...
  'name', 'type', ...
  'type', 'string', ...
  'options', [], ...
  'default', 'g');
spec(6) = struct( ...
  'comment', 'Basis function sharpness', ...
  'name', 'exp', ...
  'type', 'number', ...
  'options', [], ...
  'default', 2);
spec(7) = struct( ...
  'comment', 'Configuration for rbf_fs_2', ...
  'name', 'fsc', ...
  'type', 'structure', ...
  'options', [], ...
  'default', '[]');
spec(8) = struct( ...
  'comment', 'Display graphical timer', ...
  'name', 'timer', ...
  'type', 'string', ...
  'options', [], ...
  'default', '');
spec(9) = struct( ...
  'comment', 'Verbose output', ...
  'name', 'verb', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(10) = struct( ...
  'comment', 'Model selection criterion', ...
  'name', 'msc', ...
  'type', 'string', ...
  'options', {{'uev', 'fpe', 'gcv', 'bic'}}, ...
  'default', 'gcv');
spec(11) = struct( ...
  'comment', 'Bias unit', ...
  'name', 'bias', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);

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

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% Scales and minimum memberships.
scales = conf.scales;
nscale = length(scales);
mins = conf.minmem;
nmin = length(mins);

% Initialisation for flops and time.
stats.comps = flops;
stats.ticks = clock;

% Setup tree growing configuration.
conf1.verb = conf.verb;
conf1.rprt = conf.verb;
if conf.edge
  conf1.place = 'edge';
else
  conf1.place = 'centre';
end

% Setup tree translating configuration.
conf2.trans = conf.trans;

% Setup forward selection configuration.
conf3 = conf.fsc;

% Overide conf.fsc by parameters set up in conf.
conf3.type = conf.type;
conf3.exp = conf.exp;
conf3.msc = conf.msc;
conf3.bias = conf.bias;

% Set up a timer for minmem (if required).
if ~strcmp(conf.timer,'')
  minm_tmr = get_tmr(struct('name', [prog ' minmem ' conf.timer], 'n', nmin));
end

% Loop over mins.
berr = Inf;
for m = 1:nmin

  % Increment the timer for minmem.
  if ~strcmp(conf.timer,'')
    inc_tmr(minm_tmr)
  end

  % Grow a tree for this minmem.
  conf1.minm = mins(m);
  tree = grow_tree(X, y, conf1);

  % Set up a timer for scales (if required).
  if ~strcmp(conf.timer,'')
    scale_tmr = get_tmr(struct('name', [prog ' scales ' conf.timer], 'n', nscale));
  end
    
  for s = 1:nscale
  
    % Increment the timer for scale.
    if ~strcmp(conf.timer,'')
      inc_tmr(scale_tmr)
    end

    % Grow an RBF on the current tree.
    conf2.scale = scales(s);
    [C, R] = tree_rbf(tree, conf2);

    % Run forward selection.
    conf3.cen = C;
    conf3.rad = R;
    [C, R, w, info, conf4] = rbf_fs_2(X, y, conf3);
  
    % Is this the best so far.
    if info.err < berr
      berr = info.err;
      btree = tree;
      bscale = conf2.scale;
      bminm = conf1.minm;
      bC = C;
      bR = R;
      bw = w;
      binfo = info;
      bconf = conf4;
    end

  end

  % Close the timer for scales.
  if ~strcmp(conf.timer,'')
    close(scale_tmr)
  end

end

% Close the timer for minm.
if ~strcmp(conf.timer,'')
  close(minm_tmr)
end

% Record the computations and time taken.
stats.comps = flops - stats.comps;
stats.ticks = etime(clock,stats.ticks);

% Get the best centres, radii and weights.
C = bC;
R = bR;
w = bw;

% Replace conf.fsc with the best rbf_fs_2 conf.
conf.fsc = bconf;

% Other potentially useful results go in 'info'.
info.err = berr;
info.scale = bscale;
info.minm = bminm;
info.tree = btree;
info.stats = stats;
info.info = binfo;
info.bias = binfo.bias;

% Configuration for rbf_dm also goes in info.
info.dmc = struct('type', conf.type, 'exp', conf.exp, 'bias', binfo.bias);

