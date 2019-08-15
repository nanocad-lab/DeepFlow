function [C, R, w, info, conf] = rbf_rt_1(X, y, conf)
%
% Hybrid radial basis function network and regression tree.
%
% Solves a regression problem with inputs X and outputs y using a
% regression tree and an RBF network selected using tree-guided
% forward and backward subset selection. Returns the hidden unit
% centres C, their radii R, the hidden-to-output weights w, some
% additional information info and a fully instantiated configuration
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
prog = 'rbf_rt_1';

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
  'comment', 'Model selection criterion', ...
  'name', 'msc', ...
  'type', 'string', ...
  'options', {{'bic', 'gcv', 'loo'}}, ...
  'default', 'bic');
spec(4) = struct( ...
  'comment', 'Factor for adjusted GCV', ...
  'name', 'agcv', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 1);
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
  'comment', 'Regularisation parameter', ...
  'name', 'lambda', ...
  'type', {{'number', 'nonnegative'}}, ...
  'options', [], ...
  'default', 0);
spec(8) = struct( ...
  'comment', 'Pseudo-inverse instead of normal', ...
  'name', 'pseudo', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
  'default', 0);
spec(9) = struct( ...
  'comment', 'Extra speed', ...
  'name', 'speed', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
  'default', 1);
spec(10) = struct( ...
  'comment', 'Resolve draws randomly', ...
  'name', 'rand', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
  'default', 1);
spec(11) = struct( ...
  'comment', 'Placement of centres', ...
  'name', 'edge', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
  'default', 0);
spec(12) = struct( ...
  'comment', 'Display graphical timer', ...
  'name', 'timer', ...
  'type', 'string', ...
  'options', [], ...
  'default', '');
spec(13) = struct( ...
  'comment', 'Verbose output', ...
  'name', 'verb', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(14) = struct( ...
  'comment', 'Print final summary', ...
  'name', 'rprt', ...
  'type', 'number', ...
  'options', {{0, 1}}, ...
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
%stats.comps = flops;
stats.ticks = clock;

% Set up static part of tree growing configuration.
tcfg.verb = conf.verb;
tcfg.rprt = conf.rprt;
if conf.edge
  tcfg.place = 'edge';
else
  tcfg.place = 'centre';
end

% Set up static part of RBF growing configuration.
rcfg.msc = conf.msc;
rcfg.agcv = conf.agcv;
rcfg.type = conf.type;
rcfg.exp = conf.exp;
rcfg.lambda = conf.lambda;
rcfg.pseudo = conf.pseudo;
rcfg.speed = conf.speed;
rcfg.rand = conf.rand;
rcfg.rprt = conf.rprt;

% Set up a timer for minmem (if required).
if ~strcmp(conf.timer,'')
  tconf.name = [prog ' minmem ' conf.timer];
  tconf.n = nmin;
  minm_tmr = get_tmr(tconf);
end

% Loop over mins.
for m = 1:nmin

  % Increment the timer for minmem.
  if ~strcmp(conf.timer,'')
    inc_tmr(minm_tmr)
  end

  % Grow a tree for this minmem.
  tcfg.minm = mins(m);
  tree = grow_tree(X, y, tcfg);

  % Set up a timer for scales (if required).
  if ~strcmp(conf.timer,'')
    tconf.name = [prog ' scales ' conf.timer];
    tconf.n = nscale;
    scale_tmr = get_tmr(tconf);
  end
    
  for s = 1:nscale
  
    % Increment the timer for scale.
    if ~strcmp(conf.timer,'')
      inc_tmr(scale_tmr)
    end

    % Grow an RBF on the current tree.
    rcfg.scale = scales(s);
    [rbf, tree] = growRBF(tree, rcfg);
  
    % Is this the best tree so far.
    if m == 1 & s == 1
      btree = tree;
      brbf = rbf;
      bscale = rcfg.scale;
      bminm = tcfg.minm;
    else
      if rbf.err < brbf.err
        btree = tree;
        brbf = rbf;
        bscale = rcfg.scale;
        bminm = tcfg.minm;
      end
    end

  end

  % Close the timer for scale.
  if ~strcmp(conf.timer,'')
    close(scale_tmr)
  end

end

% Close the timer for minmem.
if ~strcmp(conf.timer,'')
  close(minm_tmr)
end

% Record the computations and time taken.
%stats.comps = flops - stats.comps;
stats.ticks = etime(clock,stats.ticks);

% Get the centres, radii and weights of the best tree.
C = brbf.C;
R = brbf.R;
w = brbf.w;

% Other potentially useful results go in 'info'.
info.scale = bscale;
info.minm = bminm;
info.tree = btree;
info.rbf = brbf;
info.err = brbf.err;
info.stats = stats;

% Configuration for rbf_dm also goes in info.
info.dmc = struct('type', conf.type, 'exp', conf.exp, 'bias', 0);


function [rbf, tree] = growRBF(tree, conf)
%
% Grows an RBF over a tree.
%

% Initialisation for flops and time.
if conf.rprt
% comps = flops;
  ticks = clock;
end

% Initialise and grow.
[rbf, tree] = initNet(tree, conf);
okay = 1;
while okay
  [rbf, okay] = splitNet(rbf, tree, conf);
end

% Record the computations and time taken.
if conf.rprt
  %comps = flops - comps;
  ticks = etime(clock,ticks);
end

% print a report.
if conf.rprt
  fprintf('-------- growRBF report --------\n')
  fprintf('rbfs:       %d\n', rbf.m)
  fprintf('actions:    '), fprintf('%d ', rbf.stats.actions), fprintf('\n')
  fprintf('splits:     '), fprintf('%d ', rbf.stats.splitds), fprintf('\n')
  fprintf('flops:      %d\n', comps)
  fprintf('time:       %.3f\n', ticks)
  fprintf('--------------------------------\n')
end


function [rbf, tree] = initNet(tree, conf)
%
% Initialise an RBF network prior to growing one over a tree.
%

% Calculate RBF columns for every node.
dmcfg.type = conf.type;
dmcfg.exp = conf.exp;
for j = 1:tree.numn
  n = tree.node(j);
  tree.node(j).h = rbf_dm(tree.X, n.c, n.r * conf.scale, dmcfg);
end

% Initialise the network with the RBF from the root node of the tree.
n = tree.node(1);
rbf.C = n.c;
rbf.R = n.r * conf.scale;
rbf.H = n.h;
rbf.m = 1;

% MSC, variance, weights etc.
[rbf.err, rbf.A, rbf.w, rbf.e, rbf.gam, rbf.AH] = learn(rbf.H, tree.y, conf);

% Initialise the array of indexes which relate columns in H to tree nodes.
rbf.node = 1;

% Initialise the list of teminal tree nodes and their number.
rbf.term = 1;
rbf.numt = 1;

% Stats.
rbf.stats.actions = zeros(1,8);
rbf.stats.splitds = zeros(tree.d,1);


function [rbf, okay] = splitNet(rbf, tree, conf)
%
% Try to grow RBF further down the tree.
%

% For each terminal node, calculate the change in MSC
% which would occur if that split was actually taken.
% 8 possibilites: add both, add left, add right, add none;
% replace both, replace left, replace right, remove the parent.
dmsc = zeros(rbf.numt,8);
for k = 1:rbf.numt
  
  % Which node?
  ind = rbf.term(k);
  n = tree.node(ind);
  
  % Has this node got children?
  if isempty(n.split)  
    % No. It can't be split.
    dmsc(k,:) = -1 * ones(1,8);
  else
    % Yes it can be split. What are its potential RBF contributions?
    lh = tree.node(n.split.l).h;
    rh = tree.node(n.split.r).h;
    % Try adding both, left only, right only, or no change.
    if (strcmp(conf.msc,'gcv') | strcmp(conf.msc,'bic')) & conf.speed == 1
      dmsc(k,1:4) = learnQuick(rbf, tree, lh, rh, conf);
    else
      dmsc(k,1) = rbf.err - learn([rbf.H lh rh], tree.y, conf);
      dmsc(k,2) = rbf.err - learn([rbf.H lh], tree.y, conf);
      dmsc(k,3) = rbf.err - learn([rbf.H rh], tree.y, conf);
      dmsc(k,4) = 0;
    end
    % Try replacing both.
    if ~ismember(ind, rbf.node)
      dmsc(k,5:7) = dmsc(k,1:3);
      dmsc(k,8) = -1;
    else
      % Try replacing both, left only, right only, or just removing parent.
      if (strcmp(conf.msc,'gcv') | strcmp(conf.msc,'bic')) & conf.speed == 1
        % I don't know how to do this quickly yet, so avoid the issue for now.
        % Meanwhile, if you want this feature, don't use conf.speed = 1.
        dmsc(k,5:8) = -1 * ones(1,4);
      else
        keep = find(rbf.node ~= ind);
        dmsc(k,5) = rbf.err - learn([rbf.H(:,keep) lh rh], tree.y, conf);
        dmsc(k,6) = rbf.err - learn([rbf.H(:,keep) lh], tree.y, conf);
        dmsc(k,7) = rbf.err - learn([rbf.H(:,keep) rh], tree.y, conf);
        dmsc(k,8) = rbf.err - learn(rbf.H(:,keep), tree.y, conf);
      end
    end
  end
  
end

% What's the best action/terminal combo?
[mdmsc, k] = max(dmsc, [], 1);
[mmdmsc, act] = max(mdmsc, [], 2);

% If there are no favourable changes possible, signal that and quit.
if mmdmsc < 0
  okay = 0;
  return
else
  okay = 1;
end

% Record action in info stats.
rbf.stats.actions(act) = rbf.stats.actions(act) + 1;

% Index of the node (in tree.node) assuming no tie.
ind = rbf.term(k(act));

% In the case of a tie amongst two or more terminals,
% randomly select one rather than always taking the first (k(act)).
% Only do this if indeterminancy is permitted (by conf.rand == 1).
if conf.rand
  winners = find(dmsc(:,act) == mmdmsc);
  numwins = length(winners);
  if numwins > 1
    winner = winners(ceil(rand * numwins));
    ind = rbf.term(winner);
  end
end

% Tree node being split and its children.
n = tree.node(ind);
lind = n.split.l;
rind = n.split.r;
ln = tree.node(lind);
rn = tree.node(rind);

% Record the split dimension. Maybe this should depend on action.
dim = n.split.dim;
rbf.stats.splitds(dim) = rbf.stats.splitds(dim) + 1;

% Remove this parent from the terminal list.
rbf.term(find(rbf.term == ind)) = [];
rbf.numt = rbf.numt - 1;

% Make the children of this node terminals.
rbf.term = [rbf.term lind rind];
rbf.numt = rbf.numt + 2;

% Less awkward names.
p = tree.p;
y = tree.y;

% Update the RBF network. Are we replacing the parent?
if act <= 4
  % No. Keep all the old columns.
  m = rbf.m;
  keep = 1:m;
else
  % Yes. Discard the parent if it's there.
  if ismember(ind, rbf.node)
    m = rbf.m - 1;
    keep = find(rbf.node ~= ind);
  else
    m = rbf.m;
    keep = 1:m;
  end
end
% Are we adding both, one or none?
if act == 1 | act == 5
  % Both.
  m = m + 2;
  nc = [ln.c rn.c];
  nr = [ln.r*conf.scale rn.r*conf.scale];
  nh = [ln.h rn.h];
  nt = [lind rind];
elseif act == 2 | act == 6
  % Left only.
  m = m + 1;
  nc = ln.c;
  nr = ln.r*conf.scale;
  nh = ln.h;
  nt = lind;
elseif act == 3 | act == 7
  % Right only.
  m = m + 1;
  nc = rn.c;
  nr = rn.r*conf.scale;
  nh = rn.h;
  nt = rind;
else
  % None.
  nc = [];
  nr = [];
  nh = [];
  nt = [];
end

% Update the RBF parameters.
rbf.m = m;
rbf.C = [rbf.C(:,keep) nc];
rbf.R = [rbf.R(:,keep) nr];
rbf.H = [rbf.H(:,keep) nh];
[rbf.err, rbf.A, rbf.w, rbf.e, rbf.gam, rbf.AH] = learn(rbf.H, tree.y, conf);

% Update the array of indexes which relate columns in H to tree nodes.
rbf.node = [rbf.node(keep) nt];


function [msc, A, w, e, gam, AH] = learn(H, y, conf)
%
% Get MSC and other things from from H and y.
%

[p,m] = size(H);
if conf.pseudo
  A = pinv(H'*H + conf.lambda * eye(m));
else
  A = inv(H'*H + conf.lambda * eye(m));
end
w = A * (H' * y);
f = H * w;
e = y - f;
if conf.lambda == 0
  gam = m;
else
  gam = m - conf.lambda * trace(A);
end
switch conf.msc
case 'gcv'
  msc = p * (e' * e) / (p - conf.agcv * gam)^2;
case 'loo'
  dP = ones(p,1) - sum((H*A).*H,2);
  msc = sum((e ./ dP).^2) / p;
case 'bic'
  msc = (p + (log(p) - 1) * gam) * (e' * e) / (p * (p - gam));
otherwise
  error(['growRBF: learn: bad MSC (' conf.msc ')'])
end

if nargout > 5
  AH = A * H';
end


function changes = learnQuick(rbf, tree, h1, h2, conf)
%
% Speedy GCV or BIC calculation.
%

% Initialise.
msc1 = rbf.err;
H1 = rbf.H;
A1H1 = rbf.AH;
P1y = rbf.e;
gam1 = rbf.gam;
y = tree.y;
lam = conf.lambda;
changes = zeros(1,4);
[p,m1] = size(H1);

% Add h1 (left).
m2 = m1 + 1;
P1h1 = h1 - H1 * (A1H1 * h1);
h1P1h1 = h1'*P1h1;
yP1h1 = y'*P1h1;
e2 = P1y - (yP1h1/(lam+h1P1h1)) * P1h1;
sse2 = e2' * e2;
if lam == 0
  gam2 = m2;
else
  h1P1P1h1 = P1h1'*P1h1;
  gam2 = gam1 + h1P1P1h1/(lam + h1P1h1);
end
switch conf.msc
case 'gcv'
  msc2 = p * sse2 / (p - conf.agcv * gam2)^2;
case 'bic'
  msc2 = (p + (log(p) - 1) * gam2) * sse2 / (p * (p - gam2));
otherwise
  error(['growRBF: learnQuick: bad MSC (' conf.msc ')'])
end
changes(2) = msc1 - msc2;

% Add h2 (right) as well.
m3 = m2 + 1;
P2y = P1y - (yP1h1/(lam + h1P1h1)) * P1h1;
A1H1h2 = A1H1 * h2;
P1h2 = h2 - H1 * A1H1h2;
h2P1h1 = h2' * P1h1;
P2h2 = P1h2 - (h2P1h1/(lam + h1P1h1)) * P1h1;
h2P2h2 = h2'*P2h2;
yP2h2 = y'*P2h2;
e3 = P2y - (yP2h2/(lam+h2P2h2)) * P2h2;
sse3 = e3' * e3;
if lam == 0
  gam3 = m3;
else
  h2P2P2h2 = P2h2'*P2h2;
  gam3 = gam2 + h2P2P2h2/(lam + h2P2h2);
end
switch conf.msc
case 'gcv'
  msc3 = p * sse3 / (p - conf.agcv * gam3)^2;
case 'bic'
  msc3 = (p + (log(p) - 1) * gam3) * sse3 / (p * (p - gam3));
end
changes(1) = msc1 - msc3;

% What about adding h2 (right) only.
h1 = h2;
m2 = m1 + 1;
P1h1 = h1 - H1 * (A1H1 * h1);
h1P1h1 = h1'*P1h1;
yP1h1 = y'*P1h1;
e2 = P1y - (yP1h1/(lam+h1P1h1)) * P1h1;
sse2 = e2' * e2;
if lam == 0
  gam2 = m2;
else
  h1P1P1h1 = P1h1'*P1h1;
  gam2 = gam1 + h1P1P1h1/(lam + h1P1h1);
end
switch conf.msc
case 'gcv'
  msc2 = p * sse2 / (p - conf.agcv * gam2)^2;
case 'bic'
  msc2 = (p + (log(p) - 1) * gam2) * sse2 / (p * (p - gam2));
end
changes(3) = msc1 - msc2;

