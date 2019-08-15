function [tree, conf] = grow_tree(X, y, conf)
%
% Grow a regression tree by recursive splitting.
%
% Training data in X (inputs) and y (outputs), configuration
% structure conf. Returns a tree structure tree and fully
% instantiated configuration structure conf.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
%

% Program name (for error messages).
prog = 'grow_tree';

% Configuration specification.
spec(1) = struct( ...
  'comment', 'Minimum membership', ...
  'name', 'minm', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 5);
spec(2) = struct( ...
  'comment', 'Placement of centres', ...
  'name', 'place', ...
  'type', 'string', ...
  'options', {{'centre', 'edge', 'both'}}, ...
  'default', 'centre');
spec(3) = struct( ...
  'comment', 'Verbose output', ...
  'name', 'verb', ...
  'type', 'number', ...
  'options', {{0 1}}, ...
  'default', 0);
spec(4) = struct( ...
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

% Initialisation for flops and time.
if conf.rprt
  comps = flops;
  ticks = clock;
end

% Initialize the tree.
tree = initTree(X, y);

% Start splitting at the first node.
tree = splitTree(tree, 1, conf);

% Print a report.
if conf.rprt
  comps = flops - comps;
  ticks = etime(clock,ticks);
  fprintf('-------- %s report --------\n', prog)
  fprintf('minm:          %d\n', conf.minm)
  fprintf('nodes:         %d\n', tree.numn)
  fprintf('levels:        %d\n', tree.level)
  fprintf('split numbers: '), fprintf('%d ', tree.split.number), fprintf('\n')
  fprintf('split order:   '), fprintf('%d ', tree.split.order), fprintf('\n')
  fprintf('flops:         %d\n', comps)
  fprintf('time:          %.3f\n', ticks)
  fprintf('----------------------------------\n')
end


function tree = initTree(X, y)
%
% Initialises a tree and its root node ready for splitting.
%

% Size of X.
[d, p] = size(X);

% Number of patterns in root node.
n.p = p;

% Sorted x and y values.
[sX, n.iX] = sort(X,2);

% Extreme limits of data.
n.lim = zeros(d,2);
for k = 1:d
  n.lim(k,:) = [X(k,n.iX(k,1)) X(k,n.iX(k,end))];
end

% Are edges of cell also edges of data?
n.edge = ones(d,2);

% Average y value.
n.ave = sum(y) / p;

% Total squared error.
n.err = sum((y - n.ave).^2);

% Split (as yet unknown).
n.split = [];

% Centre position and size of the corresponding RBF.
n.c = (n.lim(:,2) + n.lim(:,1)) / 2;
n.r = (n.lim(:,2) - n.lim(:,1)) / 2;

% Level of node in tree.
n.level = 1;

% Index in tree's list of nodes.
n.indx = 1;

% Initially, the array of nodes contains only the root.
tree.node = n;

% Current number of nodes and terminals.
tree.numn = 1;

% Input dimensions.
tree.p = p;
tree.d = d;

% Original data.
tree.X = X;
tree.y = y;

% Initialise maximum level in tree.
tree.level = 1;

% Initialise number of splits and slit order in each dimension.
tree.split.number = zeros(1,d);
tree.split.order = zeros(1,d);


function tree = splitTree(tree, ind, conf)
%
% Split a node in the tree.
%

% What node.
n = tree.node(ind);

% Some new names for convenience.
p = n.p;
d = tree.d;

% Set up sorted y values.
Y = zeros(d,p);
for k = 1:d
  Y(k,:) = tree.y(n.iX(k,:))';
end

% Initialise on the split with minimal sized left node.
i = conf.minm;
L = Y(:,1:i);
l = sum(L,2);
la = l / i;
LA = la(:,ones(1,i));
R = Y(:,(i+1):p);
r = sum(R,2);
ra = r / (p-i);
RA = ra(:,ones(1,p-i));
E = [sum((L - LA).^2,2) sum((R - RA).^2,2)];
e = sum(E,2);
x1 = tree.X(sub2ind([d tree.p],1:d,n.iX(:,i)'));
x2 = tree.X(sub2ind([d tree.p],1:d,n.iX(:,i+1)'));
deg = find(x1 == x2);
if ~isempty(deg)
  e(deg) = Inf * e(deg);
end
[be, bd] = min(e);
bE = E(bd,:);
bi = i;
bla = la(bd);
bra = ra(bd);

% Find the best amongst of all splits over all dimensions.
for i = (conf.minm+1):(p-conf.minm)
  L = Y(:,1:i);
  l = l + Y(:,i);
  la = l / i;
  LA = la(:,ones(1,i));
  R = Y(:,(i+1):p);
  r = r - Y(:,i);
  ra = r / (p-i);
  RA = ra(:,ones(1,p-i));
  E = [sum((L - LA).^2,2) sum((R - RA).^2,2)];
  e = sum(E,2);
  x1 = tree.X(sub2ind([d tree.p],1:d,n.iX(:,i)'));
  x2 = tree.X(sub2ind([d tree.p],1:d,n.iX(:,i+1)'));
  deg = find(x1 == x2);
  if ~isempty(deg)
    e(deg) = Inf * e(deg);
  end
  [e1, d1] = min(e);
  if e1 < be
    be = e1;
    bd = d1;
    bE = E(bd,:);
    bi = i;
    bla = la(bd);
    bra = ra(bd);
  end
end

% Check split won't create zero width hyperrectangle.
if be == Inf
  return
end

% Update current node.
bv = (tree.X(bd,n.iX(bd,bi)) + tree.X(bd,n.iX(bd,bi+1))) / 2;
n.split.val = bv;
n.split.dim = bd;
n.split.err = be;

% Update tree split statistics.
tree.split.number(bd) = tree.split.number(bd) + 1;
if ~tree.split.order(bd)
  tree.split.order(bd) = max(tree.split.order) + 1;
end

% Create left node.
l.p = bi;
l.iX = zeros(d,l.p);
for k = 1:d
  if k == bd
    l.iX(k,:) = n.iX(k,1:l.p);
  else
    l.iX(k,:) = n.iX(k,find(ismember(n.iX(k,:),n.iX(bd,1:l.p))));
  end
end
l.lim = n.lim;
l.lim(bd,2) = bv;
l.edge = n.edge;
l.edge(bd,2) = 0;
l.ave = bla;
l.err = bE(1);
l.split = [];
[l.c, l.r] = get_cr(d, l.lim, l.edge, conf.place);

% Create right node.
r.p = p - bi;
r.iX = zeros(d,r.p);
for k = 1:d
  if k == bd
    r.iX(k,:) = n.iX(k,(l.p+1):end);
  else
    r.iX(k,:) = n.iX(k,find(ismember(n.iX(k,:),n.iX(bd,(l.p+1):end))));
  end
end
r.lim = n.lim;
r.lim(bd,1) = bv;
r.edge = n.edge;
r.edge(bd,1) = 0;
r.ave = bra;
r.err = bE(2);
r.split = [];
[r.c, r.r] = get_cr(d, r.lim, r.edge, conf.place);

% The children are on the next level.
nlevel = n.level + 1;
l.level = nlevel;
r.level = nlevel;
if nlevel > tree.level
  tree.level = nlevel;
end

% Update the tree with the new children.
tree.numn = tree.numn + 1;
lind = tree.numn;
l.indx = lind;
tree.node = [tree.node l];
tree.numn = tree.numn + 1;
rind = tree.numn;
r.indx = rind;
tree.node = [tree.node r];

% Update the current node in the tree.
n.split.l = l.indx;
n.split.r = r.indx;
tree.node(ind) = n;

% Try to grow the children.
if l.p >= 2 * conf.minm
  if conf.verb
    fprintf('spitting %d samples in left node %d\n', l.p, lind)
  end
  tree = splitTree(tree, lind, conf);
end
if r.p >= 2 * conf.minm
  if conf.verb
    fprintf('spitting %d samples in right node %d\n', r.p, rind)
  end
  tree = splitTree(tree, rind, conf);
end

function [c, r] = get_cr(d, lim, edge, place);

if strcmp(place,'edge') | strcmp(place,'both')
  ec = zeros(d,1);
  er = zeros(d,1);
  for k = 1:d
    if edge(k,1) == edge(k,2)
      ec(k) = (lim(k,2) + lim(k,1)) / 2;
      er(k) = (lim(k,2) - lim(k,1)) / 2;
    elseif edge(k,1)
      ec(k) = lim(k,1);
      er(k) = lim(k,2) - lim(k,1);
    else
      ec(k) = lim(k,2);
      er(k) = lim(k,2) - lim(k,1);
    end
  end
end
if strcmp(place,'centre') | strcmp(place,'both')
  cc = (lim(:,2) + lim(:,1)) / 2;
  cr = (lim(:,2) - lim(:,1)) / 2;
end

switch place
case 'edge'
  c = ec;
  r = er;
case 'centre'
  c = cc;
  r = cr;
case 'both'
  if sum(abs(ec - cc)) < d * eps & sum(abs(er - cr)) < d * eps
    c = ec;
    r = er;
  else
    c = [ec cc];
    r = [er cr];
  end
end
