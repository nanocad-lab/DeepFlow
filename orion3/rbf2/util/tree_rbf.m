function [C, R, conf] = tree_rbf(tree, conf)
%
% Translates a tree into RBF centres and radii.
%
% Inputs are the tree and some optional configuration
% parameters in the structure conf. Outputs are RBF
% centres and radii plus a fully instantiated conf.
%

% Program name (for error messages).
prog = 'tree_rbf';

% Configuration specification.
spec(1) = struct( ...
  'comment', 'Scale applied to radii', ...
  'name', 'scale', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 1);
spec(2) = struct( ...
  'comment', 'Which cells get translated into RBFs', ...
  'name', 'trans', ...
  'type', 'string', ...
  'options', {{'all', 'leaves'}}, ...
  'default', 'all');

% Check number of arguments.
switch nargin
case 1
  if isstring(tree)
    switch tree
    case 'conf'
      conf_print(prog, spec)
      return
    otherwise
      error([prog ': ''' X ''' unrecognised for single string argument'])
    end
  else
    conf = [];
  end
case 2
  if isstring(tree) & isstring(conf)
    switch tree
    case 'conf'
      conf_print(prog, spec, y)
      return
    otherwise
      error([prog ': ''' X ''' unrecognised for double string argument'])
    end
  end
otherwise
  error([prog ': illegal number of arguments'])
end

% Check type of input arguments.
if ~isstruct(tree)
  error([prog ': first argument (tree) should be a structure'])
end
if ~isempty(conf) & ~isstruct(conf)
  error([prog ': second argument (conf) should be a structure'])
end

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% Initialise.
C = [];
R = [];

% Start recursing at top of tree.
[C, R] = downTree(tree, 1, conf);


function [C, R] = downTree(t, k, conf)

% Which node are we at?
n = t.node(k);

% Are we at a terminal node?
if isempty(n.split)

  % Always include leaf nodes.
  C = n.c;
  R = n.r * conf.scale;
  
else
  
  % Contribution from this node.
  if strcmp(conf.trans,'all')
    tC = n.c;
    tR = n.r * conf.scale;
  else
    tC = [];
    tR = [];
  end

  % Split.
  s = n.split;

  % Recurse.
  [lC, lR] = downTree(t, s.l, conf);
  [rC, rR] = downTree(t, s.r, conf);
    
  % Combine.
  C = [tC lC rC];
  R = [tR lR rR];
    
end
