function y = pred_tree(tree, X)
%
% Predict using a tree structure returned from grow_tree.
%
% Predict the outputs from a tree given some inputs (X).
% The tree will have been returned from the the utility
% grow_tree (or from the methods rbf_rt_1 or rbf_rt_2).
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
% 

prog = 'pred_tree';

% Check arguments.
if nargin ~= 2
  error([prog ': needs two arguments'])
end
if ~isstruct(tree)
  error([prog ': first arg (tree) needs to be a struct'])
end
if ~(isnumeric(X) & ndims(X) == 2)
  error([prog ': second arg (X) needs to be a numerical matrix'])
end

% Check dimensionality.
[d,p] = size(X);
if tree.d ~= d
  error([prog ': tree and inputs do not share same dimension'])
end

% Recurse down the tree.
y = downTree(tree, X, 1);

function y = downTree(t, X, k)

% Which node are we at?
n = t.node(k);

% Are we at a terminal node?
if isempty(n.split)

  % Give these inputs the node's average value.  
  y = n.ave * ones(size(X,2),1);
  
else
  
  % Divide.
  s = n.split;
  l = find(X(s.dim,:) <= s.val);
  r = find(X(s.dim,:) > s.val);
    
  % Recurse.
  yl = downTree(t, X(:,l), s.l);
  yr = downTree(t, X(:,r), s.r);
    
  % Recombine.
  y = zeros(size(X,2),1);
  y(l) = yl;
  y(r) = yr;
    
end
