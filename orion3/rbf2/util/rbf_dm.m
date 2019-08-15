function H = rbf_dm(X, C, R, conf)
%
% Generate a design matrix using RBF functions.
%
% Generates an RBF design matrix, H, from inputs X,
% centres C and radii R. The type of basis function
% is controlled by the (optional) conf argument, as
% is the existence of an optional bias unit.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
% 

% Program name (for error messages) and configuration spec.
prog = 'rbf_dm';
spec(1) = struct( ...
  'comment', 'Type of RBF function', ...
  'name',    'type', ...
  'type',    'string', ...
  'options', {{{'g', 'gaussian'}, {'c', 'cauchy'}, {'m', 'multiquadric'}, {'i', 'inverse', 'inverse multiquadric'}}}, ...
  'default', 'g');
spec(2) = struct( ...
  'comment', 'Sharpness of RBF function', ...
  'name',    'exp', ...
  'type',    {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 2);
spec(3) = struct( ...
  'comment', 'Location of bias unit', ...
  'name',    'bias', ...
  'type',    {{'number', 'nonnegative', 'integer'}}, ...
  'options', [], ...
  'default', 0);

% Check number of parameters. Take special action if only one or two.
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
  if isstring(X) & isstring(C)
    switch X
    case 'conf'
      conf_print(prog, spec, C)
      return
    otherwise
      error([prog ': ''' X ''' unrecognised for two string arguments'])
    end
  else
    error([prog ': unrecognised type(s) for two arguments'])
  end
case 3
  conf = [];
case 4
otherwise
  error([prog ': wrong number of arguments'])
end

% Check type of input arguments.
if ~isnumeric(X) | ndims(X) ~= 2
  error([prog ': argument X should be a matrix'])
end
if ~isnumeric(C) | ndims(C) ~= 2
  error([prog ': argument C should be a matrix'])
end
if ~isnumeric(R) | ndims(R) ~= 2
  error([prog ': argument R should be a matrix, vector or scalar'])
end

% Check for consistent dimensions amongst X, C and R. Reshape R if necessary.
[d,p] = size(X);
if size(C,1) ~= d
  error([prog ': input dimension inconsistent between X and C'])
end
m = size(C,2);
if max(size(R)) == 1
  R = R(ones(d,1),ones(m,1));
elseif size(R,1) == 1 & size(R,2) == m
  R = R(ones(d,1),:);
elseif size(R,1) == d & size(R,2) == 1
  R = R(:,ones(m,1));
elseif size(R,1) ~= d
  error([prog ': argument R has wrong input dimension'])
elseif size(R,2) ~= m
  error([prog ': number of basis functions inconsistent between C and R'])
end

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% Check bias unit located in sensible position.
if conf.bias > (m + 1)
  error([prog ': bias unit (column ' num2str(conf.bias) ') is off the end of the matrix (columns ' num2str(m) ')'])
end

% All checks done. Calculate the design matrix.
H = zeros(p,m);
e = conf.exp;
for j = 1:m
  c = C(:,j);
  r = R(:,j);
  z = sum(((X - c(:,ones(1,p))) ./ r(:,ones(1,p))).^2, 1)';
  if e ~= 2
    z = sqrt(z).^e;
  end
  switch conf.type
  case 'g'
    H(:,j) = exp(-z);
  case 'c'
    H(:,j) = 1 ./ (1 + z);
  case 'm'
    H(:,j) = sqrt(1 + z);
  case 'i'
    H(:,j) = 1 ./ sqrt(1 + z);
  otherwise
    error([prog ': illegal basis function type ''' conf.type ''''])
  end
end

% Add the optional bias unit.
if conf.bias
  b = conf.bias;
  H = [H(:,1:(b-1)) ones(p,1) H(:,b:m)];
end
