function [X, y, conf] = get_data(conf, field)
%
% Get training or test data for a simulated learning problem.
%
% The input data is returned in X, the output in y. Optional
% parameters are controlled by fields in the structure conf.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
% 

% Program name (for error messages)
prog = 'get_data';

% Configuration spec.
spec(1) = struct( ...
  'comment', 'Problem name', ...
  'name',    'name', ...
  'type',    'string', ...
  'options', {{'sine1', 'sine2', {'mackay', 'hermite'}, {'friedman', 'sacc'}}}, ...
  'default', []);
spec(2) = struct( ...
  'comment', 'Number of samples', ...
  'name',    'p', ...
  'type',    {{'number', 'integer', 'positive'}}, ...
  'options', [], ...
  'default', []);
spec(3) = struct( ...
  'comment', 'Lower limits for input components', ...
  'name',    'x1', ...
  'type',    'column vector', ...
  'options', [], ...
  'default', []);
spec(4) = struct( ...
  'comment', 'Upper limits for input components', ...
  'name',    'x2', ...
  'type',    'column vector', ...
  'options', [], ...
  'default', []);
spec(5) = struct( ...
  'comment', 'Function parameter values', ...
  'name',    'par', ...
  'type',    'vector', ...
  'options', [], ...
  'default', []);
spec(6) = struct( ...
  'comment', 'Function parameter descriptions', ...
  'name',    'pinfo', ...
  'type',    'cellstr', ...
  'options', [], ...
  'default', []);
spec(7) = struct( ...
  'comment', 'Output component', ...
  'name',    'comp', ...
  'type',    {{'number', 'integer', 'positive'}}, ...
  'options', [], ...
  'default', []);
spec(8) = struct( ...
  'comment', 'Standard deviation of the noise', ...
  'name',    'std', ...
  'type',    {{'vector', 'nonnegative'}}, ...
  'options', [], ...
  'default', []);
spec(9) = struct( ...
  'comment', 'Ordered inputs', ...
  'name',    'ord', ...
  'type',    'number', ...
  'options', {{0 1}}, ...
  'default', []);
spec(10) = struct( ...
  'comment', 'Normalised inputs', ...
  'name',    'norm', ...
  'type',    'number', ...
  'options', {{0 1}}, ...
  'default', []);

% sine1 defaults.
sine1.p = 100;
sine1.x1 = -1;
sine1.x2 = 1;
sine1.par = [0.8 6];
sine1.pinfo = {'amplitude'; 'frequency'};
sine1.comp = 1;
sine1.std = 0.1;
sine1.ord = 0;
sine1.norm = 0;

% sine2 defaults.
sine2.p = 200;
sine2.x1 = [0; -5];
sine2.x2 = [10; 5];
sine2.par = [0.8 0.25 0.5];
sine2.pinfo = {'amplitude'; 'frequency-1'; 'frequency-2'};
sine2.comp = 1;
sine2.std = 0.1;
sine2.ord = 0;
sine2.norm = 0;

% mackay defaults.
mackay.p = 100;
mackay.x1 = -4;
mackay.x2 = 4;
mackay.par = [];
mackay.pinfo = [];
mackay.comp = 1;
mackay.std = 0.1;
mackay.ord = 0;
mackay.norm = 0;

% friedman defaults.
friedman.p = 200;
friedman.x1 = [40 * pi; 1; 0; 1e-6];
friedman.x2 = [560 * pi; 100; 1; 11e-6];
friedman.par = [];
friedman.pinfo = [];
friedman.comp = 1;
friedman.std = [175; 0.44];
friedman.ord = 0;
friedman.norm = 1;

% Input dimensions (not part of conf).
sine1.d = 1;
sine2.d = 2;
mackay.d = 1;
friedman.d = 4;

% Output dimensions (not part of conf).
sine1.n = 1;
sine2.n = 1;
mackay.n = 1;
friedman.n = 2;

% Check input argument(s).
switch nargin

case 1

  % conf should either be a string or a struct.
  if isstring(conf)

    % Take special action if it's a string.
    switch conf
    case 'conf'
      conf_print(prog, spec)
      return
    case 'demo'
      [mydemo, myclean] = eval(['demo_' prog])
      run_demo(mydemo, myclean)
      return
    case {'last', 'prev'}
      [X, y, conf] = uncache(prog);
      return
    case 'names'
      options = spec(1).options;
      fprintf('Recognised data set names are (in/out dims in brackets):\n')
      for o = 1:length(options)
        fprintf('  ')
        if ischar(options{o})
          fprintf('''%s'' (%d/%d)', options{o}, eval([options{o} '.d']), eval([options{o} '.n']))
        else
          alts = options{o};
          for a = 1:length(alts)
            if a ~= 1
              fprintf(' or ')
            end
            fprintf('''%s''', alts{a})
          end
          fprintf(' (%d/%d)', eval([alts{1} '.d']), eval([alts{1} '.n']))
        end
        fprintf('\n')
      end
      return
    otherwise
      % Assume it's a problem name.
      conf.name = conf;
    end

  elseif ~isstruct(conf)

    % If it's not a string it should be a struct.
    error([prog ': argument (conf) should be string or struct'])

  end

case 2

  % Two arguments should only mean 'conf' plus a field name.
  if isstring(conf) & isstring(field)
    conf_print(prog, spec, field)
    return
  else
    error([prog ': arguments (conf, field) should be strings'])
  end

otherwise

  error([prog ': takes either 1 or 2 arguments'])

end

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% The problem name must be supplied.
if isempty(conf.name)
  error([prog ': the problem name must be supplied'])
else
  name = conf.name;
end

% Checks and defaults that depend on problem name.
if isempty(conf.p)
  conf.p = eval([name '.p']);
end
if isempty(conf.par)
  % Special treatment because default can be [] which doesn't match 'vector'.
  def = eval([name '.par']);
  if isempty(def)
    conf = rmfield(conf, 'par');
  else
    conf.par = eval([name '.par']);
  end
else
  if length(conf.par) ~= length(eval([name '.par']))
    error([prog ': conf.par has wrong length for the ''' name ''' problem'])
  end
end
if isempty(conf.pinfo)
  % Also special treatment because default can be [] which doesn't match 'cellstr'.
  def = eval([name '.pinfo']);
  if isempty(def)
    conf = rmfield(conf, 'pinfo');
  else
    conf.pinfo = eval([name '.pinfo']);
  end
else
  if length(conf.pinfo) ~= length(eval([name '.pinfo']))
    error([prog ': conf.pinfo has wrong length for the ''' name ''' problem'])
  end
end
if isempty(conf.x1)
  conf.x1 = eval([name '.x1']);
else
  if length(conf.x1) ~= eval([name '.d'])
    error([prog ': conf.x1 has wrong length for the ''' name ''' problem'])
  end
end
if isempty(conf.x2)
  conf.x2 = eval([name '.x2']);
else
  if length(conf.x2) ~= eval([name '.d'])
    error([prog ': conf.x2 has wrong length for the ''' name ''' problem'])
  end
end
if isempty(conf.comp)
  conf.comp = eval([name '.comp']);
else
  if conf.comp > eval([name '.d'])
    error([prog ': conf.comp is too big for the ''' name ''' problem'])
  end
end
if isempty(conf.std)
  conf.std = eval([name '.std']);
else
  if length(conf.std) ~= length(eval([name '.std'])) & length(conf.std) ~= 1
    error([prog ': conf.std has wrong length for the ''' name ''' problem'])
  end
end
if isempty(conf.ord)
  conf.ord = eval([name '.ord']);
end
if isempty(conf.norm)
  conf.norm = eval([name '.norm']);
end

% Some useful parameters.
p = conf.p;
d = eval([name '.d']);
w = conf.x2 - conf.x1;
m = (conf.x2 + conf.x1) / 2;
x1 = conf.x1;
x2 = conf.x2;

% Inputs.
if conf.ord
  if d == 1
    X = linspace(x1, x2, p);
  else
    % Number of samples (conf.p) may change.
    q = round(10^(log10(p)/d));
    p = q^d;
    conf.p = p;
    x = zeros(d,q);
    for k = 1:d
      x(k,:) = linspace(x1(k), x2(k), q);
    end
    X = zeros(d,p);
    for k = 1:d
      r = q^(d-k);
      j = 1;
      for i = 1:p
        X(k,i) = x(k,j);
        if rem(i,r) == 0
          j = j + 1;
          if j > q
            j = 1;
          end
        end
      end
    end
  end
else
  X = x1(:,ones(1,p)) + rand(d,p) .* w(:,ones(1,p));
end

% OK, generate the data.
switch name
case 'sine1'
  y = conf.par(1) * sin(conf.par(2)*X)';
case 'sine2'
  y = conf.par(1) * (cos(conf.par(2)*X(1,:)) .* sin(conf.par(3)*X(2,:)))';
case 'mackay'
  y = 1 + ((1 - X + 2 * X.^2) .* exp(-X.^2))';
case 'friedman'
  f = X(1,:);
  r = X(2,:);
  i = X(3,:);
  c = X(4,:);
  if conf.comp == 1
    y = sqrt(r.^2 + (f.*i - 1./(f.*c)).^2)';
  else
    y = atan((f.*i - 1./(f.*c))./r)';
  end
end

% Add noise.
if length(conf.std) > 1
  std = conf.std(conf.comp);
else
  std = conf.std;
end
if std > 0
  y = y + std * randn(p,1);
end

% Normalise? Gives +/- 1 as extremes.
if conf.norm
  X = 2 * ((X - m(:,ones(1,conf.p))) ./ w(:,ones(1,conf.p)));
end

% Save the data set if it involves any randomness.
if conf.ord == 0 | conf.std > 0
  cache(prog, X, y, conf);
end

% Cache data.
function cache(prog, X, y, conf)

% See if we can find the cache file.
me = which([prog '.mat']);

% If not, use the location of the script.
if isempty(me)
  me = which([prog '.m']);
end

% Otherwise, give up.
if isempty(me)
  error([prog ': can''t cache results (probably means a problem with PATH)'])
end

% Work out where home is.
i = findstr(prog, me);
home = me(1:(i-1));

% Save the data.
save([home prog '.mat'], 'X', 'y', 'conf')

function [X, y, conf] = uncache(prog)

% Check [prog '.mat'] can be found on the path.
me = which([prog '.mat']);
if isempty(me)
  error([prog 'can''t locate cache file'])
end

% Load it.
load(me)

% Check it has the right data.
if ~exist('X', 'var') | ~exist('y', 'var') | ~exist('conf', 'var')
  error([prog ': cache file is corrupt'])
end
