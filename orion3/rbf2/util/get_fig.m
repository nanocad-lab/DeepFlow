function [fig, conf] = get_fig(conf, field)
%
% Create a new figure or grab an old one with the same name.
%
% The new figure handle is returned in fh. Optional parameters
% are controlled by fields in the structure conf.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
% 

% Program name (for error messages).
prog = 'get_fig';

% Configuration spec.
spec(1) = struct( ...
  'comment', 'Figure name', ...
  'name',    'name', ...
  'type',    'string', ...
  'options', [], ...
  'default', 'no name');
spec(2) = struct( ...
  'comment', 'Screen position (in pixels)', ...
  'name',    'pos', ...
  'type',    {{'row vector', 'length 2', 'nonnegative', 'integer'}}, ...
  'options', [], ...
  'default', [50 50]);
spec(3) = struct( ...
  'comment', 'Screen size (in pixels)', ...
  'name',    'size', ...
  'type',    {{'row vector', 'length 2', 'nonnegative', 'integer'}}, ...
  'options', [], ...
  'default', [600 400]);
spec(4) = struct( ...
  'comment', 'Printing size (in cms)', ...
  'name',    'psize', ...
  'type',    {{'row vector', 'length 2', 'positive'}}, ...
  'options', [], ...
  'default', [12 8]);
spec(5) = struct( ...
  'comment', 'NumberTitle on or off', ...
  'name',    'num', ...
  'type',    'string', ...
  'options', {{'on', 'off'}}, ...
  'default', 'off');
spec(6) = struct( ...
  'comment', 'MenuBar off', ...
  'name',    'menu', ...
  'type',    'number', ...
  'options', {{0 1}}, ...
  'default', 1);

% Check input arguments.
switch nargin

case 1

  % String or struct?
  if isstring(conf)
    % Take special action if it's a string.
    switch conf
    case 'conf'
      conf_print(prog, spec)
      return
    otherwise
      % Assume its a figure name.
      conf.name = conf;
    end
  elseif ~isstruct(conf)
    error([prog ': argument (conf) should be string or struct'])
  end

case 2

  % Only 'conf' plus field name makes sense.
  if isstring(conf) & isstring(field)
    switch conf
    case 'conf'
      conf_print(prog, spec, field)
      return
    otherwise
      error([prog ': ''' conf ''' unrecognised as first of two args'])
    end
  else
    error([prog ': illegal type(s) for two args'])
  end

otherwise

  % Wrong number of arguments.
  error([prog ': one or two arguments required'])

end

% Keep a note of the fields initially present in conf.
fields = fieldnames(conf);

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% Assume we're not going to find the figure already exists.
fig = 0;
new = 1;

% Search through existing figures.
figs = get(0, 'Children');
for i = 1:length(figs)
  name = get(figs(i), 'Name');
  if strcmp(name, conf.name)
    fig = figs(i);
    new = 0;
    break
  end
end

% Did we find one?
if fig == 0
  % No, so make a new figure.
  fig = figure;
end

% Bring the figure to the foreground.
figure(fig)

if new
  % If this is a new figure, set all its features.
  set(fig, 'Name', conf.name)
  set(fig, 'Position', [conf.pos conf.size])
  set(fig, 'PaperUnits', 'centimeters')
  set(fig, 'PaperPosition', [0 0 conf.psize])
  set(fig, 'PaperPositionMode', 'auto')
  set(fig, 'NumberTitle', conf.num)
  if conf.menu
    set(fig, 'MenuBar', 'none')
  end
else
  % For an old figure, only set non-defaulted features.
  for k = 1:length(fields)
    field = fields{k};
    switch field
    case 'pos'
      oldpos = get(fig, 'Position');
      set(fig, 'Position', [conf.pos oldpos([3 4])])
    case 'size'
      oldpos = get(fig, 'Position');
      set(fig, 'Position', [oldpos([1 2]) conf.size])
    case 'psize'
      set(fig, 'PaperUnits', 'centimeters')
      set(fig, 'PaperPosition', [0 0 conf.psize])
      set(fig, 'PaperPositionMode', 'auto')
    case 'num'
      set(fig, 'NumberTitle', conf.num)
    case 'menu'
      if conf.menu
        set(fig, 'MenuBar', 'none')
      end
    end
  end
end

