function [fh, conf] = get_tmr(conf, field)
%
% Setup a graphical timer to monitor the progress of a loop.
%
% The new figure handle is returned in fh. Optional parameters
% are controlled by fields in the structure conf.
% 
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
% 

% Program name (for error messages).
prog = 'get_tmr';

% Configuration specification.
spec(1) = struct( ...
  'comment', 'Figure name', ...
  'name',    'name', ...
  'type',    'string', ...
  'options', [], ...
  'default', '');
spec(2) = struct( ...
  'comment', 'Horizontal position', ...
  'name', 'x', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 50);
spec(3) = struct( ...
  'comment', 'Vertical position', ...
  'name', 'y', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', []);
spec(4) = struct( ...
  'comment', 'Width', ...
  'name', 'w', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 500);
spec(5) = struct( ...
  'comment', 'Height', ...
  'name', 'h', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 50);
spec(6) = struct( ...
  'comment', 'Number of increments (required)', ...
  'name', 'n', ...
  'type', {{'number', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', []);

% Check input arguments.
switch nargin
case 1
  % Special action if conf = 'conf'.
  if isstring(conf)
    switch conf
    case 'conf'
      conf_print(prog, spec)
      return
    otherwise
      error([prog ': ''' conf ''' unrecognised for single string argument'])
    end
  elseif isnumeric(conf) & max(size(conf)) == 1
    conf.n = conf;
  end
case 2
  % Special action.
  if isstring(conf) & isstring(field)
    switch conf
    case 'conf'
      conf_print(prog, spec, field)
      return
    otherwise
      error([prog ': ''' conf ''' unrecognised for two string arguments'])
    end
  else
    error([prog ': illegal type(s) for two arguments'])
  end
otherwise
  error([prog ': configuration argument missing'])
end

% Check the configuration is okay and set defaults (if required).
conf = conf_check(conf, spec, prog);

% Not setting conf.n is an error.
if isempty(conf.n)
  error([prog ': the number of iterations (conf.n) must be set'])
end

% The prefix added to the timer figure's title.
prefix = 'Timer: ';

% Get the screen height.
S = get(0, 'ScreenSize');
H = S(4);

% If conf.y is unset that means either a default is required or
% we want to put this timer below some other existing ones. To
% find out which, search for other timers and find the one with
% the lowest vertical position.
if isempty(conf.y)

  % Search through existing figures.
  y = H;
  figs = get(0, 'Children');
  for i = 1:length(figs)
    fig = figs(i);
    data = get(fig, 'UserData');
    if isstruct(data)
      if isfield(data, 'name')
        if strncmp(data.name, prefix, length(prefix))
          pos = get(fig, 'Position');
          if pos(2) < y
            y = pos(2);
          end
        end
      end
    end
  end

  % Set conf.y.
  conf.y = y - conf.h - 50;

end

% The width may need some adjusting.
conf.w = conf.n * max([round(conf.w/conf.n) 1]);

% Restructure parameters so they are suitable for get_fig.
conf_gf.name = [prefix conf.name];
conf_gf.pos = [conf.x conf.y];
conf_gf.size = [conf.w conf.h];

% Get (or create) this figure.
fh = get_fig(conf_gf);

% Taylor this figure to being a timer and render it now.
figure(fh)
cla
set(gca, 'Position', [0 0 1 1])
hold off
patch([0 1 1 0], [0 0 1 1], 'c')
hold on
p = patch([0 0 0 0], [0 0 1 1], 'b');
data.name = conf_gf.name;
data.conf = [0 conf.n p];
set(fh, 'UserData', data)
drawnow
