function plot_ras(data, methods, sizes, perfs, pvals, conf)
%
% Plots a Rasmussen diagram to compare different methods.
%
% Inputs are title (data), method names (methods), training
% set sizes (sizes), performance figures (perfs), probabilities
% (pvals) and a configuration structure (conf).
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
%

% Initialise.
prog = 'plot_ras';

% Configuration spec.
spec(1) = struct( ...
  'comment', 'Width of figure (cms)', ...
  'name', 'width', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 12);
spec(2) = struct( ...
  'comment', 'Height of figure (cms)', ...
  'name', 'height', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 8);
spec(3) = struct( ...
  'comment', 'Title font size', ...
  'name', 'tfs', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 14);
spec(4) = struct( ...
  'comment', 'Axis mark font size', ...
  'name', 'afs', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 12);
spec(5) = struct( ...
  'comment', 'Model name font size', ...
  'name', 'mfs', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 10);
spec(6) = struct( ...
  'comment', 'Pixels per centimeter', ...
  'name', 'ppc', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', 50);
spec(7) = struct( ...
  'comment', 'Error bar colours', ...
  'name', 'ebc', ...
  'type', 'string', ...
  'options', [], ...
  'default', 'k');
spec(8) = struct( ...
  'comment', 'Error bar sizes', ...
  'name', 'ebs', ...
  'type', {{'vector', 'positive', 'integer'}}, ...
  'options', [], ...
  'default', 1);
spec(9) = struct( ...
  'comment', 'Method name colours', ...
  'name', 'mnc', ...
  'type', 'string', ...
  'options', [], ...
  'default', 'k');
spec(10) = struct( ...
  'comment', 'Maximum scaled error', ...
  'name', 'maxse', ...
  'type', {{'number', 'positive'}}, ...
  'options', [], ...
  'default', []);

% Check input argument(s).
switch nargin
case 0
  error([prog ': illegal number of arguments'])
case 1
  % data should be the string 'conf'.
  if isstring(data)
    switch data
    case 'conf'
      % Take special action.
      conf_print(prog, spec)
      return
    otherwise
      % Error.
      error([prog ': unrecognised string for single argument'])
    end
  else
    % Error.
    error([prog ': unrecognised type for single argument'])
  end
case 2
  % data should be 'conf' and methods should be a field name.
  if isstring(data) & isstring(methods)
    switch data
    case 'conf'
      % Take special action.
      conf_print(prog, spec, methods)
      return
    otherwise
      % Error.
      error([prog ': unrecognised string for two arguments'])
    end
  else
    % Error.
    error([prog ': unrecognised types for two arguments'])
  end
case 3
  error([prog ': illegal number of arguments'])
case 4
  pvals = [];
  conf = [];
case 5
  conf = [];
end

% Check the configuration is okay.
conf = conf_check(conf, spec, prog);

% Check the input arguments and establish dimensions.
if ~isstring(data)
  error([prog ': arg data needs to be a string'])
end
if ~iscell(methods) | ndims(methods) > 2 | min(size(methods)) ~= 1
  error([prog ': arg methods needs to be a 1D cell array'])
end
for m = 1:length(methods)
  if ~isstring(methods{m})
    error([prog ': component ' num2str(m) ' of arg methods is not a string'])
  end
end
nmet = length(methods);
if ~isnumeric(sizes) | ndims(sizes) > 2 | min(size(sizes)) ~= 1
  error([prog ': arg sizes needs to be a numeric vector'])
end
nsiz = length(sizes);
if ~isnumeric(perfs) | ndims(perfs) ~= 3
  error([prog ': arg perfs needs to be a 3D numeric matrix'])
end
if size(perfs,1) ~= nsiz
  error([prog ': size(perfs,1) is inconsistent with the number of training set sizes'])
end
if size(perfs,2) ~= nmet
  error([prog ': size(perfs,2) is inconsistent with the number of methods'])
end
if size(perfs,3) ~= 2
  error([prog ': size(perfs,3) should be 2 (mean and standard deviation)'])
end
if ~isnumeric(pvals) | ndims(pvals) ~= 3
  error([prog ': arg pvals needs to be a 3d numeric matrix'])
end
if size(pvals,1) ~= nsiz
  error([prog ': size(pvals,1) is inconsistent with the number of training set sizes'])
end
if size(pvals,2) ~= nmet
  error([prog ': size(pvals,2) is inconsistent with the number of methods'])
end
if size(pvals,3) ~= nmet
  error([prog ': size(pvals,2) is inconsistent with the number of methods'])
end

% Get the figure.
fconf.name = data;
fconf.pos = [50 50];
fconf.size = conf.ppc * [conf.width conf.height];
fconf.psize = [conf.width conf.height];
fig = get_fig(fconf);
clf
axis off
hold on

% Where should the axes be placed?
ax.x = 4 * conf.afs;
ax.w = fconf.size(1) - round(1.2 * ax.x);
ax.y = (nmet + 1) * conf.mfs + conf.afs/2;
ax.h = fconf.size(2) - ax.y - 3 * conf.tfs;

% Plot the x and y axis.
plot([ax.x ax.x+ax.w], [ax.y ax.y], 'k-', 'LineWidth', 2)
plot([ax.x ax.x], [ax.y ax.y+ax.h], 'k-', 'LineWidth', 2)

% Plot title.
t = text(ax.x+ax.w/2, ax.y+ax.h+15, data);
set(t, 'FontUnits', 'pixels')
set(t, 'FontSize', conf.tfs)
set(t, 'FontName', 'Courier')
p = get(t, 'Position');
e = get(t, 'Extent');
set(t, 'Position', [p(1)-e(3)/2 p(2)+1.5*e(4) p(3)])

% Annotate the y-axis.
jmpy = 0.1;
tmk = max([1 round(ax.w/75)]); % Tick mark size.
if isempty(conf.maxse)
  maxp = ceil(max(max(perfs(:,:,1)+perfs(:,:,2)))/jmpy)*jmpy;
else
  maxp = ceil(conf.maxse/jmpy)*jmpy;
end
scaley = ax.h / maxp;
if maxp > 1
  jmpy = 0.2;
end
for i = 0:jmpy:maxp
  y = ax.y + scaley * i;
  l = sprintf('%.1f', i);
  t = text(ax.x, y, l);
  set(t, 'FontUnits', 'pixels')
  set(t, 'FontSize', conf.afs)
  set(t, 'FontName', 'Courier')
  p = get(t, 'Position');
  e = get(t, 'Extent');
  set(t, 'Position', [p(1)-e(3)-2*tmk,p(2),p(3)]);
  if i == 0
    plot([ax.x-tmk ax.x+tmk-1], [y y], 'k-', 'LineWidth', 2)
  else
    plot([ax.x-tmk ax.x+tmk-1], [y y], 'k-', 'LineWidth', 1)
  end
end

% Margin width between columns for each case.
margw = round(0.3 * ax.w / (nsiz + 1));

% Column width for models and cases.
modw = floor((ax.w - (nsiz + 1) * margw) / (nsiz * nmet));
casew = modw * nmet;

% Length of column below xaxis.
cold = nmet * conf.mfs + conf.afs/2;

% Annotate the top of the plot and draw column boundaries.
x = ax.x + 2 * margw;
for i = 1:nsiz
  % Annotate.
  t = text(x, ax.y+ax.h, num2str(sizes(i)));
  set(t, 'FontUnits', 'pixels')
  set(t, 'FontSize', conf.afs)
  set(t, 'FontName', 'Courier')
  p = get(t, 'Position');
  e = get(t, 'Extent');
  set(t, 'Position', [p(1)+(casew-e(3))/2,p(2)+e(4)/2,p(3)])
  % Column boundaries.
  plot([x x], [ax.y-cold ax.y+ax.h], 'k-')
  plot([x+casew x+casew], [ax.y-cold ax.y+ax.h], 'k-')
  plot([x x+casew], [ax.y+ax.h ax.y+ax.h], 'k-')
  plot([x x+casew], [ax.y-cold ax.y-cold], 'k-')
  % Increment x.
  x = x + casew + margw;
end

% A handy constant.
nmnc = length(conf.mnc);

% Method labels.
x = ax.x + 1.5 * margw;
y = ax.y - conf.mfs/2 - conf.afs/2;
for i = 1:nmet
  method = traditional(methods{i});
  t = text(x, y, method);
  set(t, 'FontUnits', 'pixels')
  set(t, 'FontSize', conf.mfs)
  set(t, 'FontName', 'Courier')
  set(t, 'Color', conf.mnc(rem(i-1,nmnc)+1))
  p = get(t, 'Position');
  e = get(t, 'Extent');
  set(t, 'Position', [p(1)-e(3),p(2),p(3)])
  y = y - conf.mfs;
end

% Some handy constants.
nebc = length(conf.ebc);
nebs = length(conf.ebs);

% Arrow configuration.
arw.y1 = ax.y + ax.h * 0.94;      % Bottom.
arw.y2 = ax.y + ax.h * 0.99;      % Top.
arw.hw = (arw.y2-arw.y1) * 0.25;  % Head width.
arw.hh = (arw.y2-arw.y1) * 0.50;  % Head height.

% Plot the performances.
for c = 1:nsiz
  x = ax.x + 2 * margw + (c - 1) * (casew + margw);
  xm = x + modw/2;
  for m = 1:nmet
    y = ax.y + perfs(c,m,1) * scaley;
    dy = perfs(c,m,2) * scaley;
    y1 = max([ax.y y-dy]);
    y2 = min([ax.y+ax.h y+dy]);
    col = conf.ebc(rem(m-1,nebc)+1);
    lnw = conf.ebs(rem(m-1,nebs)+1);
    if y < ax.y+ax.h
      % Plot error bars.
      plot([x x+modw], [y y], [col '-'], 'LineWidth', lnw)
      plot([xm xm], [y1 y2], [col '-'], 'LineWidth', lnw)
    else
      % Plot arrow.
      plot([xm xm], [arw.y1 arw.y2], [col '-'], 'LineWidth', lnw)
      plot([xm xm+arw.hw], [arw.y2 arw.y2-arw.hh], [col '-'], 'LineWidth', lnw)
      plot([xm xm-arw.hw], [arw.y2 arw.y2-arw.hh], [col '-'], 'LineWidth', lnw)
    end
    x = x + modw;
    xm = xm + modw;
  end
end

% Plot the relative performances.
for c = 1:nsiz
  cx = ax.x + 2*margw + (c-1)*(casew+margw) + modw/2;
  for m2 = 1:nmet
    x = cx + (m2-1) * modw;
    y = ax.y - conf.mfs/2 - conf.afs/2;
    for m1 = 1:nmet
      if m1 ~= m2
        n = floor(100 * pvals(c, m1, m2));
        if n < 10 & n >= 0
          t = text(x, y, num2str(n));
        else
          t = text(x, y, '*');
        end
        set(t, 'FontUnits', 'pixels')
        set(t, 'FontSize', conf.mfs)
        set(t, 'FontName', 'Courier')
        p = get(t, 'Position');
        e = get(t, 'Extent');
        set(t, 'Position', [p(1)-e(3)/3,p(2),p(3)]);
      end
      y = y - conf.mfs;
    end
  end
end

function name = traditional(name)
%
% Replaces '_' with '-' for traditional DELVE names.
%

% Character to replace with.
trad = '-';

% Deal with underlines.
undr = find(name == '_');
name(undr) = trad(1,ones(1,length(undr)));
