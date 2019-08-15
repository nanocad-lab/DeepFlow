function inc_timer(fh)
%
% Increment a graphical timer during one iteration of a loop.
%
% Input is a figure handle made by get_tmr.m.
%
% For further details of the function see:
%
%  'Matlab Routines for RBF Networks', 1999.
%

% Bring the figure to the fore and get its data.
figure(fh)
data = get(fh, 'UserData');

% Get what we need from the data.
d = data.conf;
i = d(1);
n = d(2);
p = d(3);
name = data.name;

% Increment.
i = i + 1;

% If the increment is valid, update the timer.
if i > 0 & i <= n

  % Calculate and set the new patch parameters.
  x = i/n;
  v = get(p, 'Vertices');
  v(2,1) = x;
  v(3,1) = x;
  set(p, 'Vertices', v);

  % Add 'i of n' to the title.
  set(fh, 'Name', [name ' (' num2str(i) ' of ' num2str(n) ')'])

  % Store the updated parametes back in the figure.
  data.conf = [i n p];
  set(fh, 'UserData', data);

  % Make sure it's rendered now.
  drawnow

end
