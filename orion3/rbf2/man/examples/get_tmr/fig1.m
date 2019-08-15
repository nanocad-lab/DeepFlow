% Figure 1 in the get_timer manual page.
clear

% Get a timer.
tmr = get_tmr(struct('name', 'demo', 'n', 100));

% Increment it a few times.
for i = 1:42
  inc_tmr(tmr)
end

% Now use separate software to capture
% an image of the timer window.