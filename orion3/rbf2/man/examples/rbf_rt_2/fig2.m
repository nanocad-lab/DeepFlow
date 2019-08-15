% Figure 2 in the rbf_rt_2 manual page.
clear

% Hermite test data.
test.name = 'hermite';
test.p = 1000;
test.ord = 1;
test.std = 0;
[xt, yt] = get_data(test);

% Timer.
tmr = get_tmr(struct('name', 'fig 2', 'n', 2*100));

% Loop over replications training and testing both rt versions.
for rep = 1:100
  
  % Hermite training data.
  [x, y] = get_data('hermite');
  
  % First version.
  [c, r, w, info] = rbf_rt_1(x, y);
  Ht = rbf_dm(xt, c, r, info.dmc);
  ft = Ht * w;
  err(rep,1) = sqrt((yt - ft)' * (yt - ft) / length(yt));
  
  % Increment timer.
  inc_tmr(tmr)
  
  % Second version.
  [c, r, w, info] = rbf_rt_2(x, y);
  Ht = rbf_dm(xt, c, r, info.dmc);
  ft = Ht * w;
  err(rep,2) = sqrt((yt - ft)' * (yt - ft) / length(yt));
  
  % Increment timer.
  inc_tmr(tmr)
  
end

% Close the timer.
close(tmr)

% Print average errors.
fprintf('rbf_rt_1: %.3f +/- %.3f\n', mean(err(:,1)), std(err(:,1)))
fprintf('rbf_rt_2: %.3f +/- %.3f\n', mean(err(:,2)), std(err(:,2)))

% Get figure.
fig = get_fig('Figure 1');

% Plot.
de = 0.1;
e1 = de*floor(min(min(err))/de);
e2 = de*ceil(max(max(err))/de);
hold off
plot([e1 e2], [e1 e2], 'k--')
hold on
plot(err(:,1), err(:,2), 'r.')

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.15 0.15 0.8 0.8])
set(gca, 'XLim', [e1 e2])
set(gca, 'YLim', [e1 e2])
set(gca, 'XTick', e1:de:e2)
set(gca, 'YTick', e1:de:e2)
xlabel('rbf\_rt\_1', 'FontSize', 16)
ylabel('rbf\_rt\_2', 'FontSize', 16)

% Save postscript.
print -depsc fig2
