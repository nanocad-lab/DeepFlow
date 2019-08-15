% Figure 1 in the get_data manual page.
clear

% Sine1 training data.
[x, y, conf] = get_data('sine1');

% Display some of the default parameters.
fprintf('conf.p = %d\n', conf.p)
fprintf('conf.x1, x2 = %.1f, %.1f\n', conf.x1, conf.x2)
fprintf('conf.par = '), fprintf('%.1f ', conf.par), fprintf('\n')
fprintf('conf.std = %.1f\n', conf.std)

% Test data (unordered).
conf.p = 200;
conf.std = 0;
[xt1, yt1] = get_data(conf);

% Test data (ordered).
conf.ord = 1;
[xt2, yt2] = get_data(conf);

% Get figure.
fig = get_fig('Figure 1');

% Plot.
hold off
plot(x, y, 'r*', 'MarkerSize', 10)
hold on
plot(xt1, yt1, 'b+', 'MarkerSize', 10)
plot(xt2, yt2, 'g-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-1 1])
set(gca, 'YLim', [-1 1])
set(gca, 'XTick', [-1 1])
set(gca, 'YTick', [-1 1])
xlabel('x', 'FontSize', 16)
ylabel('y', 'FontSize', 16, 'Rotation', 0)
legend('training set', 'test (unordered)', 'test (ordered)', 3)

% Save postscript.
print -depsc fig1
