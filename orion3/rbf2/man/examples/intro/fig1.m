% Figure 1 in the Tutorial Introduction.
clear

% Training set.
conf.name = 'hermite';
conf.p = 100;
[x, y] = get_data(conf);

% Test set.
conf.p = 1000;
conf.ord = 1;
conf.std = 0;
[xt, yt] = get_data(conf);

% Get figure.
fig = get_fig('Figure 1');

% Plot.
hold off
plot(xt, yt, 'k--', 'LineWidth', 1)
hold on
plot(x, y, 'r*', 'MarkerSize', 8)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-4 4])
set(gca, 'YLim', [0 3])
set(gca, 'XTick', -4:2:4)
set(gca, 'YTick', 0:3)
xlabel('x', 'FontSize', 16)
ylabel('y', 'FontSize', 16, 'Rotation', 0)
legend('test', 'training')

% Save postscript.
print -depsc fig1


