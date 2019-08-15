% Figure 3 in the Tutorial Introduction.
clear

% Test set.
x = linspace(0, 1, 100);

% Get figure.
fig = get_fig('Figure 3');

% Centre and radius.
c = 0.5;
r = 0.2;

% Gaussian, Cuachy and Multiquadric RBFs.
h1 = rbf_dm(x, c, r, struct('type', 'm'));
h2 = rbf_dm(x, c, r, struct('type', 'c'));
h3 = rbf_dm(x, c, r, struct('type', 'g'));

% Plot.
hold off
plot(x, h1, 'r-', 'LineWidth', 2)
hold on
plot(x, h2, 'g-', 'LineWidth', 2)
plot(x, h3, 'b-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [0 1])
set(gca, 'YLim', [-0.5 1.5])
set(gca, 'XTick', [0 1])
set(gca, 'YTick', [0 1])
xlabel('x', 'FontSize', 16)
ylabel('h', 'FontSize', 16, 'Rotation', 0)
legend('type ''m''', 'type ''c''', 'type ''g''', 4)

% Save postscript.
print -depsc fig3


