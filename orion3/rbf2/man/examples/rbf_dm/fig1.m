% Figure 1 in the rbf_dm manual page.
clear

% Set of ordered, evenly spaced x-values.
x = linspace(-4, 4, 1000);

% Gaussian, Cauchy, Multiquadric and Inverse RBFs.
m = rbf_dm(x, 0, 1, struct('type', 'm'));
i = rbf_dm(x, 0, 1, struct('type', 'i'));
c = rbf_dm(x, 0, 1, struct('type', 'c'));
g = rbf_dm(x, 0, 1, struct('type', 'g'));

% Get figure.
fig = get_fig('Figure 1');

% Plot.
hold off
plot(x, m, 'm-', 'LineWidth', 2)
hold on
plot(x, i, 'r-', 'LineWidth', 2)
plot(x, c, 'c-', 'LineWidth', 2)
plot(x, g, 'g-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-4 4])
set(gca, 'YLim', [0 2])
set(gca, 'XTick', [-4 -2 0 2 4])
set(gca, 'YTick', [0 1 2])
xlabel('x', 'FontSize', 16)
ylabel('h', 'FontSize', 16, 'Rotation', 0)
legend('Multiquadric', 'Inverse', 'Cauchy', 'Gaussian')

% Save postscript.
print -depsc fig1
