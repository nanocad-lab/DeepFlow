% Figure 2 in the rbf_dm manual page.
clear

% Set of ordered, evenly spaced x-values.
x = linspace(-2, 2, 1000);

% Gaussian, Cauchy, Multiquadric and Inverse RBFs.
g1 = rbf_dm(x, 0, 1, struct('exp', 1));
g2 = rbf_dm(x, 0, 1, struct('exp', 2));
g3 = rbf_dm(x, 0, 1, struct('exp', 3));
g4 = rbf_dm(x, 0, 1, struct('exp', 4));

% Get figure.
fig = get_fig('Figure 2');

% Plot.
hold off
plot(x, g4, 'b-', 'LineWidth', 2)
hold on
plot(x, g3, 'c-', 'LineWidth', 2)
plot(x, g2, 'g-', 'LineWidth', 2)
plot(x, g1, 'y-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-2 2])
set(gca, 'YLim', [0 1])
set(gca, 'XTick', [-2 0 2])
set(gca, 'YTick', [0 1])
xlabel('x', 'FontSize', 16)
ylabel('h', 'FontSize', 16, 'Rotation', 0)
legend('4', '3', '2', '1')

% Save postscript.
print -depsc fig2
