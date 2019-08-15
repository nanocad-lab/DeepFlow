% Figure 4 in the Tutorial Introduction.
clear

% Hermite training data.
[x, y, conf] = get_data('hermite');

% Test data.
conf.p = 1000;
conf.ord = 1;
conf.std = 0;
[xt, yt] = get_data(conf);

% Set up network centres and radii.
c = x;
r = 0.4;

% Compute training set design matrix (use defaults).
H = rbf_dm(x, c, r);

% Solve for the weights. Probable numerical errors so use pinv not inv.
w = pinv(H' * H) * (H' * y);

% Now do the test set prediction.
Ht = rbf_dm(xt, c, r);
ft = Ht * w;

% Get figure.
fig = get_fig('Figure 4');

% Plot.
hold off
plot(xt, yt, 'k--')
hold on
plot(xt, ft, 'r-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-4 4])
set(gca, 'YLim', [0 3])
set(gca, 'XTick', -4:2:4)
set(gca, 'YTick', 0:3)
xlabel('x', 'FontSize', 16)
ylabel('y', 'FontSize', 16, 'Rotation', 0)
legend('target', 'prediction')

% Save postscript.
print -depsc fig4

