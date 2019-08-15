% Figure 3 in the rbf_dm manual page.
clear

% Use get_data for grid of x-values.
conf.name = 'sine2';
conf.p = 1600;
conf.ord = 1;
[X, y, conf] = get_data(conf);

% Setup centres.
C = [5 7 2; 2 0 -3];

% Setup radii.
R = [2 1 1; 1 3 1];

% Design matrix.
H = rbf_dm(X, C, R);

% Weights.
w = [2; 1; 0.5];

% Function.
f = H * w;

% Grid the function.
q = sqrt(conf.p);
F = zeros(q, q);
F(:) = f;
v1 = linspace(conf.x1(1), conf.x2(1), q);
v2 = linspace(conf.x1(2), conf.x2(2), q);

% Get a figure.
fig = get_fig('Figure 3');

% Plot.
hold off
mesh(v1, v2, F);

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [conf.x1(1) conf.x2(1)])
set(gca, 'YLim', [conf.x1(2) conf.x2(2)])
set(gca, 'ZLim', [0 2])
set(gca, 'XTick', conf.x1(1):5:conf.x2(1))
set(gca, 'YTick', conf.x1(2):5:conf.x2(2))
set(gca, 'ZTick', [0 1 2])
xlabel('x_1', 'FontSize', 16)
ylabel('x_2', 'FontSize', 16)
zlabel('y', 'FontSize', 16)

% Save postscript.
print -depsc fig3
