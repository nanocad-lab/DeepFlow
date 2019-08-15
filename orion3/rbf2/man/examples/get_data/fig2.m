% Figure 2 in the get_data manual page.
clear

% Sine2 training data.
[X, y, conf] = get_data('sine2');

% Test data.
conf.ord = 400;
conf.std = 0;
conf.ord = 1;
[Xt, yt, conf] = get_data(conf);

% Stuff needed for a mesh plot.
q = sqrt(conf.p);
x1 = linspace(conf.x1(1), conf.x2(1), q);
x2 = linspace(conf.x1(2), conf.x2(2), q);
Yt = zeros(q,q); Yt(:) = yt;

% Get figure.
fig = get_fig('Figure 2');

% Plot.
hold off
plot3(X(1,:), X(2,:), y, 'ko', 'MarkerSize', 10)
hold on
mesh(x1, x2, Yt)

% Configure plot.
grid on
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.15 0.20 0.75 0.75])
set(gca, 'XLim', [conf.x1(1) conf.x2(1)])
set(gca, 'YLim', [conf.x1(2) conf.x2(2)])
set(gca, 'ZLim', [-1 1])
set(gca, 'XTick', conf.x1(1):5:conf.x2(1))
set(gca, 'YTick', conf.x1(2):5:conf.x2(2))
set(gca, 'ZTick', [-1 0 1])
xlabel('x_1', 'FontSize', 16)
ylabel('x_2', 'FontSize', 16)
zlabel('y', 'FontSize', 16)

% Save postscript.
print -depsc fig2
