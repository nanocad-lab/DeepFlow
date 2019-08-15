% Figures 1 and 2 in the grow_tree manual page.
clear

% Sine2 training data.
[X, y] = get_data('sine2');

% Test data.
test.name = 'sine2';
test.p = 400;
test.std = 0;
test.ord = 1;
[Xt, yt, test] = get_data(test);

% Run the method.
tree = grow_tree(X, y, conf);

% Predict the test set.
ft = pred_tree(tree, Xt);

% Stuff needed for a mesh plot.
q = sqrt(test.p);
x1 = linspace(test.x1(1), test.x2(1), q);
x2 = linspace(test.x1(2), test.x2(2), q);
Yt = zeros(q,q); Yt(:) = yt;
Ft = zeros(q,q); Ft(:) = ft;

% Get figures.
fig1 = get_fig('Figure 1');
fig2 = get_fig('Figure 2');

% Plot.
figure(fig1)
hold off
mesh(x1, x2, Yt)
figure(fig2)
hold off
mesh(x1, x2, Ft)

% Configure both plots.
figure(fig1)
grid on
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.15 0.15 0.75 0.7])
set(gca, 'XLim', [test.x1(1) test.x2(1)])
set(gca, 'YLim', [test.x1(2) test.x2(2)])
set(gca, 'ZLim', [-1 1])
set(gca, 'XTick', test.x1(1):5:test.x2(1))
set(gca, 'YTick', test.x1(2):5:test.x2(2))
set(gca, 'ZTick', [-1 0 1])
xlabel('x_1', 'FontSize', 16)
ylabel('x_2', 'FontSize', 16)
zlabel('y', 'FontSize', 16)
figure(fig2)
grid on
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.15 0.15 0.75 0.7])
set(gca, 'XLim', [test.x1(1) test.x2(1)])
set(gca, 'YLim', [test.x1(2) test.x2(2)])
set(gca, 'ZLim', [-1 1])
set(gca, 'XTick', test.x1(1):5:test.x2(1))
set(gca, 'YTick', test.x1(2):5:test.x2(2))
set(gca, 'ZTick', [-1 0 1])
xlabel('x_1', 'FontSize', 16)
ylabel('x_2', 'FontSize', 16)
zlabel('y', 'FontSize', 16)

% Save postscript.
figure(fig1)
print -depsc fig1
figure(fig2)
print -depsc fig2
