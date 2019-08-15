% Figure 2 in the rbf_rr_2 manual page.
clear

% Hermite training data.
[x, y] = get_data('hermite');

% Test data.
test.name = 'hermite';
test.p = 1000;
test.ord = 1;
test.std = 0;
[xt, yt] = get_data(test);

% Configure the method.
conf.lambdas = 10.^[-2:-2:-10];
conf.scales = [0.2 0.1];
conf.msc = 'GCV';

% Run the method.
[c, r, w, info] = rbf_rr_2(x, y, conf);

% Calculate GCV as a function of lambda.
H = info.H; HH = H' * H; [p,m] = size(H);
lams = 10.^linspace(-10,0,50);
for i = 1:length(lams)
  A = inv(HH + lams(i)*eye(m));
  P = eye(p) - H * A * H';
  errs(i) = p * (y' * P) * (P * y) / trace(P)^2;
end

% Get figure.
fig = get_fig('Figure 2');

% Plot.
hold off
plot(log10(info.lams), log10(info.errs), 'r+', 'MarkerSize', 10)
hold on
plot(log10(lams), log10(errs), 'm-', 'LineWidth', 2)

% Configure plot.
dy = 0.1;
y1 = dy * floor(min(log10(info.errs)) / dy);
y2 = dy * ceil(max(log10(info.errs)) / dy);
set(gca, 'FontSize', 14)
set(gca, 'Position', [0.15 0.15 0.8 0.8])
set(gca, 'XLim', [-10 0])
set(gca, 'YLim', [y1 y2])
set(gca, 'XTick', -10:2:0)
set(gca, 'YTick', y1:dy:y2)
xlabel('log(\lambda)', 'FontSize', 14)
ylabel('log(GCV)', 'FontSize', 14)

% Save postscript.
% print -depsc fig2
