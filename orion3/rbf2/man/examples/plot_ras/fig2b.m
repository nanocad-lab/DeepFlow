% Add MARS results and plot Figure 2 of the plot_ras manual page.
clear

% Load results (methods, sizes, perfs)
% calculated by fig2a.m and put in fig2.mat.
load fig2

% Add best MARS results.
perfs(1, 3, 1) = 0.28;  % mean for 100 cases
perfs(1, 3, 2) = 0.17;  % std for 100 cases
perfs(2, 3, 1) = 0.12;  % mean for 200 cases
perfs(2, 3, 2) = 0.07;  % std for 200 cases

% Diagram title.
data = 'friedman/Z';

% Dummy pvals.
pvals = ones(nsiz, nmet, nmet);
pvals(1, :, :) = [ 0  0  0; -1  0 -1; -1 -1 0];
pvals(2, :, :) = [ 0  0  0; -1  0 -1; -1 -1 0];

% Create the Rasmussen diagram.
plot_ras(data, methods, sizes, perfs, pvals)

% Save postscript.
print -depsc fig2

% Save results for the fig3.m script.
save fig3 data methods sizes perfs pvals
