% Figure 3 of the plot_ras manual page.
clear

% Load results (data, methods, sizes, perfs)
% calculated by fig2b.m and put in fig3.mat.
load fig3

% Configure the Rasmuusen plot.
conf.width = 8;
conf.mfs = 14;
conf.afs = 16;
conf.ebc = 'mrb';
conf.ebs = [2 3 2];
conf.mnc = 'mrb';

% Create the Rasmussen diagram.
plot_ras(data, methods, sizes, perfs, pvals, conf)

% Save postscript.
print -depsc fig3
