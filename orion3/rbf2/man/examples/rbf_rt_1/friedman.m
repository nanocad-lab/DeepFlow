% The 4D example in the rbf_rt_1 manual page.
clear

% 4D data set.
[X, y] = get_data('friedman');

% Configure the method.
conf.minmem = [3 4 5];
conf.scales = [7 8 9];
conf.timer = '4D';

% Run the method.
[C, R, w, info] = rbf_rt_1(X, y, conf);

% Print out split statistics.
disp(info.tree.split.number)
disp(info.tree.split.order)