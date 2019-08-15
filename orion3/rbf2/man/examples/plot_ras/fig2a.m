% Get the data for Figure 2 of the plot_ras manual page.
clear

% Method names.
methods = {'stupid', 'rbf_rt_1', 'mars'};
nmet = length(methods);

% Test set from get_data.
test.name = 'friedman';
test.p = 5000;
test.std = 0;
[xt, yt] = get_data(test);

% Get the variance of the test set outputs.
tvar = test.p * std(yt)^2;

% Configuration structures for rbf_rt_1.
conf = struct('scales', [7 9], 'minmem', [3 4]);

% Sizes of test sets.
sizes = [100 200];
nsiz = length(sizes);

% Number of replications of each training set.
nrep = 10;

% Initialise.
results = zeros(nsiz, nrep, 2);

% Loop over sizes.
t1 = get_tmr(struct('name', 'sizes', 'n', nsiz));
for i = 1:nsiz

  % What size?
  p = sizes(i);

  % Loop over replications.
  t2 = get_tmr(struct('name', 'replications', 'n', nrep));
  for j = 1:nrep

    % Get a new training set.
    [x, y] = get_data(struct('name', 'friedman', 'p', p));

    % Predict with the stupid method.
    ft = sum(y) / length(y);
    results(i, j, 1) = (yt - ft)' * (yt - ft) / tvar;

    % Predict with rbf_rt_1.
    [c, r, w, info] = rbf_rt_1(x, y, conf);
    H = rbf_dm(xt, c, r, info.dmc);
    ft = H * w;
    results(i, j, 2) = (yt - ft)' * (yt - ft) / tvar;

    % Increment the replication timer.
    inc_tmr(t2)

  end

  % Close the replication timer.
  pause(0.5)
  close(t2)

  % Increment the sizes timer.
  inc_tmr(t1)

end

% Close the sizes timer.
pause(0.5)
close(t1)

% Convert replication results into performance figures.
perfs = zeros(nsiz, nmet, 2);
for i = 1:nsiz
  perfs(i, 1, 1) = mean(results(i, :, 1));
  perfs(i, 1, 2) = std(results(i, :, 1));
  perfs(i, 2, 1) = mean(results(i, :, 2));
  perfs(i, 2, 2) = std(results(i, :, 2));
end

% Save these results for the fig2b.m script.
save fig2 methods nmet sizes nsiz perfs
