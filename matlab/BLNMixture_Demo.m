% BLNMixture_Demo.m
% Demo: simulate → fit → evaluate pdf for a BLN + Uniform mixture

rng(1);

% Ground-truth mixture
trueObj = BLNMixture();
trueObj.K = 3;
trueObj.mu = [-0.6; 0; 0.8];
trueObj.sigma = [0.155; .13; 0.145];
trueObj.uniform_weight = 1e-3;
trueObj.pi = (1-trueObj.uniform_weight)*[0.4; .15; 0.55];
trueObj.QuadNodes = 30;

% Totals and sampling
N = 2000;
n = randi([10 1000], N, 1);
[x, xc, z] = trueObj.rnd(n);

% Sprinkle some obvious outliers into the uniform tail (optional sanity check)
x(1:30) = arrayfun(@(ni) randi([0 ni],1,1), n(1:30));
xc(1:30) = n(1:30) - x(1:30);

% Fit model
fprintf('Fitting BLNMixture...\n');
fitted = BLNMixture.fit(x, xc, 3, 'Restarts', 3, 'UniformWeight', 1e-3, 'Display','iter');

fprintf('Converged: %d | LL: %.3f | iters: %d\n', fitted.converged, fitted.loglik, fitted.iters);
tab = tabulate(fitted.labels);
disp('Label counts (1..K=BLN comps, K+1=uniform):');
disp(tab);

% Evaluate pmf on the observed samples
p = fitted.pdf(x, n);
fprintf('Mean log pmf on data: %.3f\n', mean(log(max(p, realmin))));

% Quick visualization: true vs estimated logits and sigmas
fprintf('True vs estimated params (rows: components):\n');
disp(table(trueObj.mu, fitted.mu, trueObj.sigma, fitted.sigma, 'VariableNames', ...
    {'mu_true','mu_hat','sigma_true','sigma_hat'}));

% Simple scatter of observed fraction vs component MAP label
phat = x ./ n;
figure; scatter(phat, fitted.labels, 8, 'filled'); grid on;
xlabel('Observed fraction x/n'); ylabel('MAP component'); title('BLNMixture Demo: MAP labels');


% --- Also demonstrate shared-sigma fitting ---
fprintf('\nFitting with SharedSigma=true ...\n');
fitted_shared = BLNMixture.fit(x, xc, 3, 'Restarts', 3, 'UniformWeight', 1e-3, 'Display','iter', 'SharedSigma', true);
disp(table(fitted_shared.mu, fitted_shared.sigma, 'VariableNames', {'mu_hat_shared','sigma_hat_shared'}));
3