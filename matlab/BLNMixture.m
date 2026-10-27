classdef BLNMixture
% BLNMixture  Mixture of Binomial-Logit-Normal (BLN) components with an optional
%             fixed-weight Uniform noise component.
%
%   Model:
%     For observation i with total n_i and count x_i:
%       p(x_i | params) = (1 - w0) * sum_{k=1}^K pi_k * BLN_k(x_i | n_i; mu_k, sigma_k)
%                         + w0 * 1/(n_i + 1)
%     where BLN(x|n;mu,sigma) = ∫ Binom(x|n, p)  N(logit(p) | mu, sigma^2)  dp,
%     approximated via Gauss–Hermite (GH) quadrature.
%
%   Key methods
%     - obj = BLNMixture.fit(X, Xc, K, 'Name', value, ...)   % EM fit
%     - p   = obj.pdf(x, n)                                  % mixture pmf
%     - [x, xc, z] = obj.rnd(n)                              % simulate data
%
%   Example:
%     % Simulate from a 2-component BLN + tiny uniform noise
%     obj = BLNMixture(); obj.K = 2;
%     obj.mu = [ -0.5; 0.7 ]; obj.sigma = [0.6; 0.5];
%     obj.uniform_weight = 1e-3; obj.pi = (1-obj.uniform_weight)*[0.4; 0.6];
%     n = randi([20 80], 1000, 1);
%     [x, xc] = obj.rnd(n);
%     % Fit back
%     fitted = BLNMixture.fit(x, xc, 2, 'Restarts', 3, 'UniformWeight', 1e-3, 'Display','iter');
%     % Evaluate pmf
%     p = fitted.pdf(x, n);
%
%   Pejman-style BLN + Uniform mixture, packaged as a class.

    properties
        % Model size
        K (1,1) double {mustBeInteger,mustBePositive} = 1
        % Component parameters
        mu (:,1) double = 0
        sigma (:,1) double {mustBePositive} = 0.5
        pi (:,1) double {mustBeNonnegative} = 1   % sums to (1 - uniform_weight)
        % Fixed uniform noise weight
        uniform_weight (1,1) double {mustBeGreaterThanOrEqual(uniform_weight,0), mustBeLessThanOrEqual(uniform_weight,0.5)} = 1e-3
        % Quadrature
        QuadNodes (1,1) double {mustBeInteger,mustBePositive} = 30
        % Fit diagnostics
        loglik (1,1) double = NaN
        iters (1,1) double = 0
        converged (1,1) logical = false
        resp double = []     % N x (K+1) responsibilities (last col = uniform)
        labels double = []   % N x 1 MAP in {1..K+1}
        details struct = struct()
    end

    methods (Static)
        
        function obj = fit(X, Xc, K, varargin)
            % FIT  EM fit of a K-component BLN mixture with optional shared sigma.
            %
            % obj = BLNMixture.fit(X, Xc, K, 'Name', value, ...)
            %
            % Name-Value options:
            %   'MaxIter'       (200)   maximum EM iterations
            %   'Tol'           (1e-6)  relative LL tolerance
            %   'QuadNodes'     (30)    GH nodes
            %   'Restarts'      (3)     random restarts; best LL kept
            %   'Display'       ('off') 'off' | 'iter'
            %   'Init'          (struct) fields mu (Kx1), sigma (Kx1>0 or scalar), pi (Kx1>0, sums to 1-w0)
            %   'UseFminunc'    (auto)  true/false
            %   'UniformWeight' (1e-3)  fixed w0 in [0, 0.5]
            %   'SharedSigma'   (false) if true, enforce sigma_k = sigma for all k
            p = inputParser;
            addParameter(p,'MaxIter',200);
            addParameter(p,'Tol',1e-6);
            addParameter(p,'QuadNodes',30);
            addParameter(p,'Restarts',3);
            addParameter(p,'Display','off');
            addParameter(p,'Init',struct());
            addParameter(p,'UseFminunc',[]);
            addParameter(p,'UniformWeight',1e-3);
            addParameter(p,'SharedSigma',false);
            parse(p,varargin{:});
            MaxIter    = p.Results.MaxIter;
            Tol        = p.Results.Tol;
            NN         = p.Results.QuadNodes;
            Restarts   = p.Results.Restarts;
            Disp       = p.Results.Display;
            Init       = p.Results.Init;
            UseFminunc = p.Results.UseFminunc;
            w0         = max(min(p.Results.UniformWeight, 0.5), 0);
            SharedSigma= logical(p.Results.SharedSigma);

            X = X(:); Xc = Xc(:);
            ok = isfinite(X) & isfinite(Xc) & X>=0 & Xc>=0;
            X = X(ok); Xc = Xc(ok);
            n = X + Xc;
            Nobs = numel(X);

            if isempty(UseFminunc)
                UseFminunc = ~isempty(which('fminunc'));
            end

            % constants
            logC = gammaln(n+1) - gammaln(X+1) - gammaln(Xc+1);
            logU = -log(n + 1);                      % N x 1 (uniform log pmf)
            [xh, wh] = BLNMixture.gh_nodes_weights(NN);

            best = []; bestLL = -inf;
            tStartAll = tic;

            for r = 1:max(1,Restarts)
                % ---- initialization ----
                if ~isempty(fieldnames(Init))
                    mu    = BLNMixture.col(Init, 'mu', K, 0);
                    if isfield(Init,'sigma') && isscalar(Init.sigma) && SharedSigma
                        sigma_shared = max(Init.sigma, 1e-4);
                        sigma = repmat(sigma_shared, K, 1);
                    else
                        sigma = max(BLNMixture.col(Init, 'sigma', K, 0.5), 1e-4);
                        if SharedSigma
                            sigma_shared = median(sigma);
                            sigma = repmat(sigma_shared, K, 1);
                        end
                    end
                    if isfield(Init,'pi')
                        pi = BLNMixture.col(Init, 'pi', K, (1-w0)/K);
                        s = sum(pi); if s<=0, pi = (1-w0)*ones(K,1)/K; else, pi = (1-w0)*pi/s; end
                    else
                        pi = (1-w0)*ones(K,1)/K;
                    end
                else
                    % kmeans-like init on smoothed logits
                    ph = (X + 0.5) ./ (n + 1);
                    z  = log(ph) - log1p(-ph);
                    z(~isfinite(z)) = 0;
                    if exist('kmeans','file') && K>1 && Nobs>=K
                        labels0 = kmeans(z, K, 'Replicates',3,'MaxIter',100,'Display','off');
                    else
                        edges = quantile(z, linspace(0,1,K+1));
                        labels0 = discretize(z, edges);
                        nanmask = isnan(labels0);
                        labels0(nanmask) = randi(K, sum(nanmask), 1);
                    end
                    mu = zeros(K,1); sigma = zeros(K,1); pi = zeros(K,1);
                    for k = 1:K
                        idx = (labels0==k);
                        if ~any(idx), idx = randi(Nobs, max(1,round(Nobs/K)), 1); end
                        zk  = z(idx);
                        mu(k)    = median(zk);
                        sigma(k) = max(std(zk), 0.3);
                        pi(k)    = (1-w0)*sum(idx)/Nobs;
                    end
                    s = sum(pi); if s<=0, pi = (1-w0)*ones(K,1)/K; else, pi = (1-w0)*pi/s; end
                    if SharedSigma
                        sigma_shared = median(sigma);
                        sigma = repmat(sigma_shared, K, 1);
                    end
                end

                prevLL = -inf;
                converged = false;
                tStart = tic;

                % ---- EM ----
                for it = 1:MaxIter
                    % E-step: responsibilities over K BLN + 1 uniform component
                    logpk = BLNMixture.comp_loglik_matrix(X, Xc, logC, mu, sigma, xh, wh); % N x K
                    logPi = log(max(pi(:).', realmin));                                    % 1 x K
                    logPu = log(max(w0, realmin));                                         % scalar
                    logR  = [bsxfun(@plus, logpk, logPi), logU + logPu];                   % N x (K+1)
                    [logNorm, ~] = BLNMixture.logsumexp(logR, 2);
                    logResp = bsxfun(@minus, logR, logNorm);
                    resp    = exp(logResp);                                                % N x (K+1)
                    LL      = sum(logNorm);

                    % split responsibilities
                    respBLN = resp(:,1:K);
                    respUNI = resp(:,K+1);

                    % M-step:
                    % (1) update mixture weights (BLN part sums to 1 - w0)
                    Nk  = sum(respBLN, 1)';                 % K x 1
                    SNk = sum(Nk);
                    if SNk<=0
                        pi = (1-w0)*ones(K,1)/K;
                    else
                        pi = (1-w0) * (Nk / SNk);
                    end

                                        % We'll implement separate branches for shared vs. free sigmas below.
                    if ~SharedSigma
                        for k = 1:K
                            wk = respBLN(:,k) / max(Nk(k), realmin);   % normalize per comp
                            theta0 = [mu(k), log(max(sigma(k),1e-4))];
                            if UseFminunc
                                opts = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                                    'SpecifyObjectiveGradient',true,'MaxIterations',200,'MaxFunctionEvaluations',5000);
                                [th, ~] = fminunc(@(th) BLNMixture.nll_grad_weighted(th, X, Xc, logC, wk, xh, wh), theta0, opts);
                            else
                                opts = optimset('Display','off','MaxIter',400,'MaxFunEvals',8000);
                                [th, ~] = fminsearch(@(th) BLNMixture.nll_only(th, X, Xc, logC, wk, xh, wh), theta0, opts);
                            end
                            mu(k)    = th(1);
                            sigma(k) = max(exp(th(2)), 1e-12);
                        end
                    else
                        % Shared sigma: alternate updating {mu_k} with fixed sigma, then sigma
                        % 1) Update mu_k (fixed sigma_shared)
                        for k = 1:K
                            wk = respBLN(:,k) / max(Nk(k), realmin);
                            if UseFminunc
                                opts = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                                    'SpecifyObjectiveGradient',true,'MaxIterations',200,'MaxFunctionEvaluations',5000);
                                mu(k) = fminunc(@(muk) BLNMixture.nll_grad_mu_only(muk, X, Xc, logC, wk, sigma(k), xh, wh), mu(k), opts);
                            else
                                opts = optimset('Display','off','MaxIter',400,'MaxFunEvals',8000);
                                mu(k) = fminsearch(@(muk) BLNMixture.nll_mu_only(muk, X, Xc, logC, wk, sigma(k), xh, wh), mu(k), opts);
                            end
                        end
                        % 2) Update shared sigma (log-scale) using all components
                        logsig0 = log(max(sigma(1),1e-12));
                        if UseFminunc
                            opts = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                                'SpecifyObjectiveGradient',true,'MaxIterations',200,'MaxFunctionEvaluations',5000);
                            [logsig, ~] = fminunc(@(ls) BLNMixture.nll_grad_logsigma_shared(ls, X, Xc, logC, respBLN, Nk, mu, xh, wh), logsig0, opts);
                        else
                            opts = optimset('Display','off','MaxIter',400,'MaxFunEvals',8000);
                            logsig = fminsearch(@(ls) BLNMixture.nll_logsigma_shared(ls, X, Xc, logC, respBLN, Nk, mu, xh, wh), logsig0, opts);
                        end
                        sig = max(exp(logsig), 1e-12);
                        sigma = repmat(sig, K, 1);
                    end

                    % convergence test
                    relImpr = (LL - prevLL) / (1 + abs(prevLL));
                    if strcmpi(Disp,'iter')
                        if SharedSigma
                            fprintf('Restart %d | iter %3d  LL = %.6f  relImpr = %.3e  (w0=%.2e, shared sigma)\n', r, it, LL, relImpr, w0);
                        else
                            fprintf('Restart %d | iter %3d  LL = %.6f  relImpr = %.3e  (w0=%.2e)\n', r, it, LL, relImpr, w0);
                        end
                    end
                    if relImpr < Tol && it>2
                        converged = true;
                        break;
                    end
                    prevLL = LL;
                end
                elapsed = toc(tStart);

                if LL > bestLL
                    bestLL = LL;
                    best = BLNMixture();
                    best.K = K; best.mu = mu; best.sigma = sigma; best.pi = pi;
                    best.uniform_weight = w0; best.QuadNodes = NN;
                    best.loglik = LL; best.iters = it; best.converged = converged;
                    best.resp = [respBLN respUNI];
                    best.labels = double(BLNMixture.maxcomp([respBLN respUNI]));
                    best.details = struct('elapsed_sec',elapsed,'QuadNodes',NN,'Restarts',Restarts, ...
                                          'total_elapsed',NaN,'UseFminunc',UseFminunc,'SharedSigma',SharedSigma);
                end
            end

            best.details.total_elapsed = toc(tStartAll);
            obj = best;
        end
    end

    methods
        function p = pdf(this, x, n, returnLog)
            % PDF  Mixture pmf p(x|n) for counts x with totals n.
            % p = obj.pdf(x, n) returns numeric pmf values (vector).
            % p = obj.pdf(x, n, true) returns log-pmf values.
            if nargin < 4, returnLog = false; end
            x = x(:); n = n(:);
            if numel(x) ~= numel(n)
                error('x and n must be the same length.');
            end
            X  = x; Xc = n - x;
            if any(X<0) || any(Xc<0)
                error('Counts must satisfy 0 <= x <= n.');
            end

            logC = gammaln(n+1) - gammaln(X+1) - gammaln(Xc+1);
            [xh, wh] = BLNMixture.gh_nodes_weights(this.QuadNodes);
            logpk = BLNMixture.comp_loglik_matrix(X, Xc, logC, this.mu, this.sigma, xh, wh); % N x K
            % mixture over BLN comps
            logPi = log(max(this.pi(:).', realmin));      % 1 x K
            logBLNmix = BLNMixture.logsumexp(bsxfun(@plus, logpk, logPi), 2); % N x 1
            % add uniform
            logU = -log(n + 1);
            logwBLN = log(max(1 - this.uniform_weight, realmin));
            logwU   = log(max(this.uniform_weight, realmin));
            logp = BLNMixture.logsumexp([logwBLN + logBLNmix, logwU + logU], 2);
            if returnLog
                p = logp;
            else
                p = exp(logp);
            end
        end

        function [x, xc, z] = rnd(this, n)
            % RND  Draw counts from the fitted mixture given totals n (vector or scalar).
            % [x, xc, z] = obj.rnd(n)
            %   n : Nx1 (or scalar) binomial totals
            %   x : Nx1 successes, xc : Nx1 failures, z : component labels in {1..K+1}
            if isscalar(n), n = repmat(n, 1, 1); end
            n = n(:);
            N = numel(n);

            % mixture weights across K BLN + uniform
            wBLN = this.pi(:);           % sums to (1 - w0)
            w0   = this.uniform_weight;
            mixw = [wBLN; w0];           % length K+1
            mixw = mixw / sum(mixw);

            % sample components
            edges = [0; cumsum(mixw)];
            u = rand(N,1);
            z = sum(u >= edges.', 2);    % component labels

            x  = zeros(N,1);
            xc = zeros(N,1);

            % BLN components
            for k = 1:this.K
                idx = (z==k);
                nk  = n(idx);
                if ~any(idx), continue; end
                Z   = this.mu(k) + this.sigma(k)*randn(sum(idx),1);
                pk  = 1 ./ (1 + exp(-Z));
                x(idx)  = binornd(nk, pk);
                xc(idx) = nk - x(idx);
            end
            % Uniform component (K+1)
            idxU = (z==this.K+1);
            if any(idxU)
                nk = n(idxU);
                % sample uniformly over {0..n}
                x(idxU)  = arrayfun(@(ni) randi([0 ni],1,1), nk);
                xc(idxU) = nk - x(idxU);
            end
        end
    end

    % ---------- helpers ----------
    methods (Static, Access=private)
        function idx = maxcomp(R)
            [~, idx] = max(R, [], 2);
        end

        function v = col(S, fname, K, defaultVal)
            if isfield(S, fname)
                a = S.(fname);
                if numel(a) ~= K, error('Init.%s must be length %d', fname, K); end
                v = a(:);
            else
                v = repmat(defaultVal, K, 1);
            end
        end

        function [xh, wh] = gh_nodes_weights(N)
            % Golub-Welsch GH nodes/weights for std normal expectations
            persistent cacheN cacheX cacheW
            if ~isempty(cacheN) && cacheN==N
                xh = cacheX; wh = cacheW; return;
            end
            i = (1:N-1)'; a = zeros(N,1); b = sqrt(i/2);
            J = diag(a) + diag(b,1) + diag(b,-1);
            [V,D] = eig(J);
            [xh, idx] = sort(diag(D));
            V = V(:, idx);
            wh = V(1,:).^2;     % sums to 1 for E[g(Z)] with Z~N(0,1)
            xh = xh(:); wh = wh(:);
            cacheN = N; cacheX = xh; cacheW = wh;
        end

        function L = comp_loglik_matrix(X, Xc, logC, mu, sigma, xh, wh)
            % N x K log p(x_i | mu_k, sigma_k) under BLN via GH quadrature
            N = numel(X); K = numel(mu);
            Zbase = sqrt(2)*xh.';     % 1 x NN
            L = -inf(N, K);
            for k = 1:K
                Z  = mu(k) + sigma(k) * Zbase;           % 1 x NN
                P  = 1 ./ (1 + exp(-Z));                 % 1 x NN
                LL = bsxfun(@plus, logC, X.*log(P) + Xc.*log1p(-P));  % N x NN
                LL = bsxfun(@plus, LL, log(wh.'));                   % add GH weights
                L(:,k) = BLNMixture.logsumexp(LL, 2);
            end
        end

        function [f,g] = nll_grad_weighted(theta, X, Xc, logC, w, xh, wh)
            % Weighted negative log-likelihood and gradient for single BLN component
            mu    = theta(1);
            sigma = max(exp(theta(2)), 1e-12);

            Z  = mu + sigma*sqrt(2)*xh.';      % 1 x NN
            P  = 1 ./ (1 + exp(-Z));           % 1 x NN

            LL = bsxfun(@plus, logC, X.*log(P) + Xc.*log1p(-P));  % N x NN
            LL = bsxfun(@plus, LL, log(wh.'));                    % add GH weights
            m  = max(LL, [], 2);
            s  = m + log(sum(exp(LL - m), 2));                    % log-sum-exp per obs

            % weighted NLL
            f = -sum(w .* s);

            if nargout>1
                % posterior over nodes
                A = exp(bsxfun(@minus, LL, s));                   % N x NN, rows sum to 1
                n  = X + Xc;
                sc = bsxfun(@minus, X, n.*P);                     % N x NN (score wrt z)
                Ez   = sum(A .* sc, 2);                           % N x 1
                Esg  = sum(A .* bsxfun(@times, sc, sqrt(2)*xh.'), 2); % N x 1

                gmu    = -sum(w .* Ez);
                dsigma = -sum(w .* Esg);
                gsig   = dsigma * sigma;                           % chain rule for log-sigma
                g = [gmu, gsig];
            end
        end

        function f = nll_only(theta, X, Xc, logC, w, xh, wh)
            f = BLNMixture.nll_grad_weighted(theta, X, Xc, logC, w, xh, wh);
        end

        function [f,gmu] = nll_grad_mu_only(muk, X, Xc, logC, w_norm, sigma_fixed, xh, wh)
            % Negative log-likelihood (normalized weights) as a function of mu only (sigma fixed).
            theta = [muk, log(max(sigma_fixed,1e-12))];
            [f,g] = BLNMixture.nll_grad_weighted(theta, X, Xc, logC, w_norm, xh, wh);
            if nargout>1, gmu = g(1); end
        end

        function f = nll_mu_only(muk, X, Xc, logC, w_norm, sigma_fixed, xh, wh)
            f = BLNMixture.nll_grad_mu_only(muk, X, Xc, logC, w_norm, sigma_fixed, xh, wh);
        end

        function [f,gls] = nll_grad_logsigma_shared(logsig, X, Xc, logC, respBLN, Nk, mu, xh, wh)
            % Total NLL across components with shared sigma (on log-scale).
            % Uses normalized per-component weights internally then scales by Nk.
            K = numel(mu);
            sigma_shared = max(exp(logsig), 1e-12);
            f = 0; gls = 0;
            for k = 1:K
                if Nk(k) <= 0, continue; end
                wk = respBLN(:,k) / max(Nk(k), realmin);
                theta = [mu(k), log(sigma_shared)];
                [fk, gk] = BLNMixture.nll_grad_weighted(theta, X, Xc, logC, wk, xh, wh);
                f   = f   + Nk(k) * fk;
                gls = gls + Nk(k) * gk(2);     % derivative wrt log-sigma
            end
        end

        function f = nll_logsigma_shared(logsig, X, Xc, logC, respBLN, Nk, mu, xh, wh)
            f = BLNMixture.nll_grad_logsigma_shared(logsig, X, Xc, logC, respBLN, Nk, mu, xh, wh);
        end

        function [s, m] = logsumexp(A, dim)
            if nargin<2, dim = 1; end
            m = max(A, [], dim);
            s = m + log(sum(exp(bsxfun(@minus, A, m)), dim));
        end
    end
end
