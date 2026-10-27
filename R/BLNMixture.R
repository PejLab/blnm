## BLNMixture.R
## R implementation of a Binomial-Logit-Normal mixture with optional uniform noise

`%||%` <- function(a, b) if (!is.null(a)) a else b

.machine_realmin <- .Machine$double.xmin
.machine_eps <- .Machine$double.eps

# ------- numerical helpers -------

.gh_nodes_weights <- local({
  cache <- new.env(parent = emptyenv())
  function(N) {
    stopifnot(length(N) == 1L, N >= 1)
    key <- as.character(N)
    if (exists(key, envir = cache, inherits = FALSE)) {
      return(get(key, envir = cache, inherits = FALSE))
    }
    if (N == 1L) {
      res <- list(nodes = 0, weights = 1)
      assign(key, res, envir = cache)
      return(res)
    }
    i <- seq_len(N - 1L)
    b <- sqrt(i / 2)
    J <- matrix(0, nrow = N, ncol = N)
    J[cbind(i, i + 1L)] <- b
    J[cbind(i + 1L, i)] <- b
    eig <- eigen(J, symmetric = TRUE, only.values = FALSE)
    idx <- order(eig$values)
    nodes <- eig$values[idx]
    weights <- eig$vectors[1L, idx]^2
    res <- list(nodes = nodes, weights = weights)
    assign(key, res, envir = cache)
    res
  }
})

.row_logsumexp <- function(mat) {
  if (!is.matrix(mat)) mat <- as.matrix(mat)
  m <- apply(mat, 1L, max)
  m + log(rowSums(exp(mat - m)))
}

.col_vec <- function(x) matrix(x, ncol = 1L)

.comp_loglik_matrix <- function(X, Xc, logC, mu, sigma, nodes, weights) {
  N <- length(X)
  K <- length(mu)
  NN <- length(nodes)
  log_wh <- log(weights)
  Zbase <- sqrt(2) * nodes
  out <- matrix(-Inf, nrow = N, ncol = K)
  logC_mat <- matrix(logC, nrow = N, ncol = NN)
  for (k in seq_len(K)) {
    Z <- mu[k] + sigma[k] * Zbase
    P <- plogis(Z)
    logP <- log(P)
    log1mP <- log1p(-P)
    LL <- logC_mat + outer(X, logP, "*") + outer(Xc, log1mP, "*")
    LL <- sweep(LL, 2L, log_wh, "+")
    out[, k] <- .row_logsumexp(LL)
  }
  out
}

.nll_grad_weighted <- function(theta, X, Xc, logC, w, nodes, weights) {
  mu <- theta[1L]
  sigma <- max(exp(theta[2L]), 1e-12)
  NN <- length(nodes)
  N <- length(X)
  Z <- mu + sigma * sqrt(2) * nodes
  P <- plogis(Z)
  logP <- log(P)
  log1mP <- log1p(-P)
  logC_mat <- matrix(logC, nrow = N, ncol = NN)
  LL <- logC_mat + outer(X, logP, "*") + outer(Xc, log1mP, "*")
  LL <- sweep(LL, 2L, log(weights), "+")
  m <- apply(LL, 1L, max)
  s <- m + log(rowSums(exp(LL - m)))
  f <- -sum(w * s)
  if (!is.null(w) && !all(w == 0)) {
    A <- exp(LL - s)
  } else {
    A <- matrix(0, nrow = N, ncol = NN)
  }
  n <- X + Xc
  sc <- matrix(X, nrow = N, ncol = NN) - matrix(n, nrow = N, ncol = NN) * matrix(P, nrow = N, ncol = NN, byrow = TRUE)
  Ez <- rowSums(A * sc)
  Esg <- rowSums(A * sweep(sc, 2L, sqrt(2) * nodes, "*"))
  gmu <- -sum(w * Ez)
  dsigma <- -sum(w * Esg)
  gsig <- dsigma * sigma
  list(value = f, grad = c(gmu, gsig))
}

.nll_weighted_value <- function(theta, X, Xc, logC, w, nodes, weights) {
  .nll_grad_weighted(theta, X, Xc, logC, w, nodes, weights)$value
}

.nll_weighted_grad <- function(theta, X, Xc, logC, w, nodes, weights) {
  .nll_grad_weighted(theta, X, Xc, logC, w, nodes, weights)$grad
}

.nll_grad_mu_only <- function(mu, X, Xc, logC, w_norm, sigma_fixed, nodes, weights) {
  theta <- c(mu, log(max(sigma_fixed, 1e-12)))
  res <- .nll_grad_weighted(theta, X, Xc, logC, w_norm, nodes, weights)
  list(value = res$value, grad = res$grad[1L])
}

.nll_mu_only_value <- function(mu, X, Xc, logC, w_norm, sigma_fixed, nodes, weights) {
  .nll_grad_mu_only(mu, X, Xc, logC, w_norm, sigma_fixed, nodes, weights)$value
}

.nll_mu_only_grad <- function(mu, X, Xc, logC, w_norm, sigma_fixed, nodes, weights) {
  .nll_grad_mu_only(mu, X, Xc, logC, w_norm, sigma_fixed, nodes, weights)$grad
}

.nll_grad_logsigma_shared <- function(logsigma, X, Xc, logC, respBLN, Nk, mu, nodes, weights) {
  sigma_shared <- max(exp(logsigma), 1e-12)
  total_value <- 0
  total_grad <- 0
  for (k in seq_along(mu)) {
    if (Nk[k] <= .machine_eps) next
    wk <- respBLN[, k] / max(Nk[k], .machine_realmin)
    theta <- c(mu[k], log(sigma_shared))
    res <- .nll_grad_weighted(theta, X, Xc, logC, wk, nodes, weights)
    total_value <- total_value + Nk[k] * res$value
    total_grad <- total_grad + Nk[k] * res$grad[2L]
  }
  list(value = total_value, grad = total_grad)
}

.nll_logsigma_shared_value <- function(logsigma, X, Xc, logC, respBLN, Nk, mu, nodes, weights) {
  .nll_grad_logsigma_shared(logsigma, X, Xc, logC, respBLN, Nk, mu, nodes, weights)$value
}

.nll_logsigma_shared_grad <- function(logsigma, X, Xc, logC, respBLN, Nk, mu, nodes, weights) {
  .nll_grad_logsigma_shared(logsigma, X, Xc, logC, respBLN, Nk, mu, nodes, weights)$grad
}

# ------- core API -------

bln_mixture_fit <- function(X, Xc, K,
                            max_iter = 200,
                            tol = 1e-6,
                            quad_nodes = 30,
                            restarts = 3,
                            display = c("off", "iter"),
                            init = list(),
                            use_bfgs = TRUE,
                            uniform_weight = 1e-3,
                            shared_sigma = FALSE,
                            random_seed = NULL) {
  if (!is.null(random_seed)) set.seed(random_seed)
  display <- match.arg(display)
  shared_sigma <- isTRUE(shared_sigma)
  X <- as.numeric(X)
  Xc <- as.numeric(Xc)
  keep <- is.finite(X) & is.finite(Xc) & X >= 0 & Xc >= 0
  X <- X[keep]
  Xc <- Xc[keep]
  n <- X + Xc
  if (length(X) == 0L) stop("No valid observations after filtering.")
  K <- as.integer(K)
  if (K <= 0) stop("K must be positive.")
  w0 <- max(min(uniform_weight, 0.5), 0)
  logC <- lgamma(n + 1) - lgamma(X + 1) - lgamma(Xc + 1)
  nodes_weights <- .gh_nodes_weights(quad_nodes)
  nodes <- nodes_weights$nodes
  weights <- nodes_weights$weights
  best <- NULL
  bestLL <- -Inf
  time_start_all <- proc.time()[3L]
  for (r in seq_len(max(1L, restarts))) {
    if (length(init)) {
      if (!is.null(init$mu) && length(init$mu) != K) {
        stop("init$mu must have length K.")
      }
      mu <- init$mu %||% rep(0, K)
      if (!is.null(init$sigma) && !shared_sigma && length(init$sigma) != K) {
        stop("init$sigma must have length K when shared_sigma = FALSE.")
      }
      if (!is.null(init$sigma)) {
        if (shared_sigma && length(init$sigma) == 1L) {
          sigma_shared <- max(init$sigma, 1e-4)
          sigma <- rep(sigma_shared, K)
        } else {
          sigma <- pmax(init$sigma, 1e-4)
          if (shared_sigma) {
            sigma_shared <- median(sigma)
            sigma <- rep(sigma_shared, K)
          }
        }
      } else {
        sigma <- rep(0.5, K)
        if (shared_sigma) sigma <- rep(0.5, K)
      }
      if (!is.null(init$pi)) {
        if (length(init$pi) != K) stop("init$pi must have length K.")
        pi <- init$pi
        s <- sum(pi)
        if (s <= 0) {
          pi <- rep(1 / K, K)
        } else {
          pi <- pi / s
        }
        pi <- (1 - w0) * pi
      } else {
        pi <- rep((1 - w0) / K, K)
      }
    } else {
      ph <- (X + 0.5) / (n + 1)
      z <- log(ph) - log1p(-ph)
      z[!is.finite(z)] <- 0
      if (K > 1 && length(z) >= K) {
        km <- try(stats::kmeans(z, centers = K, nstart = 3, iter.max = 100), silent = TRUE)
        if (inherits(km, "try-error")) {
          labels0 <- cut(z, breaks = quantile(z, probs = seq(0, 1, length.out = K + 1), na.rm = TRUE), include.lowest = TRUE, labels = FALSE)
        } else {
          labels0 <- km$cluster
        }
      } else {
        labels0 <- cut(z, breaks = quantile(z, probs = seq(0, 1, length.out = K + 1), na.rm = TRUE), include.lowest = TRUE, labels = FALSE)
      }
      if (anyNA(labels0)) {
        labels0[is.na(labels0)] <- sample.int(K, sum(is.na(labels0)), replace = TRUE)
      }
      mu <- sigma <- pi <- numeric(K)
      for (k in seq_len(K)) {
        idx <- which(labels0 == k)
        if (!length(idx)) idx <- sample.int(length(z), max(1L, round(length(z) / K)))
        zk <- z[idx]
        mu[k] <- stats::median(zk)
        sigma[k] <- max(stats::sd(zk), 0.3)
        pi[k] <- (1 - w0) * length(idx) / length(z)
      }
      s <- sum(pi)
      if (s <= 0) {
        pi <- rep((1 - w0) / K, K)
      } else {
        pi <- (1 - w0) * pi / s
      }
      if (shared_sigma) {
        sigma_shared <- stats::median(sigma)
        sigma <- rep(sigma_shared, K)
      }
    }
    prevLL <- -Inf
    converged <- FALSE
    start_time <- proc.time()[3L]
    resp <- NULL
    respBLN <- NULL
    respUNI <- NULL
    LL <- NA_real_
    for (it in seq_len(max_iter)) {
      logpk <- .comp_loglik_matrix(X, Xc, logC, mu, sigma, nodes, weights)
      logPi <- log(pmax(pi, .machine_realmin))
      logPu <- log(max(w0, .machine_realmin))
      logU <- -log(n + 1)
      logR <- cbind(sweep(logpk, 2L, logPi, "+"), logU + logPu)
      logNorm <- .row_logsumexp(logR)
      logResp <- sweep(logR, 1L, logNorm, "-")
      resp <- exp(logResp)
      LL <- sum(logNorm)
      respBLN <- resp[, seq_len(K), drop = FALSE]
      respUNI <- resp[, K + 1L, drop = FALSE]
      Nk <- colSums(respBLN)
      SNk <- sum(Nk)
      if (SNk <= .machine_realmin) {
        pi <- rep((1 - w0) / K, K)
      } else {
        pi <- (1 - w0) * Nk / SNk
      }
      if (!shared_sigma) {
        for (k in seq_len(K)) {
          if (Nk[k] <= .machine_eps) next
          wk <- respBLN[, k] / max(Nk[k], .machine_realmin)
          theta0 <- c(mu[k], log(max(sigma[k], 1e-4)))
          if (use_bfgs) {
            opt <- optim(theta0,
                         fn = .nll_weighted_value,
                         gr = .nll_weighted_grad,
                         method = "BFGS",
                         control = list(maxit = 200, fnscale = 1),
                         X = X, Xc = Xc, logC = logC,
                         w = wk, nodes = nodes, weights = weights)
          } else {
            opt <- optim(theta0,
                         fn = .nll_weighted_value,
                         method = "Nelder-Mead",
                         control = list(maxit = 400),
                         X = X, Xc = Xc, logC = logC,
                         w = wk, nodes = nodes, weights = weights)
          }
          mu[k] <- opt$par[1L]
          sigma[k] <- max(exp(opt$par[2L]), 1e-12)
        }
      } else {
        for (k in seq_len(K)) {
          if (Nk[k] <= .machine_eps) next
          wk <- respBLN[, k] / max(Nk[k], .machine_realmin)
          if (use_bfgs) {
            opt_mu <- optim(mu[k],
                            fn = .nll_mu_only_value,
                            gr = .nll_mu_only_grad,
                            method = "BFGS",
                            control = list(maxit = 200, fnscale = 1),
                            X = X, Xc = Xc, logC = logC,
                            w_norm = wk,
                            sigma_fixed = sigma[k],
                            nodes = nodes, weights = weights)
          } else {
            opt_mu <- optim(mu[k],
                            fn = .nll_mu_only_value,
                            method = "Nelder-Mead",
                            control = list(maxit = 400),
                            X = X, Xc = Xc, logC = logC,
                            w_norm = wk,
                            sigma_fixed = sigma[k],
                            nodes = nodes, weights = weights)
          }
          mu[k] <- opt_mu$par
        }
        logsig0 <- log(max(sigma[1L], 1e-12))
        if (use_bfgs) {
          opt_sig <- optim(logsig0,
                           fn = .nll_logsigma_shared_value,
                           gr = .nll_logsigma_shared_grad,
                           method = "BFGS",
                           control = list(maxit = 200, fnscale = 1),
                           X = X, Xc = Xc, logC = logC,
                           respBLN = respBLN, Nk = Nk, mu = mu,
                           nodes = nodes, weights = weights)
        } else {
          opt_sig <- optim(logsig0,
                           fn = .nll_logsigma_shared_value,
                           method = "Nelder-Mead",
                           control = list(maxit = 400),
                           X = X, Xc = Xc, logC = logC,
                           respBLN = respBLN, Nk = Nk, mu = mu,
                           nodes = nodes, weights = weights)
        }
        sigma_shared <- max(exp(opt_sig$par), 1e-12)
        sigma <- rep(sigma_shared, K)
      }
      relImpr <- (LL - prevLL) / (1 + abs(prevLL))
      if (identical(display, "iter")) {
        msg <- sprintf("Restart %d | iter %3d LL = %.6f relImpr = %.3e (w0 = %.2e%s)",
                       r, it, LL, relImpr, w0,
                       if (shared_sigma) ", shared sigma" else "")
        message(msg)
      }
      if (relImpr < tol && it > 2L) {
        converged <- TRUE
        break
      }
      prevLL <- LL
    }
    elapsed <- proc.time()[3L] - start_time
    if (LL > bestLL) {
      bestLL <- LL
      best <- list(
        K = K,
        mu = mu,
        sigma = sigma,
        pi = pi,
        uniform_weight = w0,
        QuadNodes = quad_nodes,
        loglik = LL,
        iters = it,
        converged = converged,
        resp = resp,
        labels = max.col(resp, ties.method = "first"),
        details = list(
          elapsed_sec = elapsed,
          QuadNodes = quad_nodes,
          Restarts = restarts,
          total_elapsed = NA_real_,
          UseBFGS = use_bfgs,
          SharedSigma = shared_sigma
        )
      )
      class(best) <- "BLNMixture"
    }
  }
  best$details$total_elapsed <- proc.time()[3L] - time_start_all
  best
}

bln_mixture_pdf <- function(object, x, n, log = FALSE) {
  stopifnot(inherits(object, "BLNMixture"))
  x <- as.numeric(x)
  n <- as.numeric(n)
  if (length(x) != length(n)) stop("x and n must have the same length.")
  X <- x
  Xc <- n - x
  if (any(X < 0 | Xc < 0)) stop("Counts must satisfy 0 <= x <= n.")
  logC <- lgamma(n + 1) - lgamma(X + 1) - lgamma(Xc + 1)
  nodes_weights <- .gh_nodes_weights(object$QuadNodes)
  logpk <- .comp_loglik_matrix(X, Xc, logC, object$mu, object$sigma,
                               nodes_weights$nodes, nodes_weights$weights)
  logPi <- log(pmax(object$pi, .machine_realmin))
  logBLN <- .row_logsumexp(sweep(logpk, 2L, logPi, "+"))
  logU <- -log(n + 1)
  logwBLN <- log(max(1 - object$uniform_weight, .machine_realmin))
  logwU <- log(max(object$uniform_weight, .machine_realmin))
  logp <- .row_logsumexp(cbind(logwBLN + logBLN, logwU + logU))
  if (log) {
    logp
  } else {
    exp(logp)
  }
}

bln_mixture_rnd <- function(object, n) {
  stopifnot(inherits(object, "BLNMixture"))
  n <- as.numeric(n)
  if (length(n) == 1L) n <- rep(n, 1L)
  n <- as.numeric(n)
  N <- length(n)
  wBLN <- object$pi
  w0 <- object$uniform_weight
  mixw <- c(wBLN, w0)
  mixw <- mixw / sum(mixw)
  edges <- cumsum(c(0, mixw))
  u <- stats::runif(N)
  z <- findInterval(u, edges, rightmost.closed = TRUE)
  x <- xc <- numeric(N)
  for (k in seq_len(object$K)) {
    idx <- which(z == k)
    if (!length(idx)) next
    nk <- n[idx]
    Z <- object$mu[k] + object$sigma[k] * stats::rnorm(length(idx))
    pk <- plogis(Z)
    x[idx] <- stats::rbinom(length(idx), size = nk, prob = pk)
    xc[idx] <- nk - x[idx]
  }
  idxU <- which(z == (object$K + 1L))
  if (length(idxU)) {
    nk <- n[idxU]
    x[idxU] <- vapply(nk, function(ni) sample.int(ni + 1L, 1L) - 1L, numeric(1L))
    xc[idxU] <- nk - x[idxU]
  }
  list(x = x, xc = xc, z = z)
}

print.BLNMixture <- function(x, ...) {
  cat("BLNMixture object\n")
  cat(" Components:", x$K, "\n")
  cat(" Uniform weight:", x$uniform_weight, "\n")
  cat(" Log-likelihood:", x$loglik, "\n")
  invisible(x)
}
