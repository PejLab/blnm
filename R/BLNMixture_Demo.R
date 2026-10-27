## BLNMixture_Demo.R
## Demonstration of the R Binomial-Logit-Normal mixture implementation

.script_args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("^--file=", "", .script_args[grepl("^--file=", .script_args)])
if (!length(.script_path)) {
  .script_path <- "R/BLNMixture_Demo.R"
}
.script_dir <- dirname(normalizePath(.script_path))
old_dir <- getwd()
setwd(.script_dir)
on.exit(setwd(old_dir), add = TRUE)

source("BLNMixture.R")

set.seed(1)

# Ground-truth mixture
true_obj <- list(
  K = 3L,
  mu = c(-0.6, 0, 0.8),
  sigma = c(0.155, 0.13, 0.145),
  uniform_weight = 1e-3,
  pi = (1 - 1e-3) * c(0.4, 0.15, 0.55),
  QuadNodes = 30L
)
class(true_obj) <- "BLNMixture"

# Sample totals and data
N <- 2000L
n <- sample(10:1000, N, replace = TRUE)
sim <- bln_mixture_rnd(true_obj, n)
x <- sim$x
xc <- sim$xc
z_true <- sim$z

# Inject a few uniform outliers (sanity check)
idx_out <- seq_len(min(30L, N))
x[idx_out] <- vapply(n[idx_out], function(ni) sample.int(ni + 1L, 1L) - 1L, numeric(1L))
xc[idx_out] <- n[idx_out] - x[idx_out]

cat("Fitting BLN mixture (free sigma)...\n")
fit <- bln_mixture_fit(x, xc, K = 3,
                       restarts = 3,
                       uniform_weight = 1e-3,
                       display = "iter",
                       quad_nodes = 30)
cat(sprintf("Converged: %s | LL: %.3f | iters: %d\n", fit$converged, fit$loglik, fit$iters))
cat("Label counts (1..K BLN, K+1 uniform):\n")
print(table(fit$labels))

p_hat <- bln_mixture_pdf(fit, x, n)
mean_log_p <- mean(log(pmax(p_hat, .Machine$double.xmin)))
cat(sprintf("Mean log pmf on data: %.3f\n", mean_log_p))

cat("True vs estimated parameters:\n")
param_tab <- data.frame(
  mu_true = true_obj$mu,
  mu_hat = fit$mu,
  sigma_true = true_obj$sigma,
  sigma_hat = fit$sigma
)
print(param_tab)

plots_dir <- "plots"
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)
phat <- x / n
png(file.path(plots_dir, "fraction_vs_labels.png"), width = 800, height = 600)
plot(phat, fit$labels,
     pch = 19, cex = 0.6,
     xlab = "Observed fraction x/n",
     ylab = "MAP component label",
     main = "BLN mixture: MAP labels vs observed fraction")
grid()
dev.off()

label_counts <- table(fit$labels)
png(file.path(plots_dir, "label_counts.png"), width = 800, height = 600)
barplot(label_counts,
        main = "BLN mixture: component label counts",
        xlab = "Component (K+1 = uniform)",
        ylab = "Count",
        col = "steelblue")
grid()
dev.off()

components <- seq_len(true_obj$K)
png(file.path(plots_dir, "mu_comparison.png"), width = 800, height = 600)
plot(components, true_obj$mu, type = "b", pch = 19, col = "dodgerblue4",
     ylim = range(c(true_obj$mu, fit$mu)),
     xlab = "Component", ylab = expression(mu),
     main = "Mu: true vs estimated (free sigma fit)")
lines(components, fit$mu, type = "b", pch = 17, col = "firebrick")
legend("topleft", legend = c("true", "estimated"),
       col = c("dodgerblue4", "firebrick"), pch = c(19, 17), lty = 1)
grid()
dev.off()

png(file.path(plots_dir, "sigma_comparison.png"), width = 800, height = 600)
plot(components, true_obj$sigma, type = "b", pch = 19, col = "darkgreen",
     ylim = range(c(true_obj$sigma, fit$sigma)),
     xlab = "Component", ylab = expression(sigma),
     main = "Sigma: true vs estimated (free sigma fit)")
lines(components, fit$sigma, type = "b", pch = 17, col = "orange3")
legend("topleft", legend = c("true", "estimated"),
       col = c("darkgreen", "orange3"), pch = c(19, 17), lty = 1)
grid()
dev.off()

max_count <- max(c(x, xc))
png(file.path(plots_dir, "counts_scatter_overlay.png"), width = 800, height = 800)
plot(x, xc,
     pch = 19,
     cex = 0.4,
     col = grDevices::adjustcolor("black", alpha.f = 0.3),
     xlab = "x (success counts)",
     ylab = "xc (failure counts)",
     main = "Observed counts with true vs inferred component directions",
     xlim = c(0, max_count),
     ylim = c(0, max_count),
     asp = 1)
grid()
draw_component_line <- function(mu, col, lty) {
  p <- plogis(mu)
  if (p <= 0 || p >= 1) return()
  slope <- (1 - p) / p
  abline(a = 0, b = slope, col = col, lwd = 2, lty = lty)
}
cols_true <- c("dodgerblue4", "darkgreen", "purple4")
cols_est <- c("firebrick", "orange3", "goldenrod3")
for (k in seq_len(true_obj$K)) {
  draw_component_line(true_obj$mu[k], cols_true[(k - 1L) %% length(cols_true) + 1L], lty = 2)
  draw_component_line(fit$mu[k], cols_est[(k - 1L) %% length(cols_est) + 1L], lty = 1)
}
legend("topright",
       legend = c(paste0("true mu[", components, "]"),
                  paste0("estimated mu[", components, "]")),
       col = c(cols_true[seq_along(components)], cols_est[seq_along(components)]),
       lty = c(rep(2, length(components)), rep(1, length(components))),
       lwd = 2,
       bty = "n")
dev.off()

cat("\nFitting with shared sigma...\n")
fit_shared <- bln_mixture_fit(x, xc, K = 3,
                              restarts = 3,
                              uniform_weight = 1e-3,
                              display = "iter",
                              quad_nodes = 30,
                              shared_sigma = TRUE)
print(data.frame(mu_hat_shared = fit_shared$mu,
                 sigma_hat_shared = fit_shared$sigma))

png(file.path(plots_dir, "shared_sigma_params.png"), width = 800, height = 600)
plot(components, fit_shared$mu, type = "b", pch = 19, col = "purple4",
     xlab = "Component", ylab = expression(mu),
     main = "Shared sigma fit: component means")
abline(h = true_obj$mu, col = "grey70", lty = 2)
grid()
legend("bottomright", legend = c("estimated mu", "true mu"),
       col = c("purple4", "grey70"), pch = c(19, NA), lty = c(1, 2))
dev.off()

cat("\nDone.\n")
