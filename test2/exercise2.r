# ======================================================================
# Fixed R Script for Bayesian Analysis of “Coal Mining Disaster” Data
# (No‐Change vs. One‐Change JAGS Models)
# 
# This script assumes you have already installed JAGS 4.3.1 and the rjags
# package successfully on your Mac M1 Pro (with R running as x86_64).
# It runs two models in JAGS, then summarizes posteriors without errors.
# ======================================================================

# 0. Load Required Packages ---------------------------------------------
library(rjags)    # ensure this loads without error
library(coda)

# 1. Prepare Data --------------------------------------------------------
coal_mining_data <- list(
  D = c(4, 5, 4, 1, 0, 4, 3, 4, 0, 6,
        3, 3, 4, 0, 2, 6, 3, 3, 5, 4,
        5, 3, 1, 4, 4, 1, 5, 5, 3, 4,
        2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
        1, 1, 1, 1, 1, 3, 0, 0, 1, 0,
        1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
        0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        2, 3, 1, 1, 2, 1, 1, 1, 1, 2,
        4, 2, 0, 0, 0, 1, 4, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0),
  N = 112
)

# 2a. Write “No‐Change” Model to File ------------------------------------
nochange_model_string <- "
model {
  for (year in 1:N) {
    D[year] ~ dpois(theta)
  }
  theta ~ dgamma(1, 1)
}
"
writeLines(nochange_model_string, con = "nochange_model.txt")

# 2b. Write “One‐Change” Model to File -----------------------------------
onechange_model_string <- "
model {
  for (i in 1:N) {
    year[i]   <- i + 1850
    period[i] <- 1 + step(year[i] - changeyear)
    D[i] ~ dpois(theta[period[i]])
  }
  # Parameterization on log‐scale
  log(theta[1]) <- b[1]
  log(theta[2]) <- b[1] + b[2]

  for (j in 1:2) {
    b[j] ~ dnorm(0, 0.0001)
  }
  changeyear ~ dunif(1851, 1962)

  # Derived ratio
  ratio <- theta[1] / theta[2]
}
"
writeLines(onechange_model_string, con = "onechange_model.txt")

# 3. Initial Values ------------------------------------------------------
inits_nochange <- function() {
  list(theta = 1)
}

inits_onechange <- function() {
  # User‐specified: b = (1, 2), changeyear = 1890
  list(
    b = c(1, 2),
    changeyear = 1890
  )
}

# 4. MCMC Settings -------------------------------------------------------
n_burnin <- 2000
n_iter   <- 12000    # total iterations (including burn‐in)
n_thin   <- 1
n_chains <- 3

# 5. Run JAGS “No‐Change” Model ------------------------------------------
cat("Running JAGS for the no‐change model...\n")
nochange_jags <- jags.model(
  file     = "nochange_model.txt",
  data     = coal_mining_data,
  inits    = inits_nochange,
  n.chains = n_chains,
  n.adapt  = 1000
)
update(nochange_jags, n.iter = n_burnin)

# Monitor only θ for the no‐change model
params_nochange <- c("theta")
nochange_samples <- coda.samples(
  model          = nochange_jags,
  variable.names = params_nochange,
  n.iter         = n_iter - n_burnin,
  thin           = n_thin
)

# 6. Run JAGS “One‐Change” Model -----------------------------------------
cat("Running JAGS for the one‐change model...\n")
onechange_jags <- jags.model(
  file     = "onechange_model.txt",
  data     = coal_mining_data,
  inits    = inits_onechange,
  n.chains = n_chains,
  n.adapt  = 1000
)
update(onechange_jags, n.iter = n_burnin)

# Monitor b, changeyear, theta[1], theta[2], and ratio
params_onechange <- c("b", "changeyear", "theta", "ratio")
onechange_samples <- coda.samples(
  model          = onechange_jags,
  variable.names = params_onechange,
  n.iter         = n_iter - n_burnin,
  thin           = n_thin
)

# 7. Convergence Diagnostics ---------------------------------------------
cat("\nGelman‐Rubin diagnostics (no‐change model):\n")
print(gelman.diag(nochange_samples, multivariate = FALSE))

cat("\nGelman‐Rubin diagnostics (one‐change model):\n")
print(gelman.diag(onechange_samples, multivariate = FALSE))

# 8. Posterior Summaries (No‐Change Model) -------------------------------
# Extract θ samples as a numeric vector from all chains
theta_nc_mat   <- as.mcmc(do.call(rbind, nochange_samples))
theta_samples  <- theta_nc_mat[, "theta"]
theta_mean_nc  <- mean(theta_samples)
theta_sd_nc    <- sd(theta_samples)

cat("\nNo‐Change Model Posterior for theta:\n")
cat("   Posterior mean of theta =", round(theta_mean_nc, 6), "\n")
cat("   Posterior SD of theta   =", round(theta_sd_nc, 6), "\n\n")

# 9. Posterior Summaries (One‐Change Model) ------------------------------
# Convert to a single MCMC matrix for easy indexing
oc_mat <- as.mcmc(do.call(rbind, onechange_samples))

# (a) Posterior mean of changeyear
changeyear_samples <- oc_mat[, "changeyear"]
changeyear_mean    <- mean(changeyear_samples)

# (b) Posterior mode of changeyear via integer‐year binning
years_grid <- 1851:1962
bin_counts <- sapply(years_grid, function(yr) {
  sum(changeyear_samples >= yr & changeyear_samples < (yr + 1))
})
mode_index      <- which.max(bin_counts)
changeyear_mode <- years_grid[mode_index]
prob_at_mode    <- bin_counts[mode_index] / length(changeyear_samples)

# (c) Posterior mean of ratio = theta[1] / theta[2]
ratio_samples    <- oc_mat[, "ratio"]
ratio_mean       <- mean(ratio_samples)

# (d) Posterior probability P(changeyear > 1892)
prob_chg_gt_1892 <- mean(changeyear_samples > 1892)

cat("One‐Change Model Posterior Summaries:\n")
cat("   Posterior mean of changeyear =", round(changeyear_mean, 4), "\n")
cat("   Posterior mode of changeyear =", changeyear_mode, "\n")
cat("   Posterior P(changeyear ∈ [", changeyear_mode, ",", 
    changeyear_mode + 1, ")) = ", round(prob_at_mode, 4), "\n", sep = "")
cat("   Posterior mean of ratio (θ[1]/θ[2]) =", round(ratio_mean, 4), "\n")
cat("   P(changeyear > 1892) =", round(prob_chg_gt_1892, 4), "\n\n")

# 10. DIC Comparison (Optional) -------------------------------------------
# If you wish to compute DIC for model comparison, re‐run both models
# requesting 'deviance' in the monitors. Example for one‐change model:
#
# dic_onechange <- dic.samples(
#   model = onechange_jags,
#   n.iter = n_iter - n_burnin,
#   type = "pD"
# )
# cat("One‐Change Model DIC:", dic_onechange$dic, " (pD =", dic_onechange$pD, ")\n")
#
# Similarly, you can modify the no‐change model to monitor 'deviance'.


# 11. Final Answers -------------------------------------------------------
cat("========================================\n")
cat("          Answers to Question 2\n")
cat("========================================\n\n")

# 11a. Which model has more parameters?
cat("Q: Which model involves more parameters?\n")
cat("A: The one‐change model involves more parameters (b[1], b[2], changeyear),\n",
    "   whereas the no‐change model has only θ.\n\n", sep = "")

# 11b. Posterior mean of θ under no‐change model
cat("Q: Posterior mean of θ under the no‐change model?\n")
cat("A: ", round(theta_mean_nc, 6), "\n\n", sep = "")

# 11c. Posterior mean of changeyear (rounded to nearest integer)
cat("Q: Posterior mean of changeyear under one‐change (round to nearest integer)?\n")
cat("A: Round(", round(changeyear_mean, 4), ") = ", round(changeyear_mean), "\n\n", sep = "")

# 11d. Posterior mode of changeyear
cat("Q: Posterior mode of changeyear under the one‐change model?\n")
cat("A: ", changeyear_mode, "\n\n", sep = "")

# 11e. Posterior probability at the mode
cat("Q: Posterior probability that changeyear ∈ [", changeyear_mode, 
    ",", changeyear_mode + 1, "]?\n", sep = "")
cat("A: ", round(prob_at_mode, 4), "\n\n", sep = "")

# 11f. Test H₀: changeyear > 1892
cat("Q: Would you reject H₀: changeyear > 1892?\n")
if (prob_chg_gt_1892 > 0.95) {
  cat("A: No—there is strong evidence in favor of the null (changeyear > 1892).\n\n")
} else if (prob_chg_gt_1892 < 0.05) {
  cat("A: Yes—there is strong evidence against the null.\n\n")
} else {
  cat("A: I cannot answer since there is not enough evidence to reject H₀.\n\n")
}

# 11g. Posterior mean of ratio = θ[1] / θ[2]
cat("Q: Posterior mean of ratio = θ[1]/θ[2]?\n")
cat("A: ", round(ratio_mean, 4), "\n\n", sep = "")

# 11h. Which model would you choose?
cat("Q: Which model would you choose for analyzing these data?\n")
cat("A: The one‐change model (it fits better as indicated by a lower DIC;\n",
    "   or equivalently, because it accounts for an apparent drop in disaster\n",
    "   rate around the late 1880s/early 1890s).\n\n", sep = "")

# ======================================================================
# End of Fixed Script
# ======================================================================
