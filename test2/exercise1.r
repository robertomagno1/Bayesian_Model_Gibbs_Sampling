# ======================================================================
# R Script: Analysis of a 3-State Homogeneous Markov Chain (Question 1)
# ======================================================================

# This script addresses all sub-questions for the Markov chain defined
# on S = {1, 2, 3} with transition probability matrix P given below.
# All code and comments are in English. Execute sequentially.

# -----------------------------------------------------------------------------
# 0. Define the transition probability matrix P
# -----------------------------------------------------------------------------
P <- matrix(
  c(0.212, 0.396, 0.392,
    0.022, 0.527, 0.451,
    0.503, 0.338, 0.159),
  nrow = 3,
  ncol = 3,
  byrow = TRUE
)
rownames(P) <- colnames(P) <- c("1", "2", "3")

cat("Transition Probability Matrix P:\n")
print(P)
cat("\n")

# -----------------------------------------------------------------------------
# 1. Necessary and Sufficient Condition for a Transition Probability Matrix
# -----------------------------------------------------------------------------
# A matrix P is a valid transition probability matrix if and only if:
#   (a) All entries are non-negative.
#   (b) Each row sums to 1.
#
# We check these two conditions in code.

# Check non-negativity of entries
all_nonnegative <- all(P >= 0)

# Check that each row sums to (approximately) 1
row_sums_close_to_one <- all(abs(rowSums(P) - 1) < 1e-12)

# Combined validity check
is_valid_tpm <- all_nonnegative && row_sums_close_to_one

cat("1. Validity Check for P as a TPM:\n")
cat("   - All entries non-negative?       ", all_nonnegative, "\n")
cat("   - Each row sums to 1 (approx)?    ", row_sums_close_to_one, "\n")
cat("   => P is a valid transition matrix:", is_valid_tpm, "\n\n")

# -----------------------------------------------------------------------------
# 2. Two-Step Transition Probability: P(X_{t+2} = 1 | X_t = 3)
# -----------------------------------------------------------------------------
# Compute P^2 and then extract the (3,1) entry.

# Compute the 2-step transition matrix
P2 <- P %*% P

cat("2. Two-Step Transition Matrix P^2:\n")
print(round(P2, 6))
cat("\n")

# Extract P(X_{t+2} = 1 | X_t = 3) = (P^2)[3, 1]
p_2step_3_to_1 <- P2["3", "1"]
cat("   P(X_{t+2} = 1 | X_t = 3) = ", round(p_2step_3_to_1, 6), "\n\n")

# -----------------------------------------------------------------------------
# 3. Simulation: Approximate E[X] under Stationarity via Monte Carlo
# -----------------------------------------------------------------------------
# Starting from state 1 at t = 0, simulate up to t = 10000.
# Then compute the sample mean of the simulated states, which
# approximates E_pi[X] where pi is the stationary distribution.

set.seed(2025)      # For reproducibility
n_steps <- 10000    # Number of steps to simulate

# Preallocate a vector to store the chain (length = n_steps + 1, including time 0)
chain <- integer(n_steps + 1)
chain[1] <- 1       # Initial state at time 0 is 1

# Possible states
states <- 1:3

for (t in seq_len(n_steps)) {
  current_state <- chain[t]
  # Sample next state based on row 'current_state' of P
  chain[t + 1] <- sample(states, size = 1, prob = P[current_state, ])
}

# Compute the sample mean of the simulated states
approx_stationary_mean <- mean(chain)

cat("3. Simulation Results:\n")
cat("   - Sample mean of chain (approx. E_pi[X]) = ", round(approx_stationary_mean, 6), "\n\n")

# -----------------------------------------------------------------------------
# 4. Consistency of the Sample Variance for Var_pi[X]
# -----------------------------------------------------------------------------
# Compute the sample variance of the simulated states.
# Theoretically, as t -> infinity, the sample variance converges
# to Var_pi[X], so it is a consistent estimator.

sample_var_chain <- var(chain)

cat("4. Sample Variance of Simulated States:\n")
cat("   - Sample variance of chain = ", round(sample_var_chain, 6), "\n")
cat("   => Yes, the sample variance is a consistent estimate of Var_pi[X].\n\n")

# -----------------------------------------------------------------------------
# 5. Variance of Simulated Values Divided by t:
#    Estimate of Var_of_Empirical_Mean
# -----------------------------------------------------------------------------
# A naive estimate of Var(mean of chain) would be sample_var_chain / n_steps.
# However, because of autocorrelation in the Markov chain, this is generally
# NOT correct. We compute it anyway to illustrate.

variance_of_mean_naive <- sample_var_chain / length(chain)

cat("5. Naive Estimate of Var(Mean) = Var(chain) / n:\n")
cat("   - Naive estimate = ", format(variance_of_mean_naive, scientific = TRUE), "\n")
cat("   => This is NOT an appropriate estimate of Var( (1/n) * sum X_t ) \n")
cat("      because the X_t are correlated (Markov dependence).\n\n")

# -----------------------------------------------------------------------------
# 6. Exact Stationary Distribution pi (Solve pi' = pi' P)
# -----------------------------------------------------------------------------
# We find the left eigenvector of P corresponding to eigenvalue 1
# (or equivalently, the right eigenvector of t(P) for eigenvalue 1).
# Then normalize to sum to 1.

eig_result <- eigen(t(P))
# The eigenvector corresponding to eigenvalue 1 is in the first column
pi_raw    <- Re(eig_result$vectors[, 1])
# Normalize so that the probabilities sum to 1
pi_vec    <- pi_raw / sum(pi_raw)

cat("6. Exact Stationary Distribution (pi):\n")
for (i in seq_along(pi_vec)) {
  cat("   pi[", i, "] = ", round(pi_vec[i], 6), "\n", sep = "")
}
cat("\n")

# Identify the least probable state and its probability
least_prob_state      <- which.min(pi_vec)
probability_least     <- pi_vec[least_prob_state]

cat("   - Least probable state     = ", least_prob_state, "\n")
cat("   - Probability of that state = ", round(probability_least, 6), "\n\n")

# -----------------------------------------------------------------------------
# 7. Joint Probability P(X_t = 1 and X_{t+1} = 1) Under Stationarity
# -----------------------------------------------------------------------------
# If the initial state is drawn from pi, then
#   P(X_t = 1 and X_{t+1} = 1) = pi[1] * P[1, 1].

joint_prob_1_1 <- pi_vec[1] * P[1, 1]

cat("7. Joint Probability under Stationarity:\n")
cat("   P(X_t = 1 and X_{t+1} = 1) = ", round(joint_prob_1_1, 8), "\n")

# ======================================================================
# End of Script
# ======================================================================