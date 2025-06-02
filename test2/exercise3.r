# ======================================================================
# R Script: Analysis of a 3-State Homogeneous Markov Chain (Question 2)
# ======================================================================
#
# This script performs the following tasks for the given transition
# probability matrix P on S = {1,2,3}:
#
# 1. Check that P is a valid TPM (all entries ≥ 0, each row sums to 1).
# 2. Compute P^4 and extract P(X_{t+4}=1 | X_t=3).
# 3. Simulate the chain from state 1 up to t = 10000 and estimate
#    the stationary expectation E[X].
# 4. Compute sample variance of simulated states and check consistency.
# 5. Compute sample variance divided by t as a naïve estimate of Var(mean).
# 6. Find the exact invariant distribution π, then:
#    - Identify the least probable state and its probability.
#    - Compute P(X_t=1, X_{t+1}=1) under stationarity.
#
# Run sequentially in R (version ≥ 3.6). Requires only base R.

# 0. Set seed for reproducibility
set.seed(123)

# 1. Define the transition probability matrix P
P <- matrix(
  c(0.502, 0.003, 0.495,
    0.269, 0.363, 0.368,
    0.345, 0.299, 0.356),
  nrow = 3,
  ncol = 3,
  byrow = TRUE
)
rownames(P) <- colnames(P) <- c("1", "2", "3")

cat("Transition Probability Matrix P:\n")
print(P)
cat("\n")

# 1a. Check that P is a valid TPM
all_nonnegative   <- all(P >= 0)
row_sums_close_to_one <- all(abs(rowSums(P) - 1) < 1e-12)
is_valid_tpm     <- all_nonnegative && row_sums_close_to_one

cat("1. Validity Check for P as a TPM:\n")
cat("   - All entries non-negative?        ", all_nonnegative, "\n")
cat("   - Each row sums to 1 (approx)?     ", row_sums_close_to_one, "\n")
cat("   => P is a valid transition matrix: ", is_valid_tpm, "\n\n")

# 2. Compute the 4-step transition matrix P^4
P4 <- P %*% P %*% P %*% P
cat("2. Four-Step Transition Matrix P^4:\n")
print(round(P4, 6))
cat("\n")

# 2a. Extract P(X_{t+4} = 1 | X_t = 3) = (P^4)[3, 1]
p_4step_3_to_1 <- P4["3", "1"]
cat("   P(X_{t+4} = 1 | X_t = 3) =", round(p_4step_3_to_1, 6), "\n\n")

# 3. Simulate the Markov chain for t = 0..10000 starting from state 1
n_steps <- 10000
states  <- 1:3
chain   <- integer(n_steps + 1)
chain[1] <- 1  # X_0 = 1

for (t in seq_len(n_steps)) {
  current_state <- chain[t]
  chain[t + 1]  <- sample(states, size = 1, prob = P[current_state, ])
}

# 3a. Approximate E_pi[X] by the sample mean of the simulated states
approx_stationary_mean <- mean(chain)
cat("3. Simulation Results:\n")
cat("   - Approximate E_pi[X] = ", round(approx_stationary_mean, 6), "\n\n")

# 4. Sample variance of the simulated states
sample_variance_chain <- var(chain)
cat("4. Sample Variance of Simulated States:\n")
cat("   - Sample variance = ", round(sample_variance_chain, 6), "\n")
cat("   => This is a consistent estimator of Var_pi[X] (as t→∞).\n\n")

# 5. Naïve estimate of Var(empirical mean) = Var(chain)/n
variance_of_mean_naive <- sample_variance_chain / length(chain)
cat("5. Naïve Estimate of Var(Mean) = Var(chain) / n:\n")
cat("   - Naïve estimate = ", format(variance_of_mean_naive, scientific = TRUE), "\n")
cat("   => NOT appropriate, because {X_t} are correlated (Markov dependence).\n\n")

# 6. Compute the exact invariant (stationary) distribution via eigen
eig_out <- eigen(t(P))
pi_raw  <- Re(eig_out$vectors[, 1])
pi_vec  <- pi_raw / sum(pi_raw)

cat("6. Exact Stationary Distribution π:\n")
for (i in seq_along(pi_vec)) {
  cat("   π[", i, "] =", round(pi_vec[i], 6), "\n", sep = "")
}
cat("\n")

# 6a. Identify least probable state and its probability
least_prob_state   <- which.min(pi_vec)
probability_least  <- pi_vec[least_prob_state]
cat("   - Least probable state    =", least_prob_state, "\n")
cat("   - Prob(least probable)    =", round(probability_least, 6), "\n\n")

# 7. Compute joint probability P(X_t=1 and X_{t+1}=1) under stationarity
joint_prob_1_1 <- pi_vec[1] * P[1, 1]
cat("7. Joint Probability under Stationarity:\n")
cat("   P(X_t=1, X_{t+1}=1) =", round(joint_prob_1_1, 8), "\n")

# ======================================================================
# End of Script
# ======================================================================
