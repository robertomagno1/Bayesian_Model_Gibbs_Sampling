### Domanda 2: Multivariate Normal Calculations

mu <- matrix(c(1, 5, 8), ncol = 1)
Sigma <- matrix(
  c( 4.696, -0.615,  3.998,
     -0.615,  7.945, -3.277,
     3.998, -3.277, 12.592),
  nrow = 3, byrow = TRUE
)

# Marginal Y2
Y2_mean     <- mu[2]
Y2_variance <- Sigma[2,2]
cat("Y2 ~ Normal(mean =", Y2_mean, ", variance =", Y2_variance, ")\n")
cat("E[Y2] =", Y2_mean, "\n")
cat("Var[Y2] =", Y2_variance, "\n\n")

# Corr(Y1, Y3)
corr_Y1_Y3 <- Sigma[1,3] / sqrt(Sigma[1,1] * Sigma[3,3])
cat("Corr(Y1, Y3) =", round(corr_Y1_Y3, 4), "\n\n")

# E[Y2 | Y1=2, Y3=6]
mu_1_3      <- rbind(mu[1], mu[3])
Sigma_11_33 <- Sigma[c(1,3), c(1,3)]
Sigma_2_13   <- matrix(c(Sigma[2,1], Sigma[2,3]), nrow = 1)
shift        <- matrix(c(2, 6), ncol = 1) - mu_1_3
E_Y2_given   <- mu[2] + Sigma_2_13 %*% solve(Sigma_11_33) %*% shift
cat("E[Y2 | Y1=2, Y3=6] =", round(E_Y2_given, 4), "\n\n")

# W = -Y1 + 5Y2 + Y3
a    <- matrix(c(-1, 5, 1), ncol = 1)
W_mean <- as.numeric(t(a) %*% mu)
W_var  <- as.numeric(t(a) %*% Sigma %*% a)
cat("W ~ Normal(mean =", round(W_mean, 6), ", variance =", round(W_var, 6), ")\n")





### Domanda 3: A/R with Cauchy → Normal

f_norm   <- function(x) dnorm(x)
h_cauchy <- function(x) dcauchy(x)

# Check k = 4
k4 <- 4
xg <- seq(-10, 10, length.out = 2001)
ok_k4 <- all(k4 * h_cauchy(xg) >= f_norm(xg))
cat("1) Possible with k=4? ", if (ok_k4) "Yes\n" else "No\n")

# Expected accepted with N_prop = 10^4, k=4
N_prop <- 1e4
E_acc_k4 <- N_prop / k4
cat("2) Expected accepted (k=4) =", E_acc_k4, "\n")

# Find optimal k
ratio_fx_hx <- function(x) f_norm(x) / h_cauchy(x)
k_opt <- optimize(ratio_fx_hx, c(-100, 100), maximum = TRUE)$objective
cat("3) Optimal k =", round(k_opt, 6), "\n")

# Alternative k condition
cat("4) Different k? Yes, any k >= ", round(k_opt, 6), "\n")

# Distribution of # of proposals before first acceptance
p_opt <- 1 / k_opt
cat("5) Distribution of trials before 1st accept: Geometric(p =", round(p_opt, 6), ")\n")
