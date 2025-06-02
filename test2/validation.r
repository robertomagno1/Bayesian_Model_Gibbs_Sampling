# ----------------------------
# Verifica numerica di Domanda 3
# ----------------------------

# 0) Preparazione
set.seed(2025)

# 1) Definiamo target f(x) = standard Normal e g(x) = standard Cauchy
f <- function(x) dnorm(x, mean = 0, sd = 1)        # = (1/sqrt(2π)) e^{-x^2/2}
g <- function(x) dcauchy(x, location = 0, scale = 1)  # = 1/[π (1 + x^2)]

# 2) Calcoliamo la costante di dominanza k* = sup_x f(x)/g(x)
#    Equivalente a sup_x [ sqrt(π/2) (1 + x^2) e^{-x^2/2} ].

h <- function(x) sqrt(pi/2) * (1 + x^2) * exp(-x^2/2)
# Il massimo di h(x) si ottiene in x = ±1
x_star <- 1
k_star <- h(x_star)
cat("k* (valore minimo che soddisfa k*g(x) >= f(x)) = ", round(k_star, 6), "\n\n")

# 3) Verifichiamo che per k = 4 la dominanza valga:
k_test <- 4
max_ratio <- optimize(function(x) f(x)/(g(x)*k_test), interval = c(-10, 10), maximum = TRUE)$objective
cat("Max_x [ f(x) / (k g(x)) ] con k=4 = ", round(max_ratio, 6),
    " (< 1 ), quindi la dominanza è vera.\n\n", sep = "")

# 4) Se si fanno n = 1e4 proposte X ~ Cauchy, la
#    probabilità di accettare ciascuna è p_acc = 1 / k.
n <- 1e4
p_acc <- 1 / k_test
expected_accepts <- n * p_acc
cat("Expected #accettati con n=1e4, k=4: ", round(expected_accepts), "\n\n")

# 5) La legge di N_acc è Binomial(n, p_acc)
#    Verifichiamo “empiricamente” il conteggio medio su molte repliche:
R <- 2000
counts <- numeric(R)
for (i in seq_len(R)) {
  U <- runif(n)            # Uniform(0,1) per accettazione
  # Proposte Cauchy:
  X_prop <- rcauchy(n, 0, 1)
  # Calcoliamo f(X_prop)/(k g(X_prop)):
  ratio_vals <- f(X_prop) / (k_test * g(X_prop))
  counts[i] <- sum(U < ratio_vals)
}
cat("Empirical mean of accepted counts (su", R, "repliche) = ",
    round(mean(counts)), "\n")
cat(" teorico = ", round(expected_accepts), "\n\n", sep = "")

# 6) Possiamo simulare la catena e contare quanti successi (Binomial):
cat("Law of N_acc ~ Binomial(n=", n, ", p=", round(p_acc,3), ")\n", sep = "")

# Fine script
