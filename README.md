# Bayesian Model Gibbs Sampling

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![R](https://img.shields.io/badge/R-4.0%2B-blue.svg)
![JAGS](https://img.shields.io/badge/JAGS-4.3.0-blue.svg)

## 📌 Overview

This repository showcases the implementation of Bayesian models using Gibbs sampling techniques in R, focusing on modeling count data through Poisson processes. The primary objective is to analyze the number of coal mining disasters over time, employing two distinct models:

- **No-Change Model**: Assumes a constant rate of disasters throughout the observed period.
- **One-Change Model**: Introduces a change point, allowing the disaster rate to shift at a specific year.

By leveraging the `R2jags` package, the project facilitates Bayesian inference through JAGS (Just Another Gibbs Sampler), enabling robust statistical analysis and model comparison.

## 🗂️ Repository Structure

```
Bayesian_Model_Gibbs_Sampling/
├── Final_Attempt.R
├── nochangemodelBM2022.txt
├── onechangemodelBM2022.txt
├── README.md
```

## 🔍 Models Description

### 1. No-Change Model

```jags
model {
  for(year in 1:N) {
    D[year] ~ dpois(theta)
  }
  theta ~ dgamma(1, 1)
}
```

- `D`: Vector of disaster counts per year.
- `N`: Total number of years.
- `theta`: Rate parameter for the Poisson distribution.

### 2. One-Change Model

```jags
model {
  for(i in 1:N) {
    year[i] <- i + 1850
    period[i] <- 1 + step(year[i] - changeyear)
    D[i] ~ dpois(theta[period[i]]) 
  }
  log(theta[1]) <- b[1]
  log(theta[2]) <- b[1] + b[2]
  for(j in 1:2){
    b[j] ~ dnorm(0, 0.0001)
  }
  changeyear ~ dunif(1851, 1962)
  ratio <- theta[1] / theta[2]
}
```

- `changeyear`: Year when the rate changes.
- `theta[1]`, `theta[2]`: Rates before and after the change point.
- `b[1]`, `b[2]`: Log-linear coefficients.
- `ratio`: Ratio of rates before and after the change point.

## 🚀 Getting Started

### Prerequisites

Ensure the following are installed on your system:

- R (version 4.0 or higher)
- JAGS (Just Another Gibbs Sampler)
- R packages: `R2jags`

### Installation

```bash
git clone https://github.com/robertomagno1/Bayesian_Model_Gibbs_Sampling.git
cd Bayesian_Model_Gibbs_Sampling
```

```r
install.packages("R2jags")
```

```r
source("Final_Attempt.R")
```

## 📊 Results & Analysis

The `Final_Attempt.R` script performs the following analyses:

- Transition Probability Matrix (TPM)
- Simulation of Markov Chains
- Acceptance-Rejection Algorithm
- Bayesian Inference with JAGS
- Model Comparison via DIC

## 📈 Visualizations

- Density plots for the A/R algorithm
- Histograms and density plots for posterior distributions
- MCMC diagnostics and convergence plots

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

MIT License

## 📬 Contact

Maintainer: [robertomagno1](https://github.com/robertomagno1)

