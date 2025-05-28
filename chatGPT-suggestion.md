# Detailed Feedback – Homework #02

Below is a structured, point‑by‑point critique of your submission, written as if I were marking the work for **Statistical Methods in Data Science II (a.y. 2024‑2025)**.  Where appropriate I suggest concrete improvements or extensions.

---

## General Presentation

* **R markdown hygiene** – Excellent use of `opts_chunk` to limit printed output, but remember to *hide* long helper functions once they have been shown once (`echo=FALSE`).  This will keep the PDF concise.
* **Title block** – Good LaTeX customisation, but Italian `babel` will translate automatic words (e.g. *Figure 1* → *Figura 1*); decide whether you want a fully‐English or fully‐Italian document and load babel accordingly.
* **Seed discipline** – You call `set.seed()` multiple times (123, 321).  Document *why* each seed is being reset; otherwise use a single global seed for full reproducibility.

---

## 1  Acceptance–Rejection (A/R) Section

### 1a  Normalising constant

* ✔ Correctly identifies $Z = \sqrt{2\pi}$.
* ❌ Minor notation: the symbol `f(x)` is *already* a density in the exercise text; keep a separate symbol (e.g. $q(x)$) for the un‑normalised kernel to avoid ambiguity.

### 1b  Geometric interpretation

* Improvement: include an actual sketch or embed a small TikZ picture showing the area under the curve equalling $\sqrt{2\pi}$.  A figure frequently clarifies the ‘area under the curve’ argument better than text.

### 1c  Choice of *k*

* You optimise the ratio **numerically** over $[-100,100]$.  Good, but:

  * Argue that the supremum occurs at $x = 0$ analytically (because the ratio is even and unimodal).  Then $k = \pi / \sqrt{2\pi} \approx 1.253314$.  Including the analytic derivation would earn full marks and avoids numerical overkill.
  * Store *k* with a descriptive name (`cauchy_envelope`) for readability.

### 1d  A/R implementation

* The loop is correct but **inefficient**:

  * Pre‑allocate `rejected` with a rough size (`vector("numeric", 5e4)`) then trim.  Repeated `c()` concatenation reallocates memory many times.
  * Vectorise: generate proposals in blocks of size 1000 and filter, which is \~10× faster in R.
* Track **total proposals** inside the loop, but also save them; the variable `trials` is overwritten but disappears from the workspace once the chunk ends (unless you save it).

### 1e  Acceptance probability

* Acceptance rate ≈ 0.73 – this is surprisingly *higher* than the theoretical 1/$k$=0.798?  Check your calculation:

  * The theoretical rate is **1/k ≈ 0.798** because $k≈1.253$.  Your Monte‑Carlo estimate is 0.729; some deviation is expected (SE ≈ 0.004 with 10 k accepted) but state that comparison explicitly.

### 1g  Rejected distribution

* Good idea to visualise.  However your density estimate uses `adjust = 2`; justify the bandwidth choice (e.g. reference Silverman’s rule).  Better: overlay the *Cauchy* and *Normal* PDFs too so the reader sees why rejections cluster in the heavy tails.

### 1h  Theoretical density of rejections

* Correct derivation.  Mention that the normalising constant equals (k−1)/k so that the rejection density integrates to 1.
* Your numerical integration with `integrate()` is appropriate; add error estimate (`integrate(...)$abs.error`).

### 1i  Why N cannot be predetermined

* Provide a short *probabilistic* argument: the number of proposals needed is a negative–binomial with mean k n and variance k n(k−1).  This quantifies the stochastic overhead exactly.

---

## 2  Discrete 3‑state Markov chain (frog)

### 2a–b  Single chain simulation

* Nice compact function.  Suggestion: store the chain as a *factor* so `table()` prints the states as "1,2,3" rather than numeric coerced names.

### 2c  500 chains final state

* You replicate chains but do **not** reset the seed inside the replicate – good.  Mention that independence between chains is assumed.
* Provide a **histogram of empirical distribution** of the 1 000‑step positions (not only final state) to illustrate convergence visually.

### 2d  Stationary distribution

* Calculated via eigenvector – correct.  Alternative: solve $\pi = \pi P$ with `solve(t(diag(3)-P+1), rep(1,3))` which is numerically stabler for nearly reducible chains.

### 2e  Comparison table

* Good.  Report also the **Chi‑square distance** or total variation distance between empirical and theoretical to quantify convergence.

### 2f  Different initial state

* You show empirical convergence.  Add theoretical justification: chain is **aperiodic** (self‑loop probability > 0 somewhere) and **irreducible**, so convergence is guaranteed by the ergodic theorem.

### Extra possible extension

* Compute **autocorrelation time** of the chain (e.g., via spectral method) so students appreciate how many *effective* samples the 1000‑step run represents.

---

## Coding Style & Efficiency

* Prefer `set.seed()` **once per Rmd**, then use `withr::with_seed()` inside functions if deterministic sub‑chunks are needed.
* Use `tibble` / `dplyr` pipelines instead of base loops when building summary tables – improves readability.
* Load libraries at the top (you load `ggplot2` twice).

---

## Documentation & Theory

* Provide *brief derivations* before each code block to demonstrate understanding; code alone is not self‑explanatory.
* When quoting theory (e.g., negative‑binomial variance) give a reference (Casella & Berger, §3.4).

---

## Summary of Key Improvements

1. **Analytic justification** for k and for stationary distribution complements numerical work.
2. **Vectorised or block sampling** in A/R boosts speed; pre‑allocate memory.
3. Provide **convergence diagnostics** (trace plot, TV distance) for the Markov chain.
4. Tighten **bandwidth choice** and overlay theoretical PDFs in rejection plot.
5. Uniform seeding strategy and minor LaTeX cleanup.

Incorporating these points will raise the submission from *good* to *excellent* and demonstrate deeper mastery of Bayesian computation principles.
