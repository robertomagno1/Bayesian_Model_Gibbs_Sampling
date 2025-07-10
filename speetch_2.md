# Bayesian Hierarchical Analysis of Titanic Passenger Survival: 
## A Comprehensive Statistical Investigation - Oral Defense

**Roberto Magno Mazzotta**  
**MSc Data Science - Bayesian Statistics**  
**Date: July 11, 2025**

---

## **1. ABSTRACT & INTRODUCTION** *(3 minutes)*

Good morning, Professor. Today I present a comprehensive Bayesian hierarchical analysis that transforms the RMS Titanic disaster into a powerful framework for understanding advanced statistical methodology and social inequality dynamics.

**The central research question**: How can modern Bayesian statistics quantify the systematic discrimination patterns that determined survival on the Titanic, while demonstrating advanced methodological competency in hierarchical modeling?

This study presents a comprehensive Bayesian hierarchical analysis of survival patterns among Titanic passengers using MCMC methods implemented through JAGS. We investigate probabilistic relationships between passenger characteristics and survival outcomes, employing a hierarchical logistic regression model with random effects for embarkation ports, incorporating weakly informative priors following current Bayesian best practices.

**Why this matters statistically:**
1. **Methodological showcase**: Demonstrates hierarchical Bayesian modeling with proper uncertainty quantification
2. **Real-world complexity**: Natural clustering (embarkation ports), missing data patterns, interaction effects
3. **Social science application**: Quantifying inequality through rigorous statistical inference

---

## **2. INTRODUCTION AND MOTIVATION** *(2 minutes)*

### **2.1 Rationale for Bayesian Methodology**

The sinking of RMS Titanic on April 14-15, 1912, provides an exceptional opportunity for statistical analysis. The Bayesian approach offers several methodological advantages over classical frequentist methods:

#### **Four fundamental advantages for this analysis:**

**1. Natural Uncertainty Quantification**: Bayesian inference provides complete posterior distributions for all parameters, enabling direct probability statements—"There's a 95% probability that female survival advantage is between 10-22 times higher."

**2. Hierarchical Structure**: The natural grouping of passengers by embarkation port creates correlation within groups that Bayesian methods handle elegantly through random effects.

**3. Prior Information Integration**: Bayesian methods allow incorporation of historical knowledge about human behavior in emergency situations through informative prior distributions.

**4. Model Comparison**: Bayesian model selection criteria (DIC, WAIC, LOO) provide principled approaches to compare competing hypotheses without arbitrary p-value thresholds.

### **2.2 Research Objectives**

This investigation aims to:
- Quantify the probabilistic impact of demographic and socioeconomic factors on survival
- Develop a predictive model with proper uncertainty quantification
- Demonstrate methodological superiority of Bayesian approaches through rigorous validation
- Provide complete posterior distributions for all inferences

---

## **3. DATASET DESCRIPTION AND EXPLORATORY ANALYSIS** *(4 minutes)*

### **3.1 Data Structure and Characteristics**

Our dataset contains **891 passengers** with **12 variables**, revealing striking initial patterns:
- **Overall survival**: 38.4% 
- **Female vs Male**: 74.2% vs 18.9%
- **1st vs 3rd class**: 63.0% vs 24.2%

But raw percentages only tell part of the story—we need sophisticated statistical modeling to understand the underlying probabilistic mechanisms.

### **3.2 Missing Data Analysis**

**Pattern analysis revealed systematic missingness:**
- **Cabin**: 77.1% missing (social stratification effect)
- **Age**: 19.9% missing (historical record-keeping issues)
- **Embarked**: 0.2% missing (port documentation crucial for business)

**MCAR Test Results**:
- **Test statistic**: 598.542
- **p-value < 0.001**
- **Conclusion**: Strong evidence against Missing Completely At Random

**Statistical implication**: Missingness is systematically related to observed variables, likely reflecting social class discrimination in record-keeping practices of 1912.

### **3.3 Enhanced Data Preprocessing with Multiple Imputation**

**Method**: Predictive Mean Matching (PMM) via `mice` package

**Theoretical justification**:
1. **Preserves distributional properties**: Maintains original data structure
2. **Uncertainty preservation**: Acknowledges we don't know true missing values
3. **Efficiency**: Uses all available information for better inference

**Comparison results**:
- Original mean age: 29.70 years
- Multiple imputation mean: 29.94 years  
- Median imputation would have been: 28.00 years

**Why this matters**: Multiple imputation preserves uncertainty and avoids bias from oversimplified imputation strategies.

### **3.4 Exploratory Data Analysis**

**Survival patterns revealed dramatic disparities**:
- Overall: 38.4%
- Female: 74.2%
- Male: 18.9%
- First Class: 63.0%
- Second Class: 47.3%
- Third Class: 24.2%

**Visual analysis showed**:
- Age distribution with median at 28 years
- Non-linear family size effects (optimal survival at 2-4 people)
- Clear gender-class interaction patterns
- Port-specific boarding compositions

---

## **4. STATISTICAL MODEL SPECIFICATION** *(5 minutes)*

### **4.1 Theoretical Foundation**

We employ a hierarchical logistic regression model for several theoretical reasons:

1. **Binary Outcome**: Survival is inherently binary, making the Bernoulli distribution the natural choice
2. **Logit Link Function**: Maps probabilities to real line, enabling linear modeling
3. **Hierarchical Structure**: Passengers naturally grouped by embarkation port
4. **Random Effects**: Port-specific random intercepts capture unobserved heterogeneity

### **4.2 Mathematical Formulation**

**Level 1 (Individual)**: 
$$y_i \sim \text{Bernoulli}(p_i)$$

**Level 2 (Linear Predictor)**:
$$\text{logit}(p_i) = \mathbf{X}_i^T\boldsymbol{\beta} + u_{j[i]}$$

**Where**:
$$\mathbf{X}_i^T\boldsymbol{\beta} = \beta_0 + \beta_{\text{female}} \cdot \text{Female}_i + \beta_{\text{age}} \cdot \text{Age}_i^* + \beta_{\text{fare}} \cdot \text{Fare}_i^* + \beta_{\text{family}} \cdot \text{FamSize}_i^* + \beta_{\text{family}^2} \cdot (\text{FamSize}_i^*)^2 + \beta_{\text{class1}} \cdot \text{Class1}_i + \beta_{\text{class2}} \cdot \text{Class2}_i$$

**Level 3 (Group Effects)**:
$$u_j \sim \mathcal{N}(0, \sigma_u^2), \quad j \in \{1,2,3\}$$

**Priors**:
$$\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, 10^2 \mathbf{I})$$
$$\sigma_u \sim \text{Half-Cauchy}(0, 5)$$

### **4.3 Prior Sensitivity Analysis**

**Tested three precision levels**:
- Strong (τ=0.001): SD = 31.6
- Moderate (τ=0.01): SD = 10.0  
- Weak (τ=0.1): SD = 3.2

**Results demonstrated robustness**:
- β_female: 2.691, 2.684, 2.671 (highly stable)
- β_class1: 2.054, 2.058, 2.036 (consistent across priors)

**Statistical conclusion**: Data dominance over prior specification ensures robust inference.

### **4.4 Data Preparation for MCMC**

**Standardization strategy**:
- All continuous variables standardized (mean=0, SD=1)
- Categorical variables coded as indicators
- Random effect indexing for embarkation ports

**Quality checks confirmed**:
- Standardized age range: [-2.11, 3.58]
- Standardized fare range: [-0.65, 9.66]
- Port distribution: Southampton (646), Cherbourg (168), Queenstown (77)

---

## **5. MCMC IMPLEMENTATION AND MODEL FITTING** *(3 minutes)*

### **5.1 Base Model Implementation**

**JAGS model specification**:
```jags
model {
  for(i in 1:N) {
    y[i] ~ dbern(p[i])
    logit(p[i]) <- beta0 + beta_female * female[i] + 
                   beta_age * age_std[i] + beta_fare * fare_std[i] + 
                   beta_family * family_size_std[i] + 
                   beta_family_sq * family_size_sq_std[i] + 
                   beta_class1 * class_1[i] + beta_class2 * class_2[i] + 
                   u_embarked[embarked_idx[i]]
  }
  
  # Weakly informative priors
  beta0 ~ dnorm(0, 0.01)  # precision = 0.01 => variance = 100
  # ... other beta priors
  
  # Half-Cauchy for random effects
  tau_u <- pow(sigma_u, -2)
  sigma_u ~ dt(0, pow(5, -2), 1) T(0,)
}
```
Computational configuration:

4 parallel chains: For convergence assessment
5,000 adaptation: Algorithm tuning phase
15,000 burn-in: Reach stationary distribution
30,000 sampling: Collect posterior samples
Thinning by 20: Reduce autocorrelation
Final sample: 6,000 draws total
5.2 Alternative Model with Interactions
Extended model includes gender × class interactions:

β_int_fem_c1: Female × First Class interaction
β_int_fem_c2: Female × Second Class interaction
Same computational setup with additional parameters tracked

5.3 Prior Sensitivity Analysis Implementation
Systematic testing across three prior scales confirmed robustness of all major conclusions, with parameter estimates varying by less than 1% across specifications.

6. RESULTS AND POSTERIOR INFERENCE (6 minutes)
6.1 Posterior Summary Statistics
Main parameter estimates (Base Model):

Parameter	Mean	SD	95% CI	Significant
β₀	-2.245	0.662	(-3.39, -0.73)	Yes
β_female	2.689	0.202	(2.30, 3.09)	Yes
β_age	-0.471	0.108	(-0.68, -0.27)	Yes
β_class1	2.051	0.303	(1.46, 2.66)	Yes
β_class2	1.074	0.241	(0.61, 1.55)	Yes
σ_u	0.710	1.014	(0.02, 3.66)	Yes
6.2 Odds Ratio Interpretation
Gender Effect: Mathematical Monument to "Women First"
Odds Ratio = 14.71 (95% CI: 9.93, 21.91)

Bayesian interpretation: There's a 95% probability that women had between 10-22 times higher survival odds than men. Historical significance: Quantifies the rigorous implementation of maritime chivalry protocols.

Class Effects: Hierarchy in Crisis
First class vs Third class: OR = 7.77 (4.29, 14.28)
Second class vs Third class: OR = 2.93 (1.83, 4.71)
Economic interpretation: A first-class ticket purchased approximately 677% higher survival odds compared to third class.

Age Effect: Physical Reality Quantified
Odds Ratio = 0.62 per standard deviation

Human interpretation: Each additional 13 years of age reduced survival odds by 38%, capturing physical mobility constraints in extreme evacuation conditions.

Family Size: Behavioral Economics in Crisis
Quadratic relationship detected:

Optimal family size: 2-4 people
Mechanism: Balance of mutual aid without coordination chaos
Theory: Mathematical evidence of cooperation vs. coordination trade-offs
6.3 Interaction Effects Analysis
Gender × Class interaction results:

Interaction	Mean	95% CI
Female × 1st Class	2.142	(0.86, 3.66)
Female × 2nd Class	2.551	(1.47, 3.77)
Predicted survival probabilities:

Male 3rd Class: 14.9%
Female 1st Class: 95.4%
Female 3rd Class: 48.7%
Male 1st Class: 44.1%
Sociological interpretation: Even "women first" protocol couldn't completely overcome class divisions—first-class women approached survival certainty while third-class men faced near-certain death.

6.4 MCMC Convergence Diagnostics
Convergence excellence achieved:

Parameter	R̂	ESS	ESS per Chain
β_female	1.0000	5,773	1,443
β_class1	1.0007	6,074	1,518
β_age	1.0002	5,838	1,460
All diagnostics exceed recommended thresholds:

R̂ < 1.01 for all parameters ✓
ESS > 1,000 for all parameters ✓
Trace plots show excellent mixing ✓
Posterior densities are smooth and unimodal ✓
7. ENHANCED MODEL COMPARISON AND SELECTION (3 minutes)
Information Criteria Analysis
DIC Comparison Results:

Model	Deviance	pD	DIC	ΔDIC	Bayes Factor
Base Model	787.99	9.55	797.54	0.00	1.0
Interaction Model	761.71	11.75	773.46	-24.08	169,231
Evidence interpretation framework:

|ΔDIC| ≤ 2: Weak evidence
2 < |ΔDIC| ≤ 5: Moderate evidence
5 < |ΔDIC| ≤ 10: Strong evidence
|ΔDIC| > 10: Very strong evidence ← Our case
Bayes Factor = 169,231: Overwhelming evidence favoring interaction model

Model Selection Decision
Selected model: Interaction model Evidence strength: Very strong evidence Justification:

ΔDIC = -24.08 indicates very strong statistical evidence
Substantively meaningful gender-class dynamics
Improved predictive performance with modest complexity increase
Theoretical significance: Gender and class effects don't simply add—they interact in complex ways reflecting 1912 social dynamics.

8. MODEL VALIDATION AND POSTERIOR PREDICTIVE CHECKS (3 minutes)
Posterior Predictive Check Methodology
Generated 500 replicated Titanic disasters from posterior distribution to test model adequacy.

Test statistics evaluated:

Overall survival rate
Gender-specific survival rates
Class-specific survival rates
Validation Results
Excellent model fit demonstrated:

Test Statistic	Observed	Bayesian p-value
Overall survival	0.384	0.508 ✓
Female survival	0.742	0.474 ✓
Male survival	0.189	0.492 ✓
Statistical interpretation: Bayesian p-values near 0.5 indicate excellent model fit—our model successfully recreates historical reality.

Visual validation: Histograms of replicated test statistics encompass observed values, confirming model adequacy across multiple dimensions.

Three Levels of Validation Achieved:
Prior sensitivity: Stable across different prior specifications
Posterior predictive: Model recreates observed patterns
Cross-method: Agreement with frequentist approaches (next section)
9. COMPARISON WITH FREQUENTIST METHODS (2 minutes)
Cross-Paradigm Validation
Comparison with alternative methods:

GLM: Standard logistic regression
GLMER: Mixed-effects logistic regression
Parameter concordance demonstrated:

Parameter	Bayesian	GLM	GLMER
β_female	2.689	2.681	2.681
β_class1	2.051	2.037	2.037
β_age	-0.471	-0.463	-0.463
Maximum absolute difference: 0.093 across all parameters

Cross-paradigm validation: High concordance strengthens confidence—we've found statistical truth that transcends methodological boundaries.

Bayesian Advantages Demonstrated
1. Direct probability statements: Can state "95% probability female advantage is 10-22 times" 2. Natural hierarchical modeling: Elegant accommodation of grouped data structure
3. Complete uncertainty quantification: Full posterior distributions for all parameters 4. Principled model comparison: Information-theoretic selection without arbitrary thresholds

10. DISCUSSION AND CONCLUSIONS (4 minutes)
10.1 Primary Findings
This enhanced Bayesian hierarchical analysis revealed key insights:

Gender as Primary Determinant: Women had ~15x higher survival odds, quantifying "women and children first" implementation
Social Stratification Effects: First-class passengers had ~8x higher odds than third-class, proving economic inequality's life-death impact
Age-Related Vulnerability: 38% reduction in odds per SD increase, reflecting physical constraints
Non-linear Family Effects: Optimal survival at medium family sizes (2-4 people)
Interaction Effects: Class advantages amplified for women, indicating complex social dynamics
Port-Specific Variation: Meaningful cultural differences captured through random effects
10.2 Methodological Contributions
Advanced Bayesian techniques demonstrated:

1. Hierarchical Modeling Excellence
Challenge: Grouped data with correlation structure
Solution: Random effects for embarkation ports with Half-Cauchy priors
Innovation: Proper uncertainty quantification for clustered data
2. Missing Data Integration
Challenge: 20% missing age data with systematic patterns
Solution: Multiple imputation preserving uncertainty
Advance: Honest statistical inference acknowledging unknowns
3. Model Comparison Framework
Tools: DIC, Bayes factors with clear evidence interpretation
Result: Very strong evidence (ΔDIC = -24.08) for interaction patterns
Advantage: Principled selection without arbitrary thresholds
4. Comprehensive Validation
Posterior predictive checks: Multiple test statistics with excellent fit
Cross-paradigm verification: Agreement with frequentist methods
Prior sensitivity: Robustness across different prior specifications
10.3 Model Selection and Averaging
Final model selection summary:

Selected model: Interaction model based on DIC
Evidence strength: Very strong evidence (ΔDIC = -24.08)
Bayes Factor: 169,231 (overwhelming support)
Key substantive findings quantified:

Female passengers: ~15x higher survival odds
1st class vs 3rd class: ~8x higher survival odds
Age effect: Significant negative impact (38% reduction per SD)
Family size: Non-linear relationship with optimal at medium sizes
Embarkation port: Meaningful random variation (σ_u = 0.710)
Gender-class interactions: Statistically significant and substantively meaningful
10.4 Limitations and Future Directions
Acknowledged limitations:

Causal inference constraints: Observational data precludes definitive causal claims
Missing data assumptions: MAR may still be violated despite sophisticated imputation
Model specification: Potential unmodeled nonlinearities
External validity: Generalization requires careful historical context consideration
Future research directions:

Spatial modeling: Geographic passenger distribution effects
Survival analysis: Time-to-rescue with censoring
Causal inference: Potential outcomes framework
Machine learning integration: Bayesian neural networks
10.5 Broader Implications
Contemporary relevance:

Emergency response planning: Understanding demographic factors in evacuation success
Social inequality research: Quantifying how stratification affects crisis outcomes
Intersectionality studies: Mathematical evidence of identity intersection effects
Policy development: Informing resource allocation for vulnerable populations
10.6 Methodological Contributions to Bayesian Practice
This analysis demonstrates key aspects of modern Bayesian methodology:

Hierarchical modeling: Proper treatment of grouped data structures
Prior specification: Theoretically justified and sensitivity-tested priors
Model comparison: Multiple criteria for robust selection
Validation: Comprehensive posterior predictive checking
Uncertainty communication: Full probabilistic inference and reporting
CONCLUSION: THE STATISTICAL TESTAMENT (2 minutes)
Methodological Excellence Achieved
We have demonstrated mastery of modern Bayesian hierarchical modeling with proper uncertainty quantification, sophisticated missing data treatment, and comprehensive validation across multiple dimensions.

Substantive insights revealed: Mathematical signatures of systematic discrimination operating with statistical precision in one of history's most tragic events, providing probabilistic evidence of how social hierarchies determined life-and-death outcomes.

Theoretical contributions: Showcased advantages of Bayesian paradigm for complex real-world problems requiring uncertainty quantification and hierarchical structure accommodation.

The Numbers That Tell Human Stories
Every coefficient represents thousands of individual human decisions:

β_female = 2.689: Mathematical monument to maritime chivalry protocols
β_class1 = 2.051: Economic privilege translated directly to survival probability
β_age = -0.471: Physical reality's cruel calculus in extreme conditions
σ_u = 0.710: Cultural and economic differences across embarkation ports
The Bayesian Legacy
Through this comprehensive analysis, we've proven that Bayesian statistics transcends mathematical machinery—it provides a powerful framework for understanding our world with appropriate humility about uncertainty while delivering actionable insights for decision-making.

The Titanic's enduring statistical lesson: Even in humanity's darkest hour, when death seemed inevitable and time impossibly short, the social hierarchies of 1912 operated with mathematical precision that sophisticated Bayesian analysis can detect, quantify, and interpret more than a century later.

This investigation reminds us that:

Behind every coefficient lies a human story
Behind every credible interval lies uncertainty demanding intellectual humility
Behind every statistical model lies the power to illuminate fundamental truths about human society
Modern Bayesian methodology can transform historical tragedy into contemporary insight
Our analysis proves that uncertainty quantification enhances rather than diminishes understanding, hierarchical thinking captures real-world complexity, proper validation ensures reliable inference, and statistical rigor serves human comprehension.

Thank you for your attention. I welcome your questions.

APPENDIX: ANTICIPATED QUESTIONS & RESPONSES
Technical Questions
Q: "Justify your choice of Half-Cauchy prior for variance parameters." A: "Following Gelman (2006), Half-Cauchy provides optimal behavior for variance parameters in hierarchical models. It's conservative when little between-group variation exists but allows large variation when data supports it. The heavy tails handle uncertainty well with small numbers of groups like our three embarkation ports."

Q: "How do you interpret the interaction coefficients statistically?" A: "The positive interaction coefficients (β_int_fem_c1 = 2.142) indicate that class advantages are amplified for women. This means the class effect isn't simply additive—first-class women achieved near-certainty of survival (95.4%) while maintaining the overall female advantage."

Q: "What about potential model misspecification?" A: "We addressed this through comprehensive posterior predictive checking across multiple test statistics. All Bayesian p-values near 0.5 indicate excellent fit. However, we acknowledge potential unmodeled nonlinearities—future work could explore Gaussian process priors for flexible functional forms."

Methodological Questions
Q: "Why multiple imputation over complete case analysis?" A: "With 20% missing age data and evidence against MCAR (p < 0.001), complete case analysis would introduce bias. Multiple imputation preserves uncertainty while using all available information, maintaining statistical power and providing honest inference about missing values."

Q: "How sensitive are your results to MCMC implementation choices?" A: "We used conservative settings: 4 chains, 15,000 burn-in, 30,000 sampling iterations. All convergence diagnostics exceeded standards (R̂ < 1.01, ESS > 1,000). Sensitivity analysis confirmed robustness across different chain lengths and starting values."

Q: "Defend your model comparison approach." A: "DIC provides a principled information-theoretic approach, penalizing complexity while rewarding fit. Our ΔDIC = -24.08 represents very strong evidence (>10) for interactions. The Bayes factor of 169,231 provides overwhelming confirmation through a different computational approach."

