# QRLF (Quantum Rational Lock Finder)

This folder contains the `megecell.py` script (V13.13.9 Code/Output), a comprehensive statistical engine designed to test for "Quantum Rational Locks" in experimental data, along with its output.

## Purpose & Methodology

The script analyzes datasets where a measured probability is extremely small but non-zero. It tests the hypothesis that these values are not random noise but are "locked" to simple, low-complexity **dyadic fractions** (i.e., fractions of the form $1/2^n$).

To do this, the script subjects the data to a gauntlet of modern statistical tests to determine the strength of the evidence.

### Summary of Findings

A full run of the script on the project's 8 data blocks is provided in `megacell_output.txt`(V13.13.9 Code/Output). The analysis provides strong, multi-faceted evidence for the existence of rational locks in this data:

* **Statistical Significance:** The primary Monte Carlo analysis shows a p-value of **p ≤ 0.0005**, indicating the observed number of dyadic locks is highly unlikely to be the result of random chance.
* **Bayesian Evidence:** A formal Bayesian model comparison concludes that the data are over **60 times more likely** under the "rational lock" hypothesis than under the standard null hypothesis (a Bayes Factor of ~61).
* **Robustness:** Leave-one-out and Jackknife resampling tests confirm that this result is stable and not dependent on any single data point.

### The Statistical Gauntlet

The `megacell.py` (V13.13.9 Code/Output) script employs a wide range of validation techniques to ensure the rigor of its conclusion, including:
* Bayesian Model Comparison (Spike-and-Slab vs. Beta-binomial)
* Monte Carlo P-Values ("PB-tails")
* Family-Wise Error Rate (FWER) correction for multiple comparisons
* Predictive Hold-Out Testing (Leave-One-Out)
* Robustness Checks (Jackknife Resampling)
* E-Value Meta-Analysis
