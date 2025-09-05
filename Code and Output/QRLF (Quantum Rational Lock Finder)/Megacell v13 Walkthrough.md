Megacell v13.13.9 — Section‑by‑Section Code Walkthrough

Purpose: This document explains what each part of the code does, how it works, and why we designed it that way. It doubles as a reviewer’s guide for replication and audit.

0\) High‑level overview

Goal: Test whether observed signals line up with a registry/ledger of simple fractions (esp. dyadics 1/2^k), using multiple, preregistered endpoints:

Per‑block and per‑cluster PB tails (Monte Carlo exceedance probabilities)

Family‑wise error rate (FWER) across the prereg family

Bayes factors (spike prior on dyadics, slab \= Beta(1/2,1/2))

E‑values and meta‑combinations

Stability checks: leave‑one‑out (dataset), jackknife (block), and a distance‑style T statistic

Reproducibility: Fixed RNG seed, ledger SHA256s \+ registry hash, code fingerprint, and an artifact bundle (CSV/JSON \+ README) output to results/run\_YYYYMMDD-HHMMSS/.

1\) Imports & global utilities  
Files / RNG / numerics

os, sys, hashlib, inspect, random, numpy as np: filesystem, hashing, runtime introspection, RNG.

dataclasses.dataclass, fractions.Fraction, typing.

Artifact helpers (Why: professional receipts)

ensure\_outdir: creates results/run\_.../, plus sections/ and ledger/ subfolders.

write\_text / write\_json / write\_csv: thin wrappers to standardize encoding and CSV headers.

Tee: utility to broadcast stdout to multiple streams (used if you tee logs in the run block).

dump\_environment: writes Python/NumPy versions and platform info.

Why: Reviewers need deterministic, traceable context. We generate machine‑readable receipts for every section.

2\) Registry (ledger) I/O and hashing  
Forms we compute

sha256\_of\_file: content hash of each ledger file path.

parse\_fracs\_from\_text: Tokenizes to avoid accidental overlaps (e.g., 11/2/4 fragments). Extracts fractions a/b and decimals strictly in (0,1).

try\_load\_ledger: Reads candidate ledger files, parses to Fractions, records each file’s SHA256.

working\_ledger: Merges a small FALLBACK set with parsed file fractions, filters to strict probabilities 0\<p\<1, deduplicates.

sha256\_of\_ledger: Canonical hash of the working registry: sort reduced num/den, newline‑join, hash. This is the preregistration anchor.

Why these choices

Tokenization prevents false positives from regex greediness.

Sorting & reducing ensures the registry hash is deterministic across machines.

Strict (0,1) filter: our tests are about probabilities, not boundary cases.

3\) MDL\* complexity measure & nearest fractions  
Definitions

bitlen\_nonneg: Special conventions bitlen(0)=1, bitlen(1)=0, else built‑in bit length. Gives a unique tag for zero and no penalty for the identity one.

mdl\_star(fr): Complexity of a reduced fraction a/b as bitlen(a)+bitlen(b).

nearest\_fraction(p, frs, max\_mdl): Among candidate fractions, choose the closest to p; break ties by smaller MDL\*. Ignore ultra‑complex fractions via max\_mdl (default 30\) to keep displays intelligible.

Why this MDL\*

We want a simple, data‑independent complexity prior: shorter numerators/denominators → simpler fractions. This induces weights 2^{-MDL\*} (see spike prior below), biasing toward interpretable rationals without hand‑tuning.

4\) Dyadics & blocks

Dyadics: 1/2^k pools: ALL\_DYADICS \= k∈\[2,16\], TINY\_DYADICS \= k∈\[TINY\_K\_MIN,16\] (default k≥8, i.e., ≤1/256). These define our spike support sets.

Block dataclass: Holds dataset id, sheet, tag, total n, XOR count kX, and two A/B proportions (for separate sanity checks).

DATASET\_NAMES, DS\_INDEX: Map blocks to cluster labels for cluster‑level tests.

Why dyadics

They are the simplest, highly compressible probabilities; they anchor the “simple fraction” hypothesis family. Restricting to TINY tests sensitivity near very small dyadics, our prereg primaries.

5\) Confidence intervals (Wilson) & CI‑coverage tests

wilson\_ci(k,n,z): Numerically stable Wilson interval with guardrails (never under the sqrt; clamp to \[0,1\]).

ci\_contains\_any\_dyadic(k,n,z, pool): Returns whether the CI covers any fraction in the pool.

Why Wilson

Wilson’s interval has good coverage, is well behaved near 0/1, and is standard for binomial proportions. The guardrails avoid numerical problems for extreme counts.

6\) Monte Carlo PB tails (blocks & clusters)

mc\_pb\_tail: Compute observed block hits; simulate under p \~ Beta(a,b) (default 1/2,1/2) and k \~ Binomial(n,p). PB tail is Pr(sim\_hits ≥ obs\_hits).

cluster\_pb\_tail: Same, but at the dataset/cluster level (whether any block in a dataset hits ⇒ dataset marked 1). PB tail is Pr(sim\_cluster\_hits ≥ obs\_cluster\_hits).

mc\_pb\_tail\_both: Combined simulator that reuses the same draws to compute both block and cluster PB tails for a given (z, pool).

Why Beta(1/2,1/2)

The Jeffreys‑like slab is symmetric and non‑informative for Bernoulli in this context. It preserves exchangeability across blocks and supports analytical forms in the Bayes section.

Floors & display

We floor Monte Carlo zeros to ≤ 1/sims for display (conservative), and compute the display SE on the floored value to avoid printing ±0.

7\) Bayes factors (spike vs slab, and spike+slab mixture)

Log‑domain numerics: log\_beta, logsumexp, and log\_beta\_binom\_noC (omit combinatorial constant since it cancels in BFs).

Spike (prior on dyadics): prepare\_spike\_weights assigns weights ∝ 2^{-MDL\*} over the dyadic pool; precomputes log(p) and log(1-p) for each atom.

Slab: Beta(1/2,1/2) over \[0,1\].

total\_lnBF:

Compute spike‑only lnBF vs slab across all blocks.

Compute spike+slab mixture lnBF by optimizing mixture weight ε∈(0,1) via a numerically stable ternary search over \[1e-9,1-1e-9\].

Why MDL\* weights and log math

MDL\* reflects Occam’s razor: simpler rationals get more prior mass.

Working in the log domain avoids underflow in products like p^k(1-p)^{n-k} at large n.

8\) Ancillary tests  
A/B sign tests

binom\_two\_sided\_p: Exact two‑sided sign test for whether pA/pB sit above/below 1/2 across blocks. Sanity lens around 0.5 (not part of primary endpoints).

Per‑block p‑values for E‑values

per\_block\_pvals: For each block, if the observed CI hits the pool, simulate under Beta(1/2,1/2) to estimate a per‑block p. Floor to 1/sims for multiplicative E‑value stability.

Optional fast path (FAST\_PVALS=1): A conservative early‑exit that checks whether a tiny set of small k could ever reach the smallest dyadic; if impossible, returns 1 early. Off by default to keep behavior maximally conservative.

Nearest‑dyadic standardized distances

nearest\_dyadic\_z: For each block’s XOR rate kX/n, find nearest dyadic, compute standardized distance |ph − dy|/SE(ph).

Why include these

E‑values give an alternative evidence scale (Markov‑bound interpretation). The z distances act as an interpretable effect‑size dial.

9\) Reporting sections (stdout) — what they print and why

section\_candidates

Prints each candidate block with A/B nearest registry fractions (using MDL\* tie‑breaks) and the XOR Wilson CI vs nearest dyadic.

Why: Transparency about where observed proportions lie relative to prereg fractions.

section\_ab\_sign

Exact binomial p‑values for A and B around 1/2.

Why: Quick sanity check that neither arm is drifting in an obvious direction.

section\_pb

For each z ∈ {1.96, 2.24, 2.58} and pool ∈ {ALL, TINY}: prints block & cluster PB tails with display floors and SE. Primary prereg endpoint (z=2.24, TINY) is flagged.

Why: This is Proof P1, the main prereg inferential lens.

familywise\_mc

Monte‑Carlo family‑wise exceedance: probability that any prereg endpoint meets/exceeds the observed hits under the null.

Why: Controls multiplicity across related endpoints.

section\_nearest\_dyadic\_z

Prints standardized distances to nearest dyadic.

Why: Intuitive effect‑size readout; complements PB tails.

section\_bayes

Reports spike‑only and spike+slab lnBF for ALL/TINY dyadic pools and the optimized ε\*.

Why: Independent evidence scale; sensitive to MDL\*‑weighted alignment.

section\_predictive\_holdout

Leave‑one‑dataset‑out: tune ε\* on train, evaluate mixture lnBF on the held‑out dataset. Notes when ε\* collapses to boundary (0 → pure spike; 1 → pure slab) and why that implies a zero predictive BF in the latter.

Why: Shows cross‑dataset predictive signal rather than in‑sample fit.

section\_evalues

Prints observed hits, Tippett min‑p, Simes p, and the product E‑value (with Markov‑bound note).

Why: Another inferential lens robust to dependence modeling choices.

section\_min\_T

Exploratory distance statistic: T \= Σ\_z |ph−dy|/SE across blocks; compares to a null MC distribution.

Note in output: the dyadic set excludes 1/2, so null draws near 0.5 look far from any allowed dyadic → large null T.

Why: Effect‑size aggregation that’s easy to visualize; explicitly marked exploratory.

section\_cluster\_LOO

Leave‑one‑dataset‑out cluster PB tails at the primary endpoint.

Why: Stability of the cluster result to dropping any one dataset.

section\_jackknife

Jackknife over blocks for the primary endpoint.

Why: Block‑level influence diagnostics.

10\) Self‑tests (sanity invariants)

MDL\* convention checks (bitlen, dyadic vs non‑dyadic non‑inferiority).

Wilson symmetry around 0.5.

Nearest‑fraction MDL\* tie‑break behavior.

Spike prior normalization (weights sum to 1 in prob domain).

Why: Quick smoke tests catch regressions immediately.

11\) Run block & artifacts

Loads/merges the ledger; prints per‑file SHA256 and preregistered registry hash.

Executes all reporting sections in a deterministic order.

Artifacts: After printing, writes:

config.json, environment.json (repro context)

ledger/registry\_fractions.txt, ledger\_hash.txt, file\_hashes.json

sections/\*.csv|json for every section

script\_snapshot.py (best‑effort source capture)

README.md describing the bundle

Why: Everything a reviewer needs to re‑run and verify is emitted automatically.

12\) Configuration knobs

SEED — master RNG seed for numpy & Python random.

MC\_SIMS\_PRIMARY, MC\_SIMS\_FWER, MC\_SIMS\_EVAL — simulation budgets.

Z\_FAMILY — the prereg endpoints.

TINY\_K\_MIN — lower bound for tiny dyadics (default 8 ⇒ ≤1/256).

MAX\_MDL\_NEAREST — hide ultra‑complex fractions when printing “nearest fraction”.

LEDGER\_PATHS — search order for the ledger file(s).

Env flags:

ASCII\_ONLY=1 ⇒ prints \<= instead of ≤ for floors.

FAST\_PVALS=1 ⇒ enables conservative early‑exit in per‑block p‑values (kept off by default for strict conservatism).

13\) Interpretation checklist (for readers)

Primary evidence: z=2.24 \+ TINY dyadics. Look at both blocks and clusters → PB tails.

Multiplicity: confirm FWER.

Bayes: spike+slab lnBF (report ε\*).

Meta: min‑p, Simes, product E‑value.

Stability: LOO (dataset) and jackknife (block).

Repro: verify preregistered-ledger-hash and file SHA256s.

14\) Design tradeoffs & alternatives

Wilson vs. exact Clopper–Pearson: Wilson has better average coverage and is smoother; CP is conservative and wider near extremes.

Beta(1/2,1/2) slab: symmetric and standard; alternatives (e.g., Beta(1,1)) change tails slightly but don’t offer clear theoretical advantages here.

Dyadic spike vs broader rational spike: We emphasize dyadics for interpretability and compressibility; the framework can accept any discrete pool.

MDL definition:\* We use bit lengths of numerator and denominator; you could add a power‑of‑two “bonus” if you want to strictly prefer dyadics (kept neutral here to avoid post‑hoc bias).

15\) Known footnotes

Floors: Display floors (≤ 1/sims) ensure we don’t report impossible zeros; SE printed alongside is computed on the displayed p to avoid ±0.

T statistic: Marked exploratory, with a note about 1/2 exclusion’s effect on null magnitudes.

Source snapshot: script\_snapshot.py is best‑effort and may fall back to a short placeholder string when introspection is limited.

16\) How to extend

Add more endpoints by following the pattern: (a) observed count, (b) null simulator, (c) CSV/JSON exporter, (d) register it in the artifact README.

Swap or augment the spike pool by editing dyadics\_set or providing a new candidate fraction list.

Parameterize sims and flags via argparse if running from CLI; hook Tee to persist stdout.txt automatically.  
