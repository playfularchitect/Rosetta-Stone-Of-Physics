Colab Link - https://colab.research.google.com/drive/1Xmp8t0_03dy5vmqJLlpyPgfmeNmEU4qL?usp=sharing

# Big Null Time — Null Code V2

This folder is part of my larger **Rosetta Stone of Physics** project.

Here, I do one very specific job:

> **I smash our universe against huge null ensembles and measure how “compressible” its structure is, in a fair, look-elsewhere-aware way.**

The two main tools are:

1. **Snap-null tests** –  
   How often does a random number land as close to a “simple” fraction as the real number does, once you include the look-elsewhere effect?

2. **MDL / ledger tests** –  
   How many bits does it take to encode a simple ledger that ties several physical quantities together, and how often does random data reach that same level of compression?

Snap-null is the **control group**.  
The **MDL + ledger** result is the **signal**.

This folder is where I show, with code and raw logs, that the “fraction physics” picture is not wishful thinking. Under explicit null models, the real ledger is **dramatically more structured and compressible** than random.

---

## TL;DR — What this actually shows

Short version, no fluff:

1. **Individual Standard Model numbers as a control**

   I run snap-null tests on many familiar scalars:

   - CKM angles  
   - α and α_s  
   - sin²θ_W  
   - mass ratios like M_W/v, M_Z/v, M_H/v, m_t/v, etc.

   When you do snap-null correctly (with look-elsewhere built in), these numbers:

   - **Behave exactly like random draws should** in this framework.  
   - Empirical p-values live comfortably in the **0.27–0.98** range.

   This is what a healthy control group looks like. The method is not “rigged” to spit out miracles.

2. **ρ² and Koide Q under fixed ledger targets**

   When I switch from “slide to best `p/q`” to **fixed ledger targets**:

   - ρ² and Koide Q become **strong outliers**:
     - ρ² ledger p ≈ **10⁻⁶**  
     - Koide Q ledger p ≈ **10⁻⁵**
   - sin²θ_W alone is less extreme (p ~ 10⁻¹), but it is part of the same ledger.

   Same math. Same random machinery.  
   Different rule (target fixed in advance) → **very different story** for these two scalars.

3. **Joint ledger triple (ρ², Koide Q, sin²θ_W)**

   I then test the **triple**:

   - Ledger target for ρ²  
   - Ledger target 2/3 for Koide Q  
   - Ledger target for sin²θ_W  

   From the marginal ledger p’s, the geometric model gives:

   - **Expected joint probability ≈ 2.26 × 10⁻¹²** (about 1 in 4.4 × 10¹¹).

   In **3 × 10 million** simulated triples, I see:

   - **0** random triples that match the real one’s combined accuracy.

   So under this null model, the real triple does not look like a typical fluctuation.  
   It sits in a region of parameter space that the random triples simply do not reach at this sample size.

4. **MDL: the main highlight**

   This is the core result of this folder.

   - Real ledger MDL ≈ **353 bits**.  
   - Null MDL ≈ **462 ± 5 bits** across widths and seeds.  
   - The real MDL lies about **20–21σ below** the null mean.  
   - Across tens of millions of Monte Carlo universes, I get:
     - `count(MDL <= real) = 0`.

   In plain language:

   > **Given the rules of this test, the ledger that fits our universe is vastly more compressible than the ledgers of random universes.**

   This is not a tiny effect hiding in noise.  
   It is a **very large gap** in code length, sitting deep in the tail of the null distribution.

5. **The contrast that matters**

   Putting it together:

   - Most **individual Standard Model scalars** behave like ordinary random-ish numbers under the same snap-null machinery.
   - The **specific ledger structure** linking ρ², Koide Q, and sin²θ_W:
     - Hits three tiny ledger targets at once (joint triple).  
     - Produces an MDL score that null universes simply do not match.

   The evidence in this folder is very straightforward:

   > Under explicit null models, the ledger is **much simpler** (in bits) than what randomness produces.
   >
   > The contrast between “ordinary scalars” and “ledger structure” is extremly sharp and it is all repeatable/verifiable.

---

## Core Goal (in simple terms)

In the bigger project for Rosetta, my axioms say:

- Reality is built from **relations and ratios**, not floating-point accidents.
- A brilliant test of that claim is: **How short is the code that describes what we see, compared to random alternatives?**

This folder runs that test in a concrete way.

Picture it like this:

- Each physical quantity is **a point between 0 and 1**:
  - ρ² = (M_W / M_Z)²  
  - Koide Q(e, μ, τ)  
  - sin²θ_W  
  - CKM s_12, s_13, s_23, etc.  
  - Ratios like M_W / v, M_Z / v, M_H / v, m_t / v

- I build a big library of **“nice” fractions** `p/q`:
  - denominators up to 1000  
  - bit-size ≤ 20  
  - limited to bands like `[0.6, 0.9]`, `[0.15, 0.35]`, or ±1 decade around the real value

Then I compare two worlds:

1. **Random universes**  
   - Draw numbers uniformly in those bands.  
   - Check how close they land to the nearest allowed fraction.  
   - Check how many bits it takes to encode a ledger built on those hits.

2. **Our universe**  
   - Use the actual measured values.  
   - Run the exact same procedures.

**Analogy 1 – Dartboard**

> The allowed fractions are tiny bullseyes on a dartboard.  
> Random universes throw random darts.  
> Our universe is one special throw.  
> I measure how often random throws do at least as well as the real throw.

**Analogy 2 – Zip file**

> The ledger is a zip file for a bundle of Standard Model numbers.  
> MDL is the size of that zip file in bits.  
>  
> Random universes need about 462 bits on average.  
> The real universe gets away with ~353 bits.  
> That is not “just a small improvement” – it is a ** undeniably huge compression jump**.

Under these tests, the data is clear:

- **Controls**: ordinary scalars look like random draws.  
- **Ledger**: the pattern that ties key scalars together is far too compressed to be explained by the same null.

---

## What lives in this folder

- **`Null Code V2`**
  - Full source code for every test in this README.
  - Builds rational families (`p/q`, `q ≤ 1000`, bits ≤ 20).
  - Runs big Monte Carlo ensembles (10 million universes or triples per run, multiple seeds and widths).
  - Reports summary stats: means, p-values, MDL distributions, z-scores, etc.

- **`Null Output V2`**
  - The complete raw logs from `Null Code V2`.
  - Every number quoted here comes straight out of those logs.

You do not have to trust my summary.  
You can read the logs, or re-run the notebooks, and watch the numbers fall out yourself.
Prove it to yourself do not believe my words. I have given all the evidence/work you need.

---

## Module guide (with the key takeaways)

### MODULE 2 — Big Snap-Null for ρ² = (M_W / M_Z)²

- Band: `[0.6, 0.9]`.  
- Fractions: `q ≤ 1000`, bits ≤ 20.  
- Define ε_real as the distance from the real ρ² to the nearest allowed fraction.  
- Run 3 × 10 million random universes and compare.

Result:

- Snap-null p ≈ **0.10**.

Takeaway:

> Under look-elsewhere-aware snap-null, ρ² is mildly interesting but not extreme.  
> About one in ten random universes does at least this well at snapping to *some* fraction in the band.

This sets the baseline for how ρ² behaves when you’re allowed to slide the target.

---

### MODULE 3 — Big Snap-Null for Koide Q(e, μ, τ)

Same snap-null setup, now for **Koide’s Q**:

- Band: `[0.4, 0.9]`.  
- Same rational family.  
- 3 × 10 million universes.

Result:

- Snap-null p ≈ **0.95**.

Takeaway:

> With proper look-elsewhere treatment, Koide’s Q behaves like a totally normal random-ish scalar in this framework.  
> Its closeness to some `p/q` in the allowed family is not rare.

This is an important control: Koide Q does **not** automatically generate tiny p-values here.

---

### MODULE 4 — Ledger Nulls for ρ², Koide Q, sin²θ_W

Now I switch to **ledger mode**:

- No more sliding targets.  
- I fix one **single** rational target in advance for each scalar:
  - Ledger target for ρ².  
  - Ledger target **2/3** for Koide Q.  
  - Ledger target for sin²θ_W.

For each scalar:

1. Draw 10 million random values in the band (3 independent seeds).  
2. Measure the distance to that fixed target.  
3. Compare to the real universe’s distance.

Results:

- ρ² ledger p ≈ **10⁻⁶**  
- Koide Q ledger p ≈ **10⁻⁵**  
- sin²θ_W ledger p ≈ **10⁻¹** (~0.08)

Takeaway:

> Under fixed-target ledger rules, ρ² and Koide Q jump from “ordinary” to **very strong outliers**.  
> The same random machinery, with the target set ahead of time, exposes how tight these locks are.

---

### MODULE 5 — Joint Ledger Triple (ρ², Koide Q, sin²θ_W)

Here I enforce all three ledger targets at once:

- (ρ² close to its ledger target)  
- (Koide Q close to 2/3)  
- (sin²θ_W close to its ledger target)

I:

- Estimate joint probability using the geometric model:
  - **p_joint ≈ 2.26 × 10⁻¹²**.  
- Run 3 × 10 million random triples in the appropriate bands.

Result:

- In every run, **0** random triples land as close (or closer) than the real triple in all three components simultaneously.

Takeaway:

> Under this null model, the real triple sits in a region of probability space that the sampled random triples do not reach.  
> The three ledger hits are not behaving at all like independent, casual coincidences.

---

### MODULE 6 — MDL Width Robustness (main MDL result)

This is the core compression test.

- Define an **MDL in bits** for a concrete ledger that encodes:
  - ρ², Koide Q, sin²θ_W, and related ratios.  
- Compute this MDL for:

  - The **real universe** → **353 bits**.  
  - Large ensembles of **null universes**, across multiple widths (±0.3, ±1, ±2 decades around the real grid).

For each width:

1. Simulate 3 × 10 million null universes.  
2. Compute MDL for each.  
3. Build the null MDL distribution and compare with 353 bits.

Null summary:

- Mean MDL ≈ **462 bits**.  
- Std dev ≈ **5.2 bits**.  
- Real MDL ≈ **353 bits** → about **20–21σ** below the null mean.  
- In all runs:
  - `count(MDL <= real) = 0`.

Takeaway:

> The ledger that fits our universe compresses its target scalars **far beyond** what these null models produce.  
> The gap is large, stable in width, and robust across independent seeds.

This is the main quantitative support, in this folder, for the proof that the “ledger pattern” is not typical random noise.

---

### MODULE 7 — Big Snap-Null for sin²θ_W (standalone sanity check)

Dedicated snap-null test for **sin²θ_W**:

- Band: `[0.15, 0.35]`.  
- `q ≤ 1000`, bits ≤ 20.  
- 3 × 10 million universes.

Result:

- Snap-null p ≈ **0.40**.

Takeaway:

> As a lone scalar in snap-null mode, sin²θ_W is perfectly ordinary.  
> Its special role shows up in the **joint ledger** and in the **MDL score**, not in its isolated snap-null behavior.

---

### MODULE 8 — Big Snap-Null Registry Scan (broad control)

This is the broad **control scan** over a registry of important parameters:

- **CKM**
  - CKM_s12, CKM_s13, CKM_s23, CKM_delta_over_pi  
- **Couplings**
  - α, α_s(M_Z), sin²θ_W  
- **Electroweak ratios**
  - M_W / v, M_Z / v  
- **Higgs**
  - M_H / v  
- **Heavy quark**
  - m_t / v  

For each parameter:

1. Define a ±1 decade band around the real value (clipped to [1e−10, 1]).  
2. Build the rational family (`q ≤ 1000`, bits ≤ 20).  
3. Compute ε_real = distance to nearest allowed fraction.  
4. Run 3 × 10 million snap-null trials and estimate p_emp.

Outcome:

- p-values fall in a **normal, non-pathological range**: ~0.27 to 0.98.  
- No “automatic miracles.” No hidden bias forcing tiny p-values.

Takeaway:

> The snap-null machinery behaves exactly like a proper statistical tool on this registry.  
> It treats most Standard Model scalars as ordinary random-ish inputs.
>  
> That’s why the **contrast** with the ledger-based results carries real weight.

---

## Why this matters in the bigger picture

In the full **Rosetta Stone of Physics** project, I start from simple axioms:

- Reality is fundamentally **relational** (differences, ratios, paradox).  
- At the deepest level, it should look like a system of **simple, exact fractions**, not floating-point noise.  
- If that’s true, the code that describes our universe’s constants should be **highly compressible (It is)**.

This folder is another slice of proof where I push that idea against aggressive null tests:

- **Controls:** show that ordinary numbers behave like ordinary numbers.  
- **Ledger + MDL:** show that the interlocking pattern picked out by the theory is **far more compressed** than random alternatives, under explicit rules.


- The code is here.  
- The logs are here.  
- The compression gap is here.
- The math is clear

