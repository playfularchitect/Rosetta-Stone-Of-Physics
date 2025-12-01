Colab Link - https://colab.research.google.com/drive/1Xmp8t0_03dy5vmqJLlpyPgfmeNmEU4qL?usp=sharing

# Big Null Time — Null Code V2

At a one-sentence level, what I’m doing here is:

> **I compare our universe to millions of fake universes to ask:  
> “How surprising is the pattern structure in a fair, look-elsewhere-aware way?”**

The two main pieces are:

1. **Snap-null tests** – “How often does a random number look as close to a ‘nice’ fraction as the real one does?”  
2. **MDL / ledger tests** – “How many bits does it take to encode a simple ledger that ties certain physical quantities together, and how rare is that level of compression under random data?”

Snap-null is the control group.  
The **MDL + ledger** result is the main signal.

---

## TL;DR — What this actually shows

Here’s the compressed story:

1. **Snap-null on lots of individual Standard Model numbers**  
   - CKM angles, α, α_s, sin²θ_W, mass ratios like M_W/v, M_Z/v, M_H/v, m_t/v, etc.  
   - Under a correct, look-elsewhere-aware snap-null,  
     their behaviour is **statistically ordinary**.  
   - p-values cluster in the 0.27–0.98 range.  
   - This is exactly what a proper control group should look like.

2. **ρ² and Koide Q under ledger targets**  
   - When I fix specific ledger fractions (instead of sliding to the best `p/q`),  
     ρ² and Koide Q become **strong outliers**:
     - ρ² ledger p ~ 10⁻⁶  
     - Koide Q ledger p ~ 10⁻⁵  
   - sin²θ_W alone is less extreme, but it still plays a role in the ledger structure.

3. **Joint ledger triple (ρ², Koide Q, sin²θ_W)**  
   - Expected joint probability from the geometric model ≈ **2.3 × 10⁻¹²**.  
   - In 3 × 10 million simulated triples, I see **0** that match the real triple’s accuracy.  
   - This confirms that, under this null model, the real three-way hit is **extraordinarily unlikely** by chance.

4. **MDL: the main highlight**  
   - Real ledger MDL ≈ **353 bits**.  
   - Null MDL ≈ **462 ± 5 bits** across widths and seeds.  
   - The real MDL sits about **20–21σ below** the null mean.  
   - In tens of millions of random universes, **none** compress as well as the real ledger.  

   In plain language:

   > Given these rules, the ledger that fits our universe is  
   > **far more compressible than what random data generates.**

5. **Putting it all together**

   - **Most individual scalar values** in the Standard Model look like ordinary random-ish numbers under fair snap-null tests.  
   - **The specific ledger structure** tying together ρ², Koide Q, and sin²θ_W is what stands out:
     - It hits three tiny ledger targets at once (joint triple test).  
     - It produces an MDL score that lies deep in the tail of the null distribution.

---

## Core Goal 

Think of each physical quantity as **a point on a line between 0 and 1**:

- ρ² = (M_W / M_Z)²  
- Koide’s Q(e, μ, τ)  
- sin²θ_W  
- mixing angles, couplings, mass ratios like M_W / v, etc.

Separately, I build a large library of **“nice” fractions** `p/q`:

- denominators up to 1000  
- total bit size ≤ 20  
- restricted to some band like `[0.6, 0.9]` or `[0.15, 0.35]`

Then:

- A **random universe** = draw random numbers in the band(s), and measure how well they “snap” to the nearest allowed fraction.
- The **real universe** = our actual measured values, run through the exact same procedure.

Analogy:

> Imagine a dartboard full of tiny bullseyes (all the allowed fractions).  
> Random universes are random darts.  
> I check how often random darts hit at least as close to a bullseye as the real darts do.

On top of that, there’s a **ledger**:

- A simple bit-encoded structure tying together:
  - ρ²,
  - Koide Q,
  - sin²θ_W,
  - and some related ratios.
- The **MDL (Minimum Description Length)** is the number of bits needed to specify this scheme.

Second analogy:

> The ledger is a **zip file for certain Standard Model numbers**.  
> The MDL score is the size of that zip: fewer bits = stronger compression.

The key result of this project is:

> **The real ledger compresses *far* better than the ledgers of random universes,  
> across tens of millions of Monte Carlo samples.**

Everything else (snap-null, registry scan) is there to show that the method behaves correctly on typical scalars, and that the standout behaviour is specific to the ledger structure.

---

## What lives in this folder

- **`Null Code V2`**
  - Full code to reproduce all results.
  - Builds rational families (`p/q`, `q ≤ 1000`, bits ≤ 20).
  - Runs large Monte Carlo ensembles (10M universes or triples per run).
  - Prints detailed summaries: means, p-values, z-scores, etc.

- **`Null Output V2`**
  - The raw logs from running `Null Code V2`.
  - This is the direct record of what the simulations produced.

---

## Module guide (short, but keeps the key numbers)

### MODULE 2 — Big Snap-Null for ρ² = (M_W / M_Z)²

- Band: `[0.6, 0.9]`.
- Rational family: `q ≤ 1000`, bits ≤ 20.
- I find the closest allowed fraction to the real ρ² and call that distance **ε_real**.
- Then I run 3 × 10 million random universes:
  - For each random draw, find its nearest fraction.
  - Compare that distance to ε_real.

Result:

- Empirical snap-null p-values ≈ **0.10** (about 10%).

Meaning:

> Under look-elsewhere-aware snap-null, ρ² is mildly interesting but not extreme.  
> Around one in ten random universes produces a ρ² that snaps at least as well to some allowed fraction.

This is an important contrast with the **ledger-fixed** version of ρ².

---

### MODULE 3 — Big Snap-Null for Koide Q(e, μ, τ)

Same snap-null idea, now for **Koide’s Q**:

- Band: `[0.4, 0.9]`.  
- Same fraction constraints.  
- 3 × 10 million universes.

Result:

- Empirical p-values ≈ **0.95**.

Meaning:

> Once the look-elsewhere effect is handled correctly (searching over many `p/q`),  
> Koide’s Q is **statistically ordinary** in this snap-null framework.  
> Its closeness to some fraction in the family is not rare.

So Koide Q, by itself under snap-null, is a clean control: the method does not force it to look special.

---

### MODULE 4 — Ledger Nulls for ρ², Koide Q, sin²θ_W

Here I change the rules to match the ledger idea.

Instead of letting the target slide to the **best** fraction anywhere in the band, I:

- Fix one specific **ledger fraction** in advance for each scalar:
  - ρ² has a fixed ledger target.
  - Koide Q has fixed target **2/3**.
  - sin²θ_W has its own fixed ledger target.

For each quantity:

1. Draw 10 million random values in the band, 3 runs.
2. Compute the distance to that **single fixed target**.
3. Count how often random draws are at least as close as the real value.

Key results:

- ρ² ledger p ≈ **10⁻⁶**  
- Koide Q ledger p ≈ **10⁻⁵**  
- sin²θ_W ledger p ≈ **10⁻¹** (~0.08)

So:

- Under **ledger** rules, ρ² and Koide Q become **strong statistical outliers**.  
- sin²θ_W, by itself, stays within a fairly ordinary range.

This shows that “target fixed in advance” and “target chosen after the fact” are very different regimes, and the ledger lives firmly in the fixed-target regime.

---

### MODULE 5 — Joint Ledger Triple (ρ², Koide Q, sin²θ_W together)

Now I treat the three ledger targets as a single joint condition:

- Target for ρ² (ledger fraction),  
- Target 2/3 for Koide Q,  
- Target for sin²θ_W.

Procedure:

- 3 runs × 10 million **triples** (ρ², Q, sin²W), sampled uniformly on their bands.  
- Expected joint probability from the individual ledger p’s:
  - **Geometric p_joint ≈ 2.26 × 10⁻¹²** (about 1 in 4.4 × 10¹¹).  
- In all runs:
  - **count(all three ≤ their ε_real)** = 0 / 10,000,000 per run, consistent with that tiny probability.

Meaning:

> Under this null model, the probability of randomly matching the real triple’s accuracy in all three ledger quantities at once is extremely small.  
> The simulations behave exactly as they should given that tiny p_joint, and the real triple sits in a region that the random ensembles simply do not visit.

This is one key pillar of the conclusion that the ledger is unusually good.

---

### MODULE 6 — MDL Width Robustness (the main result)

Here I evaluate how well the ledger **compresses** the data.

- I define an **MDL (Minimum Description Length)** in bits for the ledger that ties a specific set of Standard Model scalars together.
- For the real universe’s ledger:
  - **Real MDL ≈ 353 bits**.

Then I build large null ensembles for different width choices:

- ±0.3 decades, ±1 decade, ±2 decades around the real-value grid.  
- 3 runs × 10 million universes per width.

For each random universe:

1. Construct the corresponding ledger-like structure.  
2. Compute its MDL bits.  
3. Build the null distribution and count how often MDL(null) ≤ MDL(real).

What the null shows:

- Null mean MDL ≈ **462 bits**.  
- Null standard deviation ≈ **5.2 bits**.  
- Real MDL (353 bits) lies about **20–21σ below** the null mean.  
- Across **tens of millions** of null universes:
  - `count(MDL <= real) = 0`.

Analogy:

> The ledger is a zip file that compresses a chosen set of physical numbers.  
> Null universes typically need about **462 bits** of code.  
> Our universe does it in **~353 bits**.  
> That difference is not a small optimization; it is a **large, statistically extreme compression gap**.

This is the central result:

- The MDL test shows that the real ledger is **far more compressible** than the null ensemble.  
- The effect is stable across wide changes in width and seeds.

---

### MODULE 7 — Big Snap-Null for sin²θ_W (standalone check)

This is a focused snap-null test just for **sin²θ_W**:

- Band: `[0.15, 0.35]`.  
- Fractions: `q ≤ 1000`, bits ≤ 20.  
- 3 × 10 million universes.

Result:

- Empirical p ≈ **0.40**.

Meaning:

> As a single number trying to “snap” to some nice fraction, sin²θ_W is **unremarkable**.  
> Its snap-null behaviour is typical.  
> The interesting part is how it participates in the **joint ledger triple** and in the **global MDL score**, not its standalone snap-null value.

---

### MODULE 8 — Big Snap-Null Registry Scan (control group)

This module is a broad **sanity check** for the snap-null machinery.

I apply the same style of snap-null tests to a small registry of important SM parameters:

- **CKM sector**
  - CKM_s12  
  - CKM_s13  
  - CKM_s23  
  - CKM_delta_over_pi  
- **Couplings**
  - α (fine-structure constant)  
  - α_s(M_Z)  
  - sin²θ_W (again, as a registry entry)  
- **Electroweak ratios**
  - M_W / v  
  - M_Z / v  
- **Higgs**
  - M_H / v  
- **Heavy quark**
  - m_t / v  

For each parameter:

1. Define a band of ±1 decade around the real value, clipped to [1e−10, 1].  
2. Build the rational family (`q ≤ 1000`, bits ≤ 20).  
3. Compute ε_real = distance from the real value to the nearest allowed fraction.  
4. Run 3 × 10 million snap-null trials and estimate p_emp.

Outcome:

- Empirical p-values sit comfortably in the **0.27 to 0.98** range.  
- None of these parameters shows an anomalously tiny snap-null p-value.

Meaning:

> The snap-null test, when done with proper look-elsewhere accounting,  
> does **not** turn Standard Model numbers into automatic “miracles.”  
> Most individual scalars are statistically ordinary in this framework.

This validates the method itself and sharpens the contrast with:

- The **joint ledger triple**, and  
- The **MDL compression result**,  

which **do** show extreme behaviour.

---

So **Big Null Time / Null Code V2** is:

- A direct, code-level test of a specific **simple ledger hypothesis** against large, explicit null ensembles.  
- A demonstration that **ordinary SM scalars are not generically special** under the same tests.  
- And a clear statement that the **ledger + MDL combination** is what stands out as highly non-random in this setup.

