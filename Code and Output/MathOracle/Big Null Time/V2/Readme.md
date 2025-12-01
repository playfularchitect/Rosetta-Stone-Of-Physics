Colab Link - https://colab.research.google.com/drive/1Xmp8t0_03dy5vmqJLlpyPgfmeNmEU4qL?usp=sharing

# Big Null Time — Null Code V2

At a one-sentence level, what I’m doing here is:

> **I compare our universe to millions of fake universes to ask:  
> “How surprising is the pattern structure in a fair, look-elsewhere-aware way?”**

The two main pieces are:

1. **Snap-null tests** – “How often does a random number look as close to a ‘nice’ fraction as the real one does?”  
2. **MDL / ledger tests** – “How many bits does it take to encode a simple ledger that ties certain physical quantities together, and how rare is that level of compression under random data?”

Snap-null is the control group.  
The **MDL + ledger** result is the actual big deal.
---
## TL;DR — What this actually shows

Here’s the compressed story:

1. **Snap-null on lots of individual Standard Model numbers**  
   - CKM angles, α, α_s, sin²θ_W, mass ratios like M_W/v, M_Z/v, M_H/v, m_t/v, etc.  
   - When you do snap-null correctly (look-elsewhere included),  
     their behaviour is **completely normal**.  
   - p-values cluster in the 0.27–0.98 range.  
   - This is the control group, and it behaves like a control group should.

2. **ρ² and Koide Q under ledger targets**  
   - When you fix specific ledger fractions (instead of sliding to the best `p/q`),  
     ρ² and Koide Q become **highly non-generic**:
     - ρ² ledger p ~ 10⁻⁶  
     - Koide Q ledger p ~ 10⁻⁵  
   - sin²θ_W alone is less dramatic, but still participates in the ledger.

3. **Joint ledger triple (ρ², Koide Q, sin²θ_W)**  
   - Expected joint probability from the geometric model ≈ **2.3 × 10⁻¹²**.  
   - In 3 × 10 million triples, I see **0** matches, as expected for such a tiny probability.  
   - This says: under this model, that three-way accuracy is *extremely* unlikely by chance.

4. **MDL: the main highlight**  
   - Real ledger MDL ≈ **353 bits**.  
   - Null MDL ≈ **462 ± 5 bits** across widths and seeds.  
   - The real MDL sits **~20–21σ below** the null mean.  
   - In tens of millions of random universes, **none** compresses as well as the real ledger.  

   In plain language:

   > Given the rules I’m using, the ledger that fits our universe is  
   > **far more compressible than what random data produces.**

5. **Putting it all together**

   - **Most individual scalar values** in the Standard Model behave like normal random-ish numbers under fair snap-null tests.  
   - **The specific ledger structure** tying together ρ², Koide Q, and sin²θ_W is what stands out:
     - It hits three tiny ledger targets at once (joint triple test).  
     - It yields an MDL score that is way out in the tail of the null distribution.
---

## Core Goal 

Think of each physical quantity as **a point on a line between 0 and 1**:

- ρ² = (M_W / M_Z)²  
- Koide’s Q(e, μ, τ)  
- sin²θ_W  
- mixing angles, couplings, mass ratios like M_W / v, etc.

Separately, I build a big library of **“nice” fractions** `p/q`:

- denominators up to 1000  
- total bit size ≤ 20  
- restricted to some band like `[0.6, 0.9]` or `[0.15, 0.35]`

Now:

- A **random universe** = draw random numbers in the band(s), then see how well they “snap” to the nearest allowed fraction.
- The **real universe** = our actual measured values, treated the same way.

Analogy:

> Imagine a dartboard full of tiny bullseyes (all the allowed fractions).  
> Random universes = random darts.  
> I check how often random darts hit as close to a bullseye as the real darts do.

On top of that, there’s a **ledger**:

- A simple bit-encoded structure that ties together:
  - ρ²,
  - Koide Q,
  - sin²θ_W,
  - and some related ratios.
- The **MDL (Minimum Description Length)** is the number of bits needed to write down this scheme.

Second analogy:

> Think of the ledger as a **zip file for certain Standard Model numbers**.  
> The MDL score is like the size of the zip: fewer bits = better compression.

The main surprise of this project is:

> **The real ledger compresses *way* better than any random universe’s ledger,  
> across tens of millions of Monte Carlo samples.**

Everything else (snap-null, registry scan) is there to show that I’m not just fooling myself: most *individual* numbers behave like ordinary random-ish quantities under the same style of tests.

---

## What lives in this folder

- **`Null Code V2`**
  - The full code allowing anyone to reproduce the data.
  - Builds rational families (`p/q`, `q ≤ 1000`, bits ≤ 20).
  - Runs big Monte Carlo ensembles (10M universes or triples per run).
  - Prints detailed summaries: means, p-values, z-scores, etc.

- **`Null Output V2`**
  - The raw logs from running `Null Code V2`.
  - This is the “what actually happened” evidence for all the details below.

---

## Module guide (short, but keeps the key numbers)

### MODULE 2 — Big Snap-Null for ρ² = (M_W / M_Z)²

- Band: `[0.6, 0.9]`.
- Rational family: `q ≤ 1000`, bits ≤ 20.
- I find the closest allowed fraction to the real ρ² and call that distance **ε_real**.
- Then 3 runs × 10 million random universes:
  - For each random draw, find its nearest fraction,
  - Compare that distance to ε_real.

Result (rough):

- Empirical p-values ≈ **0.10** (about 10%).
- So: ρ² is mildly interesting under snap-null, but not astronomically rare.

Interpretation:

> If you let yourself look over all allowed fractions in that band,  
> one in ~10 random universes will land at least as close to *some* fraction as our ρ² does.

This matters later as a contrast with the **ledger** version of ρ².

---

### MODULE 3 — Big Snap-Null for Koide Q(e, μ, τ)

Same snap-null idea, but for **Koide’s Q**:

- Band: `[0.4, 0.9]`.
- Same fraction constraints.
- 3 runs × 10 million universes.

Result:

- Empirical p-values ≈ **0.95** (about 95%).

Interpretation:

> Once you correctly account for the “look-elsewhere” effect (scanning over many `p/q`),  
> Koide’s Q looks **very typical** under the snap-null test.  
> Its closeness to some fraction in that family is *not* rare.

Again: useful as a control. Koide Q is not an automatically tiny-p-value machine once the test is fair.

---

### MODULE 4 — Ledger Nulls for ρ², Koide Q, sin²θ_W

Now I switch rules.

Instead of letting the target slide to the **best** fraction anywhere in the band, I:

- fix one specific **ledger fraction** for each scalar:
  - ρ² has a fixed ledger target.
  - Koide Q has fixed target **2/3**.
  - sin²θ_W has its own fixed ledger target.

Then, for each quantity:

1. Draw 10 million random values in the band, 3 runs.
2. Compute distance to that **one fixed target**.
3. Count how often random draws are at least as close as the real value.

Key results (ballpark):

- ρ² ledger p ≈ **10⁻⁶**
- Koide Q ledger p ≈ **10⁻⁵**
- sin²θ_W ledger p ≈ **10⁻¹** (~0.08)

So:

- Under **ledger** rules, ρ² and Koide Q become **highly non-generic**.
- sin²θ_W is still reasonably typical by itself.

This already shows that fixing the target in advance (ledger style) completely changes the story for ρ² and Koide Q.

---

### MODULE 5 — Joint Ledger Triple (ρ², Koide Q, sin²θ_W together)

Here I glue the three ledger targets into one joint test:

- Target for ρ² (ledger fraction),
- Target 2/3 for Koide Q,
- Target for sin²θ_W.

Then:

- 3 runs × 10 million **triples** (ρ², Q, sin²W), sampled uniformly on their bands.
- I estimate the expected joint probability from the individual ledger p’s:
  - **Geometric p_joint ≈ 2.26 × 10⁻¹²** (about one in a trillion).
- In all runs:
  - **count(all three ≤ their ε_real)** = 0 out of 10M per run, which matches that tiny expectation.

Interpretation:

> Hitting one specific ledger target to that accuracy is rare.  
> Hitting *two* together is far rarer.  
> Hitting **three** (ρ², Q, sin²θ_W) with those tiny bands goes into “essentially never by chance” territory under this model.

This is one pillar of the “this ledger is weirdly good” story.

---

### MODULE 6 — MDL Width Robustness (the big result)

This is where I give the ledger a **compression score**.

- I define an **MDL (Minimum Description Length)** in bits for the ledger that ties certain Standard Model scalars together.
- The real universe’s ledger has:
  - **Real MDL ≈ 353 bits**.
- Then I generate big null ensembles for different width choices:
  - ±0.3 decades, ±1 decade, ±2 decades around the real value grid.
  - 3 runs × 10 million universes per width.

For each random universe:

1. Compute MDL bits for the corresponding ledger-like structure.
2. Build the null distribution: mean, standard deviation, min, max.
3. Count how often MDL(null) ≤ MDL(real).

What comes out:

- Null mean MDL ≈ **462 bits**.
- Null standard deviation ≈ **5.2 bits**.
- Real MDL (353 bits) is about **20–21σ below** the null mean.
- Across **tens of millions** of Monte Carlo universes:
  - `count(MDL <= real) = 0`.

Analogy:

> Think of the ledger like a zip file that compresses a particular set of physical numbers.  
> Random universes need around **462 bits** to describe this pattern.  
> Our universe manages to do it in **~353 bits**.  
> That’s not just “a bit better”; it’s **way off the tail** of the random distribution.

This is the main “make a big deal out of this” result:

- MDL says the real ledger is far more compressible than random,  
- and that conclusion is stable across multiple widths and seeds.

---

### MODULE 7 — Big Snap-Null for sin²θ_W (standalone check)

This is a dedicated snap-null test just for **sin²θ_W**:

- Band: `[0.15, 0.35]`.
- Fractions: `q ≤ 1000`, bits ≤ 20.
- 3 runs × 10 million universes.

Result:

- Empirical p ≈ **0.40** (roughly 40%).

Interpretation:

> As an isolated number trying to “snap” to some fraction, sin²θ_W is **completely ordinary**.  
> The interesting behaviour comes from how it fits into the **joint ledger** with ρ² and Koide Q,  
> and from the overall MDL score — not from sin²θ_W alone.

---

### MODULE 8 — Big Snap-Null Registry Scan (control group)

This module is basically my **sanity check / control group**.

I repeat snap-null tests for a small registry of important SM parameters:

- **CKM sector**
  - CKM_s12  
  - CKM_s13  
  - CKM_s23  
  - CKM_delta_over_pi  
- **Couplings**
  - α (fine-structure constant)  
  - α_s(M_Z)  
  - sin²θ_W (again, but just as another registry entry)  
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
3. Find ε_real = distance from real value to nearest allowed fraction.
4. Run 3 × 10 million snap-null trials and estimate p_emp.

Outcome:

- Empirical p-values land in a **very ordinary range**, roughly **0.27 to 0.98**.
- None of these numbers (including sin²θ_W in this context) looks crazy special under this fair snap-null test.

This is important:  

> **The method does *not* automatically spit out tiny p-values for “any” Standard Model number.**  
> Most individual scalars behave exactly like you’d expect random-ish quantities to behave once you account for look-elsewhere.

The contrast with the **ledger joint triple + MDL** is the point.

---

So this project, **Big Null Time / Null Code V2**, is basically me:

- Stress-testing a particular **simple ledger hypothesis** against large, explicit null ensembles,  
- Showing that **ordinary SM numbers are not automatically special** under this machinery,  
- And highlighting that the **ledger + MDL combination** is the thing that refuses to look like a typical random draw.
