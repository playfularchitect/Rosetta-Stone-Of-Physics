# README: Big Null Time — Null Code V2

**Colab Link: [Run the Code Yourself](https://colab.research.google.com/drive/1Xmp8t0_03dy5vmqJLlpyPgfmeNmEU4qL?usp=sharing)**

This folder is part of my larger project, **The Rosetta Stone of Physics**.

My goal here is very simple and very direct:

> **I smash our universe against huge null ensembles and measure how “compressible” its structure is.  
> If reality is built from simple rational locks, that should show up as extreme compression.**

The result is exactly that:  
the ledger that fits our universe is **far more compressed** than anything the null produces.

---

## TL;DR: What the data says

I compare:

- **Our universe’s ledger** (ρ², Koide Q, sin²θ_W, plus related ratios), vs  
- **Tens of millions of random “null universes”** built under explicit, look-elsewhere-aware rules.

What the numbers say:

- **Most individual Standard Model scalars** behave like ordinary random draws in this framework.  
- The **specific ledger structure** that ties together ρ², Koide Q, and sin²θ_W does **not**.
- The MDL (code length) of the real ledger is about **20–21σ shorter** than the null mean.  
- In all the null samples, **no random universe** reaches that level of compression.

In other words:  

> The pattern I’m encoding is **too simple** to be explained as a typical random fluctuation under the null.  
> The structure is real. The compression is real. The math shows it.

---

## How I test “simple vs random”

I use two main tools:

### 1. Snap-null tests — the dartboard

Picture a dartboard:

- Tiny bullseyes = all the “nice” fractions `p/q` with:
  - `q ≤ 1000`
  - bit-size ≤ 20
  - restricted to some band (like `[0.6, 0.9]` or `[0.15, 0.35]`)

Each physical scalar (ρ², Koide Q, sin²θ_W, CKM angles, α, α_s, M_W/v, etc.) is a **point between 0 and 1**.

Then I define two modes:

- **Snap-null (sliding target):**  
  For each random draw, I let it “snap” to the *best* nearby fraction in the family.  
  This is how I check that the method behaves correctly on ordinary numbers.

- **Ledger null (fixed target):**  
  For the scalars my theory singles out, I **fix one rational target ahead of time** (like Q = 2/3)  
  and ask: *how often does a random draw get as close as the real value to this exact target?*

### 2. MDL tests — the zip file

MDL = **Minimal Description Length** in bits.

Think of the ledger as a **zip file for a specific bundle of constants**:

- ρ² = (M_W / M_Z)²  
- Koide Q(e, μ, τ)  
- sin²θ_W  
- plus some related ratios

Random universes give you some messy pattern → big zip file.  
A highly structured ledger gives you a neat pattern → small zip file.

I compute:

- **MDL(real)** = code length of the actual ledger for our universe  
- **MDL(null)** = code lengths for many fake universes built under the same rules

The comparison is the core of this project.

---

## Proof 1: The controls behave like randomness

I start with a registry of important Standard Model scalars:

- CKM: s_12, s_13, s_23, δ/π  
- Couplings: α, α_s(M_Z), sin²θ_W  
- Ratios: M_W/v, M_Z/v, M_H/v, m_t/v  

For each one:

- I define a band (usually ±1 decade around the real value, clipped to [1e−10, 1]).  
- I build the allowed fraction family (`q ≤ 1000`, bits ≤ 20).  
- I run large snap-null ensembles (3 × 10 million draws).

What happens:

- Their empirical p-values sit safely in the **0.27–0.98** range.

This is exactly what a healthy control group should look like.

**Conclusion of this step:**

> The snap-null machinery is not a miracle generator.  
> It treats most individual Standard Model numbers as **ordinary random-ish scalars**.

That’s the baseline.

---

## Proof 2: The special lock — ρ², Koide Q, sin²θ_W

Now I zoom in on the triple my ledger cares about:

- ρ²  
- Koide Q(e, μ, τ)  
- sin²θ_W  

### Step 2.1: Individual ledger nulls

Here I switch to **fixed targets**:

- ρ² → fixed ledger rational  
- Q → **2/3**  
- sin²θ_W → fixed ledger rational

For each scalar:

- 3 runs × 10 million random draws in the relevant band  
- Measure distance to the fixed target  
- Compare to the real value’s distance

The result:

- ρ² ledger p ≈ **10⁻⁶**  
- Koide Q ledger p ≈ **10⁻⁵**  
- sin²θ_W ledger p ≈ **10⁻¹**

Under sliding-target snap-null, these didn’t look extreme.  
Under fixed-target ledger rules, **ρ² and Q become very strong outliers.**

### Step 2.2: The joint triple

Then I require all three conditions at once:

- ρ² close to its ledger target  
- Q close to 2/3  
- sin²θ_W close to its ledger target

From the individual ledger p’s, the geometric model gives:

- **p_triple ≈ 2.26 × 10⁻¹²** → about **1 in 440 billion**

I test this directly:

- 3 × 10 million random triples
- Count how many match or beat the real triple in *all three* distances

Result:

- **0** random triples match the real triple’s combined accuracy.

**Conclusion of this step:**

> The three scalars line up on their ledger targets in a way that random triples, under the same rules, simply don’t reach at this sampling scale.  
> The triple lock is extremely tight.

---

## Proof 3: The 21-sigma MDL gap

Finally, I measure the **code length** of the full ledger.

For each universe (real or null):

- I compute an MDL cost in bits for encoding the ledger that ties together ρ², Q, sin²θ_W, and related ratios.

Here’s the key summary:

| Ledger Type           | MDL (bits)      | Position vs null |
|-----------------------|-----------------|------------------|
| Random null universes | ≈ **462 ± 5**   | baseline (0 σ)   |
| Our real universe     | ≈ **353**       | **−20 to −21 σ** |

Across tens of millions of null universes, with different:

- width choices (±0.3, ±1.0, ±2.0 decades), and  
- independent seeds,

I get:

- **count(MDL ≤ 353) = 0**

**Conclusion of this step:**

> The ledger that actually fits our universe is **about 109 bits shorter** than the average null ledger.  
> That’s not a small tweak; that’s a stupid huge compression jump sitting ~20–21 standard deviations into the null tail.

The simplest exact-description of these linked constants in our universe is dramatically shorter than what random structure produces under the same machinery.

---

## What’s in this folder

Everything is here for you (or anyone else) to check:

- **`Null Code V2`**
  - Full source code for:
    - snap-null tests (sliding target),
    - ledger null tests (fixed target),
    - triple coincidence simulation,
    - MDL computation and width robustness.
  - Builds the rational families (`p/q`, `q ≤ 1000`, bits ≤ 20).
  - Runs large Monte Carlo ensembles (10M universes or triples per run).

- **`Null Output V2`**
  - Raw logs from `Null Code V2`.
  - Every number I quote above comes from these logs:
    - p-values, counts, MDL means, standard deviations, σ-scores, etc.

You do not need to trust my summary.  
You can open the logs, re-run the Colab, and watch the histograms and counts form.

---

## Big-picture meaning in my project

In the **Rosetta Stone of Physics**, I start from simple axioms:

- Reality is fundamentally **relational** and **ratio-based**, not a bag of floating-point accidents.  
- The deepest description of physical law should look like a system of **simple rational locks(It does)**.  
- If that’s true, then:
  - the code that describes our constants should be **highly compressible**, and  
  - simple rational ledgers should beat random alternatives by a wide statistical margin.

This folder is one of the places where I turn that into hard numbers.

The pattern is consistent:

- **Controls:** ordinary scalars look ordinary.  
- **Ledger:** the Ratio OS I’m encoding is **far too compressed** to be explained as typical null behavior.

The code is here.  
The logs are here.  
The compression gap is here.  
The math is done.

**Don’t take my word for it. Run the code.**



- The code is here.  
- The logs are here.  
- The compression gap is here.
- The math is clear

