# README: Big Null Time — Null Code V2

**Colab Link: [Run the Code Yourself](https://colab.research.google.com/drive/1Xmp8t0_03dy5vmqJLlpyPgfmeNmEU4qL?usp=sharing)**

This folder is part of my bigger project, **The Rosetta Stone of Physics**.

My core claim is simple:

> **The fundamental constants of nature are not random.  
> They are governed by a small, tightly-compressed rational ledger.**

This project is where I brutally test that claim against huge null ensembles.

I do two things:

1. Show that **ordinary Standard Model numbers look ordinary** under fair snap-null tests.  
2. Show that the **specific ledger** linking ρ², Koide Q, and sin²θ_W is **far too compressed** to be explained by the same random machinery.

The end result is clear:  
the “random parameters” story does *not* survive this level of compression testing.

---

## TL;DR — What the math actually proves here

I pit our universe against **tens of millions** of fake universes and ask:

- How well do they hit simple fractions?
- How short is the code (MDL) needed to describe them?

Here is what the data says:

- **Control scalars** (CKM angles, α, α_s, M_W/v, M_Z/v, M_H/v, m_t/v, sin²θ_W in isolation, etc.)  
  behave exactly like random draws should. Their snap-null p-values sit in a boring range: **0.27–0.98**.

- **Ledger scalars** (ρ², Koide Q, sin²θ_W) behave *nothing* like that once I lock them to specific targets:
  - ρ² ledger p ≈ **10⁻⁶**  
  - Koide Q ledger p ≈ **10⁻⁵**  
  - sin²θ_W ledger p ≈ **10⁻¹**

- The **three-way ledger lock** (ρ², Q, sin²θ_W together) has a joint probability  
  of about **2.26 × 10⁻¹²** (≈ 1 in 440 billion) under the null.  
  In **30 million** simulated triples, **0** random triples hit all three targets as well as the real one.

- The **MDL of the real ledger** is about **353 bits**, while the null MDL is about **462 ± 5 bits**.  
  That puts the real ledger **~20–21σ** below the null mean.  
  Across all the Monte Carlo runs, **no** random universe reaches MDL ≤ 353.

That is not a subtle “maybe” signal.  
That is a **crushing compression gap**.

**Conclusion of this folder:**

> The simplest exact description (in bits) of the ledger linking ρ², Koide Q, and sin²θ_W  
> is *far* shorter than anything produced by large, explicit null ensembles.  
> The “random scribble” hypothesis for these constants fails this test.

---

## How I test “simple vs random” (5th-grade level, no baby talk)

Think of the universe as **a set of special numbers** between 0 and 1:

- ρ² = (M_W / M_Z)²  
- Koide Q for (e, μ, τ)  
- sin²θ_W  
- CKM mixing s_12, s_13, s_23  
- α, α_s(M_Z)  
- ratios like M_W / v, M_Z / v, M_H / v, m_t / v  

Now imagine two tools:

### Tool 1 — Snap-Null: The Dartboard

Analogy:

> I draw a dartboard.  
> Tiny bullseyes = “nice fractions” p/q with q ≤ 1000 and bit-size ≤ 20,  
> limited to sensible bands (like [0.6, 0.9], [0.15, 0.35], or ±1 decade around the real value).

Then I throw darts:

- **Random universes** = random darts in the band.  
- **Our universe** = the one real dart that already landed.

I use two modes:

1. **Snap-null (sliding target)**  
   Each random dart is allowed to claim *whichever bullseye is closest*.  
   This checks: “Is the method fair? Do ordinary numbers look ordinary?”  
   → That’s where most Standard Model scalars land: **they look random and unremarkable.**

2. **Ledger null (fixed target)**  
   Here I pick **one bullseye in advance**:
   - e.g. Koide Q → target 2/3  
   - ρ² → a specific ledger fraction  
   - sin²θ_W → a specific ledger fraction  

   Then I check: *How often does a random dart land as close to that exact bullseye as the real dart does?*  
   → That’s where ρ² and Q explode into **10⁻⁶** and **10⁻⁵** p-values.

### Tool 2 — MDL: The Zip File

Analogy:

> Take a bunch of numbers and zip them into a file.  
> If they are random, the zip file is big.  
> If they follow a clean pattern, the zip file is small.

MDL = **Minimal Description Length**, in bits. It’s literally:

> “How many bits does it take to write down the ledger that generates these numbers?”

I build a concrete ledger that ties together:

- ρ²  
- Koide Q  
- sin²θ_W  
- and some related ratios

Then I measure:

- **MDL(real universe)**  
- **MDL(null universes)** for tens of millions of random samples following the same setup.

This is where the 353 vs 462 bits gap lives.

---

## What’s actually inside this folder

### `Null Code V2`

This is the full codebase for all of the tests in this README. It:

- Builds rational families: fractions p/q with:
  - `q ≤ 1000`
  - bit-size ≤ 20
  - bands like `[0.6, 0.9]`, `[0.15, 0.35]`, or ±1 decade around x_real  
- Runs big Monte Carlo ensembles:
  - 10 million universes or triples per run  
  - multiple independent seeds  
  - multiple width settings (for MDL robustness)  
- Outputs detailed summaries:
  - snap-null p-values  
  - ledger-null p-values  
  - MDL means, std devs, min/max  
  - Z-scores and raw counts

### `Null Output V2`

This is the full log dump:

- All module banners  
- All config JSON and SHA256 hashes  
- All summary stats and empirical p-values  
- All MDL distributions and Z-scores

Every number I quote here is taken from these logs.

You don’t have to trust my interpretation.  
You can **scroll the logs yourself** or **re-run the Colab** and watch the numbers print.

---

## Module-by-module signal

Here’s the short version of what each module shows.

### MODULE 2 — Big Snap-Null for ρ² = (M_W / M_Z)²

- Band: `[0.6, 0.9]`  
- Fractions: `q ≤ 1000`, bits ≤ 20  
- Snap-null (sliding target) p ≈ **0.10**

**Message:**  
When you’re allowed to pick the best fraction anywhere in the band, ρ² is only mildly interesting.  
Roughly **1 in 10** random universes does at least this well.

### MODULE 3 — Big Snap-Null for Koide Q(e, μ, τ)

- Band: `[0.4, 0.9]`  
- Same fraction family  
- Snap-null p ≈ **0.95**

**Message:**  
With the look-elsewhere effect handled correctly, Koide Q looks absolutely typical under snap-null.  
On its own, with a sliding target, it is **not** a miracle.

### MODULE 4 — Ledger Nulls for ρ², Koide Q, sin²θ_W

Here I lock in **fixed rational targets**:

- ρ² → ledger fraction  
- Q → **2/3**  
- sin²θ_W → ledger fraction  

For each:

- 3 runs × 10 million random draws  
- Compare distance to the fixed target vs real distance

**Results:**

- ρ² ledger p ≈ **10⁻⁶**  
- Koide Q ledger p ≈ **10⁻⁵**  
- sin²θ_W ledger p ≈ **10⁻¹**

**Message:**  
The moment the target is *fixed in advance*, ρ² and Q jump from “normal” to **extreme outliers**.  
Same machinery. Different rule. Completely different story.

### MODULE 5 — Joint Ledger Triple (ρ², Koide Q, sin²θ_W together)

Now I demand all three hits at once.

- From the individual ledger nulls, the joint probability is:
  - **p_triple ≈ 2.26 × 10⁻¹²** (≈ 1 in 440 billion)
- I test this directly with:
  - 3 × 10 million random triples

**Result:**

- **0** random triples match or beat the real triple on all three distances simultaneously.

**Message:**  
The ledger triple is sitting in an astronomically small region of the null space.  
The null simulation never gets there.

### MODULE 6 — MDL Width Robustness (main compression result)

This is the center of the project.

I compute MDL bits for a concrete ledger tying together:

- ρ², Q, sin²θ_W, and related ratios

For:

- **The real universe** → MDL ≈ **353 bits**  
- **Null universes** → MDL ≈ **462 ± 5 bits** across widths and seeds

**Key points:**

- Real ledger is about **109 bits shorter** than the average null ledger.  
- That corresponds to about **20–21σ** below the null mean.  
- Across tens of millions of null universes:
  - `count(MDL <= real) = 0`.

**Message:**  
The real ledger is *way* too compressed to be just “another random draw.”  
This is the main statistical hammer in this folder.

### MODULE 7 — Big Snap-Null for sin²θ_W (standalone)

Dedicated snap-null test for **sin²θ_W**:

- Band: `[0.15, 0.35]`  
- Fractions: `q ≤ 1000`, bits ≤ 20  
- Empirical p ≈ **0.40**

**Message:**  
On its own, in snap-null mode, sin²θ_W is completely ordinary.  
Its special role is in the **ledger triple** and the **MDL compression**, not as a solo prima donna.

### MODULE 8 — Big Snap-Null Registry Scan (broad control group)

I repeat snap-null tests across a mini-registry:

- CKM_s12, CKM_s13, CKM_s23, CKM_δ/π  
- α, α_s(M_Z), sin²θ_W  
- M_W/v, M_Z/v, M_H/v, m_t/v  

All with:

- ±1 decade bands (clipped to [1e−10, 1])  
- `q ≤ 1000`, bits ≤ 20  
- 3 × 10 million universes each

**Outcome:**

- p-values land in **0.27–0.98**.

**Message:**  
The method behaves correctly on a broad set of important constants.  
No automatic miracles, no “every number is special.”  
That’s why the ledger results stand out so sharply.

---

## How this fits the bigger Rosetta Stone picture

In the **Foundational Axioms** of my framework, I start from:

- Reality is made of **relational differences** and **ratios**, not standalone “things.”  
- The deep structure is **paradox + vibration**, not floating-point chaos.  
- If that’s true, you should see:
  - **Simple rational locks** at the core of the constants, and  
  - **Extreme compressibility** of the code that describes them.

This folder is one of the hard statistical proofs of that thesis:

- **Controls:** show that when a number is “just a number,” this machinery says so.  
- **Ledger + MDL:** show that the specific structure tying ρ², Q, sin²θ_W, and related ratios together  
  is *far* too simple (in bits) to be a typical null outcome.

The code is here.  
The logs are here.  
The compression gap is here.  
The math did exactly what my theory predicted it would do.

**Don’t believe me. Run it.**

