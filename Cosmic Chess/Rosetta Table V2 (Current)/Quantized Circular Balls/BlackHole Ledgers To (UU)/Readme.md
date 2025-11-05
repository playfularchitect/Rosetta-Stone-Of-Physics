# README (plain-language) November 5th, 2025

## TL;DR

* Identities hold (Smarr, first law, Christodoulou–Ruffini).
* Reversible extraction reaches the global bound; irreversibility creates a shortfall.
* Near-extremal scaling is clean and consistent.
* Response, duality, and geometry all line up.
* A project-wide meta-audit says the numbers are globally tight.

**If you only read one number:** `Unified Ledger Score ≈ 89/100` (higher is better).

**If you only read one sentence:** The full thermodynamic picture checks out—numerically, broadly, and reproducibly.




### What’s new/interesting here (in one breath)

> * We don’t just “compute numbers.” We **prove numerically** that the textbook thermodynamics of Kerr–Newman black holes holds across a broad state space, including near extremality, with receipts.
> * We show reversible extraction **saturates the theoretical bound** and that irreversibility costs you work, exactly as thermodynamics predicts.
> * We bring in the **information-geometry view** (Ruppeiner/Weinhold), compute curvatures and geodesics stably, and tie it all back to the usual identities.
> * We give you a project-wide **quality score** and per-module breakdown, so skepticism can be quantitative.

---

## What this project is

This is a numerical audit of Kerr–Newman black-hole thermodynamics (spinning + charged black holes) done in geometric units. We built small, focused “modules” (M10–M30) that each test one idea: identities like Smarr and Christodoulou, reversible/irreversible extraction, response coefficients, near-extremal scaling, information-geometry (Ruppeiner/Weinhold), and ensemble stability. Every module writes receipts (CSVs/JSON) so results are reproducible and easy to meta-audit.

> Think of it as a unit-test suite for black-hole thermodynamics, with a final, project-wide quality score.

---

## What we checked and confirmed

### 1) Core identities & energy accounting

* **Christodoulou–Ruffini identity** holds numerically across Kerr, Reissner–Nordström, and mixed Kerr–Newman states.
    * ↳ The irreducible mass M_ir computed from the horizon area matches energy partitions (rest/rotational/electromagnetic).
* **Smarr relation** M=2TS+2ΩJ+ΦQ closes to machine precision in all sub-extremal tests.
* **First law** dM = TdS + ΩdJ + ΦdQ: path integrals of the RHS exactly match the actual ΔM for small moves in state space (path independence verified with two different routes).

### 2) Reversible vs irreversible extraction

* With ΔA ≈ 0 (reversible) constraints:
    * **Spin-only** (fix charge) extracts up to the rotational share E_rot.
    * **Charge-only** (fix angular momentum) extracts up to the electromagnetic share E_EM.
    * Doing the two legs in either order reaches the global bound E_rot + E_EM = M - M_ir.
* With ΔA > 0 (irreversible) steps, extracted work falls short, depends on path order, and you cannot close a loop back to the starting state without violating the area theorem.

### 3) Near-extremal scaling

* Along a_*^2 + q_*^2 = 1 - ε with ε → 0+:
    * Hawking temperature scales like T_H ∼ ε^(1/2).
    * Horizon angular velocity Ω_H and electric potential Φ_H approach their extremal limits with clean power laws.
    * Microcanonical susceptibilities and related response functions also follow simple power-law tails.
    * Smarr remains closed across the entire tail.

### 4) Response coefficients, Maxwell symmetries & dual Hessians

* We compute the Jacobian/Hessian structure with complex-step derivatives (very stable numerically).
* **Maxwell symmetry** (equality of mixed partials) holds to very tight tolerances.
* **Duality** between microcanonical and isopotential ensembles works: the Gibbs-side Hessian is the negative inverse of the microcanonical one (H_G = -H_M⁻¹), and isopotential capacities/susceptibilities read off consistently.

### 5) Ruppeiner/Weinhold geometry

* We build the Weinhold metric from the symmetric part of H_M and the Ruppeiner metric by dividing by T.
* We evaluate scalar curvature R_R at representative states and along near-extremal paths without breaking Smarr/Maxwell closures.
* We integrate thermodynamic geodesics in the Ruppeiner metric and confirm the geometric machinery behaves consistently (end-point-accurate shooting, sensible lengths).

### 6) Ensemble stability maps

* Sweeps over (a_*, q_*) chart the signs and singular loci of capacities/susceptibilities in both ensembles.
* Phase-portrait summaries show where each ensemble is locally stable and where sign flips occur; Smarr/Maxwell remain tightly closed across the grid.

### 7) Project-wide meta-audit

* A unified meta-audit (M28) ingests every CSV/JSON/TXT and extracts every residual/“error-like” number (Smarr, Maxwell, misfits, gaps, curvatures, etc.).
* It reports a **Unified Ledger Score** on a 0–100 scale; our current run is **~89**, which indicates excellent global consistency, with the heaviest numerical tasks (large grids, curvature) being the only modest contributors to residual noise.
* The packager (M30) builds a release manifest + summary, so anyone can reproduce/inspect.

---

## How to run (Colab-friendly)

1.  Open a fresh Colab notebook.
2.  Copy/paste the modules you want (M10–M30). You can run them independently; each writes receipts under `/content/`.
3.  After running a set of modules, run **M28 (Meta-Audit)** to get the overall score and per-module integrity.
4.  Optionally run **M30** to package a release zip + manifest + a headliner ASCII table.

**Tip:** All modules use geometric units and `mpmath` with complex-step derivatives. No external data is required.

---

## What to look at (outputs)

* **Per-module receipts (CSV):** raw values, residuals, fits.
* **`M28_unified_records.csv`:** the cross-module ledger of everything “residual-like.”
* **`M28_module_integrity.csv`:** per-module scores & medians.
* **`M30_release_pack.zip`:** bundle of key artifacts + manifest + summary.

---

## Known numerical challenges (and how we handled them)

* **Near extremality:** we keep a tiny margin and use high precision to avoid complex/super-extremal drift.
* **Hessian inversions:** we use ridge-stable inverses and complex-step Jacobians (no subtraction cancellation).
* **Curvature (higher derivatives):** we combine complex-step + conservative finite-differences and sanity-check Smarr/Maxwell alongside.

---

## TL;DR

* Identities hold (Smarr, first law, Christodoulou–Ruffini).
* Reversible extraction reaches the global bound; irreversibility creates a shortfall.
* Near-extremal scaling is clean and consistent.
* Response, duality, and geometry all line up.
* A project-wide meta-audit says the numbers are globally tight.

**If you only read one number:** `Unified Ledger Score ≈ 89/100` (higher is better).

**If you only read one sentence:** The full thermodynamic picture checks out—numerically, broadly, and reproducibly.
