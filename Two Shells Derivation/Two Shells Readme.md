# Two-Shell α — Disk-Only Verification Bundle

This folder contains the *artifact-complete* two-shell derivation documents, code, outputs and a Colab verifier that recomputes  
**\(c_{\rm Pauli}\)** and **\(\alpha^{-1}\)** **from disk only** (no heavy integration).

Colab Verifier: https://colab.research.google.com/drive/1y-r8FuK-SknMRiHMJoySK6RIZR2OB-SL?usp=sharing

OSF: 

https://osf.io/zuy8c/

https://osf.io/pwur5/


**Key numbers (recomputed):**
- \(c_{\rm Pauli} = 0.0015308705589210\)
- \(\alpha^{-1} = 137.000011174238\)
- Continuum interval: \([136.9995288977, 137.0004824073]\)
- Lattice interval:    \([136.9988376016, 137.0011902923]\)

**Reproduce (Colab):**
- Open the provided Colab verifier cell, upload the 8–9 files provided to you in this folder, run once.
- Outputs: `Wesley_VERIFY_README.md`, `verify_summary.json`, `SHA256SUMS.txt`, zipped bundle.

**Scope:** math first verification of the released artifacts; no re integration; nor parameter fitting.

# Two-Shell α — Methods-First Release v0.1.0

**Scope.** This release ships a complete, artifact-first, reproducible computation for the two-shell fine-structure constant program:
- Integer shell geometry → angle classes (five CSVs)
- Locked denominator `∑ NB·cos²θ = 6210` (`denominator.json`)
- Per-class Pauli‐kernel continuum integrals + contributions (`pauli_integrals.json`)
- Point estimate / summary (`alpha_prediction.txt`)
- Tile-wise certified intervals (continuum and lattice) (`pauli_tilewise_bounds.json`)
- Disk-only verifier that recomputes `c_Pauli` and `α^{-1}` from the above files (no heavy integration)

**Recomputed from artifacts (disk-only).**
- `c_Pauli = 0.0015308705589210`
- `α^{-1} = 137.000011174238`
- Continuum interval (from tiles): `α^{-1} ∈ [136.9995288977, 137.0004824073]`
- Lattice interval via `[1, π²/4]`: `α^{-1} ∈ [136.9988376016, 137.0011902923]`

**Repro, zero heavy compute (Colab one-cell).**
Use `Wesley_VERIFY_README.md` / `verify_summary.json` / `SHA256SUMS.txt` produced by the included **Colab Verifier** cell. It:
1) checks the angle row identities (each table: `∑ total = 137`, `∑ NB·cosθ = 1`);
2) re-computes `DENOM`, `c_Pauli`, `α^{-1}`;
3) transforms tilewise `c` intervals → `α^{-1}` intervals;
4) emits SHA-256 manifest.

**What this release does _not_ claim (yet).**
- We are **not** claiming empirical agreement with CODATA; the current minimal two-shell mapping yields `α^{-1} ≈ 137.000011…`.
- The adaptive global certifier (tightening continuum bounds until `c_Pauli`’s interval excludes 0) is **out-of-scope** for this tag and may require longer runs than typical Colab sessions.
- Physical extensions (e.g., additional shells, screening corrections) are future work and intentionally omitted here.

**Positioning.** This is a **methods/rigor** drop:
- all inputs and outputs are shipped;
- numbers are reproducible from disk;
- certified tilewise bounds are included (and are conservative by design).

**Included files.**
- `angles_49_007.csv`, `angles_49_236.csv`, `angles_50_017.csv`, `angles_50_055.csv`, `angles_50_345.csv`
- `denominator.json`, `pauli_integrals.json`, `pauli_tilewise_bounds.json`, `alpha_prediction.txt`
- `Wesley_VERIFY_README.md`, `verify_summary.json`, `SHA256SUMS.txt`
-  **GOAT I–III**, **Two-Shell α Derivation Master Code Cell**, **Two-Shell α Derivation Master Output Cell**

**How to cite.**
> Wesley, E. (2025). Two-Shell α: Methods-First Release v0.1.0 (artifact-complete).  
> GitHub repository: `https://github.com/playfularchitect/Rosetta-Stone-Of-Physics.git`.

**License.**
- Code: MIT   
- Text & data: CC-BY-4.0 

**FAQ (anticipated).**
- *“Your `α^{-1}` is ~137.00001, not 137.036…”* — Correct. This minimal two shell mapping is a clean baseline. Agreement work (more shells/physics) is future-tagged; this tag demonstrates reproducible transduction and certified numerics.
- *“Intervals cross 0 at the `c`-level.”* — Yes, because tilewise continuum bounds are conservative. The **point estimate** and the **disk-only recomputation** remain fixed; the conservative certificate is shipped as is.
