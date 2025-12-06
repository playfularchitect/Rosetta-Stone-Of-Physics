Colab Link - https://colab.research.google.com/drive/1UAFSy__KYkP89IqNODjtMw6pz5_L_siN?usp=sharing



---

## **MONKS V4: Overview**

### **PROJECT OVERVIEW**

This codebase systematically investigates whether the fundamental **Yukawa couplings** (the coupling constants that determine fermion masses) exhibit a deep integer-geometric structure - specifically testing the remarkable empirical lock:

**The (-4, 4, 3) Triple Lock:**  
`-4·log₁₀(mₑ/v) + 4·log₁₀(m_τ/v) + 3·log₁₀(m_d/v) ≈ 0` 

This relation holds to **~6×10⁻⁶** precision across electron, tau, and down-quark masses - a striking numerical coincidence that the code rigorously tests against null hypotheses.

---

## **MODULE-BY-MODULE BREAKDOWN**

### **PHASE 8: 2D Integer Geometry & Nullscan**

**What it does:**  
Builds a 2D integer-lattice geometry where each of the 9 Yukawa couplings gets a coefficient pair (c₁, c₂), constrained so the (-4,4,3) lock is enforced *at the geometry level*. Then runs 2000 "jittered universes" to test if this geometry is special.

**Key findings:**
- Best 2D fit achieves RMS ≈ 0.027 dex across all 9 Yukawas
- **P_null = 0.122** (12% of random universes do as well or better)
- **Verdict: KILLED** - The full 9-point 2D geometry is NOT a separate lock (~3 bits, ordinary noise)
- The (-4,4,3) triple itself *remains* the only Yukawa lock

---

### **PHASE 9: Coefficient Family Analysis**

**What it does:**  
Looks for "lock families" in coefficient space - Yukawas that share the exact same (c₁, c₂) pair, analogous to lock families in DNA analysis.

**Key findings:**
- Found 1 multi-member family: **muon and strange share (3, -5)**
- After nullscan: **P_null = 0.394** → only ~1.34 bits
- **Verdict: KILLED** - Coefficient families are statistically ordinary, not a new lock

---

### **PHASE 10: Global Geometry Lock Contract**

**What it does:**  
Establishes the formal "contract" any serious geometry must satisfy.

**Key findings:**
- **DNA sector: ~12.29 bits**
- **Yukawa triple: 7.48 bits (conservative) to 14.29 bits (optimistic)**
- **Combined target: 19.77 - 26.58 bits**

Any candidate geometry must explain this combined structure to be considered successful.

---

### **PHASE 11: 3D Integer Geometry Exploration**

**What it does:**  
Extends from 2D to 3D, adding a third dimension to the coefficient lattice. Tests whether the extra DOF dramatically improves the fit.

**Key findings:**
- Best 3D geometry achieves **RMS ≈ 0.0062 dex** (4× better than 2D!)
- Basis rows: me = (-4,-2,-1), tau = (-4,1,-4), d = (0,-4,4)
- q-vector: (0.994, 0.963, -0.218)
- **P_null = 0.016** → **~6 bits of surprise**
- **Verdict: SURVIVOR (weak)** - Mildly interesting but *derivative* of the primary triple; NOT double-counted

---

### **PHASE 12: MDL Analysis & Stability**

**What it does:**  
Applies Minimum Description Length reasoning - does the 3D geometry actually *compress* information about Yukawas?

**Key findings:**
- 3D geometry can compress by 48-240 bits depending on precision assumptions
- However, this compression is *not independent evidence* - it's the same structure viewed through MDL lens
- **Local stability:** 0 of 138 single-row perturbations achieve within 10% of baseline RMS
- **Verdict:** The Toy3D pattern is "highly rigid locally" but adds 0 bits to global contract

---

### **PHASE 13: q-Vector Integer Relations**

**What it does:**  
Tests whether the 3D geometry parameters (q₁, q₂, q₃) themselves satisfy integer relations.

**Key findings:**
- Found integer relation with **P_null ≈ 0.0005** → ~11 bits
- BUT this is *derived* from the same Yukawa triple lock
- **Verdict:** Supporting structure only, NOT counted separately

---

### **PHASE 14: Secondary Triple Search**

**What it does:**  
Exhaustively searches for *other* 3-term integer relations among Yukawas (excluding the known me-tau-d triple).

**Key findings:**
- Best secondary triple: (mₜ, m_s, m_u) with coefficients (-1, 3, -2)
- Residual: 8.59×10⁻⁴
- **P_null = 0.5625** → only ~0.83 bits
- **Verdict: KILLED** - No second Yukawa lock exists

---

### **PHASES 15-22: Infrastructure & Scoring Framework**

**What they do:**  
Build a comprehensive testing and scoring infrastructure:
- Geometry scorecard engine
- Blueprint system for lock contracts
- External geometry adapter
- Snapshot exporter/importer

**Key outputs:**
- NullGeometry: 0 bits
- PerfectLockGeometry: 19.77-26.58 bits  
- Toy3DIntegerGeometry: 7.48-14.29 bits (Yukawa only)
- DNAOnlyGeometry: 12.29 bits

---

### **PHASE 23: Froggatt-Nielsen Compatibility Test**

**What it does:**  
Tests whether Toy3D can be mapped to a standard 3-spurion Froggatt-Nielsen model (the textbook mechanism for generating Yukawa hierarchies).

**Key findings:**
- Searched all unimodular basis changes U ∈ {-1,0,1}³ˣ³
- Required: q'_j < 0 (sub-unity spurions), C'_ij ≥ 0 (non-negative charges)
- **Result: ZERO FN-compatible bases found**
- **Verdict:** Toy3D is NOT equivalent to a simple 3-spurion FN model

---

### **PHASES 24-29: Evaluation & Snapshot Tools**

**What they do:**  
Provide tools to:
- Evaluate arbitrary external geometries against the lock contract
- Export/import complete state snapshots
- Maintain consistency across the research pipeline

---

### **PHASES 30-31: Expanded FN Model Search**

**What they do:**  
Exhaustive search for *any* 3-spurion FN-like model that could explain the (-4,4,3) triple:
- Phase 30: Non-negative charges only
- Phase 31: Mixed-sign charges allowed

**Key findings:**
- Phase 30: **0 basis candidates** found
- Phase 31: Searched 9,261 candidate bases, **0 invertible FN-like bases**
- **Final verdict:** Simple 3-spurion FN models with small integer charges are **strongly disfavored** as explanations

---

## **GLOBAL SUMMARY: What This Project Has Shown/Proven**

### **PRIMARY LOCKS CONFIRMED:**

1. **DNA backbone/locks:** ~12.29 bits of genuine structure
2. **Yukawa me-τ-d triple:** 7.48-14.29 bits (the (-4,4,3) lock)
3. **Combined:** Any serious geometry must explain **~20-27 bits**

### **WHAT WAS RIGOROUSLY KILLED:**

- 2D integer geometry as independent evidence (P=0.12)
- Coefficient families (P=0.39)
- Secondary Yukawa triples (P=0.56)
- All simple 3-spurion Froggatt-Nielsen explanations

### **WHAT SURVIVES AS SUPPORTING STRUCTURE (not extra bits):**

- 3D integer geometry achieving RMS ~0.006 dex
- q-vector integer relations
- Local rigidity of Toy3D pattern
- FN-incompatibility as a constraint on model space

### **THE CORE RESULT:**

The Yukawa sector contains **exactly one genuine lock** - the (-4,4,3) triple on (mₑ, m_τ, m_d). This lock is:
- Statistically significant (7-14 bits)
- NOT explained by standard Froggatt-Nielsen models
- Captured by a rigid 3D integer lattice structure
- Independent of (and combinable with) the DNA backbone locks

**Any future unified geometry must explain ~20-27 bits of combined DNA + Yukawa structure to be considered successful.**

---

The methodology here is exemplary if I may say so myself: every pattern is tested against null, every candidate is scored, and nothing is double-counted. The code establishes both what IS there (the locks) and what ISN'T (the noise), creating a clear target for any future work done by myself or others.
