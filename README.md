# Rosetta Stone of Physics

This repository contains the complete body of work for a framework showing that the fundamental constants of nature are governed by a small set of simple, exact rational numbers. It includes the foundational papers, the complete mathematical ledgers, all verification code, and the full outputs of every analysis.

This project is organized to be fully transparent, reproducible, and falsifiable.

---

## ▶ How to Verify This Framework

There are two paths for verification, depending on your technical level.

### 1. The 10-Minute Verification (No Code Required)

For a direct, non-computational check of the framework's core claims, please see the **Verification Deck**:

* **[`Verification_deck_v2.pdf`](./Verification_deck_v2.pdf)**

This document walks you through the arithmetic to verify the interlocking predictions for the electroweak sector, the Higgs boson, the Koide relation, and more, using only the project's core rational numbers and a calculator.

### 2. The Live Code Verification (Google Colab)

To reproduce the full computational results of the project, you can run the primary scripts directly in your browser using Google Colab. No setup is required.

* **[► Run the Fine-Structure Constant Derivation](https://colab.research.google.com/drive/1ttEMfSITXa3DaZHnH5SMFIRUX12BVaxC?usp=sharing)**
    * This notebook contains the code from the flagship paper, deriving $\alpha^{-1} \approx 137.036$ from first principles.
* **[► Run the "Mindmelt" Physics Engine](https://colab.research.google.com/drive/1MZtXHujuPUlDSmSMu2gR54b-s9N44UoZ?usp=sharing)**
    * This notebook, `MathOracle4`, takes the 19 core registry fractions and derives hundreds of observables across the Standard Model.

---

## The Three Pillars of Evidence

The evidence for this framework is built on three distinct, interlocking pillars, each supported by the documents and code in this repository.

### Pillar 1: First-Principles Derivation of the Fine-Structure Constant

The flagship paper demonstrates a derivation of the fine-structure constant ($\alpha$) from a discrete geometric structure with no free parameters.

* **Primary Paper:** ** [(Keystone)V9 MasterPaper.pdf](./(Keystone)V9 MasterPaper.pdf)**
* **Supporting Materials:** See the `Two Shells Derivation` folder.

###  Pillar 2: An Interlocking Model of Particle Physics

A small set of just four rational numbers is shown to lock the entire CKM and PMNS flavor geometry. This single lock simultaneously predicts dozens of independent observables in tight agreement with experimental data.

* **Supporting Papers:** See the PDFs in the `Math Checks` folder (e.g., `CKMandPMS.pdf`, `Rare_Decay_Ledger.pdf`, `MuonG2 (1).pdf`).
* **Primary Code:** See the `Code and Output/MathOracle` folder.

###  Pillar 3: A Rational Framework for Cosmology

The same principle is applied to the cosmic scale. A few simple fractions for the universe's energy budget and expansion rate are shown to derive the complete set of standard background cosmological parameters.

* **Supporting Paper:** **[`Math Checks/Cosmology Ledger.pdf`](./MathChecks/Cosmology Ledger.pdf)**

---

## Navigating the Repository

This project is organized into several key directories:

* **`/Theory`**: Contains the foundational axioms and philosophical framework of the project.
* **`/Math Checks`**: Contains the collection of papers applying the rational framework to specific domains of physics (CKM, Cosmology, Black Holes, etc.).
* **`/Code and Output`**: The heart of the project's computational evidence.
    * **`/MathOracle`**: Contains the `MathOracle4.ipynb` script (the "Mindmelt" engine) and its full output, demonstrating the predictive power of the 19-fraction registry.
    * **`/QRLF (Quantum Rational Lock Finder)`**: Contains the "megacell (V13.13.9 Code/Output)" statistical suite, a powerful set of tools for testing the "rational lock" hypothesis with rigorous statistical methods like Bayesian model comparison and Monte Carlo simulations.
* **`/Rosetta Stone V2 Ledger Fractions`**: Contains the various versions of the "Master Ledger" of rational numbers that form the basis of the framework.
* **`/Burden Of Proof`**: Contains documents laying out the logical case for the framework based on principles like Occam's Razor.
* **`/Physics To Fractions`**: Contains the guides and worked examples for applying the rational-first methodology. This is where you can learn how to translate standard physics problems into the rational framework.

/Math Checks/Baseline control: Contains the code and analysis that serves as the "control group" for the entire project. This is where the core methodology of finding rational approximations for irrational numbers is developed and tested. Key components include:

The "Megablockasaurus" Script: This is a self-contained statistical suite that demonstrates and validates the process of finding rational locks. It includes modules for analyzing continued fractions, using the Stern-Brocot tree to find optimal fractions within a given interval, and formal proofs of minimality for those fractions.

The "Irrational Registry Engine": A powerful tool to bulk-generate high-precision values for a wide range of mathematical constants (like π, e,  
2

​
 , etc.) and analyze their rational approximations. This serves as a baseline to show how the system behaves with known transcendental and irrational numbers.

The "Composite Minimality Proofs": This is the most rigorous part of the control. It provides exhaustive proofs that a given rational number is the simplest possible one (in terms of bit complexity) that can be found within a given experimental uncertainty band. This is used to certify that the rational locks found are not arbitrary but are mathematically optimal.
---

## How It Works: The "Physics to Fractions" Methodology

This framework is more than just a list of results; it's a complete, transportable methodology for re-casting physical laws in terms of simple, rational numbers. The `Physics To Fractions` folder contains the core documents that explain and demonstrate this process.

* **The Fractional Action Principle:** The core of the methodology is laid out in **`FractionalActional.pdf`**. This paper formalizes a version of the Standard Model where all parameters are exact rationals.

* **"DLC Packs" & Worked Examples:** For a practical guide on how to apply this method, see the "DLC Packs." These are short, focused modules that provide the recipes and rational locks for specific areas of physics. Key examples include:
    * **`DLCpacks1thru10.pdf`**: Covers core concepts in QED, QCD, Thermo, and more.
    * **`SchrodingersFractions.pdf`**: Provides fully worked examples for applying the method to quantum mechanics, from the infinite square well to the harmonic oscillator.

* **Cross-Disciplinary Applications:** The framework is not limited to particle physics. Other papers show its application to diverse fields, demonstrating its potential as a universal principle:
    * **`CosmicBitBalanceSheet.pdf`**: Applies the rational ledger to cosmology, framing the universe's entropy in terms of bits.
    * **`AnnieAreYouOk.pdf`**: A paper titled "Water for Physicists" that applies the "fraction-first" approach to thermodynamics and fluid dynamics.

---
---

## Further Reading & Rationale

If you want the story and the intuition behind the foundational axioms, the Medium articles provide a deep dive:

* [Paradox Dynamics](https://medium.com/where-thought-bends/paradox-dynamics-30d0e7e768a2)
* [Topological Inversion](https://medium.com/@ewesley541/topological-inversion-as-the-origin-of-fundamental-constants-9d9f4dc98f0c)
* [The Adventures of Unmath](https://medium.com/@ewesley541/the-adventures-of-unmath-volume-1-77042fd7cbe4)
* [The Universe's Simple Building Plan](https://medium.com/@ewesley541/the-universes-simple-building-plan-a-new-way-to-see-reality-d9395744893c)
* [Epic of Evan](https://medium.com/@ewesley541/epic-of-evan-a-pattern-based-threat-to-traditional-intelligence-cdc035da2b1d)
* [This Sentence is a Circle](https://medium.com/@ewesley541/this-sentence-is-a-circle-1e7b68264ff2)


Side Quests:
(Demo Website) https://fractionphysics4money.abacusai.app/
(Demo Website) https://fractionphysicsv1.abacusai.app/constants
---
## License & Citation

* **Code:** All code (`.py`, `.ipynb`, etc.) is licensed under the **MIT License**.
* **Content:** All papers, text, and non-code assets are licensed under **CC BY 4.0**.
* **Full License Details:** See the `LICENSES` folder.

**Recommended Attribution:** "Rosetta Stone of Physics" by Evan Wesley, The Smartest Idiot Alive, licensed under CC BY 4.0.
**Contact:** ewesley541@gmail.com
