# Code & Methodology

This folder contains the core scripts used to generate the evidence for the "Rosetta Stone of Physics" project.

## Primary Script: `MathOracle4.ipynb` (The "Mindmelt" Runner)

This code is the original computational engine of the project. Its purpose is to take a hardcoded **"registry"** of 19 fundamental constants, proposed as exact rational numbers, and use them to calculate a comprehensive array of observables across the Standard Model and cosmology.

The script is a self-contained "verification deck" in code form, demonstrating the interlocking nature of the framework.

### Key Calculations Performed:

The script is designed as a single, top-to-bottom run that verifies the consequences of the initial rational number registry. The major sections include:

* **Electroweak Sector:** Calculates the custodial rho parameter, snaps `sin²θW` to a simple fraction, derives the Higgs VEV from W/Z boson anchors, and predicts the full spectrum of boson and fermion masses.
* **Flavor Physics (CKM & PMNS):** Reconstructs the full CKM and PMNS matrices from the registry's base parameters, calculating the Jarlskog invariant, Wolfenstein parameters, and Unitarity Triangle angles.
* **Fundamental Symmetries & Relations:** Explicitly verifies the cancellation of all Standard Model gauge and gravitational anomalies (B-L, Hypercharge, Witten SU(2)) and calculates the Koide relation for charged leptons.
* **GUTs & Cosmology:** Includes toy models for Grand Unification running, proton lifetime estimates, dark matter candidates, and connections to Planck-scale physics.
* **MDL Scoreboard:** Concludes by calculating the information efficiency (parsimony) of the rational framework, comparing the bit-complexity of the 19 registry fractions against standard 64-bit float representations.

### How to Use

To run the verification, open the notebook in Google Colab and execute the single large code cell. The full, detailed output is logged to the `mindmelt_logs` directory, which can be compared directly to the `mindmelt_output.txt` file in the main repository.
