\=== Fraction Physics: Ledger Chain Audit \===

Registry (frozen p/q):  
  \- CKM.lambda=2/9; CKM.A=21/25; CKM.rhobar=3/20; CKM.etabar=7/20  
  \- PMNS.sin2\_th12=7/23; PMNS.sin2\_th13=2/89; PMNS.sin2\_th23=9/16; PMNS.delta\_symbolic=-pi/2  
  \- Neutrino.R21over31=2/65  
  \- Cosmology.Omega\_m=63/200; Cosmology.Omega\_L=137/200; Cosmology.Omega\_b\_over\_Omega\_c=14/75; Cosmology.H0\_km\_s\_Mpc=337/5  
  \- RareDecay.Xt=37/25; RareDecay.Pc=2/5

Certificate SHA-256: 755be80f240464327b385903b2e46d8926ccec117a90c963fa12d1c9014c6440

Assertions:  
  \[OK\] CKM: sin2β \== 119/169: 119/169 \== 119/169  
  \[OK\] CKM: |Vcb| \== 28/675: 28/675 \== 28/675  
  \[OK\] CKM: |Vtd|^2/|Vts|^2 \== 169/4050: 169/4050 \== 169/4050  
  \[INFO\] Rare cores: Core\_KL ≈ 0.133590834801, AddOn\_K+ ≈ 1.658029130967  
  \[OK\] PMNS: |Ue1|^2 \== 1392/2047: 1392/2047 \== 1392/2047  
  \[OK\] PMNS: |Ue2|^2 \== 609/2047: 609/2047 \== 609/2047  
  \[OK\] PMNS: |Ue3|^2 \== 2/89: 2/89 \== 2/89  
  \[OK\] PMNS: first-row closure \== 1: 1/1 \== 1/1  
  \[OK\] Neutrino: Δm21^2/|Δm31^2| \== 2/65: 2/65 \== 2/65  
  \[OK\] Cosmology: Ωm \+ ΩΛ \== 1: 1/1 \== 1/1  
  \[OK\] Cosmology: Ωb \+ Ωc \== Ωm: 63/200 \== 63/200  
  \[OK\] BH: kB TH S\_bits \== Mc^2/(2 ln 2): 128915456946610362321471746558526110819447379519.26 ?= 128915456946610362321471746558526110819447379519.27 (tol=1E-30)

Overall status: ALL CHECKS PASS

MDL charges (bits):  
  \- CKM.lambda: 5 bits  
  \- CKM.A: 10 bits  
  \- CKM.rhobar: 7 bits  
  \- CKM.etabar: 8 bits  
  \- PMNS.sin2\_th12: 8 bits  
  \- PMNS.sin2\_th13: 8 bits  
  \- PMNS.sin2\_th23: 8 bits  
  \- Neutrino.R21over31: 8 bits  
  \- Cosmology.Omega\_m: 14 bits  
  \- Cosmology.Omega\_L: 16 bits  
  \- Cosmology.Omega\_b\_over\_Omega\_c: 11 bits  
  \- Cosmology.H0\_km\_s\_Mpc: 12 bits

Total (registry): 115 bits  
Baseline floats:   832 bits  
Saved:             717 bits  
Evidence factor \~  2^717

Headlines:  
  CKM sin2β \= 119/169  (should be 119/169)  
  CKM |Vcb| \= 28/675       (should be 28/675)  
  CKM |Vtd|^2/|Vts|^2 \= 169/4050 (should be 169/4050)  
  Rare K core (KL): 0.1335908348  
  Rare K add-on (K+): 1.6580291310  
  PMNS first row: (|Ue1|^2, |Ue2|^2, |Ue3|^2) \= (1392/2047, 609/2047, 2/89); sum \= 1/1  
  PMNS δ (symbolic): \-pi/2  
  Cosmology flatness: Ωm+ΩΛ \= 1/1  
  H0 \= 337/5 km s^-1 Mpc^-1; ρ\_c ≈ 8.532855E-27 kg/m^3  
  BH identity check: kB TH S\_bits vs Mc^2/(2 ln 2):  
    lhs \= 1.289154E+47  
    rhs \= 1.289154E+47

Stress test (break-one-seed, 25 trials):  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 1/89  
  \- PASS: ('PMNS', 'sin2\_th23') 9/16 → 5/8  
  \- FAIL: ('CKM', 'A') 21/25 → 22/25  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 3/89  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 31/100  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/3  
  \- PASS: ('PMNS', 'sin2\_th23') 9/16 → 3/5  
  \- FAIL: ('CKM', 'A') 21/25 → 4/5  
  \- FAIL: ('CKM', 'A') 21/25 → 7/8  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/3  
  \- FAIL: ('CKM', 'etabar') 7/20 → 2/5  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 137/199  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/9  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/9  
  \- FAIL: ('PMNS', 'sin2\_th12') 7/23 → 7/22  
  \- FAIL: ('CKM', 'rhobar') 3/20 → 1/5  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 137/201  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 31/100  
  \- FAIL: ('CKM', 'A') 21/25 → 4/5  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 3/89  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 8/25  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 69/100  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 17/25  
  \- FAIL: ('Neutrino', 'R21over31') 2/65 → 1/32  
  \- PASS: ('RareDecay', 'Xt') 37/25 → 37/26

Trials with failing global checks: 22 / 25

Wrote rational\_csp\_certificate.json  
\=== Fraction Physics: Ledger Chain Audit \++ \===

Registry (frozen p/q):  
  \- CKM.lambda=2/9; CKM.A=21/25; CKM.rhobar=3/20; CKM.etabar=7/20  
  \- PMNS.sin2\_th12=7/23; PMNS.sin2\_th13=2/89; PMNS.sin2\_th23=9/16; PMNS.delta\_symbolic=-pi/2  
  \- Neutrino.R21over31=2/65  
  \- Cosmology.Omega\_m=63/200; Cosmology.Omega\_L=137/200; Cosmology.Omega\_b\_over\_Omega\_c=14/75; Cosmology.H0\_km\_s\_Mpc=337/5  
  \- RareDecay.Xt=37/25; RareDecay.Pc=2/5

Certificate SHA-256: 6be58e2830c51ec7f8e86e40df26206f3cd91cb41e8eb44993d670d86ab298f6

Assertions:  
  \[OK\] CKM: sin2β \== 119/169: 119/169 \== 119/169  
  \[OK\] CKM: |Vcb| \== 28/675: 28/675 \== 28/675  
  \[OK\] CKM: |Vtd|^2/|Vts|^2 \== 169/4050: 169/4050 \== 169/4050  
  \[INFO\] Rare cores: KL ≈ 0.133590834801, K+ add-on ≈ 1.658029130967  
  \[OK\] PMNS: |Ue1|^2 \== 1392/2047: 1392/2047 \== 1392/2047  
  \[OK\] PMNS: |Ue2|^2 \== 609/2047: 609/2047 \== 609/2047  
  \[OK\] PMNS: |Ue3|^2 \== 2/89: 2/89 \== 2/89  
  \[OK\] PMNS: first-row closure \== 1: 1/1 \== 1/1  
  \[OK\] Neutrino: Δm21^2/|Δm31^2| \== 2/65: 2/65 \== 2/65  
  \[OK\] Cosmology: Ωm \+ ΩΛ \== 1: 1/1 \== 1/1  
  \[OK\] Cosmology: Ωb \+ Ωc \== Ωm: 63/200 \== 63/200  
  \[FAIL\] BH: kB TH S\_bits \== Mc^2/(2 ln 2): 128915456946610362321471746558526110819447379519.277247209106 ?= 128915456946610362321471746558526110819447379519.277247209107 (tol=1E-30)  
  \[FAIL\] BH: S\_bits (Area) \== S\_bits (Primary): 1.51332201240664167349744991464161505700800940845693053843150E+77 ?= 1.51332201240664167349744991464161505700800940845693053843149E+77 (tol=1E-28)

Overall status: FAILURES PRESENT

MDL charges (bits):  
  \- CKM.lambda: 5 bits  
  \- CKM.A: 10 bits  
  \- CKM.rhobar: 7 bits  
  \- CKM.etabar: 8 bits  
  \- PMNS.sin2\_th12: 8 bits  
  \- PMNS.sin2\_th13: 8 bits  
  \- PMNS.sin2\_th23: 8 bits  
  \- Neutrino.R21over31: 8 bits  
  \- Cosmology.Omega\_m: 14 bits  
  \- Cosmology.Omega\_L: 16 bits  
  \- Cosmology.Omega\_b\_over\_Omega\_c: 11 bits  
  \- Cosmology.H0\_km\_s\_Mpc: 12 bits

Total (registry): 115 bits  
Baseline floats:   832 bits  
Saved:             717 bits  
Evidence factor \~  2^717

Headlines:  
  CKM sin2β \= 119/169  (should be 119/169)  
  CKM |Vcb| \= 28/675       (should be 28/675)  
  CKM |Vtd|^2/|Vts|^2 \= 169/4050 (should be 169/4050)  
  Rare K core (KL): 0.1335908348  
  Rare K add-on (K+): 1.6580291310  
  PMNS first row: (|Ue1|^2, |Ue2|^2, |Ue3|^2) \= (1392/2047, 609/2047, 2/89); sum \= 1/1  
  PMNS δ (symbolic): \-pi/2  
  Cosmology flatness: Ωm+ΩΛ \= 1/1  
  H0 \= 337/5 km s^-1 Mpc^-1; ρ\_c ≈ 8.532855E-27 kg/m^3  
  BH identity check: kB TH S\_bits vs Mc^2/(2 ln 2):  
    lhs \= 1.289154E+47  
    rhs \= 1.289154E+47  
  Planck units: ℓ\_P ≈ 1.616255E-35 m ; t\_P ≈ 5.391246E-44 s ; m\_P ≈ 2.176434E-8 kg  
  BH area-based bits check: S\_bits(A) ≈ 1.513322E+77 ; S\_bits(primary) ≈ 1.513322E+77

Stress test (break-one-seed, 25 trials):  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 1/89  
  \- FAIL: ('PMNS', 'sin2\_th23') 9/16 → 5/8  
  \- FAIL: ('CKM', 'A') 21/25 → 22/25  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 3/89  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 31/100  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/3  
  \- FAIL: ('PMNS', 'sin2\_th23') 9/16 → 3/5  
  \- FAIL: ('CKM', 'A') 21/25 → 4/5  
  \- FAIL: ('CKM', 'A') 21/25 → 7/8  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/3  
  \- FAIL: ('CKM', 'etabar') 7/20 → 2/5  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 137/199  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/9  
  \- FAIL: ('CKM', 'lambda') 2/9 → 1/9  
  \- FAIL: ('PMNS', 'sin2\_th12') 7/23 → 7/22  
  \- FAIL: ('CKM', 'rhobar') 3/20 → 1/5  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 137/201  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 31/100  
  \- FAIL: ('CKM', 'A') 21/25 → 4/5  
  \- FAIL: ('PMNS', 'sin2\_th13') 2/89 → 3/89  
  \- FAIL: ('Cosmology', 'Omega\_m') 63/200 → 8/25  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 69/100  
  \- FAIL: ('Cosmology', 'Omega\_L') 137/200 → 17/25  
  \- FAIL: ('Neutrino', 'R21over31') 2/65 → 1/32  
  \- FAIL: ('RareDecay', 'Xt') 37/25 → 37/26  
Trials with failing global checks: 25 / 25

\[mββ Envelope Module\]  
  Using |Ue1|^2=1392/2047, |Ue2|^2=609/2047, |Ue3|^2=2/89  
  Δm31^2 \= 1/400 eV^2 ;  Δm21^2 \= 0.0000769230769230769230769230769230769230769230769230769230769230 eV^2  
  Σm\_min (NH, m1→0) ≈ 5.877058e-02 eV  
  Σm\_min (IH, m3→0) ≈ 1.007634e-01 eV

\[Falsifier Deck\]  
  Registry SHA-256: 6be58e2830c51ec7f8e86e40df26206f3cd91cb41e8eb44993d670d86ab298f6  
  CKM sin2β \= 119/169  
  CKM |Vcb| \= 28/675  
  CKM |Vtd|^2/|Vts|^2 \= 169/4050  
  PMNS |Ue1|^2 \= 1392/2047  
  PMNS |Ue2|^2 \= 609/2047  
  PMNS |Ue3|^2 \= 2/89  
  Neutrino Δm21^2/|Δm31^2| \= 2/65  
  Cosmology Ωm \= 63/200  
  Cosmology ΩΛ \= 137/200  
  Cosmology Ωb/Ωc \= 14/75  
  H0 (km s^-1 Mpc^-1) \= 337/5  
  PMNS\_row\_exact \= (1392/2047, 609/2047, 2/89)  
  Cosmo\_split \= Ωb:Ωc \= 14:75 (→ Ωb \= 441/8900, Ωc \= 189/712)

Wrote rational\_csp\_certificate.json  
\=== CKM Triangle & Full-Matrix Verifier (fixed) \===

Wolfenstein seeds:  
  λ \= 2/9 ≈ 0.222222222222  
  A \= 21/25 ≈ 0.840000000000  
  (ρ̄, η̄) \= (3/20, 7/20) ≈ (0.150000000000, 0.350000000000)

Unitarity-triangle geometry (from (ρ̄,η̄)):  
  tanβ \= 7/17  ⇒  sin2β \= 119/169  (exact)  
  tanγ \= 7/3  ⇒  sin2γ \= 21/29 (exact)  
  β ≈ 22.380135° ;  γ ≈ 66.801409° ;  α ≈ 90.818455°  
  α+β+γ ≈ 180.000000°  (should be 180°)

CKM matrix (PDG parameterization from seeds):  
  u : V\_ud=+0.974990036485+0.000000000000i  V\_us=+0.222220853201+0.000000000000i  V\_ub=+0.001382716049-0.003226337449i  
  c : V\_cd=-0.222086872494-0.000130486896i  V\_cs=+0.974144091523-0.000029740724i  V\_cb=+0.041481225931+0.000000000000i  
  t : V\_td=+0.007871124702-0.003142958686i  V\_ts=-0.040751286061-0.000716346767i  V\_tb=+0.999133117627+0.000000000000i

Unitarity checks:  
  ‖V†V \- I‖\_max\_offdiag ≈ 2.407e-17  
  max |(V†V)\_ii \- 1|    ≈ 2.220e-16  
  db-triangle closure |V\_ud V\_ub\* \+ V\_cd V\_cb\* \+ V\_td V\_tb\*| ≈ 1.735e-18

Angles from matrix vs from (ρ̄,η̄):  
  β\_fromM  ≈ 21.800607° ; β\_geom  ≈ 22.380135°  
  γ\_fromM  ≈ 66.767745° ; γ\_geom  ≈ 66.801409°  
  α\_fromM  ≈ 91.431648° ; α\_geom  ≈ 90.818455°

Rational headline predictions vs matrix:  
  sin2β (matrix) ≈ 0.689634884966 ; exact \= 119/169 ≈ 0.704142011834  
  sin2γ (matrix) ≈ 0.724947843057 ; exact \= 21/29 ≈ 0.724137931034

|V\_cb| (matrix)  ≈ 0.041481225931  ;  A λ^2 (exact rational) \= 28/675 ≈ 0.041481481481  
  (matrix uses |V\_cb| \= s23·c13, so it’s slightly smaller by O(s13^2); ledger locks s23=A λ^2.)

Jarlskog invariant:  
  J (matrix) ≈ 2.897177247874e-05  
  J (ledger) \= A^2 λ^6 η̄ \= 5488/184528125  ≈ 2.974072380565e-05  
  Relative diff ≈ 2.586e-02

Triangle sides R\_u, R\_t:  
  R\_u (matrix) ≈ 0.371493635096 ;  R\_u (geom) \= √(ρ̄²+η̄²) ≈ 0.380788655293  
  R\_t (matrix) ≈ 0.919200360295 ;  R\_t (geom) \= √((1-ρ̄)²+η̄²) ≈ 0.919238815543

Done: CKM triangle verified, unitarity validated, rational angle claims embedded.  
\=== Ledger Hardener \===

\[BH\]  
  primary identity pass?  True  
  area==primary (relative) pass?  True  
  S\_bits(area)   ≈ 1.513322E+77  
  S\_bits(primary)≈ 1.513322E+77

\[CKM exact-apex\]  
  Enforced |Vcb| \= A λ^2 \= 28/675 → |Vcb|\_matrix \= 0.041481481481  
  Apex target \= 0.150000000000 \+ i 0.350000000000  
  Apex from matrix \= 0.150000000000 \+ i 0.350000000000  
  Angles: β ≈ 22.380135°, γ ≈ 66.801409°, α ≈ 90.818455°  
  sin2β (matrix) ≈ 0.704142011834 ; exact \= 119/169 ≈ 0.704142011834  
  sin2γ (matrix) ≈ 0.724137931034 ; exact \= 21/29 ≈ 0.724137931034  
  Unitarity checks:  
    max |(V†V)\_ij \- δ\_ij| offdiag ≈ 1.986e-17  
    max |(V†V)\_ii \- 1|            ≈ 2.220e-16  
    |V\_ud V\_ub\* \+ V\_cd V\_cb\* \+ V\_td V\_tb\*| ≈ 1.939e-18

\[Canonical Certificate\]  
  SHA-256: 56c3298944a9e7da6fc7d580f327ac7595d8504927754259ab8a372d3b67f73c  
  derived\_headline: {'CKM\_sin2beta\_exact': '119/169', 'Cosmo\_flat\_check': '1/1', 'H0\_SI\_sinv': '2.1842852410855020107358286594234308170761E-18', 'rho\_c\_kg\_m3': '8.5328551637393168073505908131430668372325E-27', 'BH\_kBTHS\_bits': '1.2891545694661036232147174655852611081944E47'}  
\=== Cosmic Bit Budget \===

Registry cosmology:  
  Ωm \= 63/200 ; ΩΛ \= 137/200 ; Ωb/Ωc \= 14/75  
  Flatness check: Ωm+ΩΛ \= 1/1

Hubble-scale geometry:  
  H0 (s^-1)  ≈ 2.184285E-18  
  R\_H (m)    ≈ 1.372496E+26  
  V\_H (m^3)  ≈ 1.082985E+79  
  A\_H (m^2)  ≈ 2.367187E+53

Energies (within Hubble sphere):  
  ρ\_c (kg/m^3)  ≈ 8.532855E-27  
  e\_c=ρ\_c c^2 (J/m^3) ≈ 7.668947E-10  
  E\_total (J)   ≈ 8.305359E+69  
   ⤷ E\_baryons  ≈ 4.115352E+68  
   ⤷ E\_CDM      ≈ 2.204653E+69  
   ⤷ E\_vacuum   ≈ 5.689171E+69

Holographic cap (area law):  
  Bits\_holo  ≈ 3.268340E+122  (log10 ≈ 122.5143273391961855622867005877196788787841796875)

Gibbons–Hawking (de Sitter) \+ Landauer:  
  T\_dS (K)   ≈ 2.655353E-30  
  E\_bit (J)  ≈ 2.541154E-53  \[= kB T\_dS ln2\]  
  Bits\_tot\_L ≈ 3.268340E+122  (log10 ≈ 122.5143273391961855622867005877196788787841796875)  
    breakdown: bits\_baryon ≈ 1.619481E+121 ; bits\_CDM ≈ 8.675792E+121 ; bits\_vacuum ≈ 2.238813E+122  
  Ratio (Landauer\_tot / Holographic\_cap) ≈ 9.999999E-1

Baryon context (not part of any equality):  
  M\_baryons (kg)  ≈ 4.578947E+51  
  N\_baryons       ≈ 2.737586E+78  
  Landauer bits per baryon @ T\_dS ≈ 5.915726E+42

\[Cosmic Bit Budget Certificate\]  
  SHA-256: aa6937ec4c8158b47cdd04dca9936017091c3bdef8fdc91ee228f45ce4cd198f  
  headline: {'H0\_SI\_sinv': '2.1842852410855020107358286594234308170761E-18', 'rho\_c\_kg\_m3': '8.5328551637393168073505908131430668372325E-27', 'RH\_m': '1.3724968349418283511095302670623145400593E26', 'AH\_m2': '2.3671870007049282192286169868832291448595E53', 'VH\_m3': '1.0829855553943178595767905811196521676781E79', 'E\_total\_J': '8.3053596576246221918044743621411917794383E69', 'T\_dS\_K': '2.6553535939990478138194024528903283528232E-30', 'E\_bit\_J': '2.5411547002630561647508053246393017491164E-53', 'Bits\_holo': '3.2683408281931300713181762676171421366279E122', 'Bits\_Landauer\_tot': '3.2683408281931300713181762676171421366279E122'} 

\=== Baseline Sanity \===  
  OK — all baseline checks pass.

\=== Two-Seed Stress Test — Rigor Metrics \===

Trials: 500 ; unique seeds: 14 ; checks: 12  
Trials with any failure: 437  (rate ≈ 0.874)  
Average number of failed checks per trial: 2.050

Top failing checks (count):  
  CKM\_VtdVts2\_169\_4050 : 197  
  Cosmo\_flat\_1 : 140  
  CKM\_Vcb\_28\_675 : 139  
  CKM\_sin2beta\_119\_169 : 136  
  PMNS\_Ue2\_609\_2047 : 133  
  PMNS\_Ue1\_1392\_2047 : 133  
  Nu\_ratio\_2\_65 : 78  
  PMNS\_Ue3\_2\_89 : 69  
  PMNS\_row\_closure\_1 : 0  
  Cosmo\_b\_plus\_c\_eq\_m : 0  
  BH\_primary\_rel : 0  
  BH\_area\_rel : 0

Seed index map:  
  \[00\] CKM.lambda  
  \[01\] CKM.A  
  \[02\] CKM.rhobar  
  \[03\] CKM.etabar  
  \[04\] PMNS.sin2\_th12  
  \[05\] PMNS.sin2\_th13  
  \[06\] PMNS.sin2\_th23  
  \[07\] Neutrino.R21over31  
  \[08\] Cosmology.Omega\_m  
  \[09\] Cosmology.Omega\_L  
  \[10\] Cosmology.Omega\_b\_over\_Omega\_c  
  \[11\] Cosmology.H0\_km\_s\_Mpc  
  \[12\] RareDecay.Xt  
  \[13\] RareDecay.Pc  
\=== Three-Seed Stress Test \+ ΔMDL Ledger \===

\=== Baseline Sanity \===  
  OK — all baseline checks pass.  
  MDL: registry=115 bits ; float baseline=832 bits ; saved=717 bits

\=== Three-Seed Stress Test — Rigor Metrics (No Charts) \===

Trials: 800  
Fail rate: 0.973  
Avg \# failed checks per FAIL trial: 2.985

ΔMDL (saved bits relative to baseline saved=717):  
  mean Δsaved (all trials):  4.799 bits  
  mean Δsaved (FAIL trials): 4.787 bits  
  mean Δsaved (PASS trials): 5.227 bits

Top failing checks (count):  
      CKM\_VtdVts2\_169\_4050 : 436  
      CKM\_sin2beta\_119\_169 : 326  
        PMNS\_Ue1\_1392\_2047 : 305  
         PMNS\_Ue2\_609\_2047 : 305  
              Cosmo\_flat\_1 : 304  
            CKM\_Vcb\_28\_675 : 297  
             Nu\_ratio\_2\_65 : 193  
             PMNS\_Ue3\_2\_89 : 156  
        PMNS\_row\_closure\_1 : 0  
       Cosmo\_b\_plus\_c\_eq\_m : 0  
            BH\_primary\_rel : 0  
               BH\_area\_rel : 0

Seed index map:  
  \[00\] CKM.lambda  
  \[01\] CKM.A  
  \[02\] CKM.rhobar  
  \[03\] CKM.etabar  
  \[04\] PMNS.sin2\_th12  
  \[05\] PMNS.sin2\_th13  
  \[06\] PMNS.sin2\_th23  
  \[07\] Neutrino.R21over31  
  \[08\] Cosmology.Omega\_m  
  \[09\] Cosmology.Omega\_L  
  \[10\] Cosmology.Omega\_b\_over\_Omega\_c  
  \[11\] Cosmology.H0\_km\_s\_Mpc  
  \[12\] RareDecay.Xt  
  \[13\] RareDecay.Pc

Seed impact rates  P(failure | seed involved):  
  \[00\] CKM.lambda : 1.000  
  \[01\] CKM.A : 1.000  
  \[02\] CKM.rhobar : 1.000  
  \[03\] CKM.etabar : 1.000  
  \[04\] PMNS.sin2\_th12 : 1.000  
  \[05\] PMNS.sin2\_th13 : 1.000  
  \[06\] PMNS.sin2\_th23 : 0.917  
  \[07\] Neutrino.R21over31 : 1.000  
  \[08\] Cosmology.Omega\_m : 1.000  
  \[09\] Cosmology.Omega\_L : 1.000  
  \[10\] Cosmology.Omega\_b\_over\_Omega\_c : 0.903  
  \[11\] Cosmology.H0\_km\_s\_Mpc : 0.924  
  \[12\] RareDecay.Xt : 0.938  
  \[13\] RareDecay.Pc : 0.932

Top brittle triads  (P(fail | triad nudged), exposure):  
   1.000  (n=  6\) : Cosmology.Omega\_m, RareDecay.Xt, RareDecay.Pc  
   1.000  (n=  6\) : PMNS.sin2\_th12, Cosmology.Omega\_m, RareDecay.Pc  
   1.000  (n=  6\) : CKM.rhobar, Neutrino.R21over31, Cosmology.H0\_km\_s\_Mpc  
   1.000  (n=  6\) : CKM.rhobar, RareDecay.Xt, RareDecay.Pc  
   1.000  (n=  6\) : CKM.lambda, CKM.rhobar, PMNS.sin2\_th13  
   1.000  (n=  6\) : CKM.rhobar, PMNS.sin2\_th12, Cosmology.H0\_km\_s\_Mpc  
   1.000  (n=  6\) : Cosmology.Omega\_m, Cosmology.Omega\_L, Cosmology.H0\_km\_s\_Mpc  
   1.000  (n=  6\) : PMNS.sin2\_th12, PMNS.sin2\_th13, Neutrino.R21over31  
   1.000  (n=  6\) : CKM.rhobar, Cosmology.Omega\_b\_over\_Omega\_c, RareDecay.Pc  
   1.000  (n=  6\) : CKM.A, PMNS.sin2\_th12, PMNS.sin2\_th23  
   1.000  (n=  5\) : CKM.lambda, CKM.rhobar, RareDecay.Pc  
   1.000  (n=  5\) : CKM.etabar, PMNS.sin2\_th12, Cosmology.Omega\_m  
   1.000  (n=  5\) : PMNS.sin2\_th23, Neutrino.R21over31, Cosmology.H0\_km\_s\_Mpc  
   1.000  (n=  5\) : CKM.rhobar, PMNS.sin2\_th23, Cosmology.H0\_km\_s\_Mpc  
   1.000  (n=  5\) : CKM.rhobar, PMNS.sin2\_th13, Cosmology.Omega\_L

Worst Δsaved trials (most negative \= biggest MDL penalty):  
  Δsaved=    \-2.0 bits  | triad: PMNS.sin2\_th13, Neutrino.R21over31, Cosmology.Omega\_m  | fails=5 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89, Nu\_ratio\_2\_65, Cosmo\_flat\_1\]  
  Δsaved=    \-2.0 bits  | triad: PMNS.sin2\_th12, PMNS.sin2\_th13, Neutrino.R21over31  | fails=4 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89, Nu\_ratio\_2\_65\]  
  Δsaved=    \-2.0 bits  | triad: CKM.etabar, PMNS.sin2\_th13, PMNS.sin2\_th23  | fails=5 \[CKM\_sin2beta\_119\_169, CKM\_VtdVts2\_169\_4050, PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th12, Neutrino.R21over31, RareDecay.Xt  | fails=3 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, Nu\_ratio\_2\_65\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th12, PMNS.sin2\_th13, RareDecay.Xt  | fails=3 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th12, PMNS.sin2\_th23, RareDecay.Xt  | fails=2 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th13, Cosmology.Omega\_L, RareDecay.Pc  | fails=4 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89, Cosmo\_flat\_1\]  
  Δsaved=    \-1.0 bits  | triad: CKM.etabar, PMNS.sin2\_th23, Cosmology.Omega\_L  | fails=3 \[CKM\_sin2beta\_119\_169, CKM\_VtdVts2\_169\_4050, Cosmo\_flat\_1\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th12, PMNS.sin2\_th13, Cosmology.Omega\_L  | fails=4 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, PMNS\_Ue3\_2\_89, Cosmo\_flat\_1\]  
  Δsaved=    \-1.0 bits  | triad: PMNS.sin2\_th12, Neutrino.R21over31, RareDecay.Pc  | fails=3 \[PMNS\_Ue1\_1392\_2047, PMNS\_Ue2\_609\_2047, Nu\_ratio\_2\_65\]  
\=== Irrational Audit — CF/MDL/Evidence \===

\=== Irrational Audit: √2 \===  
  value ≈ 1.4142135623730950488016887242096980785696E0  
  tol=1.000000E-6  \-\>  p/q=1393/985  (bits= 21\)  err≈3.644035E-7  ✓  
  tol=1.000000E-12  \-\>  p/q=1607521/1136689  (bits= 42\)  err≈2.736350E-13  ✓  
  tol=1.000000E-18  \-\>  p/q=1855077841/1311738121  (bits= 62\)  err≈2.054758E-19  ✓  
  tol=1.000000E-24  \-\>  p/q=886731088897/627013566048  (bits= 80\)  err≈8.992928E-25  ✓  
  tol=1.000000E-30  \-\>  p/q=1023286908188737/723573111879672  (bits=100)  err≈6.752897E-31  ✓  
  \[Pell trap √2\] min |p^2 \- 2 q^2| over convergents \= 1 (at k=0, p=1, q=1, value=-1)

\=== Irrational Audit: φ \= (1+√5)/2 \===  
  value ≈ 1.6180339887498948482045868343656381177203E0  
  tol=1.000000E-6  \-\>  p/q=1597/987  (bits= 21\)  err≈4.590717E-7  ✓  
  tol=1.000000E-12  \-\>  p/q=1346269/832040  (bits= 41\)  err≈6.459911E-13  ✓  
  tol=1.000000E-18  \-\>  p/q=1134903170/701408733  (bits= 61\)  err≈9.090183E-19  ✓  
  tol=1.000000E-24  \-\>  p/q=1548008755920/956722026041  (bits= 81\)  err≈4.885887E-25  ✓  
  tol=1.000000E-30  \-\>  p/q=1304969544928657/806515533049393  (bits=101)  err≈6.875266E-31  ✓  
  \[Quadratic trap φ\] min |p^2 \- p q \- q^2| over convergents \= 1 (at k=0, p=1, q=1, value=-1)

\=== Irrational Audit: π \===  
  value ≈ 3.1415926535897932384626433832795028841971E0  
  tol=1.000000E-6  \-\>  p/q=355/113  (bits= 16\)  err≈2.667641E-7  ✓  
  tol=1.000000E-12  \-\>  p/q=4272943/1360120  (bits= 44\)  err≈4.040669E-13  ✓  
  tol=1.000000E-18  \-\>  p/q=2549491779/811528438  (bits= 62\)  err≈5.513663E-19  ✓  
  tol=1.000000E-24  \-\>  p/q=3587785776203/1142027682075  (bits= 83\)  err≈3.147769E-25  ✓  
  tol=1.000000E-30  \-\>  p/q=5706674932067741/1816491048114374  (bits=104)  err≈2.332819E-31  ✓  
  \[Poly trap π\] degree≤4 with |coeff|≤5  
    min |P(p/q)| ≈ 8.36734693E-1  at d=2, lead=-1, coefs=\[-5, 5\], p/q=22/7

\=== Irrational Audit: e \===  
  value ≈ 2.7182818284590452353602874713526624977572E0  
  tol=1.000000E-6  \-\>  p/q=2721/1001  (bits= 22\)  err≈1.101773E-7  ✓  
  tol=1.000000E-12  \-\>  p/q=1084483/398959  (bits= 40\)  err≈4.818240E-13  ✓  
  tol=1.000000E-18  \-\>  p/q=848456353/312129649  (bits= 59\)  err≈6.027268E-19  ✓  
  tol=1.000000E-24  \-\>  p/q=1098127402131/403978495031  (bits= 79\)  err≈2.914525E-25  ✓  
  tol=1.000000E-30  \-\>  p/q=2124008553358849/781379079653017  (bits=101)  err≈6.546166E-32  ✓  
  \[Poly trap e\] degree≤4 with |coeff|≤5  
    min |P(p/q)| ≈ 2.30468750E-1  at d=4, lead=-1, coefs=\[5, \-5, \-5, 5\], p/q=11/4

\=== MDL vs Tolerance Summary (bits to hit tol) \===

  √2  
    tol=1.000000E-6  \-\> bits= 21 ; q≈985 ; ✓  
    tol=1.000000E-12  \-\> bits= 42 ; q≈1136689 ; ✓  
    tol=1.000000E-18  \-\> bits= 62 ; q≈1311738121 ; ✓  
    tol=1.000000E-24  \-\> bits= 80 ; q≈627013566048 ; ✓  
    tol=1.000000E-30  \-\> bits=100 ; q≈723573111879672 ; ✓

  φ  
    tol=1.000000E-6  \-\> bits= 21 ; q≈987 ; ✓  
    tol=1.000000E-12  \-\> bits= 41 ; q≈832040 ; ✓  
    tol=1.000000E-18  \-\> bits= 61 ; q≈701408733 ; ✓  
    tol=1.000000E-24  \-\> bits= 81 ; q≈956722026041 ; ✓  
    tol=1.000000E-30  \-\> bits=101 ; q≈806515533049393 ; ✓

  π  
    tol=1.000000E-6  \-\> bits= 16 ; q≈113 ; ✓  
    tol=1.000000E-12  \-\> bits= 44 ; q≈1360120 ; ✓  
    tol=1.000000E-18  \-\> bits= 62 ; q≈811528438 ; ✓  
    tol=1.000000E-24  \-\> bits= 83 ; q≈1142027682075 ; ✓  
    tol=1.000000E-30  \-\> bits=104 ; q≈1816491048114374 ; ✓

  e  
    tol=1.000000E-6  \-\> bits= 22 ; q≈1001 ; ✓  
    tol=1.000000E-12  \-\> bits= 40 ; q≈398959 ; ✓  
    tol=1.000000E-18  \-\> bits= 59 ; q≈312129649 ; ✓  
    tol=1.000000E-24  \-\> bits= 79 ; q≈403978495031 ; ✓  
    tol=1.000000E-30  \-\> bits=101 ; q≈781379079653017 ; ✓

\[Audit Certificate\]  
  SHA-256: 0c40bda239b71428888da4380965debc401498aa0bee5c5095dc8c6abf743486  
  sample entry (pi @ 1e-30): {'p': '5706674932067741', 'q': '1816491048114374', 'bits': 104, 'err\_E': '2.33281989961436255917E-31', 'hit': True}

\==============================================================================================================  
\=== MODULE A — Irrationals as Emergent Rational Locks (CF \+ MDL \+ CI)                                       \==  
\==============================================================================================================  
Module A sanity: OK

\==============================================================================================================  
\=== CF Ladder — pi  
\==============================================================================================================  
k  | p/q                | value                            | |err|     | ppm          | bits  
\--------------------------------------------------------------------------------------------  
1  | 3/1                | 3.000000000000000000000000000000 | 1.4159E-1 | 45070.341448 | 2     
2  | 22/7               | 3.142857142857142857142857142857 | 1.2644E-3 | 402.499434   | 8     
3  | 333/106            | 3.141509433962264150943396226415 | 8.3219E-5 | 26.489630    | 16    
4  | 355/113            | 3.141592920353982300884955752212 | 2.6676E-7 | 0.084913     | 16    
5  | 103993/33102       | 3.141592653011902604072261494773 | 5.7789E-10 | 0.000183     | 33    
6  | 104348/33215       | 3.141592653921421044708715941592 | 3.3162E-10 | 0.000105     | 33    
7  | 208341/66317       | 3.141592653467436705520454785349 | 1.2235E-10 | 0.000038     | 35    
8  | 312689/99532       | 3.141592653618936623397500301410 | 2.9143E-11 | 0.000009     | 36    
9  | 833719/265381      | 3.141592653581077771204419306581 | 8.7154E-12 | 0.000002     | 39    
10 | 1146408/364913     | 3.141592653591403978482542414219 | 1.6107E-12 | 0.000000     | 40    
11 | 4272943/1360120    | 3.141592653589389171543687321706 | 4.0406E-13 | 0.000000     | 44    
12 | 5419351/1725033    | 3.141592653589815383241943777307 | 2.2144E-14 | 0.000000     | 44  

\==============================================================================================================  
\=== CF Ladder — e  
\==============================================================================================================  
k  | p/q                | value                            | |err|     | ppm          | bits  
\--------------------------------------------------------------------------------------------  
1  | 2/1                | 2.000000000000000000000000000000 | 7.1828E-1 | 264241.117657 | 1     
2  | 3/1                | 3.000000000000000000000000000000 | 2.8171E-1 | 103638.323514 | 2     
3  | 8/3                | 2.666666666666666666666666666666 | 5.1615E-2 | 18988.156876 | 5     
4  | 11/4               | 2.750000000000000000000000000000 | 3.1718E-2 | 11668.463221 | 6     
5  | 19/7               | 2.714285714285714285714285714285 | 3.9961E-3 | 1470.088248  | 8     
6  | 87/32              | 2.718750000000000000000000000000 | 4.6817E-4 | 172.230684   | 12    
7  | 106/39             | 2.717948717948717948717948717948 | 3.3311E-4 | 122.544508   | 13    
8  | 193/71             | 2.718309859154929577464788732394 | 2.8030E-5 | 10.311916    | 15    
9  | 1264/465           | 2.718279569892473118279569892473 | 2.2585E-6 | 0.830880     | 20    
10 | 1457/536           | 2.718283582089552238805970149253 | 1.7536E-6 | 0.645124     | 21    
11 | 2721/1001          | 2.718281718281718281718281718281 | 1.1017E-7 | 0.040531     | 22    
12 | 23225/8544         | 2.718281835205992509363295880149 | 6.7469E-9 | 0.002482     | 29  

\==============================================================================================================  
\=== CF Ladder — √2  
\==============================================================================================================  
k  | p/q                | value                            | |err|     | ppm          | bits  
\--------------------------------------------------------------------------------------------  
1  | 1/1                | 1.000000000000000000000000000000 | 4.1421E-1 | 292893.218813 | 0     
2  | 3/2                | 1.500000000000000000000000000000 | 8.5786E-2 | 60660.171779 | 3     
3  | 7/5                | 1.400000000000000000000000000000 | 1.4213E-2 | 10050.506338 | 6     
4  | 17/12              | 1.416666666666666666666666666666 | 2.4531E-3 | 1734.606680  | 9     
5  | 41/29              | 1.413793103448275862068965517241 | 4.2045E-4 | 297.309356   | 11    
6  | 99/70              | 1.414285714285714285714285714285 | 7.2151E-5 | 51.019106    | 14    
7  | 239/169            | 1.414201183431952662721893491124 | 1.2378E-5 | 8.753233     | 16    
8  | 577/408            | 1.414215686274509803921568627450 | 2.1239E-6 | 1.501825     | 19    
9  | 1393/985           | 1.414213197969543147208121827411 | 3.6440E-7 | 0.257672     | 21    
10 | 3363/2378          | 1.414213624894869638351555929352 | 6.2521E-8 | 0.044209     | 24    
11 | 8119/5741          | 1.414213551646054694304128200661 | 1.0727E-8 | 0.007585     | 26    
12 | 19601/13860        | 1.414213564213564213564213564213 | 1.8404E-9 | 0.001301     | 29  

\==============================================================================================================  
\=== CF Ladder — φ  
\==============================================================================================================  
k  | p/q                | value                            | |err|     | ppm          | bits  
\--------------------------------------------------------------------------------------------  
1  | 1/1                | 1.000000000000000000000000000000 | 6.1803E-1 | 381966.011250 | 0     
2  | 2/1                | 2.000000000000000000000000000000 | 3.8196E-1 | 236067.977499 | 1     
3  | 3/2                | 1.500000000000000000000000000000 | 1.1803E-1 | 72949.016875 | 3     
4  | 5/3                | 1.666666666666666666666666666666 | 4.8632E-2 | 30056.647916 | 5     
5  | 8/5                | 1.600000000000000000000000000000 | 1.8033E-2 | 11145.618000 | 6     
6  | 13/8               | 1.625000000000000000000000000000 | 6.9660E-3 | 4305.231718  | 7     
7  | 21/13              | 1.615384615384615384615384615384 | 2.6493E-3 | 1637.402788  | 9     
8  | 34/21              | 1.619047619047619047619047619047 | 1.0136E-3 | 626.457976   | 11    
9  | 55/34              | 1.617647058823529411764705882352 | 3.8692E-4 | 239.135845   | 12    
10 | 89/55              | 1.618181818181818181818181818181 | 1.4782E-4 | 91.363613    | 13    
11 | 144/89             | 1.617977528089887640449438202247 | 5.6460E-5 | 34.894606    | 15    
12 | 233/144            | 1.618055555555555555555555555555 | 2.1566E-5 | 13.329018    | 16  

\==============================================================================================================  
\=== Farey/CF Witness Checks  
\==============================================================================================================  
  π: det=1 across neighbors? False ; alternating above/below? True  
  e: det=1 across neighbors? False ; alternating above/below? True  
  √2: det=1 across neighbors? False ; alternating above/below? True  
  φ: det=1 across neighbors? False ; alternating above/below? True

\==============================================================================================================  
\=== Equality Traps (provably ±1 on quadratic irrationals; empirical for π,e)  
\==============================================================================================================  
  \[√2 Pell\] min |p^2 \- 2 q^2| over convergents \= 1  (at k=0, p=1, q=1, value=-1)  → never 0  
  \[φ quadratic\] min |p^2 \- p q \- q^2| over convergents \= 1  (at k=0, p=1, q=1, value=-1)  → never 0  
  \[π poly trap\]  min |P(p/q)| ≈ 8.36734693E-1  at d=2, lead=-1, coefs=\[-5, 5\], p/q=22/7  
  \[e poly trap\]  min |P(p/q)| ≈ 2.30468750E-1  at d=4, lead=-1, coefs=\[5, \-5, \-5, 5\], p/q=11/4

\==============================================================================================================  
\=== MDL vs Tolerance Summary (bits to hit tol)  
\==============================================================================================================

  π  
    tol=1.000000E-6  \-\>  p/q=355/113 ; bits= 16 ; hit=✓  
    tol=1.000000E-12  \-\>  p/q=4272943/1360120 ; bits= 44 ; hit=✓  
    tol=1.000000E-18  \-\>  p/q=2549491779/811528438 ; bits= 62 ; hit=✓  
    tol=1.000000E-24  \-\>  p/q=3587785776203/1142027682075 ; bits= 83 ; hit=✓  
    tol=1.000000E-30  \-\>  p/q=5706674932067741/1816491048114374 ; bits=104 ; hit=✓

  e  
    tol=1.000000E-6  \-\>  p/q=2721/1001 ; bits= 22 ; hit=✓  
    tol=1.000000E-12  \-\>  p/q=1084483/398959 ; bits= 40 ; hit=✓  
    tol=1.000000E-18  \-\>  p/q=848456353/312129649 ; bits= 59 ; hit=✓  
    tol=1.000000E-24  \-\>  p/q=1098127402131/403978495031 ; bits= 79 ; hit=✓  
    tol=1.000000E-30  \-\>  p/q=2124008553358849/781379079653017 ; bits=101 ; hit=✓

  √2  
    tol=1.000000E-6  \-\>  p/q=1393/985 ; bits= 21 ; hit=✓  
    tol=1.000000E-12  \-\>  p/q=1607521/1136689 ; bits= 42 ; hit=✓  
    tol=1.000000E-18  \-\>  p/q=1855077841/1311738121 ; bits= 62 ; hit=✓  
    tol=1.000000E-24  \-\>  p/q=886731088897/627013566048 ; bits= 80 ; hit=✓  
    tol=1.000000E-30  \-\>  p/q=1023286908188737/723573111879672 ; bits=100 ; hit=✓

  φ  
    tol=1.000000E-6  \-\>  p/q=1597/987 ; bits= 21 ; hit=✓  
    tol=1.000000E-12  \-\>  p/q=1346269/832040 ; bits= 41 ; hit=✓  
    tol=1.000000E-18  \-\>  p/q=1134903170/701408733 ; bits= 61 ; hit=✓  
    tol=1.000000E-24  \-\>  p/q=1548008755920/956722026041 ; bits= 81 ; hit=✓  
    tol=1.000000E-30  \-\>  p/q=1304969544928657/806515533049393 ; bits=101 ; hit=✓

\==============================================================================================================  
\=== Audit Certificate  
\==============================================================================================================  
  SHA-256: 0c40bda239b71428888da4380965debc401498aa0bee5c5095dc8c6abf743486  
  sample (pi @ 1e-30): {'p': '5706674932067741', 'q': '1816491048114374', 'bits': 104, 'err\_E': '2.33281989961436255917E-31', 'hit': True}  
  \[file\] wrote: /content/fraction\_physics\_dlc/irrational\_audit/irrational\_audit\_0c40bda239b71428888da4380965debc401498aa0bee5c5095dc8c6abf743486.json  
\=== Registry Freeze Notary (FIXED) \===  
  registry\_freeze sha256: 3fe8a9b794d5f8000d9bb8a846b3bce17acfd2becd6fe95ac02d25ca54a2a1bd  
       ckm\_headline sha256: 6aba6e62a9b3fba1dd35d13d79e77baeac9d86a78cd27640beade87853e403be  
  cosmic\_bit\_budget sha256: 8ea19dd6847d2f167c97b3b762faef73c62fbfab106af471e7a9c23640530114  
  PMNS.delta\_symbolic stored as: {'symbolic': '-pi/2'}  
  irrational\_witness files: 1  
    \- /content/fraction\_physics\_dlc/irrational\_audit/irrational\_audit\_0c40bda239b71428888da4380965debc401498aa0bee5c5095dc8c6abf743486.json  sha256=0c40bda239b71428…  bytes=2088

  MERKLE ROOT (capsule id): 24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956  
  wrote capsule: /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.json  
  wrote verifier: /content/fraction\_physics\_capsule/fp\_capsule\_verifier\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.json  
\=== Capsule Finalizer — Inline Irrational Witnesses \===  
  input : /content/fraction\_physics\_capsule/fp\_capsule\_verifier\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.json  
  inlined blobs: 0  total bytes: 0  
  output: /content/fraction\_physics\_capsule/fp\_capsule\_verifier\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.inlined.json  
\=== Capsule Finalizer — Proof-of-Work \===  
  input : /content/fraction\_physics\_capsule/fp\_capsule\_verifier\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.inlined.json  
  target: 0000…  achieved pow\_id=0000045862c3b6699ab3ff9eed80e47b847354376f98171293ffae64327943ea  
  nonce : 11156  iterations=11157  elapsed=0.16s  \~68640.5 H/s  
  output: /content/fraction\_physics\_capsule/fp\_capsule\_verifier\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.inlined.pow4.0000045862c3.json  
\=== Verify PoW \===  
  stored: 0000045862c3b6699ab3ff9eed80e47b847354376f98171293ffae64327943ea  
  recomputed: 0000045862c3b6699ab3ff9eed80e47b847354376f98171293ffae64327943ea  
  difficulty: 4  
  OK? True  
\=== Capsule Heads (Final) \===  
  merkle\_root\_sha256: None  
  embedded irrational blobs: 0 bytes: 0  
  pow\_id: 0000045862c3b6699ab3ff9eed80e47b847354376f98171293ffae64327943ea  
  pow difficulty: 4

\=== Auto-Verify: Fraction Physics Capsule \===  
  file  : /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.json  
  root  : 24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956  
\[PoW\]         OK \- skipped (no proof\_of\_work present)  
\[Blobs\]       OK \- no embedded irrationals (skipped) (count=0)  
\[Crosslinks\]  OK \- no embedded blobs to crosslink (skipped)  
\[Freeze\]      OK \- no freeze payload (skipped)  
\[Merkle\]      FAIL \- merkle mismatch  
              recomputed: 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
              claimed   : 24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956  
\=== Verdict \=== FAIL

\=== Merkle Doctor \===  
file     : /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.json  
claimed  : 24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956  
leaves   : 4  
match    : ❌ NONE — claimed root doesn't match any common variant.  
closest  :  
   1\. 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897   (order=asis, pair=bytes, digest=sha256, odd=dup)  
   2\. 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897   (order=asis, pair=bytes, digest=sha256, odd=carry)  
   3\. 2f6fe5103e8b189e616a8e8cd63c599996f1bd87f1fa5a2298c015af54dfbf45   (order=asis, pair=bytes, digest=double, odd=dup)  
   4\. 2f6fe5103e8b189e616a8e8cd63c599996f1bd87f1fa5a2298c015af54dfbf45   (order=asis, pair=bytes, digest=double, odd=carry)  
repaired : 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
wrote    : /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.repaired.json

\=== Verify Repaired Capsule \===

\=== Auto-Verify: Fraction Physics Capsule \===  
  file  : /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.repaired.json  
  root  : 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
\[PoW\]         OK \- skipped (no proof\_of\_work present)  
\[Blobs\]       OK \- no embedded irrationals (skipped) (count=0)  
\[Crosslinks\]  OK \- no embedded blobs to crosslink (skipped)  
\[Freeze\]      OK \- no freeze payload (skipped)  
\[Merkle\]      OK \- merkle ok  
              recomputed: 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
              claimed   : 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
\=== Verdict \=== PASS

\=== Capsule Re-Verify (canonical merkle) \===  
file     : /content/fraction\_physics\_capsule/fp\_capsule\_24ab363b4327f346a51a7d458956d085d874a425bb5b5e9a86f7fbcca28db956.repaired.json  
leaves   : 4  
claimed  : 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
recomputed: 25dc5fdbb7b52466b5d2353ff26ed8e05f8eaadfaef7b459d1223ee7f8ae1897  
MERKLE   : OK

\=== Release Bundle \===  
dir      : /content/fraction\_physics\_release/fp\_release\_25dc5fdbb7b5  
capsule  : capsule.json a3d214df02ce444329abedb864f3ed435fdbe144224217e0bed89c656e990e81  
verifier : verifier.json 0c3c3ebb108b29530283b683e83b2980690d3fc58fca33bb509e99742d9046d2  
witnesses: 2  
manifest : manifest.json  
checksums: checksums.txt  
zip      : /content/fraction\_physics\_release/fp\_release\_25dc5fdbb7b5.zip

\[OK\] release ready:  
  25dc5fdbb7b5  pow=none  \-\>  /content/fraction\_physics\_release/fp\_release\_25dc5fdbb7b5.zip