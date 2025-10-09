# Two-Shell NB Variance: MHA, T₂=0 Theorem, Odd-Parity Suppression, and Constant Pack

**Object:** union of two integer-lattice shells on \(\mathbb Z^3\): \(S = S_R \cup S_{R+1}\).
**Rule:** Non‑backtracking (NB) row‑sum variance of Legendre \(P_\ell\), with optional same/cross‑shell masks.
No tunable parameters.

## 1) MHA selection (even \(\ell=4\))
- Baseline is the pair with smaller \(T_4\): (\(49,50\)).
- \(T_4\) baseline value = 1.252622401165

## 2) T₂ = 0 theorem (quadrupole cancellation)
- Using second‑moment isotropy: \(M=\sum_t \hat t\hat t^T = (d/3)I\) and NB exclusion of \(t=-s\),
  \(\Xi_2(s)\) is constant in \(s\), hence \(T_2=0\) exactly.

## 3) Odd‑parity suppression (all odd \(\ell\))
- For sets closed under inversion (\(S=-S\)) with NB, the odd‑\(\ell\) row sums cancel pairwise: \(P_\ell(-x)=-P_\ell(x)\).
- Numerically: for \(\ell=1,3,5\), all full/same/cross variances are zero for both candidate pairs.

## 4) Constant Pack (baseline)
- **C1** \(=T_4^{\rm cross}/(T_4^{\rm same}+T_4^{\rm cross})\) = 0.291218742359
- **C2** \(=T_6/(T_4+T_6)\) = 0.966082629016
- **C3** \(=T_6^{\rm cross}/(T_4^{\rm cross}+T_6^{\rm cross})\) = 0.541945035734
- **Even spectral weights**:  w4=0.028201197469, w6=0.803266473856, w8=0.168532328675
- **k68** \(=T_6/T_8\) = 4.766245622859
