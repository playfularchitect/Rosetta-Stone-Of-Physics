# Two-Shell α — Verification & Packaging Report

_Generated: 2025-09-10 14:26:15 UTC_

## 1) Inputs discovered

- `angles_49_007.csv`
- `angles_49_236.csv`
- `angles_50_017.csv`
- `angles_50_055.csv`
- `angles_50_345.csv`
- `denominator.json`
- `pauli_integrals.json`
- `pauli_tilewise_bounds.json`
- `alpha_prediction.txt`

## 2) Angle table row-sum checks (per source-type)

| file | ∑ total | OK? | ∑ cos·count | error vs 1 |
|---|---:|:---:|---:|---:|
| `angles_49_007.csv` | 137 | ✅ | 1 | 0 |
| `angles_49_236.csv` | 137 | ✅ | 1 | 0 |
| `angles_50_017.csv` | 137 | ✅ | 1 | 0 |
| `angles_50_055.csv` | 137 | ✅ | 1 | 0 |
| `angles_50_345.csv` | 137 | ✅ | 1 | 0 |

_Expected: ∑ total = 137 and ∑ cos·count = 1 for each source-type (non-backtracking row identities)._ 

## 3) Denominator identity

- `sum_NB_cos2` = **6210** (raw = 6209.999999999871)

## 4) Recomputed from artifacts (no heavy numerics)

- `total_contrib` (∑ contrib_sum over all tables) = **9.5067061708991343**
- `c_Pauli` = total_contrib / sum_NB_cos2 = **0.0015308705589210**
- `alpha^-1` = 137 + c/137 = **137.000011174238**

Per-table contributions to `total_contrib`:

- 49_007: 1.901341234180
- 49_236: 1.901341234180
- 50_017: 1.901341234180
- 50_055: 1.901341234180
- 50_345: 1.901341234180

If present in `alpha_prediction.txt`:

- reported `c_Pauli` = 0.001530870559
- reported `alpha^-1` = 137.000011174238

## 5) Certified intervals (tile-wise bounds → α^{-1})

- Continuum `c` ∈ [-0.0645410168, 0.0660898001] ⇒ `alpha^-1` ∈ [136.9995288977, 137.0004824073]
- Lattice via [1, π^2/4] `c` ∈ [-0.1592485759, 0.1630700454] ⇒ `alpha^-1` ∈ [136.9988376016, 137.0011902923]
- Heuristic (continuum) `alpha^-1` ∈ [136.9999251887, 137.0000971359]
- Heuristic (lattice)   `alpha^-1` ∈ [136.9998154105, 137.0002396732]

## 6) SHA-256 checksums (bytes, filename)

a3a13fce87606a46b8e4f36eb5ed886f189e8224214528ff2ea056e6554e734c         604  alpha_prediction.txt
14d392f95d51de8ad0aa703659d4c50af8c88fde0493200b69af9b0e7035edc8         646  angles_49_007.csv
6664741d05334b03945f970fc46d80b122d4e379d661f4bc298a35057a3eaa95        3013  angles_49_236.csv
b113fb7988123984c80bd7f1addad3cb59be4ae437ea5c7ae7021132ed8965fe        2286  angles_50_017.csv
c50e2296d0b9853f475494e4dd59a1c6f119a5b853923bac7209cbaa0a4296fd        1040  angles_50_055.csv
86778f54210242c72d5435930a759625aeda0cf80ddb0e91a53b03ec950fb7cd        3276  angles_50_345.csv
fa4dedb29cc2effd36298840130d12138116971d56d7b3ed35297f05cbc0be06          53  denominator.json
e42b427de4c4590a7c6b28241a230bb34c84b7c72acc587debaab1624740fb98       68663  pauli_integrals.json
c411e5120132bc05f770e61daf9d1a60c654e285f522cd02e1a4b000b9ca0f4e       84439  pauli_tilewise_bounds.json

> This report is auto-generated from the released artifacts only. It recomputes `c_Pauli` and `alpha^{-1}`
> without any re-integration, and verifies the structural identities implied by the two-shell geometry.
