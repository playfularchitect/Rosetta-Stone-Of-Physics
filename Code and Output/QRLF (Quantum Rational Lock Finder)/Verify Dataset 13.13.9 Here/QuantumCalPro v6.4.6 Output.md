\================================================================================  
QuantumCalPro — v6.4.6 CHAN+MLE (Self-Contained, Notebook-Safe CLI)  
\================================================================================

—— METRICS (from ρ\_psd unless noted) ——  
S (from T) \= 2.2560  \[95% CI: 2.2469, 2.2697\]  (median 2.2587)  
F(Φ⁺) from T-only \= 0.8442   |   F(Φ⁺) from ρ\_psd \= 0.8442   |   F(Φ⁺) from ρ\_mle \= 0.8442  
Concurrence \= 0.6884 (psd) / 0.6884 (mle)   Negativity \= 0.3442 / 0.3442   Purity \= 0.7208 / 0.7208

T (counts) \=   
  \+0.7872 \-0.0004 \-0.0068  
  \+0.0010 \-0.7822 \+0.0014  
  \+0.0046 \-0.0048 \+0.8073

Σ (from SVD of T) \=   
  \+0.8078 \+0.0000 \+0.0000  
  \+0.0000 \+0.7873 \+0.0000  
  \+0.0000 \+0.0000 \-0.7817

T (after frames) \=   
  \+0.8078 \+0.0000 \-0.0000  
  \+0.0000 \+0.7873 \+0.0000  
  \+0.0000 \+0.0000 \-0.7817

—— FRAME QUALITY ——  
Off-diag L2: before=9.680291231677e-03, after=2.641421381926e-16,  Δ=+9.680291231677e-03  
Diag error ‖diag(T\_after) − diag(Σ)‖₂ \= 3.140184917368e-16

—— COMPILER LINES (DECIMAL) ——  
A: Rz(-7.826°) · Rz(+180.000°) · Ry(+82.703°) · Rz(-90.000°)  
B: Rz(+7.826°) · Rz(+180.000°) · Ry(+97.297°) · Rz(-90.000°)

—— COMPILER LINES (SYMBOLIC π-rational, MDL-aware) ——  
A: Rz(-1π/23) · Rz(π) · Ry(17π/37) · Rz(-1π/2)  
B: Rz(1π/23) · Rz(π) · Ry(20π/37) · Rz(-1π/2)

—— SANITY RADAR (singles bias z-scores) ——  
A.X: mean=+0.0015, N=40960, z=+0.30 OK  
A.Y: mean=+0.0025, N=40960, z=+0.50 OK  
A.Z: mean=+0.0011, N=40960, z=+0.22 OK  
B.X: mean=+0.0005, N=40960, z=+0.10 OK  
B.Y: mean=+0.0012, N=40960, z=+0.25 OK  
B.Z: mean=+0.0014, N=40960, z=+0.28 OK

—— CHSH-OPTIMAL SETTINGS (Bloch vectors) ——  
{  
  "Alice": {  
    "a1": \[  
      \-0.05242634785506012,  
      0.11865900494878134,  
      0.991550058542253  
    \],  
    "a2": \[  
      0.9872065480776716,  
      0.15587854378008406,  
      0.0335426746334926  
    \]  
  },  
  "Bob": {  
    "b1": \[  
      \-0.045306580978465266,  
      \-0.12080682290998276,  
      0.9916415810455096  
    \],  
    "b2": \[  
      0.987480038764712,  
      \-0.1555588011013239,  
      0.026165481863737843  
    \]  
  },  
  "S\_pred": 2.255976250818879  
}

—— Rational Error Parameter Search (top π-approximants) ——  
dAz: \~ \-1π/23  
dBz: \~ 1π/23  
Aα: \~ π  
Aβ: \~ 17π/37  
Aγ: \~ \-1π/2  
Bα: \~ π  
Bβ: \~ 20π/37  
Bγ: \~ \-1π/2

—— Integer-relation miner (pairs/triples over π) ——  
  \-8·dAz/π \-8·dBz/π   (resid=0.000e+00)  
  \-7·dAz/π \-7·dBz/π   (resid=0.000e+00)  
  \-6·dAz/π \-6·dBz/π   (resid=0.000e+00)  
  \-5·dAz/π \-5·dBz/π   (resid=0.000e+00)  
  \-4·dAz/π \-4·dBz/π   (resid=0.000e+00)  
  \-3·dAz/π \-3·dBz/π   (resid=0.000e+00)  
  \-2·dAz/π \-2·dBz/π   (resid=0.000e+00)  
  \-1·dAz/π \-1·dBz/π   (resid=0.000e+00)  
  \+1·dAz/π \+1·dBz/π   (resid=0.000e+00)  
  \+2·dAz/π \+2·dBz/π   (resid=0.000e+00)  
  \+3·dAz/π \+3·dBz/π   (resid=0.000e+00)  
  \+4·dAz/π \+4·dBz/π   (resid=0.000e+00)

—— SYMBOLIC PATCH VERIFICATION (IDEAL SIM) ——  
Raw-ideal (diagnostic; inverse patch on perfect |Φ+|):  
T\_verified(raw\_ideal) \=   
  \-0.1432 \+0.1178 \+0.9827  
  \+0.9897 \+0.0090 \+0.1431  
  \-0.0080 \-0.9930 \+0.1178  
Off-diag L2 \= 1.722070282e+00,  diag error vs diag(1,-1,1) \= 1.761597410e+00

Forward-model (misalign then inverse-patch → should be ideal):  
Status: SUCCESS  
T\_verified(forward\_model) \=   
  \+1.0000 \+0.0000 \+0.0000  
  \+0.0000 \-1.0000 \+0.0000  
  \+0.0000 \+0.0000 \+1.0000  
Off-diag L2 \= 0.000000000e+00,  diag error vs diag(1,-1,1) \= 0.000000000e+00

—— LIKELIHOOD MODELS ——  
           zero-singles:  logL=-466832.14,  AIC=933664.28,  BIC=933664.28

—— RESIDUALS (obs − pred) under AIC-best ——  
XX: 00:+1.5137e-03  01:+9.0332e-04  10:-9.0332e-04  11:-1.5137e-03  
XY: 00:-1.9531e-03  01:-1.1475e-03  10:+1.1475e-03  11:+1.9531e-03  
XZ: 00:+1.1841e-03  01:+1.7456e-03  10:-1.7456e-03  11:-1.1841e-03  
YX: 00:+1.5869e-04  01:+1.7700e-03  10:-1.7700e-03  11:-1.5869e-04  
YY: 00:+2.0752e-04  01:-2.0752e-04  10:+2.0752e-04  11:-2.0752e-04  
YZ: 00:+8.4229e-04  01:+9.1553e-04  10:-9.1553e-04  11:-8.4229e-04  
ZX: 00:+1.9287e-03  01:+2.1973e-04  10:-2.1973e-04  11:-1.9287e-03  
ZY: 00:-1.3428e-04  01:-2.3804e-03  10:+2.3804e-03  11:+1.3428e-04  
ZZ: 00:+2.3437e-03  01:-3.6621e-04  10:+3.6621e-04  11:-2.3437e-03

—— TOMOGRAPHY CHECKS ——  
‖T\_meas − T\_from ρ\_lin‖\_F \= 1.110259155e-16   |   ‖T\_meas − T\_from ρ\_psd‖\_F \= 2.840154456e-16  
ρ\_lin min eigenvalue \= \+4.663e-02   (negative mass clipped \= \+0.000e+00)  
‖ρ\_lin − ρ\_psd‖\_F \= 2.506661970e-16

—— MLE TOMOGRAPHY ——  
Converged in 156 iters (Δ=9.816e-11).  
‖T\_meas − T\_from ρ\_mle‖\_F \= 1.828964080e-05  
ρ\_mle vs ρ\_psd: ‖ρ\_mle − ρ\_psd‖\_F \= 9.378509711e-04

—— LOCAL CHANNEL FIT (frames-aligned Σ) ——  
Products (Px,Py,Pz) \= (0.8078, 0.7873, 0.7817)  
Symmetric split per-axis r (A=B): rx=0.8988, ry=0.8873, rz=0.8841  
Depolarizing fit: r=0.8901 ⇒ p\_dep=0.0825, residual=1.187e-04

Done v6.4.6 CHAN+MLE.

\====================================================================================================  
Project1 — Reality Transduction Ledger (ADD-ONLY; runs after v6.4.6 output)  
\====================================================================================================

— METRICS —  
S (from T) \= 2.2560  \[95% CI: 2.2469, 2.2697\]  (median 2.2587)  
F(Φ⁺) from T \= 0.8442   |   F(Φ⁺) from ρ\_psd \= 0.8442  
Concurrence \= 0.6884   Negativity \= 0.3442   Purity \= 0.7208

T (counts) \=   
  \+0.7872 \-0.0004 \-0.0068  
  \+0.0010 \-0.7822 \+0.0014  
  \+0.0046 \-0.0048 \+0.8073

Σ (from SVD of T) \=   
  \+0.8078 \+0.0000 \+0.0000  
  \+0.0000 \+0.7873 \+0.0000  
  \+0.0000 \+0.0000 \-0.7817

T (after frames) \=   
  \+0.8078 \+0.0000 \-0.0000  
  \+0.0000 \+0.7873 \+0.0000  
  \+0.0000 \+0.0000 \-0.7817

— FRAME QUALITY —  
Off-diag L2: before=9.680291231677e-03, after=2.641421381926e-16,  Δ=+9.680291231677e-03  
Diag error ‖diag(T\_after) − diag(Σ)‖₂ \= 3.140184917368e-16

— SO(3) FRAME ZYZ (best π-rational approx, max\_den=41) —  
RA ≈ ZYZ: \-14π/31, 17π/37, \-π  
RB ≈ ZYZ: \-17π/31, 20π/37, \-π

— DYADIC TRANSDUCTION (even-parity probabilities per measured basis, z=2.24) —  
pair   N        E         p\_even     CI\_lo     CI\_hi   nearest(ALL) MDL\*  Δ      HIT  nearest(TINY) MDL\*  Δ      HIT  
XX   40960  \+0.7872  0.893604  0.890142  0.896968        1/4   3  0.6436  0       1/256   9  0.8897  0  
XY   40960  \-0.0004  0.499805  0.494271  0.505338        1/4   3  0.2498  0       1/256   9  0.4959  0  
XZ   40960  \-0.0068  0.496606  0.491073  0.502140        1/4   3  0.2466  0       1/256   9  0.4927  0  
YX   40960  \+0.0010  0.500513  0.494979  0.506046        1/4   3  0.2505  0       1/256   9  0.4966  0  
YY   40960  \-0.7822  0.108911  0.105511  0.112407        1/8   4  0.0161  0       1/256   9  0.1050  0  
YZ   40960  \+0.0014  0.500708  0.495174  0.506242        1/4   3  0.2507  0       1/256   9  0.4968  0  
ZX   40960  \+0.0046  0.502295  0.496761  0.507828        1/4   3  0.2523  0       1/256   9  0.4984  0  
ZY   40960  \-0.0048  0.497583  0.492050  0.503117        1/4   3  0.2476  0       1/256   9  0.4937  0  
ZZ   40960  \+0.8073  0.903662  0.900347  0.906878        1/4   3  0.6537  0       1/256   9  0.8998  0

\[Project1 artifacts\] wrote: proj1\_results/run\_20250907-113158

— COMPLEMENT-AWARE DYADIC TRANSDUCTION (closest in {d, 1−d}) —  
pair   N        p\_even     CI\_lo     CI\_hi   best     MDL\*   value     Δ        z      HIT  
XX   40960  0.893604  0.890142  0.896968    1-1/8    4  0.875000   0.0186  \+12.211  0  
XY   40960  0.499805  0.494271  0.505338      1/4    3  0.250000   0.2498  \+101.114  0  
XZ   40960  0.496606  0.491073  0.502140      1/4    3  0.250000   0.2466  \+99.822  0  
YX   40960  0.500513  0.494979  0.506046    1-1/4    3  0.750000   0.2495  \-100.985  0  
YY   40960  0.108911  0.105511  0.112407      1/8    4  0.125000   0.0161  \-10.452  0  
YZ   40960  0.500708  0.495174  0.506242    1-1/4    3  0.750000   0.2493  \-100.906  0  
ZX   40960  0.502295  0.496761  0.507828    1-1/4    3  0.750000   0.2477  \-100.265  0  
ZY   40960  0.497583  0.492050  0.503117      1/4    3  0.250000   0.2476  \+100.216  0  
ZZ   40960  0.903662  0.900347  0.906878    1-1/8    4  0.875000   0.0287  \+19.660  0  
\[Project1 artifacts+\] wrote: proj1\_results/run\_20250907-113158/sections/dyadic\_transduction\_complement.csv

— BELL-DIAGONAL DECOMPOSITION (lab frame via diag(T)) —  
c\_lab \= (T\_xx, T\_yy, T\_zz) \= (+0.7872, \-0.7822, \+0.8073)  
weights (Φ+, Φ−, Ψ+, Ψ−) \= \['0.8442', '0.0494', '0.0595', '0.0469'\]  
Werner p (Φ+) \= 0.7922  ⇒  S\_pred \= 2.2408,  F\_pred(Φ+) \= 0.8442  
Measured:  S \= 2.2560,  F(Φ⁺) \= 0.8442,  Tsirelson gap Δ \= 0.572451

— BELL-DIAGONAL DECOMPOSITION (SVD frame via diag(T\_after)) —  
c\_svd \= (Σ\_x, Σ\_y, Σ\_z) \= (+0.8078, \+0.7873, \-0.7817)  
weights (Φ+, Φ−, Ψ+, Ψ−) \= \['0.0597', '0.8442', '0.0495', '0.0466'\]  
Werner p (Φ+) \= \-0.2537  ⇒  S\_pred \= \-0.7177,  F\_pred(Φ+) \= 0.0597  
Measured S \= 2.2560,  Tsirelson gap Δ \= 0.572451  
\[Project1 artifacts++\] wrote: proj1\_results/run\_20250907-113158/sections/bell\_mixture\_lab.csv  
\[Project1 artifacts++\] wrote: proj1\_results/run\_20250907-113158/sections/bell\_mixture\_svd.csv  
\[Project1 artifacts++\] wrote: proj1\_results/run\_20250907-113158/qasm/inverse\_patch.qasm  
\[Project1 artifacts++\] wrote: proj1\_results/run\_20250907-113158/qasm/misalign\_then\_inverse.qasm  
\[Project1 artifacts+++\] wrote: proj1\_results/run\_20250907-113158/sections/model\_selection\_fixed\_frames.csv  
\[Project1 artifacts+++\] wrote: proj1\_results/run\_20250907-113158/sections/per\_basis\_KL\_Werner.csv  
\[Project1 artifacts+++\] wrote: proj1\_results/run\_20250907-113158/sections/per\_basis\_KL\_BellDiag3.csv

— MODEL SELECTION (fixed symbolic frames) —  
Werner(p):        p\_hat=-0.008459   logL=-511039.16   AIC=1022080.31   BIC=1022091.13   raw-2logL=1022078.31  
BellDiag(3c):     c\_hat=(+0.088197,-0.003548,-0.117125)   logL=-510604.99   AIC=1021215.99   BIC=1021248.44   raw-2logL=1021209.99  
Saturated (ZS):   logL=-466832.14   AIC=933682.28   BIC=933779.64   raw-2logL=933664.28

Winner by AIC: ZS   |   Winner by BIC: ZS

— PER-BASIS KL (obs||model) —  
pair  |  KL\_Werner            |  KL\_BellDiag3        |  note  
XX   |  0.353300 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |  0.355891 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |    
XY   |  0.000021 \#\#                                       |  0.000095 \#\#\#\#                                     |  \<-- Werner fits better  
XZ   |  0.000019 \#\#                                       |  0.000071 \#\#\#                                      |  \<-- Werner fits better  
YX   |  0.000057 \#\#\#                                      |  0.006516 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#         |  \<-- Werner fits better  
YY   |  0.348856 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |  0.328392 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |    
YZ   |  0.000010 \#                                        |  0.000136 \#\#\#\#\#                                    |  \<-- Werner fits better  
ZX   |  0.000025 \#\#                                       |  0.000252 \#\#\#\#\#\#                                   |  \<-- Werner fits better  
ZY   |  0.000110 \#\#\#\#                                     |  0.003157 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#                   |  \<-- Werner fits better  
ZZ   |  0.377006 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |  0.374294 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# |    
\[Project1 artifacts+++\] wrote: proj1\_results/run\_20250907-113158/sections/model\_selection\_snapshot.json  
\[Project1 artifacts++++\] wrote: proj1\_results/run\_20250907-113158/sections/pred\_counts\_Werner.csv  
\[Project1 artifacts++++\] wrote: proj1\_results/run\_20250907-113158/sections/pred\_counts\_BellDiag3.csv  
\[Project1 artifacts++++\] wrote: proj1\_results/run\_20250907-113158/sections/werner\_posterior\_grid.csv

— WERNER POSTERIOR (uniform prior on \[-1,1\]) —  
p\_mode=-0.008000,  p\_mean=-0.008459,  95% CI=\[-0.014605, \-0.003318\],  95% HPD=\[-0.014000, \-0.002000\],  SE≈0.002852

— DEVIANCE GOF vs Saturated zero-singles —  
Werner:    dev=88414.05  df=8  p≈0.000e+00  
BellDiag3: dev=87545.70  df=6  p≈0.000e+00

— FRAME DELTA (Symbolic → SVD) —  
Alice ΔZYZ: \-18π/19, 24π/41, 22π/39    (deg: \-170.438°, \+105.337°, \+101.574°)  
Bob   ΔZYZ: 33π/34, 15π/37, 11π/23    (deg: \+174.631°, \+72.726°, \+86.090°)  
\[Project1 artifacts++++\] wrote: proj1\_results/run\_20250907-113158/sections/werner\_gof\_frames\_summary.json  
\[compat\] numpy.trapz → numpy.trapezoid alias active; DeprecationWarning silenced.  
\[Project1 artifacts+++++\] wrote: proj1\_results/run\_20250907-113158/sections/bootstrap\_deltaAIC\_under\_Werner.csv  
\[Project1 artifacts+++++\] wrote: proj1\_results/run\_20250907-113158/sections/bootstrap\_deltaAIC\_under\_Werner\_summary.json  
\[Project1 artifacts+++++\] wrote: proj1\_results/run\_20250907-113158/sections/bootstrap\_deltaAIC\_under\_BellDiag3.csv  
\[Project1 artifacts+++++\] wrote: proj1\_results/run\_20250907-113158/sections/bootstrap\_deltaAIC\_under\_BellDiag3\_summary.json  
— PARAMETRIC BOOTSTRAP ΔAIC — done (B=200 per generator).  
\[Project1 artifacts++++++\] wrote: proj1\_results/run\_20250907-113158/sections/parametric\_S\_under\_Werner.json  
\[Project1 artifacts++++++\] wrote: proj1\_results/run\_20250907-113158/sections/parametric\_S\_under\_BellDiag3.json  
\[Project1 artifacts++++++\] wrote: proj1\_results/run\_20250907-113158/sections/ppc\_kl\_per\_basis\_Werner.csv  
\[Project1 artifacts++++++\] wrote: proj1\_results/run\_20250907-113158/sections/ppc\_kl\_per\_basis\_BellDiag3.csv  
\[Project1 artifacts++++++\] wrote: proj1\_results/run\_20250907-113158/sections/snap\_project1.json  
— PARAMETRIC S intervals \+ PPC KL \+ SNAPSHOT — done.