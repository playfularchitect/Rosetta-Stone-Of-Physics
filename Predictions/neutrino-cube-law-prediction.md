# -----------------------------------------------------------
# Neutrino Phenomenology in one cell 
#   - Normal ordering (Δm21, Δm31 > 0)
#   - m_beta, m_betabeta envelope (2 Majorana phases)
#   - prints a 1 meV benchmark and plots envelope vs m1
# -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ----- inputs (oscillation) -----
Dm21 = 7.53e-5   # eV^2  (solar)
Dm31 = 2.453e-3  # eV^2  (atmospheric, NO)

# mixing angles (radians)
th12 = np.radians(33.44)
th13 = np.radians(8.57)

s12, c12 = np.sin(th12), np.cos(th12)
s13, c13 = np.sin(th13), np.cos(th13)

# PMNS electron-row squared moduli
Ue1_2 = (c12*c13)**2
Ue2_2 = (s12*c13)**2
Ue3_2 = (s13)**2
U2 = np.array([Ue1_2, Ue2_2, Ue3_2])

# ----- masses for Normal Ordering (NO) -----
def masses_NO(m1):
    m2 = np.sqrt(m1**2 + Dm21)
    m3 = np.sqrt(m1**2 + Dm31)
    return np.array([m1, m2, m3])

# ----- beta-decay effective mass -----
def m_beta(masses):
    return np.sqrt(np.sum(U2 * masses**2))

# ----- 0νββ envelope over Majorana phases -----
def mbb_envelope(masses, nsamp=2000, seed=0):
    rng = np.random.default_rng(seed)
    a21 = rng.uniform(0, 2*np.pi, size=nsamp)
    a31 = rng.uniform(0, 2*np.pi, size=nsamp)
    A1 = Ue1_2 * masses[0]
    A2 = Ue2_2 * masses[1] * np.exp(1j*a21)
    A3 = Ue3_2 * masses[2] * np.exp(1j*a31)
    vals = np.abs(A1 + A2 + A3)
    return vals.min(), vals.max()

# ----- scan lightest mass m1 -----
m1_vals = np.logspace(-4, -1, 400)  # 0.1 meV → 0.1 eV
mbb_min, mbb_max, sum_m, mb = [], [], [], []

for m1 in m1_vals:
    m = masses_NO(m1)
    mn, mx = mbb_envelope(m, nsamp=4000)
    mbb_min.append(mn); mbb_max.append(mx)
    sum_m.append(m.sum()); mb.append(m_beta(m))

# ----- print a benchmark (m1 = 1 meV) -----
def nearest(arr, x): return arr[np.argmin(np.abs(arr - x))]
m1_bench = nearest(m1_vals, 1e-3)
m_b = masses_NO(m1_bench)
mn, mx = mbb_envelope(m_b, nsamp=20000, seed=1)
print("Normal Ordering, lightest mass ≈ 1 meV")
print(f"m1 = {m_b[0]:.6e} eV | m2 = {m_b[1]:.6e} eV | m3 = {m_b[2]:.6e} eV")
print(f"Σ m_i = {m_b.sum():.6f} eV")
print(f"m_beta = {m_beta(m_b):.6e} eV")
print(f"m_beta_beta range = {mn:.6e} – {mx:.6e} eV")

# ----- plot the 0νββ envelope -----
plt.figure(figsize=(7,5))
plt.semilogy(m1_vals*1e3, mbb_min, label="min mββ (NO)")
plt.semilogy(m1_vals*1e3, mbb_max, label="max mββ (NO)")
plt.xlabel("lightest mass m₁ (meV)")
plt.ylabel(r"$m_{\beta\beta}$ (eV)")
plt.title(r"NO: $m_{\beta\beta}$ envelope vs lightest mass (Δm²: 7.53×10$^{-5}$, 2.453×10$^{-3}$ eV$^2$)")
plt.legend()
plt.tight_layout()
plt.show()
