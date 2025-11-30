# =============================================================================
# ULTRA v2 — SELF-CONTAINED TRILLION-SCALE NULL TEST (uses your kernel TU)
# Vivi The Physics Slayer! x Evan V4 — append-only, monolithic LEGO block
# =============================================================================
import os, sys, math, ctypes, subprocess, tempfile, glob

print("\n" + "="*120)
print("ULTRA v2: self-contained, trillion-scale, 256-bucket, your-kernel env".center(120))
print("="*120 + "\n")

# ---------------------------
# Config: your kernel path + run knobs
# ---------------------------
KERNEL_PATH = "/mnt/data/kcrt_k38_8b.cpp"  # <- your fast kernel TU
TOTAL_NULL  = 3_000_000_000            # <- your flex (3 trillion)
EPS_SCALES  = (0.5, 1.0, 2.0)
SEED        = 137
OMP_THREADS = 0  # set >0 to pin threads (e.g., 16). 0 = use OpenMP default.

# Optional OpenMP pinning
# os.environ["OMP_PROC_BIND"] = "true"
# os.environ["OMP_PLACES"]    = "cores"
if OMP_THREADS > 0:
    os.environ["OMP_NUM_THREADS"] = str(int(OMP_THREADS))

# ---------------------------
# Registry: 19 params, 8 shapes, eps_real
# ---------------------------
PARAMS = [
    ("CKM","CKM_s12",0.224299998),("CKM","CKM_s13",0.00394),("CKM","CKM_s23",0.042200001),
    ("CKM","CKM_delta_over_pi",0.381971862),("COUPLINGS","alpha",0.007297353),("COUPLINGS","alpha_s_MZ",0.117899999),
    ("COUPLINGS","sin2_thetaW",0.231220001),("EW","MW_over_v",0.326452417),("EW","MZ_over_v",0.370350617),
    ("HIGGS","MH_over_v",0.508692139),("LEPTON_YUKAWA","me_over_v",2.075378e-6),("LEPTON_YUKAWA","mmu_over_v",0.0004291224),
    ("LEPTON_YUKAWA","mtau_over_v",0.007216565),("QUARK_HEAVY","mb_over_v",0.016976712),("QUARK_HEAVY","mc_over_v",0.005157996),
    ("QUARK_HEAVY","mt_over_v",0.701365635),("QUARK_LIGHT","md_over_v",1.8967e-5),("QUARK_LIGHT","ms_over_v",0.000377712),
    ("QUARK_LIGHT","mu_over_v",8.773e-6),
]
BITS_FLOAT     = 53
BASELINE_MDL   = len(PARAMS) * BITS_FLOAT  # 1007
param_index    = {(g,n): i for i,(g,n,_) in enumerate(PARAMS)}
real_values    = [v for _,_,v in PARAMS]

SHAPES = [
    ("CKM_s12_shape",("CKM","CKM_s12"),1/5,4),
    ("CKM_delta_over_pi_shape",("CKM","CKM_delta_over_pi"),3/8,6),
    ("alpha_s_MZ_shape",("COUPLINGS","alpha_s_MZ"),1/8,5),
    ("sin2_thetaW_shape",("COUPLINGS","sin2_thetaW"),1/4,4),
    ("MW_over_v_shape",("EW","MW_over_v"),1/3,3),
    ("MZ_over_v_shape",("EW","MZ_over_v"),3/8,6),
    ("MH_over_v_shape",("HIGGS","MH_over_v"),1/2,3),
    ("mt_over_v_shape",("QUARK_HEAVY","mt_over_v"),5/7,6),
]

shape_vals = []
shape_bits = []
eps_abs    = []
for _, key, sval, sbits in SHAPES:
    idx = param_index[key]
    eps = real_values[idx]/sval - 1.0
    shape_vals.append(float(sval))
    shape_bits.append(int(sbits))
    eps_abs.append(abs(eps))

# Precompute deltas vs float encoding (negative numbers)
delta_bits = [b - BITS_FLOAT for b in shape_bits]  # each ≤ 0

# ---------------------------
# Build (or reuse) the ultra 256-bucket shared library
# ---------------------------
def build_or_load_ultra_lib():
    # Try to reuse any previous build
    cands = sorted(glob.glob("/tmp/ultra_null_*/libultra_codes.so"))
    for p in reversed(cands):
        try:
            L = ctypes.CDLL(p)
            getattr(L, "ew_stream_codes_256")
            print(">>> Reusing existing ultra lib:", p)
            return L
        except Exception:
            pass

    if not os.path.exists(KERNEL_PATH):
        raise FileNotFoundError(f"Kernel not found at {KERNEL_PATH}")

    code = r'''
#define main K38_DISABLED
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <algorithm>
#ifdef _OPENMP
  #include <omp.h>
#endif
// Include your TU so we compile under the same AVX2/OpenMP environment.
#include "%s"

// SplitMix64 → double in [0,1) with 53 bits
static inline uint64_t splitmix64_next(uint64_t &x){
  x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = x; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline double u01_53(uint64_t &s){ uint64_t r=splitmix64_next(s); return (double)(r>>11) * (1.0/9007199254740992.0); }
static inline int popc8(unsigned x){ return __builtin_popcount(x & 0xFFu); }

// Stream N universes → 256 code counts (exact) and snapped-counts [0..8]
extern "C" void ew_stream_codes_256(
    unsigned long long total,
    const double* low8, const double* high8,
    unsigned long long seed,
    unsigned long long* out_codes256,  // [256]
    unsigned long long* out_snaps9     // [9]
){
  int P = 1;
  #ifdef _OPENMP
    P = omp_get_max_threads();
  #endif
  std::vector< std::vector<unsigned long long> > C(P, std::vector<unsigned long long>(256,0ULL));
  std::vector< std::vector<unsigned long long> > S(P, std::vector<unsigned long long>(9,0ULL));

  #pragma omp parallel
  {
    int tid=0;
    #ifdef _OPENMP
      tid = omp_get_thread_num();
    #endif
    unsigned long long start = (total * (unsigned long long)tid) / (unsigned long long)P;
    unsigned long long end   = (total * (unsigned long long)(tid+1)) / (unsigned long long)P;

    // eight independent splitmix streams per thread
    uint64_t st[8];
    for(int k=0;k<8;++k){
      uint64_t base = seed ^ (0xD3C6A7B5C4E3F291ULL + (unsigned long long)tid*0x9E3779B97F4A7C15ULL + (unsigned long long)k*0xBF58476D1CE4E5B9ULL);
      st[k]=base; (void)splitmix64_next(st[k]); (void)splitmix64_next(st[k]);
    }

    for(unsigned long long i=start; i<end; ++i){
      double v0=u01_53(st[0]), v1=u01_53(st[1]), v2=u01_53(st[2]), v3=u01_53(st[3]);
      double v4=u01_53(st[4]), v5=u01_53(st[5]), v6=u01_53(st[6]), v7=u01_53(st[7]);

      unsigned code=0u;
      code |= (v0>=low8[0] && v0<=high8[0]) ? (1u<<0) : 0u;
      code |= (v1>=low8[1] && v1<=high8[1]) ? (1u<<1) : 0u;
      code |= (v2>=low8[2] && v2<=high8[2]) ? (1u<<2) : 0u;
      code |= (v3>=low8[3] && v3<=high8[3]) ? (1u<<3) : 0u;
      code |= (v4>=low8[4] && v4<=high8[4]) ? (1u<<4) : 0u;
      code |= (v5>=low8[5] && v5<=high8[5]) ? (1u<<5) : 0u;
      code |= (v6>=low8[6] && v6<=high8[6]) ? (1u<<6) : 0u;
      code |= (v7>=low8[7] && v7<=high8[7]) ? (1u<<7) : 0u;

      C[tid][code]++; S[tid][ popc8(code) ]++;
    }
  }

  for(int t=0;t<P;++t){
    for(int c=0;c<256;++c) out_codes256[c] += C[t][c];
    for(int s=0;s<9;++s)   out_snaps9[s]   += S[t][s];
  }
}
''' % (KERNEL_PATH.replace("\\","\\\\"))

    work = tempfile.mkdtemp(prefix="ultra_null_")
    cpp_path = os.path.join(work, "ultra_codes.cpp")
    so_path  = os.path.join(work, "libultra_codes.so")
    with open(cpp_path, "w") as f: f.write(code)
    cmd = ["g++","-std=c++17","-O3","-Ofast","-funroll-loops","-fPIC","-shared","-fopenmp","-march=native","-mtune=native", cpp_path, "-o", so_path]
    print(">>> Compiling ultra kernel:\n   "," ".join(cmd))
    subprocess.check_call(cmd)
    print("\n>>> Ultra kernel built at:", so_path, "\n")
    L = ctypes.CDLL(so_path)
    getattr(L, "ew_stream_codes_256")
    return L

lib = build_or_load_ultra_lib()
lib.ew_stream_codes_256.argtypes = [
    ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
]
lib.ew_stream_codes_256.restype = None

# ---------------------------
# Helpers: bounds, MDL per code, summarizer
# ---------------------------
def bounds_for_scale(scale):
    lo=[]; hi=[]
    for k in range(8):
        w = (scale*eps_abs[k])*(1.0+1e-12)
        lo.append(shape_vals[k]*(1.0-w))
        hi.append(shape_vals[k]*(1.0+w))
    return (ctypes.c_double*8)(*lo), (ctypes.c_double*8)(*hi)

# exact MDL for each 8-bit snap code
MDL_BY_CODE = [0.0]*256
for c in range(256):
    s = float(BASELINE_MDL)
    for k in range(8):
        if (c>>k)&1: s += delta_bits[k]
    MDL_BY_CODE[c] = s
CODES_SORTED = sorted(range(256), key=lambda c: MDL_BY_CODE[c])

def summarize_counts(code_counts, mdl_real):
    total = sum(code_counts)
    gmin = next(MDL_BY_CODE[c] for c in CODES_SORTED if code_counts[c]>0)
    gmax = next(MDL_BY_CODE[c] for c in reversed(CODES_SORTED) if code_counts[c]>0)
    s1=s2=0.0
    for c in range(256):
        cnt = code_counts[c]
        if cnt:
            x = MDL_BY_CODE[c]
            s1 += x*cnt
            s2 += (x*x)*cnt
    mean = s1/total if total else float('nan')
    var  = max(0.0, s2/total - mean*mean)
    std  = math.sqrt(var)

    def quantile(p):
        if total==0: return float('nan')
        rank = int(round(p*(total-1)))
        cum = 0
        for c in CODES_SORTED:
            cnt = code_counts[c]
            if rank < cum + cnt:
                return MDL_BY_CODE[c]
            cum += cnt
        return MDL_BY_CODE[CODES_SORTED[-1]]

    p5,p25,p50,p75,p95 = quantile(0.05), quantile(0.25), quantile(0.50), quantile(0.75), quantile(0.95)
    n_better = sum(code_counts[c] for c in range(256) if MDL_BY_CODE[c] <= mdl_real)
    p_emp = n_better/total if total else float('nan')
    return gmin,gmax,mean,std,(p5,p25,p50,p75,p95),p_emp

# ---------------------------
# Runner
# ---------------------------
print("="*120)
print("RATIO_OS_EW_SHAPE_NULLTEST_v3 — ULTRA v2 (self-contained)".center(120))
print("="*120)
print(f"#params                         : {len(PARAMS)}")
print(f"#SHAPE defs                     : {len(SHAPES)}")
print(f"Baseline all-float MDL          : {BASELINE_MDL:.1f} bits")
print(f"Null universes per scale        : {TOTAL_NULL:,d}")
print(f"ε-scales tested                 : {EPS_SCALES}")
print("")

for scale in EPS_SCALES:
    lo8, hi8 = bounds_for_scale(scale)

    # Real-universe MDL under this ε (exact snap check on the 8 shaped slots)
    mdl_real = float(BASELINE_MDL); snapped_real = 0
    for k, (_, key, sval, sbits) in enumerate(SHAPES):
        w  = (scale*eps_abs[k])*(1.0+1e-12)
        lo = sval*(1.0 - w)
        hi = sval*(1.0 + w)
        rv = real_values[param_index[key]]
        if lo <= rv <= hi:
            mdl_real += (sbits - BITS_FLOAT)
            snapped_real += 1

    # Integer seed mix per-scale (no float XOR)
    scale_tag = int(round(scale * 1_000_000))
    seed64 = ((scale_tag * 1315423911) ^ int(SEED)) & 0xFFFFFFFFFFFFFFFF

    # stream with the C++ ultra kernel
    codes256 = (ctypes.c_ulonglong * 256)()
    snaps9   = (ctypes.c_ulonglong * 9)()
    lib.ew_stream_codes_256(
        ctypes.c_ulonglong(int(TOTAL_NULL)),
        lo8, hi8,
        ctypes.c_ulonglong(seed64),
        codes256, snaps9
    )
    counts = [codes256[i] for i in range(256)]
    snaps  = [snaps9[i] for i in range(9)]

    gmin,gmax,mean,std,quants,p_emp = summarize_counts(counts, mdl_real)
    p5,p25,p50,p75,p95 = quants

    print("-"*120)
    print(f"[ε-scale = {scale:.3f}]")
    print("[Real universe vs SHAPE]")
    print(f"  MDL_real (bits)               : {mdl_real:.1f}")
    print(f"  Compression factor            : {mdl_real/BASELINE_MDL:.3f}")
    print(f"  Snapped params (real)         : {snapped_real}")
    print("")
    print("[Null ensemble stats — exact from 256 buckets]")
    print(f"  min MDL                       : {gmin:.1f}")
    print(f"  5th percentile                : {p5:.1f}")
    print(f"  25th percentile               : {p25:.1f}")
    print(f"  median                        : {p50:.1f}")
    print(f"  75th percentile               : {p75:.1f}")
    print(f"  95th percentile               : {p95:.1f}")
    print(f"  max MDL                       : {gmax:.1f}")
    print(f"  mean MDL                      : {mean:.1f}")
    print(f"  std(MDL)                      : {std:.3f}")
    print("")
    print("[Significance vs null]")
    print(f"  Real MDL                      : {mdl_real:.1f}")
    print(f"  Empirical p-value             : p ≈ {p_emp:.6g}")
    print("")
    print("[Snapped-parameter counts in null ensemble]")
    for k in range(9):
        print(f"  snapped = {k:2d}   freq = {snaps[k]}")
    print("")

print("="*120)
print("Done: ULTRA v2 — self-contained".center(120))
print("="*120)





# =============================================================================
# SIGMA ADD-ON — print rough z-significance like the old module
# Vivi The Physics Slayer! V4 — append-only, reuses the ultra 256-bucket lib
# =============================================================================
import os, math, glob, ctypes

# ----- knobs -----
TOTAL_NULL  = 3_000_000_000       # set to your massive N if you want (e.g., 3_000_000_000_000)
EPS_SCALES  = (0.5, 1.0, 2.0)
SEED        = 137
OMP_THREADS = 0                   # set >0 to pin threads

if OMP_THREADS > 0:
    os.environ["OMP_NUM_THREADS"] = str(int(OMP_THREADS))

# ----- reuse the already-built ultra library -----
def _load_ultra_lib():
    for p in sorted(glob.glob("/tmp/ultra_null_*/libultra_codes.so"))[::-1]:
        try:
            L = ctypes.CDLL(p); getattr(L, "ew_stream_codes_256"); return L
        except Exception: pass
    raise RuntimeError("Ultra lib not found — run the ULTRA v2 cell once first.")

lib = _load_ultra_lib()
lib.ew_stream_codes_256.argtypes = [
    ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
]
lib.ew_stream_codes_256.restype = None

# ----- minimal registry (same numbers as before) -----
PARAMS = [
    ("CKM","CKM_s12",0.224299998),("CKM","CKM_s13",0.00394),("CKM","CKM_s23",0.042200001),
    ("CKM","CKM_delta_over_pi",0.381971862),("COUPLINGS","alpha",0.007297353),("COUPLINGS","alpha_s_MZ",0.117899999),
    ("COUPLINGS","sin2_thetaW",0.231220001),("EW","MW_over_v",0.326452417),("EW","MZ_over_v",0.370350617),
    ("HIGGS","MH_over_v",0.508692139),("LEPTON_YUKAWA","me_over_v",2.075378e-6),("LEPTON_YUKAWA","mmu_over_v",0.0004291224),
    ("LEPTON_YUKAWA","mtau_over_v",0.007216565),("QUARK_HEAVY","mb_over_v",0.016976712),("QUARK_HEAVY","mc_over_v",0.005157996),
    ("QUARK_HEAVY","mt_over_v",0.701365635),("QUARK_LIGHT","md_over_v",1.8967e-5),("QUARK_LIGHT","ms_over_v",0.000377712),
    ("QUARK_LIGHT","mu_over_v",8.773e-6),
]
BITS_FLOAT   = 53
BASELINE_MDL = len(PARAMS)*BITS_FLOAT
param_index  = {(g,n): i for i,(g,n,_) in enumerate(PARAMS)}
real_values  = [v for _,_,v in PARAMS]
SHAPES = [
    ("CKM_s12_shape",("CKM","CKM_s12"),1/5,4),
    ("CKM_delta_over_pi_shape",("CKM","CKM_delta_over_pi"),3/8,6),
    ("alpha_s_MZ_shape",("COUPLINGS","alpha_s_MZ"),1/8,5),
    ("sin2_thetaW_shape",("COUPLINGS","sin2_thetaW"),1/4,4),
    ("MW_over_v_shape",("EW","MW_over_v"),1/3,3),
    ("MZ_over_v_shape",("EW","MZ_over_v"),3/8,6),
    ("MH_over_v_shape",("HIGGS","MH_over_v"),1/2,3),
    ("mt_over_v_shape",("QUARK_HEAVY","mt_over_v"),5/7,6),
]
shape_vals=[]; shape_bits=[]; eps_abs=[]
for _, key, sval, sbits in SHAPES:
    idx = param_index[key]
    eps_abs.append(abs(real_values[idx]/sval - 1.0))
    shape_vals.append(float(sval))
    shape_bits.append(int(sbits))
delta_bits = [b - BITS_FLOAT for b in shape_bits]    # ≤ 0

# ----- helpers -----
def bounds_for_scale(scale):
    lo=[]; hi=[]
    for k in range(8):
        w = (scale*eps_abs[k])*(1.0+1e-12)
        lo.append(shape_vals[k]*(1.0-w))
        hi.append(shape_vals[k]*(1.0+w))
    return (ctypes.c_double*8)(*lo), (ctypes.c_double*8)(*hi)

# MDL per 8-bit snap code (256 buckets)
MDL_BY_CODE = [0.0]*256
for c in range(256):
    s = float(BASELINE_MDL)
    for k in range(8):
        if (c>>k)&1: s += delta_bits[k]
    MDL_BY_CODE[c] = s
CODES_SORTED = sorted(range(256), key=lambda c: MDL_BY_CODE[c])

def summarize(code_counts, mdl_real):
    total = sum(code_counts)
    # exact mean/var (population) from 256 buckets
    s1=s2=0.0
    for c in range(256):
        cnt = code_counts[c]
        if cnt:
            x = MDL_BY_CODE[c]
            s1 += x*cnt
            s2 += (x*x)*cnt
    mean = s1/total if total else float('nan')
    var_pop = max(0.0, s2/total - mean*mean)
    std_pop = math.sqrt(var_pop)
    # rough z like before:
    #   z ≈ (mean_null - MDL_real) / std_null    (population std)
    z = (mean - mdl_real)/std_pop if std_pop>0 else float('inf')
    return mean, std_pop, z

print("\n" + "="*120)
print("RATIO_OS_EW_SHAPE_NULLTEST_v3 — ULTRA σ-significance add-on".center(120))
print("="*120)
print(f"#params                         : {len(PARAMS)}")
print(f"#SHAPE defs                     : {len(SHAPES)}")
print(f"Baseline all-float MDL          : {BASELINE_MDL:.1f} bits")
print(f"Null universes per scale        : {TOTAL_NULL:,d}")
print(f"ε-scales tested                 : {EPS_SCALES}\n")

for scale in EPS_SCALES:
    lo8, hi8 = bounds_for_scale(scale)

    # real-universe MDL under this ε
    mdl_real = float(BASELINE_MDL); snapped_real = 0
    for k, (_, key, sval, sbits) in enumerate(SHAPES):
        w=(scale*eps_abs[k])*(1.0+1e-12); lo=sval*(1.0-w); hi=sval*(1.0+w)
        rv = real_values[param_index[key]]
        if lo <= rv <= hi:
            mdl_real += (sbits - BITS_FLOAT)
            snapped_real += 1

    # integer seed mix
    scale_tag = int(round(scale*1_000_000))
    seed64 = ((scale_tag*1315423911) ^ int(SEED)) & 0xFFFFFFFFFFFFFFFF

    # stream counts
    codes256 = (ctypes.c_ulonglong * 256)()
    snaps9   = (ctypes.c_ulonglong * 9)()
    lib.ew_stream_codes_256(
        ctypes.c_ulonglong(int(TOTAL_NULL)),
        lo8, hi8,
        ctypes.c_ulonglong(seed64),
        codes256, snaps9
    )
    counts = [codes256[i] for i in range(256)]

    mean, std_pop, z = summarize(counts, mdl_real)

    print("-"*120)
    print(f"[ε-scale = {scale:.3f}]")
    print(f"  Real MDL                      : {mdl_real:.1f} bits   (snapped params = {snapped_real})")
    print(f"  mean_null(MDL)                : {mean:.2f} bits")
    print(f"  std_null(MDL)  (population)   : {std_pop:.3f} bits")
    print(f"  σ-significance (rough, z)     : z ≈ {z:.2f}")
    print("")
print("="*120)
print("Done: σ-significance add-on".center(120))
print("="*120)
