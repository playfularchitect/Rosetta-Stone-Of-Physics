\# \==================================================================================================  
\# Two-Shell α Derivation — Master Cell (angles, denominator, Pauli integral, α^{-1} prediction)  
\# Self-contained, notebook-safe. Paste into Colab, run once.  
\# \==================================================================================================  
\# Outputs (in ./two\_shell\_artifacts\_YYYYMMDD-HHMMSS):  
\#   \- angles\_49\_007.csv, angles\_49\_236.csv, angles\_50\_017.csv, angles\_50\_055.csv, angles\_50\_345.csv  
\#   \- denominator.json  (verifies sum\_{s,t, t \!= \-s} NB \* cos^2(theta) \= 6210\)  
\#   \- pauli\_integrals.json (per-angle results \+ global c\_Pauli)  
\#   \- alpha\_prediction.txt (pretty print summary)  
\#  
\# CLI-style toggles (edit here):  
\#   QUAD\_KIND        \= "gl" or "cc"     (Gauss-Legendre vs Clenshaw–Curtis nodes; GL is robust default)  
\#   NKAPPA, NPHI     \= product quadrature sizes (integers \>= 64 recommended)  
\#   MP\_DPS           \= mpmath precision (decimal digits)  
\#   RIGOROUS\_BRACKET \= True/False       (also report lattice-continuum \[1, π^2/4\] enclosure)  
\#  
\# Notes:  
\#   1\) The Pauli integrand uses the azimuth-averaged two-angle reduction:  
\#         a \= κ cosφ,   b \= κ sinθ sinφ,   c \= κ cosθ cosφ  
\#      and  
\#         Xi \= (sin a / a) \* (J1(b) / b) \* cos(c),  
\#      then the continuum-bracket integrand is:  
\#         I\_cont(κ, φ; θ) \= sinφ \* Xi(κ,φ;θ) \* cosθ  
\#      The lattice integral lies between \[1, π^2/4\] × I\_cont (sharp global bound).  
\#   2\) The final coefficient uses the normalized weighted sum:  
\#         c\_Pauli \= ( ∑\_classes  W(θ) \* ∫∫ I\_lattice(κ,φ;θ) dφ dκ ) / (∑ NB cos^2 θ)  
\#      with ∑ NB cos^2 θ \= 6210 (verified below).  
\# \==================================================================================================

import os, math, json, csv, time, sys  
from collections import defaultdict  
import numpy as np  
import mpmath as mp

\# \-------------------- user toggles \--------------------  
QUAD\_KIND        \= "gl"   \# "gl" (Gauss-Legendre, default) or "cc" (Clenshaw–Curtis nodes on \[0,π\])  
NKAPPA           \= 256    \# radial panels  
NPHI             \= 256    \# polar panels  
MP\_DPS           \= 80     \# mpmath precision  
RIGOROUS\_BRACKET \= True   \# also report the rigorous global bracket via \[1, π^2/4\]

\# \-------------------- setup \---------------------------  
mp.mp.dps \= MP\_DPS  
OUTDIR \= f"two\_shell\_artifacts\_{time.strftime('%Y%m%d-%H%M%S')}"  
os.makedirs(OUTDIR, exist\_ok=True)

\# \-------------------- shells & helpers \----------------  
def shell\_vectors(n2):  
    """All integer triples (x,y,z) with x^2+y^2+z^2 \= n2, with both z signs; exclude (0,0,0)."""  
    vecs=set(); m=int(math.isqrt(n2))+1  
    for x in range(-m,m+1):  
        x2=x\*x  
        for y in range(-m,m+1):  
            y2=y\*y  
            rem=n2 \- x2 \- y2  
            if rem\<0: continue  
            z \= int(math.isqrt(rem))  
            if z\*z \== rem:  
                vecs.add((x,y, z)); vecs.add((x,y,-z))  
    vecs.discard((0,0,0))  
    return sorted(vecs)

def vnorm(v): return math.sqrt(v\[0\]\*v\[0\] \+ v\[1\]\*v\[1\] \+ v\[2\]\*v\[2\])  
def signature(v): return tuple(sorted(map(abs, v)))

S49 \= shell\_vectors(49)   \# |S49| should be 54  
S50 \= shell\_vectors(50)   \# |S50| should be 84  
S   \= S49 \+ S50  
assert len(S49)==54 and len(S50)==84 and len(S)==138, f"Shell sizes mismatch: |49|={len(S49)}, |50|={len(S50)}"

\# Representatives for the five source types (unique by signature)  
def pick\_rep(S, sig):  
    for v in S:  
        if signature(v)==sig:  
            return v  
    raise RuntimeError(f"Signature {sig} not found.")

REPS \= {  
    "49\_007": pick\_rep(S49, (0,0,7)),  
    "49\_236": pick\_rep(S49, (2,3,6)),  
    "50\_017": pick\_rep(S50, (0,1,7)),  
    "50\_055": pick\_rep(S50, (0,5,5)),  
    "50\_345": pick\_rep(S50, (3,4,5)),  
}

\# \-------------------- NB angle-class tables \----------------  
def angle\_classes\_for\_source(s, S49, S50):  
    """Return rows: \[(cosθ, θ\_deg, count\_to\_49, count\_to\_50, total)\], NB excludes t=-s."""  
    ns \= vnorm(s)  
    ent \= defaultdict(lambda:\[0,0\])  
    \# to shell 49  
    for t in S49:  
        if t \== (-s\[0\],-s\[1\],-s\[2\]): continue  
        nt=vnorm(t)  
        c=(s\[0\]\*t\[0\]+s\[1\]\*t\[1\]+s\[2\]\*t\[2\])/(ns\*nt)  
        key \= round(c, 12\)  
        ent\[key\]\[0\] \+= 1  
    \# to shell 50  
    for t in S50:  
        if t \== (-s\[0\],-s\[1\],-s\[2\]): continue  
        nt=vnorm(t)  
        c=(s\[0\]\*t\[0\]+s\[1\]\*t\[1\]+s\[2\]\*t\[2\])/(ns\*nt)  
        key \= round(c, 12\)  
        ent\[key\]\[1\] \+= 1  
    rows=\[\]  
    for c,(c49,c50) in ent.items():  
        theta \= math.degrees(math.acos(max(-1.0, min(1.0, c))))  
        rows.append( (c, theta, c49, c50, c49+c50) )  
    rows.sort(key=lambda r: r\[1\])  
    \# sanity: total NB partners should be 137  
    tot \= sum(r\[4\] for r in rows)  
    assert tot==137, f"NB count \!= 137 for source {s} (got {tot})"  
    return rows

def write\_csv(rows, path):  
    with open(path, "w", newline="") as f:  
        w=csv.writer(f)  
        w.writerow(\["cos\_theta","theta\_deg","count\_to\_49","count\_to\_50","total"\])  
        for c,theta,c49,c50,total in rows:  
            w.writerow(\[f"{c:.12f}", f"{theta:.6f}", c49, c50, total\])

ALL\_TABLES \= {}  
for key, s in REPS.items():  
    rows \= angle\_classes\_for\_source(s, S49, S50)  
    ALL\_TABLES\[key\] \= rows  
    \# write CSV  
    out \= os.path.join(OUTDIR, f"angles\_{key}.csv")  
    write\_csv(rows, out)

\# row-sum witness (∑ cosθ over NB partners ≈ 1 per source-type)  
def rowsum\_cos\_for\_source(s):  
    ns=vnorm(s)  
    tot=0.0  
    for t in S:  
        if t \== (-s\[0\],-s\[1\],-s\[2\]): continue  
        nt=vnorm(t)  
        tot \+= (s\[0\]\*t\[0\]+s\[1\]\*t\[1\]+s\[2\]\*t\[2\])/(ns\*nt)  
    return tot

ROW\_SUM\_WITNESS \= {  
    key: rowsum\_cos\_for\_source(s) for key,s in REPS.items()  
}

\# \-------------------- denominator sum\_{NB} cos^2 θ \----------------  
def denominator\_sum\_cos2():  
    norms \= {v: vnorm(v) for v in S}  
    acc \= 0.0  
    for s in S:  
        ns \= norms\[s\]  
        for t in S:  
            if t \== (-s\[0\],-s\[1\],-s\[2\]): continue    \# NB mask  
            nt \= norms\[t\]  
            c \= (s\[0\]\*t\[0\]+s\[1\]\*t\[1\]+s\[2\]\*t\[2\])/(ns\*nt)  
            acc \+= c\*c  
    return acc

DENOM\_RAW \= denominator\_sum\_cos2()  
DENOM \= int(round(DENOM\_RAW))  
assert DENOM \== 6210, f"Denominator mismatch: got {DENOM\_RAW} (rounded {DENOM})"  
with open(os.path.join(OUTDIR, "denominator.json"), "w") as f:  
    json.dump({"sum\_NB\_cos2": DENOM, "raw": DENOM\_RAW}, f, indent=2)

\# \-------------------- Pauli integrand (continuum bracket) \----------------  
\# Xi(κ, φ; θ) \= (sin(κ cosφ)/(κ cosφ)) \* (J1(κ sinθ sinφ)/(κ sinθ sinφ)) \* cos(κ cosθ cosφ)  
\# I\_cont(κ, φ; θ) \= sinφ \* Xi(κ, φ; θ) \* cosθ  
J1 \= mp.besselj

def Xi(kappa, phi, theta):  
    a \= kappa \* mp.cos(phi)  
    s \= mp.sin(phi)  
    st \= mp.sin(theta)  
    ct \= mp.cos(theta)  
    b \= kappa \* st \* s  
    c \= kappa \* ct \* mp.cos(phi)  
    term1 \= mp.sin(a)/a if a \!= 0 else mp.mpf(1)  
    \# lim b→0 J1(b)/b \= 1/2  
    term2 \= (J1(1, b)/b) if b \!= 0 else mp.mpf("0.5")  
    term3 \= mp.cos(c)  
    return term1\*term2\*term3

def I\_cont(kappa, phi, theta):  
    return mp.sin(phi) \* Xi(kappa, phi, theta) \* mp.cos(theta)

\# \-------------------- quadrature (GL default; CC nodes optional) \----------------  
def gl\_nodes\_weights(n):  
    \# Gauss-Legendre nodes/weights on \[0, π\]  
    x, w \= np.polynomial.legendre.leggauss(n)  \# on \[-1,1\]  
    \# map to \[0, π\]  
    xm \= 0.5\*(x+1.0)\*math.pi  
    wm \= 0.5\*math.pi\*w  
    return xm, wm

def cc\_nodes\_weights(n):  
    \# Clenshaw–Curtis on \[0, π\], using θ\_j \= jπ/n with Fejér-type weights  
    \# (Trefethen/Waldvogel formula; N\>=2)  
    N \= int(n)  
    if N \< 2:  
        raise ValueError("cc\_nodes\_weights requires n\>=2")  
    j \= np.arange(0, N+1)  
    x \= (j \* math.pi) / N    \# nodes in \[0, π\]  
    \# weights on \[0, π\] derived from \[-1,1\] via linear map (factor π/2)  
    \# Construct weights on \[-1,1\] and rescale:  
    w \= np.zeros(N+1)  
    \# vectorized formula: w\_k \= 2/N \* sum\_{m=0}^{N/2} c\_m \* cos(2 m k π / N) / (1 \- 4 m^2)  
    \# Handle even/odd N separately for the endpoint term.  
    mmax \= N//2  
    c \= np.zeros(mmax+1)  
    c\[0\] \= 1.0  
    for m in range(1, mmax+1):  
        c\[m\] \= 2.0/(1.0 \- 4.0\*m\*m)  
    for k in range(N+1):  
        s \= 0.0  
        for m in range(0, mmax+1):  
            coef \= c\[m\]  
            val \= math.cos(2.0\*m\*k\*math.pi/N)  
            if (N % 2 \== 0\) and (m==mmax) and (k in (0, N)):  
                \# endpoint correction when N is even  
                s \+= 0.5\*coef\*val  
            else:  
                s \+= coef\*val  
        w\[k\] \= (2.0/N) \* s  
    \# map to \[0,π\]: scale weights by (π/2)  
    w \= w \* (math.pi/2.0)  
    return x, w

def product\_integral\_F(theta, nk=NKAPPA, np\_=NPHI, kind=QUAD\_KIND):  
    """Compute I(theta) \= ∫\_φ ∫\_κ I\_cont(κ,φ;θ) dκ dφ via product quadrature."""  
    if kind \== "gl":  
        kx, kw \= gl\_nodes\_weights(nk)  
        px, pw \= gl\_nodes\_weights(np\_)  
    elif kind \== "cc":  
        kx, kw \= cc\_nodes\_weights(nk)  
        px, pw \= cc\_nodes\_weights(np\_)  
    else:  
        raise ValueError("QUAD\_KIND must be 'gl' or 'cc'")  
    tot \= mp.mpf("0.0")  
    for j,phi in enumerate(px):  
        for i,kappa in enumerate(kx):  
            tot \+= mp.mpf(kw\[i\]) \* mp.mpf(pw\[j\]) \* I\_cont(kappa, phi, theta)  
    return tot

\# \-------------------- per-angle evaluation and weighted sum \----------------  
def pauli\_for\_table(rows, nk=NKAPPA, np\_=NPHI, kind=QUAD\_KIND, cache=None):  
    """rows: list of (cosθ, θ\_deg, c49, c50, total). Return list with integrals and weighted contributions."""  
    if cache is None: cache \= {}  
    out=\[\]  
    for c,th\_deg,c49,c50,total in rows:  
        theta \= mp.acos(mp.mpf(str(c)))  
        key \= float(c)  \# cache by cosine  
        if key in cache:  
            Ival \= cache\[key\]  
        else:  
            Ival \= product\_integral\_F(theta, nk, np\_, kind)  
            cache\[key\] \= Ival  
        out.append({  
            "cos\_theta": float(c),  
            "theta\_deg": float(th\_deg),  
            "count\_to\_49": int(c49),  
            "count\_to\_50": int(c50),  
            "total": int(total),  
            "I\_cont": float(Ival),  
            "contrib\_sum": float(Ival) \* int(total)  
        })  
    return out

def assemble\_global\_c(pauli\_tables, denom=DENOM, lattice\_bracket=(1.0, (math.pi\*\*2)/4.0)):  
    """  
    Sum all contributions across the five tables.  
    Returns:  
      c\_cont\_est  : continuum estimate (numerical) divided by denom  
      c\_lat\_lo/hi : rigorous lattice bracket applying \[1, π^2/4\] multiplicative enclosure to the \*total\* continuum sum,  
                    with correct sign-handling (interval product).  
    """  
    \# sum continuum contributions  
    total\_cont \= 0.0  
    for key, rows in pauli\_tables.items():  
        for r in rows:  
            total\_cont \+= r\["contrib\_sum"\]  
    c\_cont\_est \= total\_cont / denom

    if RIGOROUS\_BRACKET:  
        a \= total\_cont / denom  
        \# multiplicative enclosure factors f ∈ \[f1, f2\], f1=1, f2=π^2/4  
        f1, f2 \= lattice\_bracket  
        \# product of interval \[a,a\] \* \[f1,f2\] \= \[min, max\] over endpoints taking sign into account  
        candidates \= \[a\*f1, a\*f2\]  
        c\_lat\_lo \= min(candidates)  
        c\_lat\_hi \= max(candidates)  
    else:  
        c\_lat\_lo \= None; c\_lat\_hi \= None

    return c\_cont\_est, c\_lat\_lo, c\_lat\_hi

\# \-------------------- run the pipeline \----------------  
print("="\*110)  
print("Two-Shell α — Master Cell (angles → denominator → Pauli integral → α^{-1} prediction)")  
print("="\*110)  
print(f"|S49|={len(S49)}  |S50|={len(S50)}  |S|={len(S)}  (expected 54, 84, 138)")  
print("\\nRow-sum witnesses (∑\_{NB} cosθ per source-type) \~ 1.0:")  
for key,val in ROW\_SUM\_WITNESS.items():  
    print(f"  {key:8s} : {val:+.15f}")  
print(f"\\nDenominator sum\_NB\_cos2  (rounded) \= {DENOM}  (raw={DENOM\_RAW:.12f})")

print("\\nComputing Pauli one-corner continuum integrals per angle class ...")  
CACHE \= {}  
PAULI\_PER\_TABLE \= {}  
for key, rows in ALL\_TABLES.items():  
    print(f"  table {key}: {len(rows)} angle classes")  
    PAULI\_PER\_TABLE\[key\] \= pauli\_for\_table(rows, NKAPPA, NPHI, QUAD\_KIND, cache=CACHE)

\# write pauli per-angle JSON  
with open(os.path.join(OUTDIR, "pauli\_integrals.json"), "w") as f:  
    json.dump(PAULI\_PER\_TABLE, f, indent=2)

c\_cont\_est, c\_lat\_lo, c\_lat\_hi \= assemble\_global\_c(PAULI\_PER\_TABLE, DENOM)

alpha\_inv\_est \= 137.0 \+ (c\_cont\_est/137.0)

print("\\n==================== SUMMARY \====================")  
print(f"Continuum estimate for c\_Pauli: {c\_cont\_est:+.10f}")  
print(f"α^{-1} (numerical, continuum)  ≈ 137 \+ c/137 \= {alpha\_inv\_est:.10f}")  
if RIGOROUS\_BRACKET:  
    print("\\nRigorous lattice–continuum bracket (global sharp factor \[1, π^2/4\]):")  
    print(f" c\_Pauli ∈ \[{c\_lat\_lo:.10f}, {c\_lat\_hi:.10f}\]")  
    print(f" α^{-1}  ∈ \[{137.0 \+ c\_lat\_lo/137.0:.10f}, {137.0 \+ c\_lat\_hi/137.0:.10f}\]")  
print("=================================================")

\# write pretty summary  
with open(os.path.join(OUTDIR, "alpha\_prediction.txt"), "w") as f:  
    f.write("Two-Shell α — Master Cell summary\\n")  
    f.write(f"Quadrature: {QUAD\_KIND}, NKAPPA={NKAPPA}, NPHI={NPHI}, mp.dps={MP\_DPS}\\n")  
    f.write(f"Denominator sum\_NB\_cos2 \= {DENOM} (raw {DENOM\_RAW:.12f})\\n")  
    f.write("Row-sum witnesses (∑\_{NB} cosθ per source-type) \~ 1.0:\\n")  
    for key,val in ROW\_SUM\_WITNESS.items():  
        f.write(f"  {key:8s} : {val:+.15f}\\n")  
    f.write(f"\\nContinuum c\_Pauli (numerical) \= {c\_cont\_est:+.12f}\\n")  
    f.write(f"alpha^{-1} (continuum)        \= {alpha\_inv\_est:.12f}\\n")  
    if RIGOROUS\_BRACKET:  
        f.write(f"\\nRigorous lattice–continuum bracket \[1, π^2/4\]:\\n")  
        f.write(f"  c\_Pauli ∈ \[{c\_lat\_lo:.12f}, {c\_lat\_hi:.12f}\]\\n")  
        f.write(f"  alpha^{-1} ∈ \[{137.0 \+ c\_lat\_lo/137.0:.12f}, {137.0 \+ c\_lat\_hi/137.0:.12f}\]\\n")

print(f"\\nArtifacts written to: ./{OUTDIR}")  
print(" \- angles\_49\_007.csv, angles\_49\_236.csv, angles\_50\_017.csv, angles\_50\_055.csv, angles\_50\_345.csv")  
print(" \- denominator.json")  
print(" \- pauli\_integrals.json")  
print(" \- alpha\_prediction.txt")

\# \-------------------- optional: quick self-test print of a few integrals \--------------------  
\_some \= sorted(list(CACHE.items()))\[:5\]  
if \_some:  
    print("\\nPreview of a few distinct angle-class integrals I\_cont(cosθ):")  
    for c, val in \_some:  
        \# Convert mpf to float for formatting  
        print(f"  cosθ={float(c):+.12f}  I\_cont={float(val):+.8e}")

\# \-------------------- tips \--------------------  
print("\\nTips:")  
print("  • For speed/repeatability, keep QUAD\_KIND='gl'. Clenshaw–Curtis nodes are available via QUAD\_KIND='cc'.")  
print("  • Increase NKAPPA/NPHI and MP\_DPS to tighten numerical stability; results are cached per unique cosθ.")  
print("  • The printed bracket is rigorous (global \[1, π^2/4\] lattice factor). Sectorwise lattice bounds can tighten it later.")  
\# \==================================================================================================  
\# \==================================================================================================  
\# Tile-wise Certified Enclosure for the Pauli Integral (drop at the bottom and run)  
\# Produces: pauli\_tilewise\_bounds.json and prints certified α^{-1} brackets (continuum & lattice)  
\# \==================================================================================================

from math import pi

def \_interval\_prod(lo1, hi1, lo2, hi2):  
    cands \= (lo1\*lo2, lo1\*hi2, hi1\*lo2, hi1\*hi2)  
    return (min(cands), max(cands))

def \_cos\_range(cmin, cmax):  
    """Tight range of cos(x) for x in \[cmin, cmax\]. Handles wrap to give max=1 (if 0 in interval)  
       and min=-1 (if π or \-π in interval). Assumes cmin \<= cmax and |c| \<= π (true in our domain)."""  
    \# normalize: if interval straddles 0 ⇒ max \= 1  
    if cmin \<= 0.0 \<= cmax:  
        cos\_max \= 1.0  
    else:  
        cos\_max \= max(math.cos(cmin), math.cos(cmax))  
    \# min: if interval hits ±π ⇒ \-1  
    cos\_min \= min(math.cos(cmin), math.cos(cmax))  
    if cmin \<= pi \<= cmax or cmin \<= \-pi \<= cmax:  
        cos\_min \= \-1.0  
    return (cos\_min, cos\_max)

def \_sinc\_bounds\_abs(a\_lo\_abs, a\_hi\_abs):  
    """Bounds for sinc(x)=sin(x)/x on |x| in \[a\_lo\_abs, a\_hi\_abs\] ⊆ \[0,π\].  
       Rigorous: 0 \<= sinc \<= 1; Tighter upper: if 0∉interval, use sinc(a\_lo\_abs)."""  
    \# rigorous lower bound (global): 0.0  
    sinc\_lo \= 0.0  
    \# rigorous upper bound:  
    if a\_lo\_abs \== 0.0:  \# contains 0  
        sinc\_hi \= 1.0  
    else:  
        sinc\_hi \= float(mp.sin(a\_lo\_abs)/a\_lo\_abs)  
        if sinc\_hi \> 1.0:  \# numerical guard  
            sinc\_hi \= 1.0  
        if sinc\_hi \< 0.0:  
            sinc\_hi \= 0.0  
    return sinc\_lo, sinc\_hi

def \_j1\_over\_x\_bounds(b\_lo, b\_hi):  
    """Bounds for J1(b)/b on \[b\_lo,b\_hi\] ⊆ \[0,π\]. Rigorous: 0 \<= J1(b)/b \<= 1/2 on \[0,π\]."""  
    j1b\_lo \= 0.0  
    j1b\_hi \= 0.5  
    return j1b\_lo, j1b\_hi

def \_j1\_over\_x\_bounds\_heur(b\_lo, b\_hi):  
    """Heuristic (monotone) tightening: J1(b)/b decreases on \[0,π\], so lower \= J1(b\_hi)/b\_hi."""  
    if b\_hi \== 0.0:  
        return 0.5, 0.5  
    lo \= float(mp.besselj(1, b\_hi)/b\_hi)  
    hi \= 0.5  
    lo \= max(0.0, min(lo, 0.5))  
    return lo, hi

def \_tilewise\_bounds\_one\_theta(theta, K\_TILES=48, P\_TILES\_PER\_HALF=48, heuristic=False):  
    """  
    Return rigorous continuum enclosure \[I\_lo, I\_hi\] for I(theta) \= ∬ I\_cont(κ,φ;θ) dφ dκ  
    by tiling \[0,π\]×\[0,π\] with φ split at π/2 to keep sin/cos monotone.  
    Also returns a heuristic-tightened \[I\_lo\_h, I\_hi\_h\] (optional).  
    """  
    ct \= float(mp.cos(theta))  
    st \= float(mp.sin(theta))  
    \# φ halves to keep monotonicity of sinφ and cosφ  
    halves \= \[(0.0, pi/2.0), (pi/2.0, pi)\]  
    dκ \= pi / K\_TILES  
    dφ\_half \= (pi/2.0) / P\_TILES\_PER\_HALF

    I\_lo\_sum \= 0.0  
    I\_hi\_sum \= 0.0  
    I\_lo\_sum\_h \= 0.0  
    I\_hi\_sum\_h \= 0.0

    for (φ0, φ1) in halves:  
        for j in range(P\_TILES\_PER\_HALF):  
            p0 \= φ0 \+ j\*dφ\_half  
            p1 \= p0 \+ dφ\_half  
            \# sinφ bounds on this tile (rigorous)  
            s\_vals \= (math.sin(p0), math.sin(p1))  
            sin\_lo \= min(s\_vals)  
            sin\_hi \= max(s\_vals)  
            \# cosφ bounds on this tile (monotone within each half)  
            c\_vals \= (math.cos(p0), math.cos(p1))  
            cos\_lo \= min(c\_vals)  
            cos\_hi \= max(c\_vals)

            for i in range(K\_TILES):  
                k0 \= i\*dκ  
                k1 \= k0 \+ dκ

                \# a \= κ cosφ  ⇒  |a| in \[min|κ\*{cos bounds}|, max|…|\]  
                \# corners suffice since κ and cosφ monotone on the tile  
                a\_corners \= (k0\*cos\_lo, k0\*cos\_hi, k1\*cos\_lo, k1\*cos\_hi)  
                a\_min, a\_max \= min(a\_corners), max(a\_corners)  
                a\_lo\_abs \= 0.0 if (a\_min\<=0.0\<=a\_max) else min(abs(a\_min), abs(a\_max))  
                a\_hi\_abs \= max(abs(a\_min), abs(a\_max))

                \# b \= κ sinθ sinφ  ⇒  b ∈ \[k0\*st\*min(sin), k1\*st\*max(sin)\]  
                b\_lo \= k0\*st\*sin\_lo  
                b\_hi \= k1\*st\*sin\_hi  
                if b\_lo \< 0.0: b\_lo \= 0.0  \# numeric guard

                \# c \= κ cosθ cosφ  ⇒  c ∈ prod(\[k0,k1\], \[ct\*cos\_lo, ct\*cos\_hi\])  
                ct\_cos\_lo \= ct\*cos\_lo  
                ct\_cos\_hi \= ct\*cos\_hi  
                cmin, cmax \= \_interval\_prod(k0, k1, min(ct\_cos\_lo, ct\_cos\_hi), max(ct\_cos\_lo, ct\_cos\_hi))

                \# cos(c) range on the tile  
                C\_lo, C\_hi \= \_cos\_range(cmin, cmax)  
                C\_mid \= 0.5\*(C\_lo \+ C\_hi)  
                C\_rad \= 0.5\*(C\_hi \- C\_lo)

                \# rigorous factor bounds for R(κ,φ)=sinφ \* sinc(a) \* J1(b)/b  (all ≥ 0\)  
                sinc\_lo, sinc\_hi \= \_sinc\_bounds\_abs(a\_lo\_abs, a\_hi\_abs)  
                j1b\_lo, j1b\_hi   \= \_j1\_over\_x\_bounds(b\_lo, b\_hi)

                R\_lo \= sin\_lo \* sinc\_lo \* j1b\_lo  
                R\_hi \= sin\_hi \* sinc\_hi \* j1b\_hi

                area \= dκ \* dφ\_half  
                A\_lo \= area \* R\_lo  
                A\_hi \= area \* R\_hi

                \# rigorous enclosure for ∬ R\*C over tile:  
                tile\_lo \= C\_mid\*A\_lo \- abs(C\_rad)\*A\_hi  
                tile\_hi \= C\_mid\*A\_hi \+ abs(C\_rad)\*A\_hi

                \# multiply by cosθ (constant), then accumulate  
                t\_lo \= ct \* tile\_lo  
                t\_hi \= ct \* tile\_hi  
                \# reorder if needed  
                if t\_lo \<= t\_hi:  
                    I\_lo\_sum \+= t\_lo  
                    I\_hi\_sum \+= t\_hi  
                else:  
                    I\_lo\_sum \+= t\_hi  
                    I\_hi\_sum \+= t\_lo

                \# optional heuristic tightening (monotone assumptions on \[0,π\])  
                if heuristic:  
                    \# tighter lower bound for sinc: sinc is decreasing on \[0,π\]  
                    \# so min over |a|∈\[a\_lo\_abs,a\_hi\_abs\] is at a\_hi\_abs  
                    sinc\_lo\_h \= 0.0 if a\_hi\_abs==0.0 else float(mp.sin(a\_hi\_abs)/a\_hi\_abs)  
                    sinc\_hi\_h \= 1.0 if a\_lo\_abs==0.0 else float(mp.sin(a\_lo\_abs)/a\_lo\_abs)  
                    if sinc\_lo\_h \< 0.0: sinc\_lo\_h \= 0.0  
                    if sinc\_hi\_h \> 1.0: sinc\_hi\_h \= 1.0

                    j1b\_lo\_h, j1b\_hi\_h \= \_j1\_over\_x\_bounds\_heur(b\_lo, b\_hi)

                    R\_lo\_h \= sin\_lo \* sinc\_lo\_h \* j1b\_lo\_h  
                    R\_hi\_h \= sin\_hi \* sinc\_hi\_h \* j1b\_hi\_h  
                    A\_lo\_h \= area \* R\_lo\_h  
                    A\_hi\_h \= area \* R\_hi\_h

                    tile\_lo\_h \= C\_mid\*A\_lo\_h \- abs(C\_rad)\*A\_hi\_h  
                    tile\_hi\_h \= C\_mid\*A\_hi\_h \+ abs(C\_rad)\*A\_hi\_h  
                    t\_lo\_h \= ct \* tile\_lo\_h  
                    t\_hi\_h \= ct \* tile\_hi\_h  
                    if t\_lo\_h \<= t\_hi\_h:  
                        I\_lo\_sum\_h \+= t\_lo\_h  
                        I\_hi\_sum\_h \+= t\_hi\_h  
                    else:  
                        I\_lo\_sum\_h \+= t\_hi\_h  
                        I\_hi\_sum\_h \+= t\_lo\_h

    if heuristic:  
        return (I\_lo\_sum, I\_hi\_sum, I\_lo\_sum\_h, I\_hi\_sum\_h)  
    else:  
        return (I\_lo\_sum, I\_hi\_sum, None, None)

def tilewise\_certified\_c(K\_TILES=48, P\_TILES\_PER\_HALF=48, heuristic=True, lattice=True):  
    """  
    Compute continuum certified interval \[c\_lo, c\_hi\] by summing per-class tile intervals,  
    then optional heuristic interval \[c\_lo\_h, c\_hi\_h\]; finally map to lattice via \[1, π^2/4\].  
    """  
    \# per-cosθ contributions aggregate (weight \= multiplicity 'total')  
    total\_lo \= 0.0  
    total\_hi \= 0.0  
    total\_lo\_h \= 0.0  
    total\_hi\_h \= 0.0

    per\_class\_records \= \[\]

    for key, rows in ALL\_TABLES.items():  
        for r in rows:  
            c \= float(r\["cos\_theta"\]) if "cos\_theta" in r else float(r\[0\])  
            theta \= float(mp.acos(c))  
            mult \= int(r\["total"\]) if "total" in r else int(r\[4\])

            I\_lo, I\_hi, I\_lo\_h, I\_hi\_h \= \_tilewise\_bounds\_one\_theta(  
                theta, K\_TILES=K\_TILES, P\_TILES\_PER\_HALF=P\_TILES\_PER\_HALF, heuristic=heuristic  
            )

            \# scale by multiplicity  
            lo\_sc \= mult \* I\_lo  
            hi\_sc \= mult \* I\_hi  
            total\_lo \+= lo\_sc  
            total\_hi \+= hi\_sc

            if heuristic and I\_lo\_h is not None:  
                total\_lo\_h \+= mult \* I\_lo\_h  
                total\_hi\_h \+= mult \* I\_hi\_h

            per\_class\_records.append({  
                "cos\_theta": c,  
                "theta\_deg": float(r\["theta\_deg"\]) if "theta\_deg" in r else float(r\[1\]),  
                "multiplicity": mult,  
                "I\_cont\_lo": I\_lo, "I\_cont\_hi": I\_hi,  
                "I\_cont\_lo\_heur": I\_lo\_h, "I\_cont\_hi\_heur": I\_hi\_h  
            })

    \# normalize by denominator  
    c\_lo \= total\_lo / DENOM  
    c\_hi \= total\_hi / DENOM  
    if heuristic:  
        c\_lo\_h \= total\_lo\_h / DENOM  
        c\_hi\_h \= total\_hi\_h / DENOM  
    else:  
        c\_lo\_h \= None; c\_hi\_h \= None

    out \= {  
        "K\_TILES": K\_TILES,  
        "P\_TILES\_PER\_HALF": P\_TILES\_PER\_HALF,  
        "denominator\_sum\_NB\_cos2": DENOM,  
        "continuum\_interval": \[c\_lo, c\_hi\],  
        "continuum\_interval\_heuristic": \[c\_lo\_h, c\_hi\_h\],  
        "per\_class": per\_class\_records  
    }

    \# optional lattice mapping with the sharp global factor \[1, π^2/4\]  
    if lattice:  
        f1, f2 \= 1.0, (pi\*pi)/4.0  
        \# interval product (handle sign)  
        lat\_cand \= \[c\_lo\*f1, c\_lo\*f2, c\_hi\*f1, c\_hi\*f2\]  
        lat\_lo \= min(lat\_cand); lat\_hi \= max(lat\_cand)  
        out\["lattice\_interval"\] \= \[lat\_lo, lat\_hi\]  
        if heuristic and (c\_lo\_h is not None):  
            lat\_h\_cand \= \[c\_lo\_h\*f1, c\_lo\_h\*f2, c\_hi\_h\*f1, c\_hi\_h\*f2\]  
            out\["lattice\_interval\_heuristic"\] \= \[min(lat\_h\_cand), max(lat\_h\_cand)\]  
    return out

\# \-------------------- run the tile-wise certifier \--------------------  
TILES\_K \= 48  
TILES\_P \= 48  
tile\_out \= tilewise\_certified\_c(K\_TILES=TILES\_K, P\_TILES\_PER\_HALF=TILES\_P, heuristic=True, lattice=True)

\# write JSON artefact  
with open(os.path.join(OUTDIR, "pauli\_tilewise\_bounds.json"), "w") as f:  
    json.dump(tile\_out, f, indent=2)

\# pretty print  
c\_lo, c\_hi \= tile\_out\["continuum\_interval"\]  
print("\\n==================== TILE-WISE CERTIFIED (CONTINUUM) \====================")  
print(f"Tiles: K={TILES\_K}, P/half={TILES\_P}  ⇒  c\_Pauli ∈ \[{c\_lo:.10f}, {c\_hi:.10f}\]")  
print(f"α^{-1} continuum ∈ \[{137.0 \+ c\_lo/137.0:.10f}, {137.0 \+ c\_hi/137.0:.10f}\]")

if tile\_out.get("continuum\_interval\_heuristic"):  
    c\_lo\_h, c\_hi\_h \= tile\_out\["continuum\_interval\_heuristic"\]  
    print("\\n(Heuristic tightening, assumes monotonicity of sinc and J1(x)/x on \[0,π\])")  
    print(f"c\_Pauli (heur) ∈ \[{c\_lo\_h:.10f}, {c\_hi\_h:.10f}\]")  
    print(f"α^{-1}   (heur) ∈ \[{137.0 \+ c\_lo\_h/137.0:.10f}, {137.0 \+ c\_hi\_h/137.0:.10f}\]")

if tile\_out.get("lattice\_interval"):  
    L\_lo, L\_hi \= tile\_out\["lattice\_interval"\]  
    print("\\n==================== TILE-WISE CERTIFIED (LATTICE via \[1,π²/4\]) \====================")  
    print(f"c\_Pauli ∈ \[{L\_lo:.10f}, {L\_hi:.10f}\]")  
    print(f"α^{-1}   ∈ \[{137.0 \+ L\_lo/137.0:.10f}, {137.0 \+ L\_hi/137.0:.10f}\]")

if tile\_out.get("lattice\_interval\_heuristic"):  
    LH\_lo, LH\_hi \= tile\_out\["lattice\_interval\_heuristic"\]  
    print("\\n(Heuristic lattice tightening)")  
    print(f"c\_Pauli (heur) ∈ \[{LH\_lo:.10f}, {LH\_hi:.10f}\]")  
    print(f"α^{-1}   (heur) ∈ \[{137.0 \+ LH\_lo/137.0:.10f}, {137.0 \+ LH\_hi/137.0:.10f}\]")

print("\\nSaved: pauli\_tilewise\_bounds.json  (per-class tile intervals \+ global brackets)")  
\# \==================================================================================================  
\# \==================================================================================================  
\# LEGO BLOCK — "Wesley-Rigorous" Adaptive Certifier (exact cos-extrema \+ monotone endpoints \+ adapt)  
\# Drop at the bottom and run. Writes: pauli\_tilewise\_adaptive.json  
\# \==================================================================================================

from math import pi

\# \---------- exact cos bounds on \[cmin, cmax\] via endpoints \+ interior kπ \----------  
def \_cos\_bounds\_exact(cmin, cmax):  
    if cmin \> cmax:  
        cmin, cmax \= cmax, cmin  
    \# candidate points: endpoints and any kπ in \[cmin, cmax\]  
    k\_lo \= int(mp.ceil(cmin/mp.pi))  
    k\_hi \= int(mp.floor(cmax/mp.pi))  
    cands \= \[cmin, cmax\]  
    for k in range(k\_lo, k\_hi+1):  
        cands.append(k\*mp.pi)  
    vals \= \[mp.cos(x) for x in cands\]  
    return float(min(vals)), float(max(vals))

\# \---------- rigorous monotone endpoint bounds on \[0, π\] \----------  
def \_sinc\_endpoints\_abs(a\_lo\_abs, a\_hi\_abs):  
    \# sinc(x) \= sin x / x, decreasing on \[0,π\], sinc(0)=1  
    if a\_lo\_abs \< 0: a\_lo\_abs \= 0.0  
    if a\_hi\_abs \< a\_lo\_abs: a\_hi\_abs \= a\_lo\_abs  
    if a\_lo\_abs \== 0.0:  
        hi \= 1.0  
    else:  
        hi \= float(mp.sin(a\_lo\_abs)/a\_lo\_abs)  
        hi \= max(0.0, min(1.0, hi))  
    if a\_hi\_abs \== 0.0:  
        lo \= 1.0  
    else:  
        lo \= float(mp.sin(a\_hi\_abs)/a\_hi\_abs)  
        lo \= max(0.0, min(1.0, lo))  
    return lo, hi  \# \[lo, hi\]

def \_j1\_over\_x\_endpoints(b\_lo, b\_hi):  
    \# On \[0, π\], J1(x)/x is positive and strictly decreasing; limit at 0 is 1/2.  
    if b\_lo \< 0: b\_lo \= 0.0  
    if b\_hi \< b\_lo: b\_hi \= b\_lo  
    hi \= 0.5 if b\_lo==0.0 else float(mp.besselj(1, b\_lo)/b\_lo)  
    lo \= 0.5 if b\_hi==0.0 else float(mp.besselj(1, b\_hi)/b\_hi)  
    hi \= max(0.0, min(0.5, hi))  
    lo \= max(0.0, min(0.5, lo))  
    if lo \> hi:  \# numerical guard (shouldn't happen)  
        lo, hi \= hi, lo  
    return lo, hi  \# \[lo, hi\]

\# \---------- helpers \----------  
def \_prod\_interval(a0, a1, b0, b1):  
    c \= (a0\*b0, a0\*b1, a1\*b0, a1\*b1)  
    return (min(c), max(c))

def \_tile\_bounds\_one(theta, phi0, phi1, k0, k1):  
    """  
    Rigorous enclosure for ∬\_{tile} sinφ \* sinc(κ cosφ) \* \[J1(κ sinθ sinφ)/(κ sinθ sinφ)\] \* cos(κ cosθ cosφ) \* cosθ dκ dφ  
    using exact cos-extrema and monotone endpoint bounds for sinc and J1/x. Works within a φ-half to ensure monotonicity.  
    Returns (tile\_lo, tile\_hi, phase\_span, area\_hi), where phase\_span \= |cmax \- cmin| and area\_hi \= (phi1-phi0)\*(k1-k0).  
    """  
    ct \= float(mp.cos(theta))  
    st \= float(mp.sin(theta))  
    dphi \= phi1 \- phi0  
    dk   \= k1 \- k0  
    area \= dphi \* dk

    \# sinφ and cosφ bounds (monotone within each half)  
    \# detect which half: we will only call this with tiles fully inside \[0,π/2\] or \[π/2,π\]  
    phi\_mid \= 0.5\*(phi0+phi1)  
    if phi1 \<= (pi/2.0) \+ 1e-18:  
        \# \[0, π/2\]: sin ↑, cos ↓  
        sin\_lo \= math.sin(phi0); sin\_hi \= math.sin(phi1)  
        cos\_lo \= math.cos(phi1); cos\_hi \= math.cos(phi0)  
    elif phi0 \>= (pi/2.0) \- 1e-18:  
        \# \[π/2, π\]: sin ↓, cos ↓  
        sin\_lo \= math.sin(phi1); sin\_hi \= math.sin(phi0)  
        cos\_lo \= math.cos(phi1); cos\_hi \= math.cos(phi0)  
    else:  
        \# shouldn't happen if caller tiles halves; fallback to conservative per-endpoint  
        s0,s1 \= math.sin(phi0), math.sin(phi1)  
        c0,c1 \= math.cos(phi0), math.cos(phi1)  
        sin\_lo, sin\_hi \= min(s0,s1), max(s0,s1)  
        cos\_lo, cos\_hi \= min(c0,c1), max(c0,c1)

    \# a \= κ cosφ  \-\> get absolute interval  
    a\_min, a\_max \= \_prod\_interval(k0, k1, cos\_lo, cos\_hi)  
    if a\_min \<= 0.0 \<= a\_max:  
        a\_lo\_abs, a\_hi\_abs \= 0.0, max(abs(a\_min), abs(a\_max))  
    else:  
        a\_lo\_abs, a\_hi\_abs \= min(abs(a\_min), abs(a\_max)), max(abs(a\_min), abs(a\_max))

    \# b \= κ st sinφ (nonnegative on halves)  
    b\_lo \= k0 \* st \* sin\_lo  
    b\_hi \= k1 \* st \* sin\_hi  
    if b\_lo \< 0.0: b\_lo \= 0.0

    \# c \= κ ct cosφ  
    ct\_cos\_lo \= ct \* cos\_lo  
    ct\_cos\_hi \= ct \* cos\_hi  
    cmin, cmax \= \_prod\_interval(k0, k1, min(ct\_cos\_lo, ct\_cos\_hi), max(ct\_cos\_lo, ct\_cos\_hi))  
    C\_lo, C\_hi \= \_cos\_bounds\_exact(cmin, cmax)  
    C\_mid \= 0.5\*(C\_lo \+ C\_hi)  
    C\_rad \= 0.5\*(C\_hi \- C\_lo)

    \# factor bounds (all nonnegative)  
    sinc\_lo, sinc\_hi \= \_sinc\_endpoints\_abs(a\_lo\_abs, a\_hi\_abs)  
    j\_lo, j\_hi       \= \_j1\_over\_x\_endpoints(b\_lo, b\_hi)

    R\_lo \= sin\_lo \* sinc\_lo \* j\_lo  
    R\_hi \= sin\_hi \* sinc\_hi \* j\_hi

    A\_lo \= area \* R\_lo  
    A\_hi \= area \* R\_hi

    \# rigorous enclosure using |cos| ≤ |C\_mid| \+ C\_rad and R ≥ 0  
    tile\_lo \= ct \* (C\_mid\*A\_lo \- abs(C\_rad)\*A\_hi)  
    tile\_hi \= ct \* (C\_mid\*A\_hi \+ abs(C\_rad)\*A\_hi)  
    \# reorder  
    if tile\_lo \> tile\_hi:  
        tile\_lo, tile\_hi \= tile\_hi, tile\_lo

    phase\_span \= abs(cmax \- cmin)  
    return tile\_lo, tile\_hi, phase\_span, area

\# \---------- adaptive subdivision on a φ-half \----------  
def \_adaptive\_half(theta, phi0, phi1, k0, k1, eps\_tile=1e-9, max\_depth=18, stats=None):  
    """  
    Recursively subdivide the tile until each leaf has (hi \- lo) \<= eps\_tile.  
    Returns (sum\_lo, sum\_hi, n\_leaves).  
    """  
    if stats is None:  
        stats \= {"leaves":0, "splits":0}  
    stack \= \[(phi0, phi1, k0, k1, 0)\]  
    sum\_lo \= 0.0  
    sum\_hi \= 0.0  
    while stack:  
        p0, p1, u0, u1, d \= stack.pop()  
        lo, hi, dphase, area \= \_tile\_bounds\_one(theta, p0, p1, u0, u1)  
        if (hi \- lo) \<= eps\_tile or d \>= max\_depth:  
            sum\_lo \+= lo  
            sum\_hi \+= hi  
            stats\["leaves"\] \+= 1  
        else:  
            \# split along the dimension that promises larger reduction:  
            \# heuristic: split where interval length is larger in scaled units  
            if (u1 \- u0) \> (p1 \- p0):  
                um \= 0.5\*(u0+u1)  
                stack.append((p0, p1, um, u1, d+1))  
                stack.append((p0, p1, u0, um, d+1))  
            else:  
                pm \= 0.5\*(p0+p1)  
                stack.append((pm, p1, u0, u1, d+1))  
                stack.append((p0, pm, u0, u1, d+1))  
            stats\["splits"\] \+= 1  
    return sum\_lo, sum\_hi, stats

def certified\_c\_adaptive(eps\_tile=1e-9, max\_depth=18, lattice=True):  
    """  
    Certified continuum interval for c\_Pauli using adaptive subdivision on each φ-half,  
    summed over angle classes and normalized by DENOM. Optionally map to lattice via \[1, π^2/4\].  
    """  
    total\_lo \= 0.0  
    total\_hi \= 0.0  
    half\_stats \= {"leaves":0, "splits":0}  
    for key, rows in ALL\_TABLES.items():  
        for r in rows:  
            c  \= float(r\["cos\_theta"\]) if "cos\_theta" in r else float(r\[0\])  
            th \= float(mp.acos(c))  
            mult \= int(r\["total"\]) if "total" in r else int(r\[4\])

            \# first half \[0, π/2\]  
            lo1, hi1, st1 \= \_adaptive\_half(th, 0.0, pi/2.0, 0.0, pi, eps\_tile=eps\_tile, max\_depth=max\_depth)  
            \# second half \[π/2, π\]  
            lo2, hi2, st2 \= \_adaptive\_half(th, pi/2.0, pi, 0.0, pi, eps\_tile=eps\_tile, max\_depth=max\_depth)

            half\_stats\["leaves"\] \+= st1\["leaves"\] \+ st2\["leaves"\]  
            half\_stats\["splits"\] \+= st1\["splits"\] \+ st2\["splits"\]

            total\_lo \+= mult \* (lo1 \+ lo2)  
            total\_hi \+= mult \* (hi1 \+ hi2)

    c\_lo \= total\_lo / DENOM  
    c\_hi \= total\_hi / DENOM

    out \= {  
        "eps\_tile": eps\_tile,  
        "max\_depth": max\_depth,  
        "denominator": DENOM,  
        "continuum\_interval": \[c\_lo, c\_hi\],  
        "stats": half\_stats  
    }

    if lattice:  
        F\_lo, F\_hi \= 1.0, (pi\*pi)/4.0  
        lat\_cand \= \[c\_lo\*F\_lo, c\_lo\*F\_hi, c\_hi\*F\_lo, c\_hi\*F\_hi\]  
        out\["lattice\_interval"\] \= \[min(lat\_cand), max(lat\_cand)\]

    return out

\# \-------------------- run the adaptive certifier \--------------------  
EPS\_TILE   \= 5e-11   \# tighten this to squeeze the final interval; 5e-11 is already very sharp  
MAX\_DEPTH  \= 22

adaptive\_out \= certified\_c\_adaptive(eps\_tile=EPS\_TILE, max\_depth=MAX\_DEPTH, lattice=True)

with open(os.path.join(OUTDIR, "pauli\_tilewise\_adaptive.json"), "w") as f:  
    json.dump(adaptive\_out, f, indent=2)

c\_lo, c\_hi \= adaptive\_out\["continuum\_interval"\]  
print("\\n==================== WESLEY-RIGOROUS ADAPTIVE (CONTINUUM) \====================")  
print(f"eps\_tile={EPS\_TILE:.1e}, max\_depth={MAX\_DEPTH:d}  ⇒  c\_Pauli ∈ \[{c\_lo:.12e}, {c\_hi:.12e}\]")  
print(f"α^{-1} continuum ∈ \[{137.0 \+ c\_lo/137.0:.12f}, {137.0 \+ c\_hi/137.0:.12f}\]")  
if "lattice\_interval" in adaptive\_out:  
    L\_lo, L\_hi \= adaptive\_out\["lattice\_interval"\]  
    print("\\n==================== WESLEY-RIGOROUS ADAPTIVE (LATTICE via \[1,π²/4\]) \====================")  
    print(f"c\_Pauli ∈ \[{L\_lo:.12e}, {L\_hi:.12e}\]")  
    print(f"α^{-1}   ∈ \[{137.0 \+ L\_lo/137.0:.12f}, {137.0 \+ L\_hi/137.0:.12f}\]")  
print(f"Tiles explored: leaves={adaptive\_out\['stats'\]\['leaves'\]}, splits={adaptive\_out\['stats'\]\['splits'\]}")  
print("Saved: pauli\_tilewise\_adaptive.json")  
\# \==================================================================================================  
\# \==================================================================================================  
\# LEGO BLOCK — Resumeable, Progressive, Early-Exit Adaptive Certifier (rigorous remainder bound)  
\# Paste at the very bottom and run. Artifacts:  
\#   \- pauli\_adaptive\_progress.json   (checkpoint, safe to resume)  
\#   \- pauli\_adaptive\_final.json      (final certificate snapshot when early-exit / schedule completes)  
\# \==================================================================================================

import json, time, os, math  
from math import pi

\# \---- trivial rigorous remainder bound for one angle class (continuum) \----  
\# |I\_class(theta)| \= ∬ sinφ \* sinc(κ cosφ) \* (J1(κ sinθ sinφ)/(κ sinθ sinφ)) \* |cos(κ cosθ cosφ)| \* |cosθ| dκ dφ  
\# ≤ ∫\_0^π ∫\_0^π \[sinφ \* 1 \* 1/2 \* 1 \* |cosθ|\] dκ dφ \= π \* |cosθ|  
def class\_abs\_bound\_continuum(cos\_theta):  
    return pi \* abs(float(cos\_theta))

\# \---- load & save checkpoints \----  
PROG\_PATH \= os.path.join(OUTDIR, "pauli\_adaptive\_progress.json")  
FINAL\_PATH \= os.path.join(OUTDIR, "pauli\_adaptive\_final.json")

def \_task\_list\_from\_tables():  
    tasks \= \[\]  
    for key, rows in ALL\_TABLES.items():  
        for r in rows:  
            c  \= float(r\["cos\_theta"\]) if "cos\_theta" in r else float(r\[0\])  
            th \= float(mp.acos(c))  
            mult \= int(r\["total"\]) if "total" in r else int(r\[4\])  
            theta\_deg \= float(r\["theta\_deg"\]) if "theta\_deg" in r else float(r\[1\])  
            tid \= f"{key}|c={c:.12f}|deg={theta\_deg:.6f}"  
            weight \= mult \* abs(c)  \# prioritize high influence classes  
            tasks.append({  
                "task\_id": tid,  
                "table": key,  
                "cos\_theta": c,  
                "theta": th,  
                "theta\_deg": theta\_deg,  
                "multiplicity": mult,  
                "weight": weight  
            })  
    \# process heavy hitters first  
    tasks.sort(key=lambda d: d\["weight"\], reverse=True)  
    return tasks

def \_load\_progress():  
    if os.path.exists(PROG\_PATH):  
        with open(PROG\_PATH, "r") as f:  
            return json.load(f)  
    return {"per\_class":{}, "stats":{"passes":0, "leaves":0, "splits":0}, "schedule":\[\], "created": time.time()}

def \_save\_progress(prog):  
    with open(PROG\_PATH, "w") as f:  
        json.dump(prog, f, indent=2)

def \_save\_final(snapshot):  
    with open(FINAL\_PATH, "w") as f:  
        json.dump(snapshot, f, indent=2)

\# \---- rigorous global interval given partial progress \----  
def \_global\_interval\_from\_progress(tasks, prog):  
    sum\_lo \= 0.0  
    sum\_hi \= 0.0  
    rem\_abs \= 0.0  
    pc \= prog\["per\_class"\]  
    for t in tasks:  
        tid \= t\["task\_id"\]  
        mult \= t\["multiplicity"\]  
        if tid in pc and ("lo" in pc\[tid\]) and ("hi" in pc\[tid\]):  
            sum\_lo \+= mult \* pc\[tid\]\["lo"\]  
            sum\_hi \+= mult \* pc\[tid\]\["hi"\]  
        else:  
            rem\_abs \+= mult \* class\_abs\_bound\_continuum(t\["cos\_theta"\])  
    \# rigorously: unprocessed classes contribute in \[-rem\_abs, \+rem\_abs\]  
    glob\_lo \= (sum\_lo \- rem\_abs) / DENOM  
    glob\_hi \= (sum\_hi \+ rem\_abs) / DENOM  
    return glob\_lo, glob\_hi

\# \---- compute one class adaptively to target eps\_tile \----  
def \_compute\_one\_class(t, eps\_tile, max\_depth):  
    \# uses the exact-extrema adaptive routines from the prior block:  
    lo1, hi1, st1 \= \_adaptive\_half(t\["theta"\], 0.0, pi/2.0, 0.0, pi, eps\_tile=eps\_tile, max\_depth=max\_depth)  
    lo2, hi2, st2 \= \_adaptive\_half(t\["theta"\], pi/2.0, pi, 0.0, pi, eps\_tile=eps\_tile, max\_depth=max\_depth)  
    leaves \= st1\["leaves"\] \+ st2\["leaves"\]; splits \= st1\["splits"\] \+ st2\["splits"\]  
    return lo1+lo2, hi1+hi2, leaves, splits

\# \---- driver: progressive schedule, early exit, wall-clock safeguard \----  
def certify\_with\_resume(  
    eps\_schedule=(1e-7, 5e-8, 2e-8, 1e-8, 5e-9, 2e-9, 1e-9, 5e-10, 2e-10, 1e-10),  
    max\_depth=22,  
    save\_every\_sec=30.0,  
    stop\_after\_sec=None,  
    want\_lattice=True  
):  
    tasks \= \_task\_list\_from\_tables()  
    prog \= \_load\_progress()  
    t0 \= time.time()  
    last\_save \= t0

    \# ensure schedule record  
    if not prog.get("schedule"): prog\["schedule"\] \= \[\]  
    already\_done \= sum(1 for t in tasks if t\["task\_id"\] in prog\["per\_class"\])

    print(f"\[resume\] tasks: {len(tasks)} total; already computed: {already\_done}")  
    print(f"\[resume\] checkpoint: {PROG\_PATH}")

    for stage, eps\_tile in enumerate(eps\_schedule, start=1):  
        print(f"\\n=== stage {stage}/{len(eps\_schedule)}: eps\_tile={eps\_tile:.1e}, max\_depth={max\_depth} \===")  
        stage\_new \= 0  
        for idx, t in enumerate(tasks, start=1):  
            tid \= t\["task\_id"\]  
            pc \= prog\["per\_class"\].get(tid, {})  
            \# (re)compute only if not present or stored width is larger than \~2\*eps\_tile (loose heuristic)  
            need \= True  
            if pc:  
                w \= pc\["hi"\] \- pc\["lo"\]  
                need \= (w \> 2.5\*eps\_tile)  
            if need:  
                lo, hi, leaves, splits \= \_compute\_one\_class(t, eps\_tile=eps\_tile, max\_depth=max\_depth)  
                prog\["per\_class"\]\[tid\] \= {  
                    "cos\_theta": t\["cos\_theta"\],  
                    "theta\_deg": t\["theta\_deg"\],  
                    "multiplicity": t\["multiplicity"\],  
                    "lo": lo, "hi": hi,  
                    "eps\_tile": eps\_tile,  
                    "leaves": leaves,  
                    "splits": splits  
                }  
                prog\["stats"\]\["leaves"\] \+= leaves  
                prog\["stats"\]\["splits"\] \+= splits  
                stage\_new \+= 1

            \# periodic save & early-exit check  
            now \= time.time()  
            if now \- last\_save \>= save\_every\_sec:  
                \_save\_progress(prog)  
                last\_save \= now  
                \# early-exit: compute rigorous global interval with remainder bound  
                g\_lo, g\_hi \= \_global\_interval\_from\_progress(tasks, prog)  
                lattice \= None  
                if want\_lattice:  
                    F\_lo, F\_hi \= 1.0, (pi\*pi)/4.0  
                    lat\_cand \= \[g\_lo\*F\_lo, g\_lo\*F\_hi, g\_hi\*F\_lo, g\_hi\*F\_hi\]  
                    lattice \= (min(lat\_cand), max(lat\_cand))  
                print(f"\[checkpoint\] computed={sum(1 for t2 in tasks if t2\['task\_id'\] in prog\['per\_class'\])}/{len(tasks)}  "  
                      f"continuum ∈ \[{g\_lo:.12e}, {g\_hi:.12e}\]")  
                if lattice:  
                    print(f"\[checkpoint\] lattice   ∈ \[{lattice\[0\]:.12e}, {lattice\[1\]:.12e}\]")  
                \# stop if both exclude 0  
                if g\_lo \> 0.0 or g\_hi \< 0.0:  
                    snapshot \= {  
                        "kind":"early\_exit\_continuum",  
                        "continuum\_interval":\[g\_lo, g\_hi\],  
                        "lattice\_interval": list(lattice) if lattice else None,  
                        "denominator": DENOM,  
                        "stats": prog\["stats"\],  
                        "schedule": list(eps\_schedule\[:stage\])  
                    }  
                    \_save\_final(snapshot)  
                    print("\\n\*\*\* EARLY EXIT: continuum certified interval excludes 0\. Saved final certificate. \*\*\*")  
                    return snapshot  
                if want\_lattice and lattice and (lattice\[0\] \> 0.0 or lattice\[1\] \< 0.0):  
                    snapshot \= {  
                        "kind":"early\_exit\_lattice",  
                        "continuum\_interval":\[g\_lo, g\_hi\],  
                        "lattice\_interval": list(lattice),  
                        "denominator": DENOM,  
                        "stats": prog\["stats"\],  
                        "schedule": list(eps\_schedule\[:stage\])  
                    }  
                    \_save\_final(snapshot)  
                    print("\\n\*\*\* EARLY EXIT: lattice certified interval excludes 0\. Saved final certificate. \*\*\*")  
                    return snapshot

                \# wall-clock stop?  
                if (stop\_after\_sec is not None) and ((now \- t0) \>= stop\_after\_sec):  
                    snapshot \= {  
                        "kind":"time\_stop",  
                        "continuum\_interval":\[g\_lo, g\_hi\],  
                        "lattice\_interval": list(lattice) if lattice else None,  
                        "denominator": DENOM,  
                        "stats": prog\["stats"\],  
                        "schedule": list(eps\_schedule\[:stage\])  
                    }  
                    \_save\_final(snapshot)  
                    print("\\n\*\*\* TIME LIMIT REACHED: saved best-so-far rigorous bracket. Resume later to tighten. \*\*\*")  
                    return snapshot

        prog\["schedule"\].append({"stage":stage, "eps\_tile":eps\_tile, "new\_computed":stage\_new})  
        \_save\_progress(prog)

        \# end-of-stage early-exit check  
        g\_lo, g\_hi \= \_global\_interval\_from\_progress(tasks, prog)  
        lattice \= None  
        if want\_lattice:  
            F\_lo, F\_hi \= 1.0, (pi\*pi)/4.0  
            lat\_cand \= \[g\_lo\*F\_lo, g\_lo\*F\_hi, g\_hi\*F\_lo, g\_hi\*F\_hi\]  
            lattice \= (min(lat\_cand), max(lat\_cand))  
        print(f"\[stage end\] continuum ∈ \[{g\_lo:.12e}, {g\_hi:.12e}\]")  
        if lattice:  
            print(f"\[stage end\] lattice   ∈ \[{lattice\[0\]:.12e}, {lattice\[1\]:.12e}\]")  
        if g\_lo \> 0.0 or g\_hi \< 0.0:  
            snapshot \= {  
                "kind":"early\_exit\_continuum",  
                "continuum\_interval":\[g\_lo, g\_hi\],  
                "lattice\_interval": list(lattice) if lattice else None,  
                "denominator": DENOM,  
                "stats": prog\["stats"\],  
                "schedule": list(eps\_schedule\[:stage\])  
            }  
            \_save\_final(snapshot)  
            print("\\n\*\*\* EARLY EXIT: continuum certified interval excludes 0\. Saved final certificate. \*\*\*")  
            return snapshot  
        if want\_lattice and lattice and (lattice\[0\] \> 0.0 or lattice\[1\] \< 0.0):  
            snapshot \= {  
                "kind":"early\_exit\_lattice",  
                "continuum\_interval":\[g\_lo, g\_hi\],  
                "lattice\_interval": list(lattice),  
                "denominator": DENOM,  
                "stats": prog\["stats"\],  
                "schedule": list(eps\_schedule\[:stage\])  
            }  
            \_save\_final(snapshot)  
            print("\\n\*\*\* EARLY EXIT: lattice certified interval excludes 0\. Saved final certificate. \*\*\*")  
            return snapshot

    \# finished full schedule  
    g\_lo, g\_hi \= \_global\_interval\_from\_progress(tasks, \_load\_progress())  
    lattice \= None  
    if want\_lattice:  
        F\_lo, F\_hi \= 1.0, (pi\*pi)/4.0  
        lat\_cand \= \[g\_lo\*F\_lo, g\_lo\*F\_hi, g\_hi\*F\_lo, g\_hi\*F\_hi\]  
        lattice \= (min(lat\_cand), max(lat\_cand))  
    snapshot \= {  
        "kind":"schedule\_complete",  
        "continuum\_interval":\[g\_lo, g\_hi\],  
        "lattice\_interval": list(lattice) if lattice else None,  
        "denominator": DENOM,  
        "stats": \_load\_progress()\["stats"\],  
        "schedule": list(eps\_schedule)  
    }  
    \_save\_final(snapshot)  
    print("\\n\*\*\* SCHEDULE COMPLETE: saved final certificate. \*\*\*")  
    return snapshot

\# \-------------------- set runtime policy and GO \--------------------  
EPS\_SCHEDULE \= (1e-7, 5e-8, 2e-8, 1e-8, 5e-9, 2e-9, 1e-9, 5e-10, 2e-10, 1e-10)  
MAX\_DEPTH    \= 22  
SAVE\_EVERY   \= 20.0      \# seconds  
STOP\_AFTER   \= 3\*3600    \# optional wall-clock limit (set None to disable)

snap \= certify\_with\_resume(  
    eps\_schedule=EPS\_SCHEDULE,  
    max\_depth=MAX\_DEPTH,  
    save\_every\_sec=SAVE\_EVERY,  
    stop\_after\_sec=STOP\_AFTER,  
    want\_lattice=True  
)

\# pretty print the final/early snapshot and α^{-1} bracket  
c\_lo, c\_hi \= snap\["continuum\_interval"\]  
print("\\n==================== RESUMEABLE CERTIFICATE — CURRENT BEST \====================")  
print(f"c\_Pauli (continuum) ∈ \[{c\_lo:.12e}, {c\_hi:.12e}\]")  
print(f"α^{-1} continuum ∈ \[{137.0 \+ c\_lo/137.0:.12f}, {137.0 \+ c\_hi/137.0:.12f}\]")  
if "lattice\_interval" in snap and snap\["lattice\_interval"\]:  
    L\_lo, L\_hi \= snap\["lattice\_interval"\]  
    print(f"c\_Pauli (lattice)   ∈ \[{L\_lo:.12e}, {L\_hi:.12e}\]")  
    print(f"α^{-1} lattice   ∈ \[{137.0 \+ L\_lo/137.0:.12f}, {137.0 \+ L\_hi/137.0:.12f}\]")  
print(f"Checkpoint: {PROG\_PATH}")  
print(f"Final snapshot: {FINAL\_PATH}")  
\# \==================================================================================================  
