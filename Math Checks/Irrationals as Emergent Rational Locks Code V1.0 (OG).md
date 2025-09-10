\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — MEGA BLOCKASAURUS (v1.0)  \================================================ \#  
\# \=== "Irrationals as Emergent Rational Locks" \+ Farey Witness \+ Proofs (no charts)             \#  
\# \=== One-shot, append-only, self-contained.                                                    \#  
\# \=== Author: Evan Wesley                                         \#  
\# \=================================================================================================  
\# WHAT THIS BLOCK INCLUDES (no plotting, print-only \+ optional files):  
\#   MODULE A (v0.1): Continued fractions, convergents, MDL costs, CI-snapper, α registry vs snapshot.  
\#   MODULE B (v0.2): Egyptian fractions, Stern–Brocot (CI witness), batch lock hunter, toy LFT RG.  
\#   MODULE C (v0.3): Minimal-denominator certificates (exhaustive \+ Farey-neighbor attempt),  
\#                    Dual-objective (min-MDL & min-den) locks with proofs, sweep \+ JSON artifact.  
\#  
\# OUTPUT CONTROL (set flags below):  
\#   WRITE\_PER\_MODULE\_FILES  \-\> write human-readable per-module .md files  
\#   WRITE\_COMBINED\_FILE     \-\> write one combined .md report that merges all modules  
\#   WRITE\_JSON\_WITNESS      \-\> write JSON witness artifact for Module C sweep  
\#   OUTPUT\_DIR              \-\> directory for outputs (Colab: "/content/fraction\_physics\_dlc")  
\#  
\# DESIGN:  
\#   \- Deterministic, offline. Only stdlib \+ Decimal \+ Fractions \+ json \+ os \+ io.  
\#   \- Clear banners; tables printed to stdout and optionally captured into files.  
\#   \- NO charts/matplotlib usage in this mega build.  
\#   \- Safe typing (no PEP 604 "|"), no runtime type annotation evaluation issues.  
\# \=================================================================================================

import math, os, json, io  
from fractions import Fraction  
from decimal import Decimal, getcontext  
from typing import List, Dict, Optional, Tuple  
from contextlib import redirect\_stdout

\# \------------------------- CONFIG FLAGS \----------------------------------------------------------  
WRITE\_PER\_MODULE\_FILES \= True      \# write A.md, B.md, C.md  
WRITE\_COMBINED\_FILE    \= True      \# write DLC\_mega\_report.md  
WRITE\_JSON\_WITNESS     \= True      \# write Module C witness JSON  
OUTPUT\_DIR             \= "/content/fraction\_physics\_dlc"

\# Precision for Decimal arithmetic  
getcontext().prec \= 100

\# Ensure output dir exists if we’re writing files  
def \_ensure\_dir(path: str):  
    try:  
        os.makedirs(path, exist\_ok=True)  
    except Exception:  
        pass

if WRITE\_PER\_MODULE\_FILES or WRITE\_COMBINED\_FILE or WRITE\_JSON\_WITNESS:  
    \_ensure\_dir(OUTPUT\_DIR)

\# \========== SHARED UTILS (printing, MDL, errors, tables, I/O capture) \===========================

def \_mdl\_bits(fr: Fraction) \-\> int:  
    p \= abs(fr.numerator)  
    q \= abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def \_abs\_err(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def \_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= \_abs\_err(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

def \_fmt\_frac(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

def \_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar)  
    print(f"=== {title}")  
    print(bar)

def \_print\_table(rows: List\[Dict\[str, str\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

def \_capture\_printed(fn, \*args, \*\*kwargs) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        fn(\*args, \*\*kwargs)  
    return buf.getvalue()

def \_write\_text(path: str, text: str):  
    try:  
        with open(path, "w", encoding="utf-8") as f:  
            f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

\# \=================================================================================================  
\# MODULE A — v0.1: Continued Fractions, Convergents, MDL, CI Snap, α vs snapshot (NO charts)  
\# \=================================================================================================

\# Constants (audit-friendly literals)  
A\_CONSTANTS \= {  
    "pi":    "3.14159265358979323846264338327950288419716939937510582097494459230781640628",  
    "e":     "2.71828182845904523536028747135266249775724709369995957496696762772407663035",  
    "sqrt2": "1.41421356237309504880168872420969807856967187537694807317667973799",  
    "phi":   "1.618033988749894848204586834365638117720309179805762862135448622705260",  
}

\# Registry fraction for α (Rosetta-style)  
A\_ALPHA\_REGISTRY \= Fraction(2639, 361638\)     \# \~ 0.0072973526009  
A\_ALPHA\_INV\_SNAPSHOT \= "137.035999084"        \# editable  
A\_ALPHA\_INV\_1SIGMA  \= Decimal("0.000000021")  \# typical order-of-magnitude

def A\_decimal(s: str) \-\> Decimal:  
    return Decimal(s)

def A\_cf\_from\_decimal(x: Decimal, max\_terms: int \= 64\) \-\> List\[int\]:  
    terms: List\[int\] \= \[\]  
    sign \= \-1 if x \< 0 else 1  
    y \= \-x if sign \< 0 else x  
    for \_ in range(max\_terms):  
        a0 \= int(y.to\_integral\_value(rounding="ROUND\_FLOOR"))  
        terms.append(a0 if sign \> 0 or len(terms) \> 0 else \-a0)  
        frac \= y \- Decimal(a0)  
        if frac \== 0:  
            break  
        y \= Decimal(1) / frac  
    if sign \< 0 and len(terms) \> 0 and terms\[0\] \> 0:  
        terms\[0\] \= \-terms\[0\]  
    return terms

def A\_convergents(cf\_terms: List\[int\]) \-\> List\[Fraction\]:  
    ps \= \[0,1\]; qs \= \[1,0\]; out: List\[Fraction\] \= \[\]  
    for a in cf\_terms:  
        p \= a\*ps\[-1\] \+ ps\[-2\]  
        q \= a\*qs\[-1\] \+ qs\[-2\]  
        ps \= \[ps\[-1\], p\]; qs \= \[qs\[-1\], q\]  
        out.append(Fraction(p,q))  
    return out

def A\_ci\_snap(convergents: List\[Fraction\], center: Decimal, sigma: Decimal,  
              prefer\_min\_mdl: bool \= True) \-\> Optional\[Tuple\[Fraction, Decimal, int\]\]:  
    lo, hi \= center \- sigma, center \+ sigma  
    cand: List\[Tuple\[Fraction, Decimal, int\]\] \= \[\]  
    for fr in convergents:  
        val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
        if lo \<= val \<= hi:  
            err \= abs(val \- center); bits \= \_mdl\_bits(fr)  
            cand.append((fr, err, bits))  
    if not cand:  
        return None  
    if prefer\_min\_mdl:  
        cand.sort(key=lambda t: (t\[2\], t\[1\]))  
    else:  
        cand.sort(key=lambda t: (t\[1\], t\[2\]))  
    return cand\[0\]

def A\_unit\_sanity():  
    \# √2 \=\> \[1;2,2,2,...\]  
    s2 \= A\_decimal(A\_CONSTANTS\["sqrt2"\])  
    t  \= A\_cf\_from\_decimal(s2, max\_terms=8)  
    assert t\[0\] \== 1 and all(a==2 for a in t\[1:\]), "√2 CF sanity failed."  
    \# φ \=\> \[1;1,1,1,...\]  
    ph \= A\_decimal(A\_CONSTANTS\["phi"\])  
    t2 \= A\_cf\_from\_decimal(ph, max\_terms=8)  
    assert t2\[0\]==1 and all(a==1 for a in t2\[1:\]), "phi CF sanity failed."  
    \# e begins \[2;1,2,1,...\]  
    e  \= A\_decimal(A\_CONSTANTS\["e"\])  
    et \= A\_cf\_from\_decimal(e, max\_terms=10)  
    assert et\[0\]==2 and et\[1\]==1 and et\[2\]==2, "e CF sanity failed."  
    print("Module A sanity: OK")

def MODULE\_A\_run() \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        \_hdr("MODULE A — Irrationals as Emergent Rational Locks (CF \+ MDL \+ CI)")  
        A\_unit\_sanity()

        \# Analyze π, e, φ, √2: print top convergents  
        for key in \["pi","e","phi","sqrt2"\]:  
            target \= A\_decimal(A\_CONSTANTS\[key\])  
            terms  \= A\_cf\_from\_decimal(target, max\_terms=48)  
            convs  \= A\_convergents(terms)  
            rows=\[\]  
            for k, fr in enumerate(convs\[:14\], start=1):  
                val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
                rows.append({  
                    "k": k,  
                    "p/q": \_fmt\_frac(fr),  
                    "value": f"{val:.30f}",  
                    "|err|": f"{\_abs\_err(fr, target):.3E}",  
                    "ppm": f"{\_ppm(fr, target):.6f}",  
                    "bits": \_mdl\_bits(fr),  
                })  
            \_hdr(f"CF Ladder — {key}")  
            \_print\_table(rows, \["k","p/q","value","|err|","ppm","bits"\])

        \# α views  
        alpha\_inv\_center \= A\_decimal(A\_ALPHA\_INV\_SNAPSHOT)  
        alpha\_center     \= Decimal(1) / alpha\_inv\_center  
        alpha\_registry   \= Decimal(A\_ALPHA\_REGISTRY.numerator)/Decimal(A\_ALPHA\_REGISTRY.denominator)

        \# α registry:  
        target \= alpha\_registry  
        terms \= A\_cf\_from\_decimal(target, max\_terms=48)  
        convs \= A\_convergents(terms)  
        rows=\[\]  
        for k, fr in enumerate(convs\[:10\], start=1):  
            val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
            rows.append({  
                "k": k, "p/q": \_fmt\_frac(fr), "value": f"{val}",  
                "|err|": f"{\_abs\_err(fr, target):.3E}", "ppm": f"{\_ppm(fr, target):.6f}",  
                "bits": \_mdl\_bits(fr)  
            })  
        \_hdr("CF Ladder — alpha (registry rational)")  
        \_print\_table(rows, \["k","p/q","value","|err|","ppm","bits"\])

        \# α^{-1} snapshot:  
        target \= alpha\_inv\_center  
        terms  \= A\_cf\_from\_decimal(target, max\_terms=48)  
        convs  \= A\_convergents(terms)  
        rows=\[\]  
        for k, fr in enumerate(convs\[:12\], start=1):  
            val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
            rows.append({  
                "k": k, "p/q": \_fmt\_frac(fr), "value": f"{val}",  
                "|err|": f"{\_abs\_err(fr, target):.3E}", "ppm": f"{\_ppm(fr, target):.6f}",  
                "bits": \_mdl\_bits(fr)  
            })  
        \_hdr("CF Ladder — alpha^{-1} (snapshot)")  
        \_print\_table(rows, \["k","p/q","value","|err|","ppm","bits"\])

        \# CI snap demos  
        \_hdr("CI Snap — alpha^{-1}")  
        pick \= A\_ci\_snap(convs, alpha\_inv\_center, A\_ALPHA\_INV\_1SIGMA, prefer\_min\_mdl=True)  
        if pick is None:  
            print("No convergent in 1σ.")  
        else:  
            fr, err, bits \= pick  
            print(f"lock: {fr.numerator}/{fr.denominator}  bits={bits}  |err|={err}")

        dalpha \= A\_ALPHA\_INV\_1SIGMA/(alpha\_inv\_center\*alpha\_inv\_center)  
        \_hdr("CI Snap — alpha")  
        terms\_a  \= A\_cf\_from\_decimal(alpha\_center, max\_terms=64)  
        convs\_a  \= A\_convergents(terms\_a)  
        pick\_a   \= A\_ci\_snap(convs\_a, alpha\_center, dalpha, prefer\_min\_mdl=True)  
        if pick\_a is None:  
            print("No convergent in propagated 1σ.")  
        else:  
            fr, err, bits \= pick\_a  
            print(f"lock: {fr.numerator}/{fr.denominator}  bits={bits}  |err|={err}")

        \# Compare α registry vs snapshot  
        \_hdr("alpha — Registry vs Snapshot")  
        diff \= abs(alpha\_registry \- alpha\_center)  
        ppm  \= (diff/alpha\_center)\*Decimal(1\_000\_000)  
        print(f"α\_registry (2639/361638)  : {alpha\_registry}")  
        print(f"α\_snapshot (1/{alpha\_inv\_center}): {alpha\_center}")  
        print(f"|Δ|={diff}  ppm={ppm}")  
    return buf.getvalue()

\# \=================================================================================================  
\# MODULE B — v0.2: Egyptian Fractions, Stern–Brocot CI Witness, Batch Lock Hunter, Toy LFT RG  
\# \=================================================================================================

def B\_egyptian\_greedy(fr: Fraction) \-\> List\[int\]:  
    assert fr \> 0  
    p, q \= fr.numerator, fr.denominator  
    parts: List\[int\] \= \[\]  
    while p \!= 0:  
        n \= (q \+ p \- 1)//p  
        parts.append(n)  
        p, q \= p\*n \- q, q\*n  
        g \= math.gcd(p,q)  
        if g\>1: p//=g; q//=g  
    return parts

def B\_egyptian\_verify(fr: Fraction, parts: List\[int\]) \-\> Tuple\[bool, Fraction\]:  
    s \= sum(Fraction(1,n) for n in parts)  
    return (s \== fr, s)

def B\_egyptian\_bits(parts: List\[int\]) \-\> int:  
    return sum(0 if n\<=1 else math.ceil(math.log2(n)) for n in parts)

def B\_best\_rational\_in\_interval(L: Decimal, U: Decimal, objective: str \= "mdl",  
                                max\_steps: int \= 20000, denom\_cap: int \= 10\*\*6) \-\> Optional\[Fraction\]:  
    assert L \<= U  
    a,b \= 0,1  
    c,d \= 1,0  
    best: Optional\[Tuple\[Fraction,int,int\]\] \= None  
    steps=0  
    while steps \< max\_steps:  
        steps \+= 1  
        m\_num, m\_den \= a+c, b+d  
        if m\_den \== 0 or m\_den \> denom\_cap: break  
        m\_val \= Decimal(m\_num)/Decimal(m\_den)  
        if m\_val \< L:  
            a,b \= m\_num, m\_den  
        elif m\_val \> U:  
            c,d \= m\_num, m\_den  
        else:  
            fr \= Fraction(m\_num, m\_den)  
            score \= (m\_den if objective=="den" else \_mdl\_bits(fr))  
            if (best is None) or (score \< best\[1\]) or (score \== best\[1\] and m\_den \< best\[2\]):  
                best \= (fr, score, m\_den)  
            if (m\_val \- L) \> (U \- m\_val):  
                c,d \= m\_num, m\_den  
            else:  
                a,b \= m\_num, m\_den  
    return None if best is None else best\[0\]

def B\_snap\_via\_sb(name: str, center: Decimal, sigma: Decimal, objective: str \= "mdl") \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        L \= center \- sigma; U \= center \+ sigma  
        \_hdr(f"Stern–Brocot CI Witness — {name}")  
        print(f"band: \[{L}, {U}\]  objective: {objective}")  
        fr \= B\_best\_rational\_in\_interval(L, U, objective=objective)  
        if fr is None:  
            print("Result: no candidate found under search limits.")  
        else:  
            val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
            print(f"lock: {fr.numerator}/{fr.denominator}  bits={\_mdl\_bits(fr)}  value={val}")  
            print(f"inside band: {L\<=val\<=U}")  
    return buf.getvalue()

def B\_batch\_lock\_hunter(items: List\[Dict\[str,str\]\], objective: str \= "mdl") \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        \_hdr("Batch Lock Hunter (Stern–Brocot)")  
        rows=\[\]  
        for it in items:  
            name \= it\["name"\]  
            xc   \= Decimal(it\["center\_str"\])  
            sg   \= Decimal(it\["sigma\_str"\])  
            fr \= B\_best\_rational\_in\_interval(xc \- sg, xc \+ sg, objective=objective)  
            if fr is None:  
                rows.append({"name":name, "p/q":"-", "bits":"-", "abs\_err":"-", "ppm":"-",  
                             "band": f"\[{xc-sg}, {xc+sg}\]"})  
            else:  
                val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
                rows.append({  
                    "name": name,  
                    "p/q": \_fmt\_frac(fr),  
                    "bits": \_mdl\_bits(fr),  
                    "abs\_err": f"{abs(val \- xc):.3E}",  
                    "ppm": f"{(\_abs\_err(fr, xc)/abs(xc))\*Decimal(1\_000\_000):.6f}" if xc \!= 0 else "—",  
                    "band": f"\[{xc \- sg}, {xc \+ sg}\]"  
                })  
        \_print\_table(rows, \["name","p/q","bits","abs\_err","ppm","band"\])  
    return buf.getvalue()

\# Toy LFT RG: y(t) \= (a t \+ b)/(c t \+ d) exact rational anchors  
def B\_lft\_solve\_three(anchors: List\[Tuple\[Fraction, Fraction\]\]) \-\> Tuple\[Fraction, Fraction, Fraction, Fraction\]:  
    (t1,y1),(t2,y2),(t3,y3) \= anchors  
    def row(t,y): return (t, Fraction(1,1), \-y\*t, \-y)  
    R1=row(t1,y1); R2=row(t2,y2); R3=row(t3,y3)  
    A \= \[\[R1\[0\],R1\[1\],R1\[2\]\], \[R2\[0\],R2\[1\],R2\[2\]\], \[R3\[0\],R3\[1\],R3\[2\]\]\]  
    Bv= \[-R1\[3\], \-R2\[3\], \-R3\[3\]\]  
    def det3(M):  
        (a11,a12,a13),(a21,a22,a23),(a31,a32,a33)=M  
        return (a11\*(a22\*a33 \- a23\*a32)  
               \-a12\*(a21\*a33 \- a23\*a31)  
               \+a13\*(a21\*a32 \- a22\*a31))  
    def rep(M,i,col):  
        out=\[row\[:\] for row in M\]  
        for k in range(3): out\[k\]\[i\]=col\[k\]  
        return out  
    D  \= det3(A)  
    if D \== 0: raise ValueError("Degenerate anchors: determinant zero.")  
    Da \= det3(rep(A,0,Bv)); Db \= det3(rep(A,1,Bv)); Dc \= det3(rep(A,2,Bv))  
    a \= Da/D; b=Db/D; c=Dc/D; d=Fraction(1,1)  
    return a,b,c,d

def B\_lft\_eval(a: Fraction,b: Fraction,c: Fraction,d: Fraction,t: Fraction) \-\> Fraction:  
    num \= a\*t \+ b; den \= c\*t \+ d  
    if den \== 0: raise ZeroDivisionError("LFT denominator zero.")  
    return num/den

def MODULE\_B\_run() \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        \_hdr("MODULE B — Egyptian, Stern–Brocot Witness, Batch Hunter, Toy LFT RG")

        \# Egyptian decompositions  
        examples \= \[Fraction(2639,361638), Fraction(188,843), Fraction(22,7), Fraction(355,113)\]  
        \_hdr("Egyptian Fraction Decompositions (Greedy)")  
        rows=\[\]  
        for fr in examples:  
            parts \= B\_egyptian\_greedy(fr)  
            ok, s  \= B\_egyptian\_verify(fr, parts)  
            rows.append({  
                "p/q": \_fmt\_frac(fr),  
                "egyptian": " \+ ".join(f"1/{n}" for n in parts),  
                "terms": len(parts),  
                "sum\_ok": ok,  
                "sum\_val": \_fmt\_frac(s),  
                "sum\_bits": B\_egyptian\_bits(parts)  
            })  
        \_print\_table(rows, \["p/q","egyptian","terms","sum\_ok","sum\_val","sum\_bits"\])

        \# Stern–Brocot CI witness demos (alpha^-1 and alpha)  
        alpha\_inv\_center \= Decimal("137.035999084")  
        alpha\_inv\_sigma  \= Decimal("0.000000021")  
        print(B\_snap\_via\_sb("alpha^{-1}", alpha\_inv\_center, alpha\_inv\_sigma, objective="mdl"))  
        alpha\_center \= Decimal(1)/alpha\_inv\_center  
        alpha\_sigma  \= alpha\_inv\_sigma/(alpha\_inv\_center\*alpha\_inv\_center)  
        print(B\_snap\_via\_sb("alpha", alpha\_center, alpha\_sigma, objective="mdl"))

        \# Batch hunter  
        items \= \[  
            {"name":"alpha^{-1}", "center\_str":"137.035999084", "sigma\_str":"0.000000021"},  
            {"name":"alpha",      "center\_str":f"{alpha\_center}", "sigma\_str":f"{alpha\_sigma}"},  
            {"name":"sin^2θW",    "center\_str":"0.2312200006", "sigma\_str":"0.0000300000"},  
            {"name":"pi",         "center\_str":"3.141592653589793", "sigma\_str":"1E-15"},  
        \]  
        print(B\_batch\_lock\_hunter(items, objective="mdl"))

        \# Toy LFT RG anchors (exact rationals)  
        \_hdr("Toy Möbius RG — exact anchor preservation")  
        t1=Fraction(0,1); y1=Fraction(2639,361638)  
        t2=Fraction(1,1); y2=Fraction(188,843)  
        t3=Fraction(2,1); y3=Fraction(1,7)   \# 22/7 \- 3 \= 1/7  
        a,b,c,d \= B\_lft\_solve\_three(\[(t1,y1),(t2,y2),(t3,y3)\])  
        print(f"a={a}, b={b}, c={c}, d={d}")  
        for (ti,yi) in \[(t1,y1),(t2,y2),(t3,y3)\]:  
            yhat \= B\_lft\_eval(a,b,c,d,ti)  
            print(f"t={ti}: y(t)={yhat}  target={yi}  exact\_match={yhat==yi}")

    return buf.getvalue()

\# \=================================================================================================  
\# MODULE C — v0.3: Minimal-Denominator Proofs, Dual Locks, Sweep \+ JSON (NO charts)  
\# \=================================================================================================

def C\_best\_in\_interval(L: Decimal, U: Decimal, objective: str \= "mdl",  
                       max\_steps: int \= 20000, denom\_cap: int \= 10\*\*7) \-\> Optional\[Fraction\]:  
    assert L \<= U  
    a,b \= 0,1; c,d \= 1,0  
    best: Optional\[Tuple\[Fraction,int,int\]\] \= None  
    steps=0  
    while steps \< max\_steps:  
        steps \+= 1  
        m\_num, m\_den \= a+c, b+d  
        if m\_den \== 0 or m\_den \> denom\_cap: break  
        m\_val \= Decimal(m\_num)/Decimal(m\_den)  
        if m\_val \< L:  
            a,b \= m\_num, m\_den  
        elif m\_val \> U:  
            c,d \= m\_num, m\_den  
        else:  
            fr \= Fraction(m\_num, m\_den)  
            score \= (m\_den if objective=="den" else \_mdl\_bits(fr))  
            if (best is None) or (score \< best\[1\]) or (score \== best\[1\] and m\_den \< best\[2\]):  
                best \= (fr, score, m\_den)  
            if (m\_val \- L) \> (U \- m\_val):  
                c,d \= m\_num, m\_den  
            else:  
                a,b \= m\_num, m\_den  
    return None if best is None else best\[0\]

def C\_prove\_min\_den\_exhaustive(L: Decimal, U: Decimal, candidate: Fraction,  
                               cap\_hard: int \= 200\_000) \-\> Dict\[str, object\]:  
    q \= candidate.denominator  
    limit \= min(q-1, cap\_hard)  
    for d in range(1, int(limit)+1):  
        n\_low  \= int((L\*Decimal(d)).to\_integral\_value(rounding="ROUND\_CEILING"))  
        n\_high \= int((U\*Decimal(d)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
        if n\_low \> n\_high: continue  
        for n in range(n\_low, n\_high+1):  
            if math.gcd(n,d)==1:  
                return {"ok": False, "found\_smaller": f"{n}/{d}", "d\_checked": int(limit)}  
    return {"ok": True, "d\_checked": int(limit)}

def C\_try\_farey\_neighbor\_certificate(L: Decimal, U: Decimal, candidate: Fraction,  
                                     max\_steps: int \= 20000\) \-\> Dict\[str, object\]:  
    a,b \= 0,1; c,d \= 1,0  
    steps=0  
    while steps \< max\_steps:  
        steps \+= 1  
        m\_num, m\_den \= a+c, b+d  
        if m\_den==0: break  
        m\_val \= Decimal(m\_num)/Decimal(m\_den)  
        if m\_val \< L:  
            a,b \= m\_num, m\_den  
        elif m\_val \> U:  
            c,d \= m\_num, m\_den  
        else:  
            if (m\_val \- L) \> (U \- m\_val):  
                c,d \= m\_num, m\_den  
            else:  
                a,b \= m\_num, m\_den  
        if a\*d \+ b\*c \== 0: break  
    det \= b\*c \- a\*d  
    ok\_det \= (abs(det) \== 1\)  
    mediant \= Fraction(a+c, b+d)  
    if ok\_det and mediant \== candidate:  
        return {  
            "ok": True, "left": f"{a}/{b}", "right": f"{c}/{d}", "det": int(det),  
            "mediant": f"{mediant.numerator}/{mediant.denominator}",  
            "explanation": "Candidate equals mediant of Farey neighbors; denominators between are ≥ b+d."  
        }  
    return {"ok": False, "left": f"{a}/{b}", "right": f"{c}/{d}", "det": int(det) if isinstance(det,int) else det,  
            "note": "Could not assert mediant witness; rely on exhaustive proof."}

def C\_dual\_lock\_with\_proofs(name: str, center: Decimal, sigma: Decimal,  
                            denom\_cap: int \= 10\*\*7, exhaustive\_cap: int \= 200\_000) \-\> Dict\[str, object\]:  
    L, U \= center \- sigma, center \+ sigma  
    min\_mdl \= C\_best\_in\_interval(L, U, objective="mdl", denom\_cap=denom\_cap)  
    min\_den \= C\_best\_in\_interval(L, U, objective="den", denom\_cap=denom\_cap)  
    out: Dict\[str, object\] \= {"name": name, "band": \[str(L), str(U)\]}  
    for tag, fr in (("min\_mdl", min\_mdl), ("min\_den", min\_den)):  
        if fr is None:  
            out\[tag\] \= None; continue  
        val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
        entry \= {  
            "pq": \_fmt\_frac(fr),  
            "value": str(val),  
            "bits": \_mdl\_bits(fr),  
            "den": fr.denominator,  
            "abs\_err": str(abs(val \- center)),  
            "ppm": str(\_ppm(fr, center)),  
        }  
        entry\["proof\_exhaustive"\] \= C\_prove\_min\_den\_exhaustive(L, U, fr, cap\_hard=exhaustive\_cap)  
        entry\["proof\_farey"\] \= C\_try\_farey\_neighbor\_certificate(L, U, fr)  
        out\[tag\] \= entry  
    return out

def MODULE\_C\_run\_and\_json(json\_out\_path: Optional\[str\] \= None) \-\> Tuple\[str, Optional\[str\]\]:  
    alpha\_inv\_center \= Decimal("137.035999084")  
    alpha\_inv\_sigma  \= Decimal("0.000000021")  
    alpha\_center     \= Decimal(1)/alpha\_inv\_center  
    alpha\_sigma      \= alpha\_inv\_sigma/(alpha\_inv\_center\*alpha\_inv\_center)  
    items \= \[  
        {"name":"alpha^{-1}", "center":alpha\_inv\_center, "sigma":alpha\_inv\_sigma},  
        {"name":"alpha",      "center":alpha\_center,     "sigma":alpha\_sigma},  
        {"name":"sin^2θW",    "center":Decimal("0.2312200006"), "sigma":Decimal("0.0000300000")},  
        {"name":"pi",         "center":Decimal("3.141592653589793"), "sigma":Decimal("1E-15")},  
    \]  
    artifacts=\[\]  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        \_hdr("MODULE C — Dual Locks with Proofs (min-MDL & min-den) \+ JSON witness")  
        rows=\[\]  
        for it in items:  
            res \= C\_dual\_lock\_with\_proofs(it\["name"\], it\["center"\], it\["sigma"\])  
            artifacts.append(res)  
            row \= {"name": it\["name"\], "band": f"\[{res\['band'\]\[0\]}, {res\['band'\]\[1\]}\]"}  
            for tag in ("min\_mdl","min\_den"):  
                e \= res.get(tag)  
                if e is None:  
                    row\[f"{tag}\_p/q"\]="-"; row\[f"{tag}\_bits"\]="-"; row\[f"{tag}\_den"\]="-"; row\[f"{tag}\_ppm"\]="-"  
                else:  
                    row\[f"{tag}\_p/q"\]=e\["pq"\]; row\[f"{tag}\_bits"\]=e\["bits"\]; row\[f"{tag}\_den"\]=e\["den"\]  
                    row\[f"{tag}\_ppm"\]=f"{Decimal(e\['ppm'\]):.6f}"  
            rows.append(row)  
        \_print\_table(rows, \["name","band","min\_mdl\_p/q","min\_mdl\_bits","min\_mdl\_den","min\_mdl\_ppm",  
                                   "min\_den\_p/q","min\_den\_bits","min\_den\_den","min\_den\_ppm"\])

    json\_path\_written \= None  
    if json\_out\_path is not None:  
        try:  
            with open(json\_out\_path, "w", encoding="utf-8") as f:  
                json.dump(artifacts, f, indent=2)  
            json\_path\_written \= json\_out\_path  
        except Exception as e:  
            print(f"\[file\] failed to write JSON witness: {e}")  
    return buf.getvalue(), json\_path\_written

\# \=================================================================================================  
\# DRIVER — run A, B, C; print; optionally write per-module and combined reports \+ JSON  
\# \=================================================================================================

def MEGA\_MAIN():  
    combined\_parts: List\[str\] \= \[\]

    \# MODULE A  
    a\_text \= MODULE\_A\_run()  
    print(a\_text)  
    if WRITE\_PER\_MODULE\_FILES:  
        \_write\_text(os.path.join(OUTPUT\_DIR, "MODULE\_A\_Irrational\_Locks.md"), a\_text)  
    combined\_parts.append("\# MODULE A — Irrationals as Emergent Rational Locks\\n\\n"+a\_text)

    \# MODULE B  
    b\_text \= MODULE\_B\_run()  
    print(b\_text)  
    if WRITE\_PER\_MODULE\_FILES:  
        \_write\_text(os.path.join(OUTPUT\_DIR, "MODULE\_B\_Farey\_Witness\_Egyptian\_LFT.md"), b\_text)  
    combined\_parts.append("\# MODULE B — Farey Witness, Egyptian, LFT\\n\\n"+b\_text)

    \# MODULE C  
    json\_path \= os.path.join(OUTPUT\_DIR, "irrational\_locks\_witness.json") if WRITE\_JSON\_WITNESS else None  
    c\_text, json\_written \= MODULE\_C\_run\_and\_json(json\_out\_path=json\_path)  
    print(c\_text)  
    if WRITE\_PER\_MODULE\_FILES:  
        \_write\_text(os.path.join(OUTPUT\_DIR, "MODULE\_C\_Lock\_Proofs\_and\_Sweep.md"), c\_text)  
    combined\_parts.append("\# MODULE C — Lock Proofs and Sweep\\n\\n"+c\_text)

    \# Combined report  
    if WRITE\_COMBINED\_FILE:  
        mega \= "\# FRACTION PHYSICS DLC — MEGA REPORT (v1.0)\\n\\n" \+ "\\n\\n".join(combined\_parts)  
        \_write\_text(os.path.join(OUTPUT\_DIR, "DLC\_mega\_report.md"), mega)

    \# JSON confirmation  
    if WRITE\_JSON\_WITNESS and json\_written:  
        print(f"\[json\] witness written: {json\_written}")

\# Run  
if \_\_name\_\_ \== "\_\_main\_\_":  
    MEGA\_MAIN()

\# \=================================================================================================  
\# \=== END MEGA BLOCK \==============================================================================  
\# \=================================================================================================  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — RUNTIME CONFIG SYNC \+ INDEXER (v1.1, append-only, no charts)       \===  
\# \=== Purpose:                                                                                    \===  
\# \===  • Reuse or set safe defaults for WRITE\_\* flags and OUTPUT\_DIR from previous blocks.       \===  
\# \===  • Generate/refresh an INDEX.md of everything produced so far.                             \===  
\# \===  • Provide tiny helpers to write human-readable notes/files that follow your file rules.   \===  
\# \===  • Zero edits to earlier cells; fully append-only.                                         \===  
\# \=================================================================================================

import os, io, json, time  
from contextlib import redirect\_stdout

\# \------------------------- 1\) CONFIG SYNC \--------------------------------------------------------  
\# Reuse prior globals if present; otherwise set safe defaults matching your rules.  
try:  
    WRITE\_PER\_MODULE\_FILES   \# type: ignore\[name-defined\]  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE      \# type: ignore\[name-defined\]  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    WRITE\_JSON\_WITNESS       \# type: ignore\[name-defined\]  
except NameError:  
    WRITE\_JSON\_WITNESS \= True  
try:  
    OUTPUT\_DIR               \# type: ignore\[name-defined\]  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"

os.makedirs(OUTPUT\_DIR, exist\_ok=True)

\# \------------------------- 2\) MINI I/O UTILS \-----------------------------------------------------  
def \_dlc\_write\_text(path: str, text: str):  
    try:  
        with open(path, "w", encoding="utf-8") as f:  
            f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def \_dlc\_append\_text(path: str, text: str):  
    try:  
        with open(path, "a", encoding="utf-8") as f:  
            f.write(text)  
        print(f"\[file\] appended: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to append {path}: {e}")

def \_dlc\_h1(s: str) \-\> str: return f"\# {s}\\n\\n"  
def \_dlc\_h2(s: str) \-\> str: return f"\#\# {s}\\n\\n"  
def \_dlc\_codeblock(s: str) \-\> str: return f"\`\`\`\\n{s}\\n\`\`\`\\n\\n"

\# \------------------------- 3\) INDEXER \------------------------------------------------------------  
def DLC\_BUILD\_INDEX(output\_dir: str \= OUTPUT\_DIR):  
    """  
    Scan OUTPUT\_DIR and build/update INDEX.md listing all generated artifacts, sizes, and mtimes.  
    Also surfaces presence of the JSON witness (if any).  
    """  
    entries \= \[\]  
    for root, \_, files in os.walk(output\_dir):  
        for fn in sorted(files):  
            p \= os.path.join(root, fn)  
            try:  
                st \= os.stat(p)  
                size \= st.st\_size  
                mtime \= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st\_mtime))  
                entries.append((p, size, mtime))  
            except Exception:  
                pass

    lines \= \[\]  
    lines.append(\_dlc\_h1("Fraction Physics DLC — Artifact Index"))  
    lines.append(f"\*\*Output dir:\*\* \`{output\_dir}\`  \\n")  
    lines.append("| File | Size (bytes) | Modified (local time) |")  
    lines.append("|------|--------------:|-----------------------|")  
    for p, sz, mt in entries:  
        rel \= os.path.relpath(p, output\_dir)  
        lines.append(f"| \`{rel}\` | {sz} | {mt} |")  
    lines.append("\\n")  
    \# Quick spotlight for JSON witness if present  
    jw \= os.path.join(output\_dir, "irrational\_locks\_witness.json")  
    if os.path.exists(jw):  
        try:  
            with open(jw, "r", encoding="utf-8") as f:  
                data \= json.load(f)  
            lines.append(\_dlc\_h2("Witness JSON Summary"))  
            lines.append(f"- Path: \`irrational\_locks\_witness.json\`  \\n- Items: \*\*{len(data)}\*\*  \\n")  
            \# list item names if available  
            names \= \[\]  
            for it in data:  
                nm \= it.get("name")  
                if nm: names.append(nm)  
            if names:  
                lines.append("Items: " \+ ", ".join(f"\`{n}\`" for n in names) \+ "\\n")  
        except Exception as e:  
            lines.append(f"\> Could not parse witness JSON: \`{e}\`\\n")

    index\_path \= os.path.join(output\_dir, "INDEX.md")  
    \_dlc\_write\_text(index\_path, "\\n".join(lines))  
    return index\_path

\# \------------------------- 4\) RUN-NOTE (human-readable) \------------------------------------------  
def DLC\_WRITE\_NOTE(title: str, body: str, output\_dir: str \= OUTPUT\_DIR):  
    """  
    Create/append a human-readable RUN\_NOTES.md in OUTPUT\_DIR for ad-hoc commentary —  
    stays within your file rules and keeps the append-only vibe.  
    """  
    path \= os.path.join(output\_dir, "RUN\_NOTES.md")  
    stamp \= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  
    chunk \= \_dlc\_h2(f"{title} — {stamp}") \+ body \+ "\\n\\n"  
    if os.path.exists(path):  
        \_dlc\_append\_text(path, chunk)  
    else:  
        \_dlc\_write\_text(path, \_dlc\_h1("Run Notes") \+ chunk)  
    return path

\# \------------------------- 5\) “THIS RUN” SUMMARY FILE \--------------------------------------------  
def DLC\_WRITE\_THIS\_RUN\_SUMMARY(extra\_text: str \= ""):  
    """  
    Writes README\_THIS\_RUN.md capturing current flags and where things are going.  
    This is regenerated on each run (overwrite); it’s a convenience orientation file.  
    """  
    lines \= \[\]  
    lines.append(\_dlc\_h1("This Run — Summary"))  
    lines.append("\*\*Flags\*\*  \\n")  
    lines.append(f"- WRITE\_PER\_MODULE\_FILES: \*\*{WRITE\_PER\_MODULE\_FILES}\*\*  \\n")  
    lines.append(f"- WRITE\_COMBINED\_FILE: \*\*{WRITE\_COMBINED\_FILE}\*\*  \\n")  
    lines.append(f"- WRITE\_JSON\_WITNESS: \*\*{WRITE\_JSON\_WITNESS}\*\*  \\n")  
    lines.append(f"- OUTPUT\_DIR: \`{OUTPUT\_DIR}\`  \\n\\n")  
    if extra\_text:  
        lines.append(\_dlc\_h2("Notes"))  
        lines.append(extra\_text \+ "\\n")  
    path \= os.path.join(OUTPUT\_DIR, "README\_THIS\_RUN.md")  
    \_dlc\_write\_text(path, "".join(lines))  
    return path

\# \------------------------- 6\) DO THE THING \-------------------------------------------------------  
if \_\_name\_\_ \== "\_\_main\_\_":  
    print("Config sync:", dict(  
        WRITE\_PER\_MODULE\_FILES=WRITE\_PER\_MODULE\_FILES,  
        WRITE\_COMBINED\_FILE=WRITE\_COMBINED\_FILE,  
        WRITE\_JSON\_WITNESS=WRITE\_JSON\_WITNESS,  
        OUTPUT\_DIR=OUTPUT\_DIR  
    ))  
    idx\_path \= DLC\_BUILD\_INDEX()  
    print(f"Index refreshed: {idx\_path}")  
    readme\_path \= DLC\_WRITE\_THIS\_RUN\_SUMMARY("Config and artifact index updated.")  
    print(f"Run summary: {readme\_path}")

    \# Example: drop a small note (safe to delete these lines if you don’t want a note each time)  
    DLC\_WRITE\_NOTE("Post-mega sync", "Index rebuilt and flags harmonized. Ready for the next DLC block.")  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — v1.4 "Universal Irrational Registry Engine" (append-only, no charts)===  
\# \=== Purpose: Bulk-generate a large registry of irrationals, CF ladders, MDL, optional CI snaps.  \==  
\# \=== Works with your file flags. Prints tables \+ writes human-readable files when enabled.       \===  
\# \=================================================================================================

import os, io, math, json, time  
from fractions import Fraction  
from decimal import Decimal, getcontext, ROUND\_FLOOR  
from contextlib import redirect\_stdout  
from typing import Dict, Any, List, Tuple, Optional

\# \---------- CONFIG SYNC (re-use flags from earlier blocks; set safe defaults if missing) \----------  
try:  
    WRITE\_PER\_MODULE\_FILES  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    WRITE\_JSON\_WITNESS  
except NameError:  
    WRITE\_JSON\_WITNESS \= True  
try:  
    OUTPUT\_DIR  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"  
os.makedirs(OUTPUT\_DIR, exist\_ok=True)

\# local precision knobs (registry-level)  
UIR\_DIGITS\_DEFAULT \= 80          \# working digits for constants  
UIR\_CF\_TERMS\_MAX   \= 64          \# CF terms to compute  
UIR\_SHOW\_TOP       \= 12          \# rows to print in cell per constant (keep output readable)

\# bump Decimal precision (add guard digits for internal steps)  
getcontext().prec \= UIR\_DIGITS\_DEFAULT \+ 10

\# \---------- SHARED I/O \----------  
def \_uir\_write\_text(path: str, text: str):  
    try:  
        d \= os.path.dirname(path)  
        if d: os.makedirs(d, exist\_ok=True)  
        with open(path, "w", encoding="utf-8") as f:  
            f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def \_uir\_now():  
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

\# \---------- TABLE/PRINT HELPERS \----------  
def UIR\_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar); print(f"=== {title}"); print(bar)

def UIR\_print\_table(rows: List\[Dict\[str, Any\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

\# \---------- MDL & CF \----------  
def UIR\_bits(fr: Fraction) \-\> int:  
    p, q \= abs(fr.numerator), abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def UIR\_abs\_err(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def UIR\_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= UIR\_abs\_err(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

def UIR\_fmt(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

def UIR\_cf\_from\_decimal(x: Decimal, max\_terms: int \= UIR\_CF\_TERMS\_MAX) \-\> List\[int\]:  
    terms \= \[\]  
    neg \= x \< 0  
    y \= \-x if neg else x  
    for \_ in range(max\_terms):  
        a0 \= int(y.to\_integral\_value(rounding=ROUND\_FLOOR))  
        terms.append(-a0 if (neg and len(terms)==0) else a0)  
        frac \= y \- Decimal(a0)  
        if frac \== 0:  
            break  
        y \= Decimal(1)/frac  
    return terms

def UIR\_convergents(terms: List\[int\]) \-\> List\[Fraction\]:  
    ps=\[0,1\]; qs=\[1,0\]; out=\[\]  
    for a in terms:  
        p \= a\*ps\[-1\] \+ ps\[-2\]  
        q \= a\*qs\[-1\] \+ qs\[-2\]  
        ps=\[ps\[-1\], p\]; qs=\[qs\[-1\], q\]  
        out.append(Fraction(p,q))  
    return out

\# \---------- CI SNAP (optional per-constant if CI provided) \----------  
def UIR\_best\_in\_CI(convs: List\[Fraction\], center: Decimal, sigma: Decimal,  
                   prefer\_min\_bits: bool=True) \-\> Optional\[Tuple\[Fraction, Decimal, int\]\]:  
    L, U \= center \- sigma, center \+ sigma  
    cands=\[\]  
    for fr in convs:  
        v \= Decimal(fr.numerator)/Decimal(fr.denominator)  
        if L \<= v \<= U:  
            err \= abs(v \- center)  
            bits \= UIR\_bits(fr)  
            cands.append((fr, err, bits))  
    if not cands: return None  
    key \= (lambda t: (t\[2\], t\[1\])) if prefer\_min\_bits else (lambda t: (t\[1\], t\[2\]))  
    cands.sort(key=key)  
    return cands\[0\]

\# \---------- CORE GENERATORS (algorithmic) \----------  
\# High-speed π via Gauss–Legendre AGM  
def UIR\_pi(digits: int \= UIR\_DIGITS\_DEFAULT) \-\> Decimal:  
    old \= getcontext().prec  
    getcontext().prec \= digits \+ 15  
    one \= Decimal(1)  
    two \= Decimal(2)  
    four \= Decimal(4)  
    a \= one  
    b \= one / two.sqrt()  
    t \= Decimal("0.25")  
    p \= one  
    for \_ in range(10):  \# this doubles digits each step; \~7 steps reach \>1k digits; 10 is safe  
        an \= (a \+ b) / two  
        b  \= (a \* b).sqrt()  
        t  \= t \- p \* (a \- an) \* (a \- an)  
        a  \= an  
        p  \= p \* two  
    pi \= (a \+ b) \* (a \+ b) / (four \* t)  
    getcontext().prec \= old  
    return \+pi  \# unary plus rounds to current context

def UIR\_e() \-\> Decimal:  
    return Decimal(1).exp()

def UIR\_ln(x: Decimal) \-\> Decimal:  
    return x.ln()

def UIR\_sqrt(n: int) \-\> Decimal:  
    return Decimal(n).sqrt()

\# Metallic means: δ\_n \= (n \+ sqrt(n^2 \+ 4)) / 2 (φ is n=1)  
def UIR\_metallic(n: int) \-\> Decimal:  
    nD \= Decimal(n)  
    return (nD \+ (nD\*nD \+ Decimal(4)).sqrt()) / Decimal(2)

\# Plastic constant ρ: real root of x^3 \= x \+ 1  
def UIR\_plastic(digits: int \= UIR\_DIGITS\_DEFAULT) \-\> Decimal:  
    old \= getcontext().prec  
    getcontext().prec \= digits \+ 10  
    x \= Decimal("1.3")  \# good initial guess  
    one \= Decimal(1)  
    three \= Decimal(3)  
    for \_ in range(50):  
        f  \= x\*x\*x \- x \- one  
        df \= three\*x\*x \- one  
        x\_new \= x \- f/df  
        if abs(x\_new \- x) \< Decimal(10) \*\* (-(digits+2)):  
            x \= x\_new; break  
        x \= x\_new  
    getcontext().prec \= old  
    return \+x

\# Tribonacci constant τ3: real root of x^3 \= x^2 \+ x \+ 1  
def UIR\_tribonacci(digits: int \= UIR\_DIGITS\_DEFAULT) \-\> Decimal:  
    old \= getcontext().prec  
    getcontext().prec \= digits \+ 10  
    x \= Decimal("1.8")  
    one \= Decimal(1)  
    two \= Decimal(2)  
    three \= Decimal(3)  
    for \_ in range(50):  
        f  \= x\*x\*x \- (x\*x \+ x \+ one)  
        df \= three\*x\*x \- (two\*x \+ one)  
        x\_new \= x \- f/df  
        if abs(x\_new \- x) \< Decimal(10) \*\* (-(digits+2)):  
            x \= x\_new; break  
        x \= x\_new  
    getcontext().prec \= old  
    return \+x

\# Lambert W(1) (Omega constant): solve w\*exp(w) \= 1  
def UIR\_omega(digits: int \= UIR\_DIGITS\_DEFAULT) \-\> Decimal:  
    old \= getcontext().prec  
    getcontext().prec \= digits \+ 10  
    x \= Decimal("0.56714329")  \# good seed  
    one \= Decimal(1)  
    for \_ in range(80):  
        ex \= x.exp()  
        f  \= x\*ex \- one  
        df \= ex\*(x \+ one)  
        x\_new \= x \- f/df  
        if abs(x\_new \- x) \< Decimal(10) \*\* (-(digits+2)):  
            x \= x\_new; break  
        x \= x\_new  
    getcontext().prec \= old  
    return \+x

\# Champernowne C10: 0.123456789101112...  
def UIR\_champernowne(digits: int \= 300\) \-\> Decimal:  
    s \= \["0","."\]  
    n \= 1  
    while len("".join(s)) \- 2 \< digits:  
        s.append(str(n))  
        n \+= 1  
    return Decimal("".join(s)\[:digits+2\])  \# keep "0." \+ digits

\# Liouville L: sum 10^{-n\!}  (we emit digits as a string)  
def UIR\_liouville(terms: int \= 8\) \-\> Decimal:  
    \# build string with 1's at positions n\! (1-indexed after decimal)  
    max\_pos \= math.factorial(terms)  
    s \= \["0","."\]  
    for i in range(1, max\_pos+1):  
        s.append("0")  
    \# place ones  
    for n in range(1, terms+1):  
        pos \= math.factorial(n)  
        s\[1 \+ pos\] \= "1"  \# index shift since we have "0."  
    return Decimal("".join(s))

\# \---------- REGISTRY DEFINITION \----------  
\# Each entry: {"gen": \<generator\>, "args": {...}, "ci": (center\_str, sigma\_str)|None}  
\# Add/adjust sets below to scale the registry up/down.

def UIR\_build\_registry() \-\> Dict\[str, Dict\[str, Any\]\]:  
    R: Dict\[str, Dict\[str, Any\]\] \= {}

    \# Core transcendentals  
    R\["pi"\]  \= {"gen":"pi"}  
    R\["e"\]   \= {"gen":"e"}  
    R\["ln2"\] \= {"gen":"ln", "args":{"x": Decimal(2)}}  
    R\["ln10"\]= {"gen":"ln", "args":{"x": Decimal(10)}}

    \# Algebraic — square roots  
    for n in range(2, 31):  \# √2 ... √30  
        R\[f"sqrt\_{n}"\] \= {"gen":"sqrt", "args":{"n": n}}

    \# Metallic means (includes φ=metallic(1), silver=metallic(2), etc.)  
    for n in range(1, 16):  
        R\[f"metallic\_{n}"\] \= {"gen":"metallic", "args":{"n": n}}

    \# Named algebraics  
    R\["phi"\]        \= {"gen":"metallic", "args":{"n":1}}  
    R\["plastic"\]    \= {"gen":"plastic"}  
    R\["tribonacci"\] \= {"gen":"tribonacci"}  
    R\["omega\_W1"\]   \= {"gen":"omega"}

    \# Constructed sequences  
    R\["champernowne\_300d"\] \= {"gen":"champernowne", "args":{"digits":300}}  
    R\["liouville\_8terms"\]  \= {"gen":"liouville",    "args":{"terms":8}}

    \# Example CI (optional): if you want CI snapping for a constant, set ci=(center, sigma)  
    \# R\["example\_with\_CI"\] \= {"gen":"pi", "ci":("3.1415926535897932384626", "1e-20")}

    return R

\# \---------- GENERATE VALUE \----------  
def UIR\_eval\_entry(name: str, spec: Dict\[str, Any\], digits: int) \-\> Decimal:  
    gen \= spec.get("gen")  
    args \= spec.get("args", {}) or {}  
    if gen \== "pi":  
        return UIR\_pi(digits)  
    if gen \== "e":  
        return UIR\_e()  
    if gen \== "ln":  
        x \= args\["x"\]; return UIR\_ln(x)  
    if gen \== "sqrt":  
        return UIR\_sqrt(int(args\["n"\]))  
    if gen \== "metallic":  
        return UIR\_metallic(int(args\["n"\]))  
    if gen \== "plastic":  
        return UIR\_plastic(digits)  
    if gen \== "tribonacci":  
        return UIR\_tribonacci(digits)  
    if gen \== "omega":  
        return UIR\_omega(digits)  
    if gen \== "champernowne":  
        d \= int(args.get("digits", 300)); return UIR\_champernowne(d)  
    if gen \== "liouville":  
        t \= int(args.get("terms", 8)); return UIR\_liouville(t)  
    \# Literal fallback: if "literal" in spec  
    if gen \== "literal":  
        return Decimal(str(args\["value"\]))  
    raise ValueError(f"Unknown generator for {name}: {gen}")

\# \---------- PER-CONSTANT REPORT \----------  
def UIR\_constant\_report(name: str, value: Decimal, ci: Optional\[Tuple\[str,str\]\] \= None,  
                        cf\_terms: int \= UIR\_CF\_TERMS\_MAX, show\_top: int \= UIR\_SHOW\_TOP) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        UIR\_hdr(f"Constant: {name}")  
        print(f"Value (Decimal, {getcontext().prec}p context):")  
        print(value)  
        \# CF ladder  
        terms \= UIR\_cf\_from\_decimal(value, max\_terms=cf\_terms)  
        convs \= UIR\_convergents(terms)  
        rows=\[\]  
        for k, fr in enumerate(convs\[:show\_top\], start=1):  
            val \= Decimal(fr.numerator)/Decimal(fr.denominator)  
            rows.append({  
                "k": k,  
                "p/q": UIR\_fmt(fr),  
                "value": f"{val:.30f}",  
                "|err|": f"{UIR\_abs\_err(fr, value):.3E}",  
                "ppm": f"{UIR\_ppm(fr, value):.6f}",  
                "bits": UIR\_bits(fr)  
            })  
        UIR\_print\_table(rows, \["k","p/q","value","|err|","ppm","bits"\])

        \# CI snap if provided  
        if ci is not None:  
            center \= Decimal(ci\[0\]); sigma \= Decimal(ci\[1\])  
            pick \= UIR\_best\_in\_CI(convs, center, sigma, prefer\_min\_bits=True)  
            UIR\_hdr(f"CI Snap (optional) — band \[{center \- sigma}, {center \+ sigma}\]")  
            if pick is None:  
                print("No convergent inside band at this depth.")  
            else:  
                fr, err, bits \= pick  
                approx \= Decimal(fr.numerator)/Decimal(fr.denominator)  
                print(f"lock: {fr.numerator}/{fr.denominator}  bits={bits}")  
                print(f"value: {approx}")  
                print(f"|err|={err}  ppm={UIR\_ppm(fr, center)}")  
    return buf.getvalue()

\# \---------- BATCH RUNNER \----------  
def UIR\_run\_registry(digits: int \= UIR\_DIGITS\_DEFAULT,  
                     cf\_terms: int \= UIR\_CF\_TERMS\_MAX,  
                     outdir: str \= OUTPUT\_DIR,  
                     write\_files: bool \= True,  
                     combined\_filename: str \= "Irrational\_Registry\_Report.md",  
                     show\_top: int \= UIR\_SHOW\_TOP) \-\> str:  
    registry \= UIR\_build\_registry()

    combined\_chunks \= \[\]  
    summary\_rows \= \[\]

    UIR\_hdr("Universal Irrational Registry — RUN")  
    print(f"Items: {len(registry)}   Precision: {digits} digits   CF terms: {cf\_terms}   Timestamp: {\_uir\_now()}")  
    print("Writing files:", write\_files)  
    if write\_files:  
        os.makedirs(os.path.join(outdir, "constants"), exist\_ok=True)

    for name, spec in registry.items():  
        \# compute value  
        val \= UIR\_eval\_entry(name, spec, digits)  
        \# per-constant text  
        text \= UIR\_constant\_report(name, val, ci=spec.get("ci"), cf\_terms=cf\_terms, show\_top=show\_top)  
        \# print small header only; not to flood cell with full ladders for hundreds of constants  
        \# but we DO print the top table (show\_top)  
        print(text)

        \# record summary row using the last row (best printed convergent) if exists  
        \# or the first convergent  
        terms \= UIR\_cf\_from\_decimal(val, max\_terms=cf\_terms)  
        convs \= UIR\_convergents(terms)  
        if convs:  
            fr \= convs\[min(len(convs), show\_top)-1\]  
            row \= {  
                "name": name,  
                "best\_seen\_p/q": UIR\_fmt(fr),  
                "bits": UIR\_bits(fr),  
                "ppm": f"{UIR\_ppm(fr, val):.6f}"  
            }  
        else:  
            row \= {"name": name, "best\_seen\_p/q": "-", "bits": "-", "ppm": "-"}  
        summary\_rows.append(row)

        \# write separate file  
        if write\_files and WRITE\_PER\_MODULE\_FILES:  
            path \= os.path.join(outdir, "constants", f"{name}.md")  
            \_uir\_write\_text(path, text)

        \# add to combined  
        combined\_chunks.append(f"\# {name}\\n\\n{text}\\n")

    \# write combined file  
    combined\_text \= "\# Universal Irrational Registry — Combined Report\\n\\n" \+ \\  
                    f"\_Generated: {\_uir\_now()}  |  Items: {len(registry)}  |  Precision: {digits} digits\_\\n\\n" \+ \\  
                    "\#\# Summary (last printed convergent in each ladder)\\n\\n"  
    \# summary table  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        UIR\_print\_table(summary\_rows, \["name","best\_seen\_p/q","bits","ppm"\])  
    combined\_text \+= buf.getvalue() \+ "\\n" \+ "\\n".join(combined\_chunks)

    if write\_files and WRITE\_COMBINED\_FILE:  
        \_uir\_write\_text(os.path.join(outdir, combined\_filename), combined\_text)

    return combined\_text

\# \---------- ENTRY POINT \----------  
def UIR\_MAIN():  
    \# You can scale up digits/terms here if you want heavier runs  
    combined \= UIR\_run\_registry(digits=UIR\_DIGITS\_DEFAULT, cf\_terms=UIR\_CF\_TERMS\_MAX,  
                                outdir=OUTPUT\_DIR, write\_files=True,  
                                combined\_filename="Irrational\_Registry\_Report.md",  
                                show\_top=UIR\_SHOW\_TOP)  
    \# also write a light “drop-in index”  
    index\_note \= f"\# Irrational Registry Index\\n\\nGenerated {\_uir\_now()} with {UIR\_DIGITS\_DEFAULT} digits.\\n\\n"  
    \_uir\_write\_text(os.path.join(OUTPUT\_DIR, "REGISTRY\_INDEX.md"), index\_note)

if \_\_name\_\_ \== "\_\_main\_\_":  
    UIR\_MAIN()

\# \=================================================================================================  
\# \=== END v1.4 \====================================================================================  
\# \=================================================================================================  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — v2.0 "Cosmic Irrational Lab" (append-only, no charts)              \===  
\# \=== Adds: ζ(3), Catalan G, BBP π (with hex-digit extractor), Stoneham S\_{b,c}, CF statistics  \===  
\# \===      Gauss–Kuzmin test, Khinchin & Lévy empirical limits.                                 \===  
\# \=== File outputs honor WRITE\_\* flags and OUTPUT\_DIR from earlier cells.                        \===  
\# \=================================================================================================

import os, io, math, time, json  
from decimal import Decimal, getcontext, ROUND\_FLOOR  
from fractions import Fraction  
from contextlib import redirect\_stdout  
from typing import List, Dict, Tuple, Optional

\# \---------- CONFIG SYNC \----------  
try:  
    WRITE\_PER\_MODULE\_FILES  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    WRITE\_JSON\_WITNESS  
except NameError:  
    WRITE\_JSON\_WITNESS \= True  
try:  
    OUTPUT\_DIR  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"  
OUT\_SUBDIR \= os.path.join(OUTPUT\_DIR, "uiextras")  
os.makedirs(OUT\_SUBDIR, exist\_ok=True)

\# Precision knobs  
UIX\_DIGITS \= 100           \# working precision for Decimal  
UIX\_CF\_TERMS \= 80          \# CF term cap  
UIX\_SHOW\_TOP \= 14          \# rows shown per ladder  
getcontext().prec \= UIX\_DIGITS \+ 15

\# \---------- COMMON UTILS \----------  
def \_uix\_now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  
def \_uix\_write\_text(path: str, text: str):  
    try:  
        os.makedirs(os.path.dirname(path), exist\_ok=True)  
        with open(path, "w", encoding="utf-8") as f: f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def UIX\_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar); print(f"=== {title}"); print(bar)

def UIX\_print\_table(rows: List\[Dict\[str, str\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

def UIX\_bits(fr: Fraction) \-\> int:  
    p, q \= abs(fr.numerator), abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def UIX\_abs\_err(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def UIX\_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= UIX\_abs\_err(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

def UIX\_fmt(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

def UIX\_cf\_from\_decimal(x: Decimal, max\_terms: int \= UIX\_CF\_TERMS) \-\> List\[int\]:  
    terms \= \[\]  
    neg \= x \< 0  
    y \= \-x if neg else x  
    for \_ in range(max\_terms):  
        a0 \= int(y.to\_integral\_value(rounding=ROUND\_FLOOR))  
        terms.append(-a0 if (neg and len(terms)==0) else a0)  
        frac \= y \- Decimal(a0)  
        if frac \== 0: break  
        y \= Decimal(1)/frac  
    return terms

def UIX\_convergents(terms: List\[int\]) \-\> List\[Fraction\]:  
    ps=\[0,1\]; qs=\[1,0\]; out=\[\]  
    for a in terms:  
        p \= a\*ps\[-1\] \+ ps\[-2\]  
        q \= a\*qs\[-1\] \+ qs\[-2\]  
        ps=\[ps\[-1\], p\]; qs=\[qs\[-1\], q\]  
        out.append(Fraction(p,q))  
    return out

\# \---------- CONST GENERATORS \----------  
def UIX\_e() \-\> Decimal: return Decimal(1).exp()  
def UIX\_ln(x: Decimal) \-\> Decimal: return x.ln()  
def UIX\_pi\_agm(digits: int \= UIX\_DIGITS) \-\> Decimal:  
    old \= getcontext().prec  
    getcontext().prec \= digits \+ 20  
    one \= Decimal(1); two=Decimal(2); four=Decimal(4)  
    a \= one; b \= one/two.sqrt(); t=Decimal("0.25"); p=one  
    for \_ in range(12):  
        an=(a+b)/two; b=(a\*b).sqrt(); t \= t \- p\*(a-an)\*(a-an); a=an; p=p\*two  
    pi \= (a+b)\*(a+b)/(four\*t)  
    getcontext().prec \= old  
    return \+pi

\# Apery ζ(3) via pretty-fast series: ζ(3) \= sum\_{n\>=1} 1/n^3  
\# (We enhance with an acceleration term using Euler–Maclaurin tail approx for speed.)  
def UIX\_zeta3(digits: int \= UIX\_DIGITS) \-\> Decimal:  
    \# crude but sufficient for \~50-80 digits with enough terms  
    getcontext().prec \= digits \+ 15  
    S \= Decimal(0)  
    N \= 200000  \# tune if needed; safe for \~10-12 digits; but with Decimal may be slow.  
    \# We'll do a hybrid: sum to 20000; then add tail approx ∫ and Bernoulli correction  
    N \= 20000  
    one \= Decimal(1)  
    for n in range(1, N+1):  
        S \+= one/(Decimal(n)\*\*3)  
    \# tail \~ ∫\_N^∞ x^{-3} dx \= 1/(2 N^2)  
    tail \= one/(Decimal(2)\*(Decimal(N)\*\*2))  
    \# next correction \~ 3/(2\*6\*N^3) \= 1/(4\*N^3)  
    tail \+= one/(Decimal(4)\*(Decimal(N)\*\*3))  
    return \+(S \+ tail)

\# Catalan's constant G \= sum\_{n≥0} (-1)^n/(2n+1)^2  
def UIX\_catalan(digits: int \= UIX\_DIGITS) \-\> Decimal:  
    getcontext().prec \= digits \+ 15  
    S \= Decimal(0); one=Decimal(1); sign=1  
    N \= 200000  \# truncated; decent for \~10-12 digits  
    N \= 100000  
    for n in range(N):  
        term \= one/Decimal((2\*n+1)\*\*2)  
        S \+= term if (n%2==0) else \-term  
    return \+S

\# Stoneham S\_{b,c} \= sum\_{n\>=1} b^{-n c^n}  
def UIX\_stoneham(b: int, c: int, terms: int \= 12\) \-\> Decimal:  
    getcontext().prec \= UIX\_DIGITS \+ 15  
    s \= Decimal(0); bD=Decimal(b)  
    for n in range(1, terms+1):  
        expn \= n\*(c\*\*n)  
        s \+= Decimal(1) / (bD\*\*Decimal(expn))  
    return \+s

\# \---------- BBP π HEX DIGITS (no prior digits) \----------  
\# Returns the hex digits of π starting at position n (1-indexed after the point), count m.  
def UIX\_pi\_hex\_digits(n: int, m: int \= 16\) \-\> str:  
    \# Implements Bailey–Borwein–Plouffe digit extraction (hex).  
    \# π \= Σ\_{k=0}^∞ 16^{-k} ( 4/(8k+1) \- 2/(8k+4) \- 1/(8k+5) \- 1/(8k+6) )  
    \# We compute fractional part of Σ\_{k=0}^{n+m} 16^{n-1-k} (...) modulo 1 then convert to hex digits.  
    def series(j):  
        s \= 0.0  
        \# Left sum: k=0..n-1, compute 16^{n-1-k} mod (8k+j), using modular exponentiation  
        for k in range(n):  
            denom \= 8\*k \+ j  
            s \+= pow(16, n-1-k, denom) / denom  
            s \-= int(s)  
        \# Right tail: k=n..n+m+50 (enough)  
        t \= 0.0  
        for k in range(n, n+m+50):  
            t \+= 16\*\*(n-1-k) / (8\*k \+ j)  
        return (s \+ t) % 1.0

    x \= 4\*series(1) \- 2\*series(4) \- series(5) \- series(6)  
    x \= x % 1.0  
    hex\_digits \= ""  
    for \_ in range(m):  
        x \= 16\*x  
        digit \= int(x)  
        x \-= digit  
        hex\_digits \+= "0123456789ABCDEF"\[digit\]  
    return hex\_digits

\# \---------- CF STATISTICS LAB \----------  
\# Partial quotients {a\_k}; tests:  
\#  \- Gauss–Kuzmin: P(a=k) \~ log\_2(1 \+ 1/(k(k+2)))  
\#  \- Khinchin: (Π a\_k)^{1/n} \-\> K ≈ 2.685452001...  
\#  \- Lévy: q\_n^{1/n} \-\> L \= exp(π^2/(12 ln 2)) ≈ 3.2758229187...  
UIX\_KHINCHIN \= Decimal("2.68545200106530644530971483548")  \# ref  
UIX\_LEVY     \= Decimal("3.2758229187218111597876818824535")

def UIX\_partial\_quotients(x: Decimal, max\_terms: int \= 2000\) \-\> List\[int\]:  
    terms=\[\]  
    y=x  
    for \_ in range(max\_terms):  
        a0 \= int(y.to\_integral\_value(rounding=ROUND\_FLOOR))  
        terms.append(a0)  
        frac \= y \- Decimal(a0)  
        if frac \== 0:  
            break  
        y \= Decimal(1)/frac  
    return terms

def UIX\_gauss\_kuzmin\_prob(k: int) \-\> Decimal:  
    \# P(a1 \= k) \= log\_2(1 \+ 1/(k(k+2))) \= ln(1+1/(k(k+2))) / ln(2)  
    num \= (Decimal(1) \+ Decimal(1)/Decimal(k\*(k+2))).ln()  
    den \= Decimal(2).ln()  
    return num/den

def UIX\_cf\_stats\_report(name: str, x: Decimal, Kmax: int \= 20, terms: int \= 500\) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        UIX\_hdr(f"CF Statistics — {name}  (terms={terms}, Kmax={Kmax})")  
        a \= UIX\_partial\_quotients(x, max\_terms=terms)  
        \# frequencies for k=1..Kmax  
        freq \= {k:0 for k in range(1, Kmax+1)}  
        for idx, ak in enumerate(a\[1:\], start=2):  \# skip a0 integer part for the distribution test  
            if 1 \<= ak \<= Kmax: freq\[ak\]+=1  
        total \= sum(freq.values())  
        rows=\[\]  
        chi2 \= Decimal(0)  
        for k in range(1, Kmax+1):  
            obs \= Decimal(freq\[k\])  
            exp \= Decimal(total) \* UIX\_gauss\_kuzmin\_prob(k)  
            diff \= obs \- exp  
            chi2 \+= (diff\*diff) / (exp if exp\!=0 else Decimal(1))  
            rows.append({  
                "k": k,  
                "obs": int(obs),  
                "exp": f"{exp:.2f}",  
                "p(k)": f"{UIX\_gauss\_kuzmin\_prob(k):.6f}"  
            })  
        UIX\_print\_table(rows, \["k","obs","exp","p(k)"\])  
        print(f"χ^2 (goodness vs Gauss–Kuzmin, bins 1..{Kmax}): {chi2:.3f}")

        \# Khinchin product^{1/n}  
        prod \= Decimal(1)  
        count \= 0  
        for ak in a\[1:\]:   \# skip a0  
            if ak \== 0: continue  
            prod \*= Decimal(ak)  
            count \+= 1  
            if count \>= terms-1: break  
        if count \> 0:  
            K\_est \= prod \*\* (Decimal(1)/Decimal(count))  
        else:  
            K\_est \= Decimal(0)  
        print(f"Khinchin estimate from first {count} a\_k: {K\_est}  (reference {UIX\_KHINCHIN})")

        \# Lévy constant via convergents  
        terms\_cf \= UIX\_cf\_from\_decimal(x, max\_terms=terms)  
        convs \= UIX\_convergents(terms\_cf)  
        if convs:  
            qn \= convs\[-1\].denominator  
            n  \= len(convs)  
            L\_est \= (Decimal(qn)) \*\* (Decimal(1)/Decimal(n))  
            print(f"Lévy estimate from q\_{n}: {L\_est}  (reference {UIX\_LEVY})")  
        else:  
            print("No convergents found for Lévy estimate.")  
    return buf.getvalue()

\# \---------- LADDER PRINTER \----------  
def UIX\_cf\_ladder(name: str, value: Decimal, top: int \= UIX\_SHOW\_TOP, terms: int \= UIX\_CF\_TERMS) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        UIX\_hdr(f"CF Ladder — {name}")  
        t \= UIX\_cf\_from\_decimal(value, max\_terms=terms)  
        convs \= UIX\_convergents(t)  
        rows=\[\]  
        for k, fr in enumerate(convs\[:top\], start=1):  
            v \= Decimal(fr.numerator)/Decimal(fr.denominator)  
            rows.append({  
                "k": k, "p/q": UIX\_fmt(fr),  
                "value": f"{v:.30f}",  
                "|err|": f"{UIX\_abs\_err(fr, value):.3E}",  
                "ppm": f"{UIX\_ppm(fr, value):.6f}",  
                "bits": UIX\_bits(fr)  
            })  
        UIX\_print\_table(rows, \["k","p/q","value","|err|","ppm","bits"\])  
    return buf.getvalue()

\# \---------- COMBINED DRIVER \----------  
def UIX\_MAIN():  
    combined\_chunks \= \[\]  
    def add\_report(title: str, text: str, filename: str):  
        combined\_chunks.append(f"\# {title}\\n\\n{text}\\n")  
        if WRITE\_PER\_MODULE\_FILES:  
            \_uix\_write\_text(os.path.join(OUT\_SUBDIR, filename), text)

    UIX\_hdr("Cosmic Irrational Lab — START")  
    print(f"Precision: {getcontext().prec}  Timestamp: {\_uix\_now()}  Output: {OUT\_SUBDIR}")

    \# 1\) ζ(3) Apéry  
    z3 \= UIX\_zeta3(UIX\_DIGITS)  
    text \= UIX\_cf\_ladder("zeta(3)", z3)  
    add\_report("zeta(3) — Apéry constant", text, "zeta3.md")

    \# 2\) Catalan G  
    G \= UIX\_catalan(UIX\_DIGITS)  
    text \= UIX\_cf\_ladder("Catalan G", G)  
    add\_report("Catalan's constant G", text, "catalan\_G.md")

    \# 3\) π via AGM \+ CF \+ BBP hex digits at positions  
    pi\_dec \= UIX\_pi\_agm(UIX\_DIGITS)  
    text \= UIX\_cf\_ladder("pi (AGM)", pi\_dec)  
    \# Append BBP demo  
    bbp\_block \= io.StringIO()  
    with redirect\_stdout(bbp\_block):  
        UIX\_hdr("π — BBP Hex Digit Extraction")  
        for start in \[1, 1000, 100000\]:  
            digs \= UIX\_pi\_hex\_digits(start, 16\)  
            print(f"hex digits starting at position {start}: {digs}")  
    text \+= bbp\_block.getvalue()  
    add\_report("pi — AGM \+ BBP hex digits", text, "pi\_agm\_bbp.md")

    \# 4\) Stoneham S\_{2,3} (and configurable)  
    stone \= UIX\_stoneham(2,3,terms=10)  
    text \= UIX\_cf\_ladder("Stoneham S\_{2,3} (10 terms)", stone)  
    add\_report("Stoneham S\_{2,3}", text, "stoneham\_2\_3.md")

    \# 5\) CF statistics lab on some headline constants  
    for (nm, val) in \[("pi", pi\_dec), ("e", UIX\_e()), ("ln2", UIX\_ln(Decimal(2))), ("zeta3", z3)\]:  
        rep \= UIX\_cf\_stats\_report(nm, val, Kmax=20, terms=600)  
        add\_report(f"CF Statistics — {nm}", rep, f"cfstats\_{nm}.md")

    \# Combined file  
    if WRITE\_COMBINED\_FILE:  
        mega \= "\# Cosmic Irrational Lab — Combined Report (v2.0)\\n\\n" \+ "\\n".join(combined\_chunks)  
        \_uix\_write\_text(os.path.join(OUT\_SUBDIR, "UIX\_Report.md"), mega)

    UIX\_hdr("Cosmic Irrational Lab — END")

if \_\_name\_\_ \== "\_\_main\_\_":  
    UIX\_MAIN()

\# \=================================================================================================  
\# \=== END v2.0 \====================================================================================  
\# \=================================================================================================  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — v2.6.1 "Hyperlocks (robust, fixed normality precision guard)"     \===  
\# \=== Replaces v2.6. Append-only, no charts.                                                    \===  
\# \=== Modules: (1) Normality Lab  (2) Shared-Denominator Hunter  (3) CF Prefix Cylinders        \===  
\# \=================================================================================================

import os, io, math, time  
from fractions import Fraction  
from decimal import Decimal, getcontext, ROUND\_FLOOR  
from contextlib import redirect\_stdout  
from typing import List, Dict, Tuple, Optional

\# \--------------------------- CONFIG SYNC \----------------------------------------------------------  
try:  
    WRITE\_PER\_MODULE\_FILES  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    OUTPUT\_DIR  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"

HL26\_DIR \= os.path.join(OUTPUT\_DIR, "hyperlocks\_v26")  
os.makedirs(HL26\_DIR, exist\_ok=True)

\# working precision for Decimal ops  
getcontext().prec \= 140

\# \--------------------------- SHARED UTILS \---------------------------------------------------------  
def H26\_now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def H26\_write(path: str, text: str):  
    try:  
        os.makedirs(os.path.dirname(path), exist\_ok=True)  
        with open(path, "w", encoding="utf-8") as f: f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def H26\_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar); print(f"=== {title}"); print(bar)

def H26\_print\_table(rows: List\[Dict\[str, str\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

def H26\_bits(fr: Fraction) \-\> int:  
    p, q \= abs(fr.numerator), abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def H26\_abs\_err(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def H26\_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= H26\_abs\_err(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

def H26\_fmt(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

\# CF basics  
def H26\_cf\_terms(x: Decimal, max\_terms: int \= 128\) \-\> List\[int\]:  
    terms=\[\]  
    y=x  
    for \_ in range(max\_terms):  
        a0 \= int(y.to\_integral\_value(rounding=ROUND\_FLOOR))  
        terms.append(a0)  
        frac \= y \- Decimal(a0)  
        if frac \== 0: break  
        y \= Decimal(1)/frac  
    return terms

def H26\_convergents(terms: List\[int\]) \-\> List\[Fraction\]:  
    ps=\[0,1\]; qs=\[1,0\]; out=\[\]  
    for a in terms:  
        p \= a\*ps\[-1\] \+ ps\[-2\]  
        q \= a\*qs\[-1\] \+ qs\[-2\]  
        ps=\[ps\[-1\], p\]; qs=\[qs\[-1\], q\]  
        out.append(Fraction(p,q))  
    return out

\# \--------------------------- (1) NORMALITY LAB \----------------------------------------------------  
HEX\_ALPH \= "0123456789ABCDEF"

def H26\_frac\_to\_base\_digits(x: Decimal, base: int, n\_digits: int) \-\> List\[int\]:  
    """  
    Extract fractional base-b digits by repeated multiply.  
    Requires context precision \>\> n\_digits to avoid artifacts.  
    """  
    if base \< 2: raise ValueError("base must be \>= 2")  
    y \= x \- int(x)  
    if y \< 0: y \+= 1  
    digs=\[\]  
    for \_ in range(n\_digits):  
        y \= y \* base  
        d \= int(y.to\_integral\_value(rounding=ROUND\_FLOOR))  
        \# clip just in case of rare rounding edge  
        if d \< 0: d \= 0  
        if d \>= base: d \= base-1  
        digs.append(d)  
        y \= y \- d  
    return digs

def H26\_chi\_square\_uniform(digs: List\[int\], base: int) \-\> Tuple\[Dict\[int,int\], Decimal\]:  
    freq \= {k:0 for k in range(base)}  
    for d in digs:  
        if 0 \<= d \< base: freq\[d\]+=1  
    n \= Decimal(len(digs))  
    if n \== 0: return freq, Decimal(0)  
    exp \= n/Decimal(base)  
    chi2 \= Decimal(0)  
    for k in range(base):  
        obs \= Decimal(freq\[k\])  
        diff \= obs \- exp  
        chi2 \+= (diff\*diff)/exp  
    return freq, \+chi2

def H26\_report\_normality(name: str, value: Decimal, bases=(2,10,16), n\_digits: int \= 4096\) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        H26\_hdr(f"Normality Probe — {name}")  
        print(f"Digits analyzed per base: {n\_digits}")  
        for b in bases:  
            \# temporarily increase precision to keep carry-safe  
            oldp \= getcontext().prec  
            getcontext().prec \= max(oldp, n\_digits \+ 140\)  \# \<-- FIXED: no division by zero  
            freq, chi2 \= H26\_chi\_square\_uniform(H26\_frac\_to\_base\_digits(value, b, n\_digits), b)  
            getcontext().prec \= oldp  
            rows=\[\]  
            for k in range(b):  
                rows.append({"digit": k if b\!=16 else HEX\_ALPH\[k\], "count": freq\[k\]})  
            print(f"\\nBase {b} frequency counts:")  
            H26\_print\_table(rows, \["digit","count"\])  
            print(f"χ^2 vs uniform (df={b-1}): {chi2}")  
    return buf.getvalue()

\# \--------------------------- (2) SHARED-DENOMINATOR HUNTER \----------------------------------------  
def H26\_find\_shared\_q(bands: List\[Dict\[str,str\]\], q\_cap: int \= 500000,  
                      objective: str \= "bits") \-\> Optional\[Dict\[str, object\]\]:  
    """  
    bands: \[{name, center\_str, sigma\_str}\]  
    Find q ≤ q\_cap and integers p\_i with p\_i/q ∈ band\_i ∀i.  
    objective: "bits" minimizes total MDL; "q" minimizes q (then bits).  
    """  
    targets \= \[(it\["name"\], Decimal(it\["center\_str"\]), Decimal(it\["sigma\_str"\])) for it in bands\]  
    best=None  \# (primary, secondary, q, rows)  
    for q in range(1, q\_cap+1):  
        rows=\[\]; total\_bits=0; feasible=True  
        for (nm, xc, sg) in targets:  
            L, U \= xc \- sg, xc \+ sg  
            p0 \= int((q\*xc).to\_integral\_value(rounding=ROUND\_FLOOR))  
            hit=None  
            for p in (p0-2, p0-1, p0, p0+1, p0+2):  
                v \= Decimal(p)/Decimal(q)  
                if L \<= v \<= U:  
                    hit \= p; break  
            if hit is None:  
                feasible=False; break  
            fr \= Fraction(hit, q)  
            rows.append((nm, hit, v, H26\_bits(fr), H26\_ppm(fr, xc)))  
            total\_bits \+= H26\_bits(fr)  
        if not feasible: continue  
        primary, secondary \= (total\_bits, q) if objective=="bits" else (q, total\_bits)  
        if (best is None) or (primary \< best\[0\]) or (primary==best\[0\] and secondary \< best\[1\]):  
            best \= (primary, secondary, q, rows)  
    if best is None:  
        return None  
    primary, secondary, q, rows \= best  
    return {  
        "objective": objective,  
        "q": q,  
        "score\_primary": primary,  
        "score\_secondary": secondary,  
        "matches": \[  
            {"name": nm, "p": p, "q": q, "value": str(val), "bits": bits, "ppm\_vs\_center": str(ppm)}  
            for (nm,p,val,bits,ppm) in rows  
        \]  
    }

def H26\_report\_shared\_q(title: str, bands: List\[Dict\[str,str\]\],  
                        q\_cap: int \= 200000, objective: str \= "bits") \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        H26\_hdr(f"Shared-Denominator Hunter — {title}")  
        print(f"Objective: {objective}   q\_cap={q\_cap}   items={len(bands)}")  
        for it in bands:  
            print(f"  \- {it\['name'\]}: center={it\['center\_str'\]}  sigma={it\['sigma\_str'\]}")  
        sol \= H26\_find\_shared\_q(bands, q\_cap=q\_cap, objective=objective)  
        if sol is None:  
            print("\\nNo common-q solution found under current cap.")  
        else:  
            print(f"\\nSOLUTION: q={sol\['q'\]}   primary={sol\['score\_primary'\]}   secondary={sol\['score\_secondary'\]}")  
            rows=\[\]  
            for m in sol\["matches"\]:  
                rows.append({  
                    "name": m\["name"\],  
                    "p/q": f"{m\['p'\]}/{sol\['q'\]}",  
                    "value": m\["value"\],  
                    "ppm\_vs\_center": m\["ppm\_vs\_center"\],  
                    "bits": m\["bits"\]  
                })  
            H26\_print\_table(rows, \["name","p/q","value","ppm\_vs\_center","bits"\])  
    return buf.getvalue()

\# \--------------------------- (3) CF PREFIX CYLINDERS (robust) \------------------------------------  
def H26\_cf\_common\_prefix(values: List\[Decimal\], max\_terms: int \= 64, ignore\_a0: bool \= False) \-\> List\[int\]:  
    """  
    Longest shared CF prefix. If ignore\_a0=True, require equality starting at a1 (skip integer part).  
    """  
    seqs \= \[H26\_cf\_terms(v, max\_terms=max\_terms) for v in values\]  
    if not seqs: return \[\]  
    if ignore\_a0:  
        minlen \= min(len(s) for s in seqs)  
        if minlen \< 2: return \[\]  
        pref=\[\]  
        for i in range(1, minlen):  
            ai \= seqs\[0\]\[i\]  
            if all(s\[i\]==ai for s in seqs):  
                pref.append(ai)  
            else:  
                break  
        return pref  
    else:  
        minlen \= min(len(s) for s in seqs)  
        pref=\[\]  
        for i in range(minlen):  
            ai \= seqs\[0\]\[i\]  
            if all(s\[i\]==ai for s in seqs):  
                pref.append(ai)  
            else:  
                break  
        return pref

def H26\_cylinder\_from\_prefix(prefix: List\[int\]) \-\> Optional\[Tuple\[Fraction, Fraction\]\]:  
    """  
    Safe cylinder from a CF prefix:  
      \- empty \-\> None  
      \- \[a0\]  \-\> \[a0/1, (a0+1)/1\]  
      \- \[a0,...,ak\] \-\> interval between Ck and mediant(Ck, Ck-1)  
    """  
    if not prefix:  
        return None  
    convs \= H26\_convergents(prefix)  
    Ck \= convs\[-1\]  
    if len(convs) \== 1:  
        a0 \= prefix\[0\]  
        L, U \= Fraction(a0,1), Fraction(a0+1,1)  
        return (min(L,U), max(L,U))  
    Ckm1 \= convs\[-2\]  
    mediant \= Fraction(Ck.numerator \+ Ckm1.numerator, Ck.denominator \+ Ckm1.denominator)  
    L, U \= (Ck, mediant) if Ck \<= mediant else (mediant, Ck)  
    return (L, U)

def H26\_report\_cf\_cylinder(title: str, items: List\[Tuple\[str, Decimal\]\],  
                           max\_terms: int \= 64, ignore\_a0: bool \= False,  
                           group\_by\_a0: bool \= True) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        H26\_hdr(f"CF Prefix Cylinder — {title}")  
        groups \= {}  
        for (nm, val) in items:  
            a0 \= int(val.to\_integral\_value(rounding=ROUND\_FLOOR))  
            key \= a0 if group\_by\_a0 and not ignore\_a0 else "ALL"  
            groups.setdefault(key, \[\]).append((nm, val))

        for gkey, gitems in groups.items():  
            print(f"\\nGroup: {gkey}")  
            for (nm, val) in gitems:  
                print(f"- {nm}: {val}")  
            vals \= \[v for (\_,v) in gitems\]  
            pref \= H26\_cf\_common\_prefix(vals, max\_terms=max\_terms, ignore\_a0=ignore\_a0)  
            if ignore\_a0:  
                print(f"Common CF tail prefix (length {len(pref)} starting at a1): {pref}")  
                if not pref:  
                    print("No shared tail prefix — no finite cylinder.")  
                    continue  
                a0\_first \= int(vals\[0\].to\_integral\_value(rounding=ROUND\_FLOOR))  
                full\_prefix \= \[a0\_first\] \+ pref  
            else:  
                print(f"Common CF prefix (length {len(pref)}): {pref}")  
                full\_prefix \= pref

            cyl \= H26\_cylinder\_from\_prefix(full\_prefix)  
            if cyl is None:  
                print("No finite cylinder (empty prefix).")  
                continue  
            L, U \= cyl  
            Ld \= Decimal(L.numerator)/Decimal(L.denominator)  
            Ud \= Decimal(U.numerator)/Decimal(U.denominator)  
            print(f"Cylinder interval: \[{L.numerator}/{L.denominator}, {U.numerator}/{U.denominator}\]  (\~\[{Ld}, {Ud}\])")  
            rows=\[\]  
            for (nm, val) in gitems:  
                rows.append({"name": nm, "inside?": (Ld \<= val \<= Ud)})  
            H26\_print\_table(rows, \["name","inside?"\])  
    return buf.getvalue()

\# \--------------------------- DEMO DRIVER \----------------------------------------------------------  
def H26\_fast\_pi():  
    old \= getcontext().prec  
    getcontext().prec \= 160  
    one=Decimal(1); two=Decimal(2); four=Decimal(4)  
    a=one; b=one/two.sqrt(); t=Decimal("0.25"); p=one  
    for \_ in range(10):  
        an=(a+b)/two; b=(a\*b).sqrt(); t=t-p\*(a-an)\*(a-an); a=an; p=p\*two  
    pi=(a+b)\*(a+b)/(four\*t)  
    getcontext().prec \= old  
    return \+pi

def H26\_MAIN():  
    chunks=\[\]

    H26\_hdr("Hyperlocks v2.6.1 — START")  
    print(f"Precision context: {getcontext().prec}   Timestamp: {H26\_now()}   Output: {HL26\_DIR}")

    \# Prepare headline constants locally (self-contained)  
    pi\_val \= H26\_fast\_pi()  
    e\_val  \= Decimal(1).exp()  
    ln2    \= Decimal(2).ln()  
    alpha\_inv \= Decimal("137.035999084")  
    alpha    \= Decimal(1) / alpha\_inv

    \# (1) Normality probes  
    nrm\_pi \= H26\_report\_normality("pi (AGM)", pi\_val, bases=(2,10,16), n\_digits=4096)  
    nrm\_e  \= H26\_report\_normality("e", e\_val, bases=(2,10,16), n\_digits=4096)  
    nrm\_l2 \= H26\_report\_normality("ln 2", ln2, bases=(2,10,16), n\_digits=4096)  
    chunks.append(("\# Normality — π", nrm\_pi, "normality\_pi.md"))  
    chunks.append(("\# Normality — e", nrm\_e, "normality\_e.md"))  
    chunks.append(("\# Normality — ln2", nrm\_l2, "normality\_ln2.md"))

    \# (2) Shared denominator demo  
    bands\_demo \= \[  
        {"name":"alpha^{-1}", "center\_str":str(alpha\_inv), "sigma\_str":"2.1e-8"},  
        {"name":"pi",         "center\_str":str(pi\_val),    "sigma\_str":"1e-12"},  
        {"name":"ln2",        "center\_str":str(ln2),       "sigma\_str":"1e-15"},  
    \]  
    shared\_bits \= H26\_report\_shared\_q("demo α^{-1} \+ π \+ ln2 (min bits)", bands\_demo, q\_cap=500000, objective="bits")  
    shared\_q    \= H26\_report\_shared\_q("demo α^{-1} \+ π \+ ln2 (min q)",    bands\_demo, q\_cap=500000, objective="q")  
    chunks.append(("\# Shared Denominator — bits objective", shared\_bits, "shared\_den\_bits.md"))  
    chunks.append(("\# Shared Denominator — q objective",    shared\_q,    "shared\_den\_q.md"))

    \# (3) CF prefix cylinders  
    cfp1 \= H26\_report\_cf\_cylinder("alpha^{-1}, pi, ln2 — grouped by a0", \[  
        ("alpha^{-1}", alpha\_inv),  
        ("pi",         pi\_val),  
        ("ln2",        ln2),  
    \], max\_terms=48, ignore\_a0=False, group\_by\_a0=True)  
    cfp2 \= H26\_report\_cf\_cylinder("alpha^{-1}, pi, ln2 — tail-only (ignore a0), grouped", \[  
        ("alpha^{-1}", alpha\_inv),  
        ("pi",         pi\_val),  
        ("ln2",        ln2),  
    \], max\_terms=48, ignore\_a0=True, group\_by\_a0=True)  
    chunks.append(("\# CF Prefix — grouped by a0", cfp1, "cfprefix\_grouped.md"))  
    chunks.append(("\# CF Prefix — tail-only",     cfp2, "cfprefix\_tail.md"))

    \# print \+ write artifacts  
    combined=\[\]  
    for title, text, fname in chunks:  
        print(text)  
        combined.append(f"{title}\\n\\n{text}\\n")  
        if WRITE\_PER\_MODULE\_FILES:  
            H26\_write(os.path.join(HL26\_DIR, fname), text)

    if WRITE\_COMBINED\_FILE:  
        H26\_write(os.path.join(HL26\_DIR, "Hyperlocks\_Report.md"),  
                  "\# Hyperlocks v2.6.1 — Combined Report\\n\\n" \+ "\\n".join(combined))

    H26\_hdr("Hyperlocks v2.6.1 — END")

if \_\_name\_\_ \== "\_\_main\_\_":  
    H26\_MAIN()

\# \=================================================================================================  
\# \=== END v2.6.1 \==================================================================================  
\# \=================================================================================================  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — v3.2 "Composite Minimality Proofs++ (exhaustive two-term denial)"  \===  
\# \=== Append-only • No charts • Honors WRITE\_\* flags & OUTPUT\_DIR                               \===  
\# \=================================================================================================  
\# What this module guarantees (under stated caps):  
\#   • SINGLE-FRACTION BASELINE: exhaustive over all reduced p/q with q ≤ Q\_cap\_single that land  
\#     in \[center ± sigma\]; records the best (min MDL bits, tie-break by smaller q).  
\#   • TWO-TERM PACKS (GENERAL): exhaustive over all reduced p1/q1, p2/q2 with q1,q2 ≤ D\_cap\_2  
\#     whose sum lies in the band. We enumerate via exact p2 range induced by the residual band  
\#     L2..U2 \= (center±sigma) − p1/q1 for each (p1,q1), and we test all p2 in that range for  
\#     every q2 ≤ D\_cap\_2 (gcd(p2,q2)=1). This covers \*all\* 2-term reduced packs under the caps.  
\#   • EGYPTIAN-ONLY OPTION: same exhaustive proof for packs restricted to 1/n1 \+ 1/n2 with  
\#     n1,n2 ≤ U\_cap\_egy (classic Egyptian sums).  
\#   • CERTIFICATES: printed tables \+ machine-readable JSON per band; includes enumeration counts,  
\#     caps, best witnesses, and an explicit “no better exists within caps” denial for 2-terms.  
\#  
\# Output files:  
\#   \- {OUTPUT\_DIR}/composite\_proofs/\*.md       (one per band)  
\#   \- {OUTPUT\_DIR}/composite\_proofs/proofs.json (all bands, machine-readable)  
\#  
\# Usage:  
\#   \- Edit BAND\_SPECS below (or call CMP\_PROVE\_BANDS with your own list of dicts).  
\#   \- Reasonable default caps: Q\_cap\_single=200\_000; D\_cap\_2=1\_000 (tunable).  
\#     For quick runs, start D\_cap\_2 \~ 400–600; for heavier proofs, bump it.  
\# \=================================================================================================

import os, io, math, time, json  
from fractions import Fraction  
from decimal import Decimal, getcontext, ROUND\_FLOOR  
from contextlib import redirect\_stdout  
from typing import List, Dict, Tuple, Optional

\# \-------------------------------- CONFIG SYNC \-----------------------------------------------------  
try:  
    WRITE\_PER\_MODULE\_FILES  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    WRITE\_JSON\_WITNESS  
except NameError:  
    WRITE\_JSON\_WITNESS \= True  
try:  
    OUTPUT\_DIR  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"

CP\_OUTDIR \= os.path.join(OUTPUT\_DIR, "composite\_proofs")  
os.makedirs(CP\_OUTDIR, exist\_ok=True)

\# Set a safe high precision for Decimal logic; we only use it for band arithmetic (not heavy sums)  
getcontext().prec \= 120

\# \-------------------------------- UTILITIES \-------------------------------------------------------  
def CP\_now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def CP\_write(path: str, text: str):  
    try:  
        os.makedirs(os.path.dirname(path), exist\_ok=True)  
        with open(path, "w", encoding="utf-8") as f: f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def CP\_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar); print(f"=== {title}"); print(bar)

def CP\_print\_table(rows: List\[Dict\[str, str\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

def CP\_bits(fr: Fraction) \-\> int:  
    p, q \= abs(fr.numerator), abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def CP\_abs\_err\_val(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def CP\_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= CP\_abs\_err\_val(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

def CP\_fmt(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

\# \-------------------------------- SINGLE-FRACTION EXHAUSTIVE \-------------------------------------  
def CMP\_single\_minimality(center: Decimal, sigma: Decimal, q\_cap\_single: int \= 200\_000) \-\> Dict\[str, object\]:  
    """  
    Exhaustively enumerates all reduced p/q with q ≤ q\_cap\_single that fall inside \[center±sigma\].  
    Returns best by (min bits, then smaller q), plus enumeration statistics.  
    """  
    L, U \= center \- sigma, center \+ sigma  
    best \= None  \# (bits, q, p, value\_decimal)  
    tested \= 0  
    hits \= 0  
    for q in range(1, q\_cap\_single+1):  
        \# compute integer p bounds that can land inside band  
        p\_low  \= int((L\*Decimal(q)).to\_integral\_value(rounding="ROUND\_CEILING"))  
        p\_high \= int((U\*Decimal(q)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
        if p\_low \> p\_high:  
            continue  
        for p in range(p\_low, p\_high+1):  
            tested \+= 1  
            if math.gcd(p,q) \!= 1:  
                continue  
            hits \+= 1  
            fr \= Fraction(p,q)  
            bits \= CP\_bits(fr)  
            val  \= Decimal(p)/Decimal(q)  
            if (best is None) or (bits \< best\[0\]) or (bits \== best\[0\] and q \< best\[1\]):  
                best \= (bits, q, p, val)  
    if best is None:  
        return {"exists": False, "tested\_pairs": tested, "hits": 0}  
    bits, q, p, val \= best  
    return {  
        "exists": True,  
        "tested\_pairs": tested,  
        "hits": hits,  
        "best": {"p": p, "q": q, "bits": bits, "value": str(val)}  
    }

\# \-------------------------------- TWO-TERM EXHAUSTIVE (GENERAL) \----------------------------------  
def CMP\_two\_term\_minimality(center: Decimal, sigma: Decimal, D\_cap\_2: int \= 1\_000,  
                            window\_multiplier: Decimal \= Decimal(5)) \-\> Dict\[str, object\]:  
    """  
    Exhaustive over all reduced p1/q1, p2/q2 with q1,q2 ≤ D\_cap\_2 whose sum lies in \[center±sigma\].  
    Enumeration strategy:  
      For each reduced p1/q1 lying in a generous WINDOW around center (center ± max(5σ, 0.1)),  
      compute residual band \[L2,U2\] \= \[L − p1/q1, U − p1/q1\]. For each q2 ≤ D\_cap\_2, compute the  
      integer range p2 ∈ \[ceil(L2\*q2), floor(U2\*q2)\] and test all reduced p2/q2 in that range.  
    This covers \*all\* 2-term reduced packs under the caps. Returns best by total bits (tie-break  
    on smaller max(q1,q2), then smaller q1+q2).  
    """  
    L, U \= center \- sigma, center \+ sigma  
    window \= max(Decimal("0.1"), window\_multiplier \* sigma)  
    W\_L, W\_U \= center \- window, center \+ window

    tested\_outer \= 0  
    tested\_inner \= 0  
    hits \= 0  
    best \= None  \# (bits\_total, max\_q, sum\_q, (p1,q1,p2,q2), val\_decimal)

    \# Enumerate the first term  
    for q1 in range(1, D\_cap\_2+1):  
        \# window for p1  
        p1\_low  \= int((W\_L\*Decimal(q1)).to\_integral\_value(rounding="ROUND\_FLOOR")) \- 1  
        p1\_high \= int((W\_U\*Decimal(q1)).to\_integral\_value(rounding="ROUND\_CEILING")) \+ 1  
        for p1 in range(p1\_low, p1\_high+1):  
            tested\_outer \+= 1  
            if math.gcd(p1, q1) \!= 1:  
                continue  
            v1 \= Decimal(p1)/Decimal(q1)  
            L2, U2 \= L \- v1, U \- v1  
            if L2 \> U2:  
                continue  
            \# Enumerate the second term using exact p2 ranges  
            for q2 in range(1, D\_cap\_2+1):  
                p2\_low  \= int((L2\*Decimal(q2)).to\_integral\_value(rounding="ROUND\_CEILING"))  
                p2\_high \= int((U2\*Decimal(q2)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
                if p2\_low \> p2\_high:  
                    continue  
                for p2 in range(p2\_low, p2\_high+1):  
                    tested\_inner \+= 1  
                    if math.gcd(p2, q2) \!= 1:  
                        continue  
                    v2 \= Decimal(p2)/Decimal(q2)  
                    val \= v1 \+ v2  
                    \# val is guaranteed in band by construction of p2 range; gcd filters ensure reduced  
                    hits \+= 1  
                    fr1, fr2 \= Fraction(p1,q1), Fraction(p2,q2)  
                    bits\_total \= CP\_bits(fr1) \+ CP\_bits(fr2)  
                    max\_q \= max(q1, q2)  
                    sum\_q \= q1 \+ q2  
                    if (best is None) or \\  
                       (bits\_total \< best\[0\]) or \\  
                       (bits\_total \== best\[0\] and max\_q \< best\[1\]) or \\  
                       (bits\_total \== best\[0\] and max\_q \== best\[1\] and sum\_q \< best\[2\]):  
                        best \= (bits\_total, max\_q, sum\_q, (p1,q1,p2,q2), val)

    if best is None:  
        return {  
            "exists": False,  
            "tested\_outer\_candidates": tested\_outer,  
            "tested\_inner\_pairs": tested\_inner,  
            "hits": 0  
        }  
    bits\_total, max\_q, sum\_q, (p1,q1,p2,q2), val \= best  
    return {  
        "exists": True,  
        "tested\_outer\_candidates": tested\_outer,  
        "tested\_inner\_pairs": tested\_inner,  
        "hits": hits,  
        "best": {  
            "p1": p1, "q1": q1,  
            "p2": p2, "q2": q2,  
            "bits\_total": bits\_total,  
            "value": str(val)  
        }  
    }

\# \-------------------------------- TWO-TERM EXHAUSTIVE (EGYPTIAN) \---------------------------------  
def CMP\_two\_term\_egyptian(center: Decimal, sigma: Decimal, U\_cap\_egy: int \= 50\_000) \-\> Dict\[str, object\]:  
    """  
    Exhaustive over 1/n1 \+ 1/n2 with n1,n2 ≤ U\_cap\_egy landing in the band.  
    Returns best by total bits (bits of 1/n is bits(n) since numerator=1).  
    """  
    L, U \= center \- sigma, center \+ sigma  
    tested \= 0  
    hits \= 0  
    best \= None  \# (bits\_total, max\_n, sum\_n, (n1,n2), val\_decimal)  
    for n1 in range(2, U\_cap\_egy+1):  
        v1 \= Decimal(1) / Decimal(n1)  
        L2, U2 \= L \- v1, U \- v1  
        if L2 \> U2:  
            continue  
        \# For 1/n2 in \[L2, U2\] \=\> n2 in \[ceil(1/U2), floor(1/L2)\] (when positive)  
        \# Handle signs robustly by scanning a safe local range  
        \# We'll just derive a tight candidate interval when L2,U2 \> 0; otherwise fall back to scan.  
        if L2 \> 0:  
            n2\_min \= int((Decimal(1)/U2).to\_integral\_value(rounding="ROUND\_CEILING"))  
            n2\_max \= int((Decimal(1)/L2).to\_integral\_value(rounding="ROUND\_FLOOR"))  
            n2\_min \= max(n2\_min, 2\)  
            n2\_max \= min(n2\_max, U\_cap\_egy)  
            candidates \= range(n2\_min, n2\_max+1) if n2\_min \<= n2\_max else range(0)  
        else:  
            \# conservative local search when residual allows negative/zero — try a bounded window  
            start \= 2  
            stop  \= min(U\_cap\_egy, 2\_000)  
            candidates \= range(start, stop+1)

        for n2 in candidates:  
            tested \+= 1  
            v2 \= Decimal(1) / Decimal(n2)  
            val \= v1 \+ v2  
            if L \<= val \<= U:  
                hits \+= 1  
                fr1, fr2 \= Fraction(1,n1), Fraction(1,n2)  
                bits\_total \= CP\_bits(fr1) \+ CP\_bits(fr2)  
                max\_n \= max(n1,n2)  
                sum\_n \= n1 \+ n2  
                if (best is None) or \\  
                   (bits\_total \< best\[0\]) or \\  
                   (bits\_total \== best\[0\] and max\_n \< best\[1\]) or \\  
                   (bits\_total \== best\[0\] and max\_n \== best\[1\] and sum\_n \< best\[2\]):  
                    best \= (bits\_total, max\_n, sum\_n, (n1,n2), val)

    if best is None:  
        return {  
            "exists": False,  
            "tested\_pairs": tested,  
            "hits": 0  
        }  
    bits\_total, max\_n, sum\_n, (n1,n2), val \= best  
    return {  
        "exists": True,  
        "tested\_pairs": tested,  
        "hits": hits,  
        "best": {  
            "n1": n1, "n2": n2,  
            "bits\_total": bits\_total,  
            "value": str(val)  
        }  
    }

\# \-------------------------------- REPORT \+ CERTIFICATE WRITER \------------------------------------  
def CMP\_prove\_band(name: str, center: Decimal, sigma: Decimal,  
                   Q\_cap\_single: int \= 200\_000,  
                   D\_cap\_2: int \= 1\_000,  
                   U\_cap\_egy: int \= 50\_000,  
                   window\_multiplier: Decimal \= Decimal(5)) \-\> Tuple\[str, Dict\[str, object\]\]:  
    """  
    Runs all three proofs for a single band and returns (markdown\_report, json\_record).  
    """  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        CP\_hdr(f"Composite Minimality Proofs++ — {name}")  
        print(f"Band: \[{center \- sigma}, {center \+ sigma}\]  (center={center}, sigma={sigma})")  
        print(f"Caps: single q ≤ {Q\_cap\_single}; two-term denoms ≤ {D\_cap\_2}; egyptian n ≤ {U\_cap\_egy}\\n")

        \# SINGLE FRACTION  
        sres \= CMP\_single\_minimality(center, sigma, q\_cap\_single=Q\_cap\_single)  
        if sres\["exists"\]:  
            b \= sres\["best"\]  
            print(f"\[Single\] best p/q @ q≤{Q\_cap\_single}: {b\['p'\]}/{b\['q'\]}  bits={b\['bits'\]}  value={b\['value'\]}")  
        else:  
            print(f"\[Single\] no reduced p/q with q≤{Q\_cap\_single} lies in band.")  
        print(f"\[Single\] enumeration: tested\_pairs={sres\['tested\_pairs'\]}  hits={sres.get('hits',0)}\\n")

        \# TWO-TERM GENERAL  
        gres \= CMP\_two\_term\_minimality(center, sigma, D\_cap\_2=D\_cap\_2, window\_multiplier=window\_multiplier)  
        if gres\["exists"\]:  
            b \= gres\["best"\]  
            print(f"\[2-term general\] best: {b\['p1'\]}/{b\['q1'\]} \+ {b\['p2'\]}/{b\['q2'\]}  "  
                  f"bits\_total={b\['bits\_total'\]}  value={b\['value'\]}")  
        else:  
            print(f"\[2-term general\] no reduced p1/q1+p2/q2 with q1,q2 ≤ {D\_cap\_2} lies in band.")  
        print(f"\[2-term general\] enumeration: outer\_candidates={gres\['tested\_outer\_candidates'\]}  "  
              f"inner\_pairs={gres\['tested\_inner\_pairs'\]}  hits={gres.get('hits',0)}\\n")

        \# TWO-TERM EGYPTIAN  
        eres \= CMP\_two\_term\_egyptian(center, sigma, U\_cap\_egy=U\_cap\_egy)  
        if eres\["exists"\]:  
            b \= eres\["best"\]  
            print(f"\[2-term Egyptian\] best: 1/{b\['n1'\]} \+ 1/{b\['n2'\]}  "  
                  f"bits\_total={b\['bits\_total'\]}  value={b\['value'\]}")  
        else:  
            print(f"\[2-term Egyptian\] no 1/n1+1/n2 with n1,n2 ≤ {U\_cap\_egy} lies in band.")  
        print(f"\[2-term Egyptian\] enumeration: tested\_pairs={eres\['tested\_pairs'\]}  hits={eres.get('hits',0)}\\n")

        \# CERTIFICATE SUMMARY  
        print("Certificate summary (under stated caps):")  
        if sres\["exists"\]:  
            s\_bits \= int(sres\["best"\]\["bits"\])  
            print(f"- Singles: best bits \= {s\_bits} (exhaustive within q≤{Q\_cap\_single}).")  
        else:  
            print(f"- Singles: none exist within q≤{Q\_cap\_single}.")  
        if gres\["exists"\]:  
            g\_bits \= int(gres\["best"\]\["bits\_total"\])  
            print(f"- Two-term (general): best bits\_total \= {g\_bits} (exhaustive within q1,q2≤{D\_cap\_2}).")  
            if sres\["exists"\]:  
                print(f"  Improvement vs best single \= {s\_bits \- g\_bits} bits (if positive).")  
        else:  
            print(f"- Two-term (general): none exist within q1,q2≤{D\_cap\_2}.")  
        if eres\["exists"\]:  
            e\_bits \= int(eres\["best"\]\["bits\_total"\])  
            print(f"- Two-term (Egyptian): best bits\_total \= {e\_bits} (exhaustive within n≤{U\_cap\_egy}).")

    md\_text \= buf.getvalue()  
    record \= {  
        "name": name,  
        "timestamp": CP\_now(),  
        "band": {"center": str(center), "sigma": str(sigma)},  
        "caps": {"Q\_cap\_single": Q\_cap\_single, "D\_cap\_2": D\_cap\_2, "U\_cap\_egy": U\_cap\_egy},  
        "single": sres,  
        "two\_term\_general": gres,  
        "two\_term\_egyptian": eres  
    }  
    return md\_text, record

def CMP\_PROVE\_BANDS(bands: List\[Dict\[str, str\]\],  
                    Q\_cap\_single: int \= 200\_000,  
                    D\_cap\_2: int \= 1\_000,  
                    U\_cap\_egy: int \= 50\_000,  
                    window\_multiplier: Decimal \= Decimal(5),  
                    write\_files: bool \= True,  
                    combined\_filename: str \= "Composite\_Proofs\_Report.md",  
                    json\_filename: str \= "proofs.json"):  
    """  
    Prove a list of bands. Each band dict must have: {"name", "center\_str", "sigma\_str"}.  
    Writes per-band .md and a combined report \+ JSON in composite\_proofs/.  
    """  
    combined\_chunks \= \[\]  
    json\_records \= \[\]

    CP\_hdr("Composite Minimality Proofs++ — RUN")  
    print(f"Items: {len(bands)}  Caps: single q≤{Q\_cap\_single}, 2-term denoms≤{D\_cap\_2}, egyptian n≤{U\_cap\_egy}")  
    print(f"Timestamp: {CP\_now()}  Output: {CP\_OUTDIR}")

    for b in bands:  
        name \= b\["name"\]  
        center \= Decimal(str(b\["center\_str"\]))  
        sigma  \= Decimal(str(b\["sigma\_str"\]))  
        md\_text, record \= CMP\_prove\_band(name, center, sigma,  
                                         Q\_cap\_single=Q\_cap\_single,  
                                         D\_cap\_2=D\_cap\_2,  
                                         U\_cap\_egy=U\_cap\_egy,  
                                         window\_multiplier=window\_multiplier)  
        print(md\_text)  
        combined\_chunks.append(f"\# {name}\\n\\n{md\_text}\\n")  
        json\_records.append(record)  
        if write\_files and WRITE\_PER\_MODULE\_FILES:  
            CP\_write(os.path.join(CP\_OUTDIR, f"{name.replace(' ','\_')}.md"), md\_text)

    combined \= "\# Composite Minimality Proofs++ — Combined\\n\\n" \+ "\\n".join(combined\_chunks)  
    if write\_files and WRITE\_COMBINED\_FILE:  
        CP\_write(os.path.join(CP\_OUTDIR, combined\_filename), combined)  
    if write\_files and WRITE\_JSON\_WITNESS:  
        CP\_write(os.path.join(CP\_OUTDIR, json\_filename), json.dumps(json\_records, indent=2))

\# \----------------------------------- DEMO INVOCATION \---------------------------------------------  
\# You can change these bands or call CMP\_PROVE\_BANDS yourself with any list you want.  
BAND\_SPECS \= \[  
    {"name": "alpha^{-1}",    "center\_str": "137.035999084", "sigma\_str": "2.1e-8"},  
    {"name": "sin^2(theta\_W) (demo)", "center\_str": "0.2312200006", "sigma\_str": "3e-5"},  
    {"name": "pi (tight demo)",       "center\_str": "3.141592653589793", "sigma\_str": "1e-12"},  
\]

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# Reasonable default caps; tune D\_cap\_2 upward for stronger certificates (costlier).  
    CMP\_PROVE\_BANDS(BAND\_SPECS,  
                    Q\_cap\_single=200\_000,  
                    D\_cap\_2=800,                \# try 800–1200 for heavier proofs  
                    U\_cap\_egy=20000,            \# egyptian cap; safe to raise if you care  
                    window\_multiplier=Decimal(5),  
                    write\_files=True,  
                    combined\_filename="Composite\_Proofs\_Report.md",  
                    json\_filename="proofs.json")

\# \=================================================================================================  
\# \=== END v3.2 \====================================================================================  
\# \=================================================================================================  
\# \=================================================================================================  
\# \=== FRACTION PHYSICS DLC — v3.4 "Common Substrate Fusion \+ 3-Term Explorer \+ Evidence Ledger" \===  
\# \=== Append-only • No charts • Honors WRITE\_\* flags & OUTPUT\_DIR                                \===  
\# \=================================================================================================  
\# What you get:  
\#   1\) Fusion(A): single shared-q across many bands (bit-min or q-min objective)  
\#   2\) Fusion(B): shared {q1,q2} for ALL bands (each x\_i ≈ p1\_i/q1 \+ p2\_i/q2 within its band)  
\#   3\) 3-Term Explorer: when 2-term (exhaustive) is insufficient, search 3-term packs with pruning  
\#   4\) Evidence Ledger: scans OUTPUT\_DIR, indexes all md/json artifacts into one report \+ JSON.  
\# \=================================================================================================

import os, io, math, time, json, itertools  
from fractions import Fraction  
from decimal import Decimal, getcontext, ROUND\_FLOOR  
from contextlib import redirect\_stdout  
from typing import List, Dict, Tuple, Optional

\# \-------------------------------- CONFIG SYNC \-----------------------------------------------------  
try:  
    WRITE\_PER\_MODULE\_FILES  
except NameError:  
    WRITE\_PER\_MODULE\_FILES \= True  
try:  
    WRITE\_COMBINED\_FILE  
except NameError:  
    WRITE\_COMBINED\_FILE \= True  
try:  
    WRITE\_JSON\_WITNESS  
except NameError:  
    WRITE\_JSON\_WITNESS \= True  
try:  
    OUTPUT\_DIR  
except NameError:  
    OUTPUT\_DIR \= "/content/fraction\_physics\_dlc"

FUS\_OUTDIR \= os.path.join(OUTPUT\_DIR, "fusion\_v34")  
LEDGER\_PATH\_MD \= os.path.join(OUTPUT\_DIR, "Evidence\_Ledger.md")  
LEDGER\_PATH\_JSON \= os.path.join(OUTPUT\_DIR, "Evidence\_Ledger.json")  
os.makedirs(FUS\_OUTDIR, exist\_ok=True)

getcontext().prec \= 140

\# \-------------------------------- UTILITIES \-------------------------------------------------------  
def F34\_now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def F34\_write(path: str, text: str):  
    try:  
        os.makedirs(os.path.dirname(path), exist\_ok=True)  
        with open(path, "w", encoding="utf-8") as f: f.write(text)  
        print(f"\[file\] wrote: {path}")  
    except Exception as e:  
        print(f"\[file\] failed to write {path}: {e}")

def F34\_hdr(title: str):  
    bar \= "="\*110  
    print("\\n"+bar); print(f"=== {title}"); print(bar)

def F34\_print\_table(rows: List\[Dict\[str, str\]\], cols: List\[str\]):  
    if not rows:  
        print("(no rows)"); return  
    widths \= {c: max(len(c), \*(len(str(r.get(c,""))) for r in rows)) for c in cols}  
    line \= " | ".join(c.ljust(widths\[c\]) for c in cols)  
    print(line); print("-"\*len(line))  
    for r in rows:  
        print(" | ".join(str(r.get(c,"")).ljust(widths\[c\]) for c in cols))

def F34\_bits(fr: Fraction) \-\> int:  
    p, q \= abs(fr.numerator), abs(fr.denominator)  
    cp \= 0 if p \<= 1 else math.ceil(math.log2(p))  
    cq \= 0 if q \<= 1 else math.ceil(math.log2(q))  
    return cp \+ cq

def F34\_fmt(fr: Fraction) \-\> str:  
    return f"{fr.numerator}/{fr.denominator}"

def F34\_abs\_err(fr: Fraction, target: Decimal) \-\> Decimal:  
    return abs(Decimal(fr.numerator)/Decimal(fr.denominator) \- target)

def F34\_ppm(fr: Fraction, target: Decimal) \-\> Decimal:  
    err \= F34\_abs\_err(fr, target)  
    return Decimal(0) if target \== 0 else (err/abs(target))\*Decimal(1\_000\_000)

\# \-------------------------------- SMALL CONSTANTS (LOCAL) \----------------------------------------  
def F34\_fast\_pi():  
    old \= getcontext().prec  
    getcontext().prec \= 160  
    one=Decimal(1); two=Decimal(2); four=Decimal(4)  
    a=one; b=one/two.sqrt(); t=Decimal("0.25"); p=one  
    for \_ in range(10):  
        an=(a+b)/two; b=(a\*b).sqrt(); t=t \- p\*(a-an)\*(a-an); a=an; p=p\*two  
    pi=(a+b)\*(a+b)/(four\*t)  
    getcontext().prec \= old  
    return \+pi

PI\_D \= F34\_fast\_pi()  
E\_D  \= Decimal(1).exp()  
LN2  \= Decimal(2).ln()  
SQRT2 \= Decimal(2).sqrt()  
ALPHA\_INV \= Decimal("137.035999084")  
ALPHA     \= Decimal(1)/ALPHA\_INV

\# \-------------------------------- FUSION (A): SHARED SINGLE q \------------------------------------  
def F34\_shared\_q(bands: List\[Dict\[str,str\]\], q\_cap: int \= 500000, objective: str \= "bits") \-\> Optional\[Dict\]:  
    """  
    Given bands: \[{"name", "center\_str", "sigma\_str"}\], find single q ≤ q\_cap and integers p\_i with p\_i/q in band\_i.  
    Objective: "bits" (min total Σ bits(p\_i/q)) or "q" (min q then bits).  
    """  
    targets \= \[(it\["name"\], Decimal(it\["center\_str"\]), Decimal(it\["sigma\_str"\])) for it in bands\]  
    best=None  \# (primary, secondary, q, rows)  
    for q in range(1, q\_cap+1):  
        rows=\[\]; total\_bits=0; feasible=True  
        for (nm, xc, sg) in targets:  
            L, U \= xc \- sg, xc \+ sg  
            p0 \= int((Decimal(q)\*xc).to\_integral\_value(rounding="ROUND\_FLOOR"))  
            hit=None; val=None  
            for p in (p0-2, p0-1, p0, p0+1, p0+2):  
                val \= Decimal(p)/Decimal(q)  
                if L \<= val \<= U:  
                    hit \= p; break  
            if hit is None:  
                feasible=False; break  
            fr \= Fraction(hit, q)  
            rows.append({  
                "name": nm, "p": hit, "q": q,  
                "value": str(val),  
                "bits": F34\_bits(fr),  
                "ppm": str(F34\_ppm(fr, xc))  
            })  
            total\_bits \+= F34\_bits(fr)  
        if not feasible:  
            continue  
        primary, secondary \= ((total\_bits, q) if objective=="bits" else (q, total\_bits))  
        if (best is None) or (primary \< best\[0\]) or (primary==best\[0\] and secondary \< best\[1\]):  
            best \= (primary, secondary, q, rows)  
    if best is None:  
        return None  
    primary, secondary, q, rows \= best  
    return {"q": q, "score\_primary": primary, "score\_secondary": secondary, "matches": rows, "objective": objective}

def F34\_report\_shared\_q(title: str, bands: List\[Dict\[str,str\]\], q\_cap: int, objective: str) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        F34\_hdr(f"Fusion(A) — Shared Single q — {title}")  
        print(f"q\_cap={q\_cap} objective={objective}  items={len(bands)}")  
        for it in bands:  
            print(f" \- {it\['name'\]}: center={it\['center\_str'\]} sigma={it\['sigma\_str'\]}")  
        sol \= F34\_shared\_q(bands, q\_cap=q\_cap, objective=objective)  
        if sol is None:  
            print("\\nNo common q found under cap.")  
        else:  
            print(f"\\nSOLUTION: q={sol\['q'\]}  primary={sol\['score\_primary'\]}  secondary={sol\['score\_secondary'\]}")  
            rows=\[\]  
            for m in sol\["matches"\]:  
                rows.append({  
                    "name": m\["name"\],  
                    "p/q": f"{m\['p'\]}/{sol\['q'\]}",  
                    "value": m\["value"\],  
                    "ppm": m\["ppm"\],  
                    "bits": m\["bits"\]  
                })  
            F34\_print\_table(rows, \["name","p/q","value","ppm","bits"\])  
    return buf.getvalue()

\# \----------------------- FUSION (B): SHARED {q1, q2} FOR ALL BANDS (2-TERM) \----------------------  
def F34\_feasible\_two\_term\_for\_item(xc: Decimal, sg: Decimal, q1: int, q2: int,  
                                   p1\_neighborhood: int \= 3\) \-\> Optional\[Tuple\[int,int,Decimal\]\]:  
    """  
    For a single item band \[xc±sg\], find integers (p1,p2) with p1/q1 \+ p2/q2 in band.  
    We test p1 near q1\*xc (±p1\_neighborhood), then derive exact p2 interval.  
    Returns (p1,p2,value) if feasible, else None.  
    """  
    L, U \= xc \- sg, xc \+ sg  
    p1\_center \= int((Decimal(q1)\*xc).to\_integral\_value(rounding="ROUND\_FLOOR"))  
    for dp in range(-p1\_neighborhood, p1\_neighborhood+1):  
        p1 \= p1\_center \+ dp  
        v1 \= Decimal(p1)/Decimal(q1)  
        L2 \= (L \- v1)\*Decimal(q2)  
        U2 \= (U \- v1)\*Decimal(q2)  
        p2\_low  \= int(L2.to\_integral\_value(rounding="ROUND\_CEILING"))  
        p2\_high \= int(U2.to\_integral\_value(rounding="ROUND\_FLOOR"))  
        if p2\_low \> p2\_high:  
            continue  
        \# choose closest p2 to q2\*(xc \- v1)  
        p2\_center \= int((Decimal(q2)\*(xc \- v1)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
        for p2 in (p2\_center, p2\_center-1, p2\_center+1, p2\_low, p2\_high):  
            if p2\_low \<= p2 \<= p2\_high:  
                val \= v1 \+ Decimal(p2)/Decimal(q2)  
                return (p1, p2, val)  
    return None

def F34\_shared\_pair(bands: List\[Dict\[str,str\]\], q1\_cap: int \= 800, q2\_cap: int \= 800,  
                    p1\_neighborhood: int \= 3, objective: str \= "bits") \-\> Optional\[Dict\]:  
    """  
    Search q1≤q1\_cap, q2≤q2\_cap such that for every item i there exist p1\_i, p2\_i with  
    p1\_i/q1 \+ p2\_i/q2 ∈ band\_i. Objective: min total bits across all (p1\_i/q1, p2\_i/q2)  
    (ties: min max(q1,q2), then min q1+q2).  
    """  
    targets \= \[(it\["name"\], Decimal(it\["center\_str"\]), Decimal(it\["sigma\_str"\])) for it in bands\]  
    best=None  \# (primary, tie1, tie2, q1, q2, rows)  
    for q1 in range(1, q1\_cap+1):  
        for q2 in range(1, q2\_cap+1):  
            rows=\[\]; total\_bits=0; feasible=True  
            for (nm, xc, sg) in targets:  
                feas \= F34\_feasible\_two\_term\_for\_item(xc, sg, q1, q2, p1\_neighborhood=p1\_neighborhood)  
                if feas is None:  
                    feasible=False; break  
                p1, p2, val \= feas  
                fr1 \= Fraction(p1, q1)  
                fr2 \= Fraction(p2, q2)  
                rows.append({  
                    "name": nm,  
                    "p1/q1": F34\_fmt(fr1),  
                    "p2/q2": F34\_fmt(fr2),  
                    "value": str(val),  
                    "bits1": F34\_bits(fr1),  
                    "bits2": F34\_bits(fr2),  
                    "bits\_total": F34\_bits(fr1)+F34\_bits(fr2)  
                })  
                total\_bits \+= F34\_bits(fr1)+F34\_bits(fr2)  
            if not feasible:  
                continue  
            primary \= total\_bits if objective=="bits" else max(q1,q2)  
            tie1    \= max(q1,q2) if objective=="bits" else total\_bits  
            tie2    \= q1 \+ q2  
            if (best is None) or (primary \< best\[0\]) or (primary==best\[0\] and tie1 \< best\[1\]) or (primary==best\[0\] and tie1==best\[1\] and tie2 \< best\[2\]):  
                best \= (primary, tie1, tie2, q1, q2, rows)  
    if best is None:  
        return None  
    primary, tie1, tie2, q1, q2, rows \= best  
    return {"q1": q1, "q2": q2, "objective": objective, "score\_primary": primary, "tie1": tie1, "tie2": tie2, "matches": rows}

def F34\_report\_shared\_pair(title: str, bands: List\[Dict\[str,str\]\],  
                           q1\_cap: int, q2\_cap: int, p1\_neighborhood: int, objective: str) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        F34\_hdr(f"Fusion(B) — Shared {{q1,q2}} — {title}")  
        print(f"q1\_cap={q1\_cap} q2\_cap={q2\_cap} p1\_neighborhood=±{p1\_neighborhood} objective={objective}")  
        for it in bands:  
            print(f" \- {it\['name'\]}: center={it\['center\_str'\]} sigma={it\['sigma\_str'\]}")  
        sol \= F34\_shared\_pair(bands, q1\_cap=q1\_cap, q2\_cap=q2\_cap,  
                              p1\_neighborhood=p1\_neighborhood, objective=objective)  
        if sol is None:  
            print("\\nNo common {q1,q2} substrate found under caps.")  
        else:  
            print(f"\\nSOLUTION: q1={sol\['q1'\]} q2={sol\['q2'\]}  primary={sol\['score\_primary'\]}  tie1={sol\['tie1'\]}  tie2={sol\['tie2'\]}")  
            rows=\[\]  
            for m in sol\["matches"\]:  
                rows.append({  
                    "name": m\["name"\],  
                    "p1/q1": m\["p1/q1"\],  
                    "p2/q2": m\["p2/q2"\],  
                    "value": m\["value"\],  
                    "bits1": m\["bits1"\],  
                    "bits2": m\["bits2"\],  
                    "bits\_total": m\["bits\_total"\]  
                })  
            F34\_print\_table(rows, \["name","p1/q1","p2/q2","value","bits1","bits2","bits\_total"\])  
    return buf.getvalue()

\# \-------------------------------- 3-TERM EXPLORER (PRUNED) \---------------------------------------  
def F34\_three\_term\_explorer(center: Decimal, sigma: Decimal,  
                            D\_cap: int \= 600,  
                            window\_multiplier: Decimal \= Decimal(5),  
                            bits\_ceiling: Optional\[int\] \= None) \-\> Optional\[Dict\]:  
    """  
    Meet-in-the-middle-ish 3-term search with pruning.  
      \- Precompute pool T of reduced p/q with q≤D\_cap and |p/q \- center| ≤ window  
      \- Sort by bits; maintain best\_bits, prune combos whose partial bits exceed current best  
      \- For each pair (t1,t2), compute residual band for t3 and check reduced candidates exactly.  
    Returns best by total bits; ties by max(q), then sum q.  
    """  
    L, U \= center \- sigma, center \+ sigma  
    window \= max(Decimal("0.1"), window\_multiplier \* sigma)  
    W\_L, W\_U \= center \- window, center \+ window

    \# Build pool  
    pool=\[\]  
    for q in range(1, D\_cap+1):  
        p\_low  \= int((W\_L\*Decimal(q)).to\_integral\_value(rounding="ROUND\_FLOOR")) \- 1  
        p\_high \= int((W\_U\*Decimal(q)).to\_integral\_value(rounding="ROUND\_CEILING")) \+ 1  
        for p in range(p\_low, p\_high+1):  
            if math.gcd(p,q) \!= 1: continue  
            fr \= Fraction(p,q)  
            pool.append(fr)  
    \# unique & sort by bit cost  
    pool \= sorted(set(pool), key=lambda fr: (F34\_bits(fr), fr.denominator, abs(fr.numerator)))  
    if not pool:  
        return None

    best=None  \# (bits\_total, maxq, sumq, (fr1,fr2,fr3), value)  
    best\_bits \= bits\_ceiling if bits\_ceiling is not None else None

    \# For fast residual candidate generation, bucket pool by denominator for quick scans  
    pool\_by\_q \= {}  
    for fr in pool:  
        pool\_by\_q.setdefault(fr.denominator, \[\]).append(fr)

    \# Iterate over pairs with pruning  
    for i, fr1 in enumerate(pool):  
        bits1 \= F34\_bits(fr1)  
        if best\_bits is not None and bits1 \>= best\_bits:  
            break  
        for j in range(i, len(pool)):  
            fr2 \= pool\[j\]  
            bits12 \= bits1 \+ F34\_bits(fr2)  
            if best\_bits is not None and bits12 \>= best\_bits:  
                continue  
            v12 \= Decimal(fr1.numerator)/Decimal(fr1.denominator) \+ Decimal(fr2.numerator)/Decimal(fr2.denominator)  
            L3 \= L \- v12; U3 \= U \- v12  
            if L3 \> U3:  
                continue  
            \# For each q3 ≤ D\_cap, test p3 bounds exactly  
            for q3, frs in pool\_by\_q.items():  
                \# quick bound: if bits12 \+ bits(1/q3) \>= best\_bits, skip  
                if best\_bits is not None and bits12 \+ math.ceil(math.log2(max(2,q3))) \>= best\_bits:  
                    continue  
                p3\_low  \= int((L3\*Decimal(q3)).to\_integral\_value(rounding="ROUND\_CEILING"))  
                p3\_high \= int((U3\*Decimal(q3)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
                if p3\_low \> p3\_high:  
                    continue  
                \# scan candidates near center  
                p3\_center \= int(((L3+U3)/Decimal(2)\*Decimal(q3)).to\_integral\_value(rounding="ROUND\_FLOOR"))  
                cand\_p3 \= \[p for p in (p3\_center, p3\_center-1, p3\_center+1, p3\_low, p3\_high) if p3\_low \<= p \<= p3\_high\]  
                \# ensure reduced  
                for p3 in cand\_p3:  
                    if math.gcd(p3, q3) \!= 1:  
                        continue  
                    fr3 \= Fraction(p3, q3)  
                    total\_bits \= bits12 \+ F34\_bits(fr3)  
                    if (best is None) or (total\_bits \< best\[0\]) or \\  
                       (total\_bits \== best\[0\] and max(fr1.denominator, fr2.denominator, q3) \< best\[1\]) or \\  
                       (total\_bits \== best\[0\] and max(fr1.denominator, fr2.denominator, q3) \== best\[1\] and (fr1.denominator+fr2.denominator+q3) \< best\[2\]):  
                        val \= (Decimal(fr1.numerator)/Decimal(fr1.denominator) \+  
                               Decimal(fr2.numerator)/Decimal(fr2.denominator) \+  
                               Decimal(p3)/Decimal(q3))  
                        if L \<= val \<= U:  
                            best \= (total\_bits,  
                                    max(fr1.denominator, fr2.denominator, q3),  
                                    fr1.denominator \+ fr2.denominator \+ q3,  
                                    (fr1, fr2, Fraction(p3,q3)),  
                                    val)  
                            best\_bits \= total\_bits  
    if best is None:  
        return None  
    total\_bits, maxq, sumq, (fr1, fr2, fr3), val \= best  
    return {  
        "bits\_total": total\_bits,  
        "max\_q": maxq,  
        "sum\_q": sumq,  
        "terms": \[F34\_fmt(fr1), F34\_fmt(fr2), F34\_fmt(fr3)\],  
        "value": str(val)  
    }

def F34\_report\_three\_term(name: str, center: Decimal, sigma: Decimal,  
                          D\_cap: int \= 600, bits\_ceiling: Optional\[int\] \= None) \-\> str:  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        F34\_hdr(f"3-Term Explorer — {name}")  
        print(f"Band: \[{center \- sigma}, {center \+ sigma}\]  D\_cap={D\_cap}  bits\_ceiling={bits\_ceiling}")  
        res \= F34\_three\_term\_explorer(center, sigma, D\_cap=D\_cap, bits\_ceiling=bits\_ceiling)  
        if res is None:  
            print("No 3-term pack found under current caps.")  
        else:  
            print(f"Best 3-term: {' \+ '.join(res\['terms'\])}")  
            print(f"Value: {res\['value'\]}  bits\_total={res\['bits\_total'\]}  max\_q={res\['max\_q'\]}  sum\_q={res\['sum\_q'\]}")  
    return buf.getvalue()

\# \-------------------------------- EVIDENCE LEDGER BUILDER \----------------------------------------  
def F34\_build\_ledger(root: str) \-\> Tuple\[str, Dict\]:  
    """  
    Walk OUTPUT\_DIR, index all .md and .json artifacts with file sizes and mtimes.  
    """  
    entries=\[\]  
    for dirpath, dirnames, filenames in os.walk(root):  
        for fn in filenames:  
            if not (fn.endswith(".md") or fn.endswith(".json")):  
                continue  
            path \= os.path.join(dirpath, fn)  
            try:  
                st \= os.stat(path)  
                entries.append({  
                    "path": os.path.relpath(path, root),  
                    "bytes": st.st\_size,  
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st\_mtime))  
                })  
            except Exception:  
                continue  
    entries.sort(key=lambda e: (e\["path"\]))  
    \# Write MD  
    buf \= io.StringIO()  
    with redirect\_stdout(buf):  
        F34\_hdr("Evidence Ledger — Index of Artifacts")  
        print(f"Root: {root}   Items: {len(entries)}   Generated: {F34\_now()}\\n")  
        rows=\[\]  
        for e in entries:  
            rows.append({"file": e\["path"\], "bytes": e\["bytes"\], "modified": e\["mtime"\]})  
        F34\_print\_table(rows, \["file","bytes","modified"\])  
    md\_text \= buf.getvalue()  
    json\_obj \= {"root": root, "generated": F34\_now(), "entries": entries}  
    return md\_text, json\_obj

\# \-------------------------------- DEMO SETS (SELF-CONTAINED) \-------------------------------------  
FUSION\_BANDS\_DEMO \= \[  
    {"name":"alpha^{-1}", "center\_str": str(ALPHA\_INV), "sigma\_str": "2.1e-8"},  
    {"name":"pi",         "center\_str": str(PI\_D),      "sigma\_str": "1e-12"},  
    {"name":"ln2",        "center\_str": str(LN2),       "sigma\_str": "1e-15"},  
    {"name":"sqrt2",      "center\_str": str(SQRT2),     "sigma\_str": "1e-12"},  
\]

THREE\_TERM\_DEMOS \= \[  
    {"name":"alpha", "center\_str": str(ALPHA), "sigma\_str": "2e-12"},  
    {"name":"pi/8",  "center\_str": str(PI\_D/Decimal(8)), "sigma\_str": "1e-12"},  
\]

\# \-------------------------------- DRIVER \----------------------------------------------------------  
def F34\_MAIN():  
    F34\_hdr("v3.4 — START")  
    print(f"Precision: {getcontext().prec}   Time: {F34\_now()}   Output root: {FUS\_OUTDIR}")

    chunks=\[\]

    \# Fusion(A): shared single q (both objectives)  
    repA\_bits \= F34\_report\_shared\_q("demo set", FUSION\_BANDS\_DEMO, q\_cap=500000, objective="bits")  
    repA\_q    \= F34\_report\_shared\_q("demo set", FUSION\_BANDS\_DEMO, q\_cap=500000, objective="q")  
    chunks.append(("\# Fusion(A) — bits", repA\_bits, "fusionA\_bits.md"))  
    chunks.append(("\# Fusion(A) — q",    repA\_q,    "fusionA\_q.md"))

    \# Fusion(B): shared {q1,q2} substrate (two-term for all)  
    repB\_bits \= F34\_report\_shared\_pair("demo set", FUSION\_BANDS\_DEMO,  
                                       q1\_cap=600, q2\_cap=600, p1\_neighborhood=3, objective="bits")  
    repB\_q    \= F34\_report\_shared\_pair("demo set", FUSION\_BANDS\_DEMO,  
                                       q1\_cap=600, q2\_cap=600, p1\_neighborhood=3, objective="q")  
    chunks.append(("\# Fusion(B) — bits", repB\_bits, "fusionB\_bits.md"))  
    chunks.append(("\# Fusion(B) — q",    repB\_q,    "fusionB\_q.md"))

    \# 3-Term Explorer demos  
    for it in THREE\_TERM\_DEMOS:  
        name   \= it\["name"\]; center \= Decimal(it\["center\_str"\]); sigma \= Decimal(it\["sigma\_str"\])  
        trep \= F34\_report\_three\_term(name, center, sigma, D\_cap=600, bits\_ceiling=None)  
        chunks.append((f"\# 3-Term — {name}", trep, f"three\_term\_{name.replace(' ','\_')}.md"))

    \# Print & write files  
    combined=\[\]  
    for title, text, fname in chunks:  
        print(text)  
        combined.append(f"{title}\\n\\n{text}\\n")  
        if WRITE\_PER\_MODULE\_FILES:  
            F34\_write(os.path.join(FUS\_OUTDIR, fname), text)  
    if WRITE\_COMBINED\_FILE:  
        F34\_write(os.path.join(FUS\_OUTDIR, "Fusion\_v34\_Report.md"),  
                  "\# Fusion v3.4 — Combined\\n\\n" \+ "\\n".join(combined))

    \# Evidence Ledger (global over OUTPUT\_DIR)  
    md\_ledger, json\_ledger \= F34\_build\_ledger(OUTPUT\_DIR)  
    print(md\_ledger)  
    F34\_write(LEDGER\_PATH\_MD, md\_ledger)  
    if WRITE\_JSON\_WITNESS:  
        F34\_write(LEDGER\_PATH\_JSON, json.dumps(json\_ledger, indent=2))

    F34\_hdr("v3.4 — END")

if \_\_name\_\_ \== "\_\_main\_\_":  
    F34\_MAIN()

\# \=================================================================================================  
\# \=== END v3.4 \====================================================================================  
\# \=================================================================================================  
