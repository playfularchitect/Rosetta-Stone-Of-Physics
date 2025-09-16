\# QuantumCalPro\_v6\_4\_6\_CHAN\_MLE\_selfcontained.py  
\# Self-contained, notebook-safe, generates everything on every run.  
\# Optional flags (ignored if unknown flags are present, e.g. Jupyter's \-f):  
\#   \--verify  
\#   \--shots 100000  
\#   \--export-json snap.json  
\#   \--export-rho rho\_prefix  
\#   \--bootstrap 200  
\#   \--no-color

import argparse, math, json, sys  
from typing import Dict, Tuple  
import numpy as np

np.set\_printoptions(suppress=True, linewidth=140)

\# \---------- Basics \----------  
I2 \= np.eye(2, dtype=complex)  
sx \= np.array(\[\[0,1\],\[1,0\]\], dtype=complex)  
sy \= np.array(\[\[0,-1j\],\[1j,0\]\], dtype=complex)  
sz \= np.array(\[\[1,0\],\[0,-1\]\], dtype=complex)  
PAULI \= \[sx, sy, sz\]  
AXES \= \['X','Y','Z'\]  
PAIR2IDX \= {(a+b):(i,j) for i,a in enumerate(AXES) for j,b in enumerate(AXES)}

def to\_deg(x): return float(x)\*180.0/math.pi  
def clamp(x,a,b): return max(a, min(b, x))  
def frob(M): return float(np.linalg.norm(M, 'fro'))  
def ensure\_c(A): return np.asarray(A, dtype=complex)

def jsonify\_array(A: np.ndarray):  
    A \= np.asarray(A)  
    if np.iscomplexobj(A) or np.max(np.abs(np.imag(A)))\>1e-12:  
        return {"real": A.real.tolist(), "imag": A.imag.tolist()}  
    return A.astype(float).tolist()

\# \---------- Inline dataset (10× shots) \----------  
EXTERNAL\_COUNTS \= {  
 'XX': {'00': 18363, '01': 2216,  '10': 2142,  '11': 18239},  
 'XY': {'00': 10156, '01': 10197, '10': 10291, '11': 10316},  
 'XZ': {'00': 10219, '01': 10381, '10': 10238, '11': 10122},  
 'YX': {'00': 10257, '01': 10302, '10': 10157, '11': 10244},  
 'YY': {'00': 2239,  '01': 18241, '10': 18258, '11': 2222},  
 'YZ': {'00': 10289, '01': 10263, '10': 10188, '11': 10220},  
 'ZX': {'00': 10366, '01': 10202, '10': 10184, '11': 10208},  
 'ZY': {'00': 10185, '01': 10192, '10': 10387, '11': 10196},  
 'ZZ': {'00': 18603, '01': 1958,  '10': 1988,  '11': 18411},  
}

\# \---------- Formatting \----------  
def fmt\_mat3(M: np.ndarray) \-\> str:  
    M \= np.asarray(M, dtype=float)  
    return (  
        f"  {M\[0,0\]:+0.4f} {M\[0,1\]:+0.4f} {M\[0,2\]:+0.4f}\\n"  
        f"  {M\[1,0\]:+0.4f} {M\[1,1\]:+0.4f} {M\[1,2\]:+0.4f}\\n"  
        f"  {M\[2,0\]:+0.4f} {M\[2,1\]:+0.4f} {M\[2,2\]:+0.4f}"  
    )

\# \---------- Counts → T, singles \----------  
def basis\_E(c: Dict\[str,int\]) \-\> float:  
    n00,n01,n10,n11 \= c\['00'\],c\['01'\],c\['10'\],c\['11'\]  
    N \= n00+n01+n10+n11  
    if N==0: return 0.0  
    return (n00+n11 \- n01-n10)/N

def singles\_A(c):  
    n00,n01,n10,n11 \= c\['00'\],c\['01'\],c\['10'\],c\['11'\]; N=n00+n01+n10+n11  
    return 0.0 if N==0 else (n00+n01 \- n10-n11)/N

def singles\_B(c):  
    n00,n01,n10,n11 \= c\['00'\],c\['01'\],c\['10'\],c\['11'\]; N=n00+n01+n10+n11  
    return 0.0 if N==0 else (n00+n10 \- n01-n11)/N

def counts\_to\_T\_and\_singles(data) \-\> Tuple\[np.ndarray,np.ndarray,np.ndarray\]:  
    T \= np.zeros((3,3), float)  
    a \= np.zeros(3, float)  
    b \= np.zeros(3, float)  
    for pair,c in data.items():  
        i,j \= PAIR2IDX\[pair\]; T\[i,j\] \= basis\_E(c)  
    for i,Ai in enumerate(AXES):  
        a\[i\] \= np.mean(\[singles\_A(data\[Ai+Bj\]) for Bj in AXES\])  
    for j,Bj in enumerate(AXES):  
        b\[j\] \= np.mean(\[singles\_B(data\[Ai+Bj\]) for Ai in AXES\])  
    return T,a,b

\# \---------- CHSH \----------  
def chsh\_from\_T(T: np.ndarray):  
    M \= T.T @ T  
    w,\_ \= np.linalg.eigh(M)  
    w \= np.sort(w)\[::-1\]  
    S \= float(2.0\*math.sqrt(max(0.0, w\[0\]+w\[1\])))

    TA \= T @ T.T  
    wa,va \= np.linalg.eigh(TA); idx \= np.argsort(wa)\[::-1\]  
    a1 \= va\[:,idx\[0\]\]; a2 \= va\[:,idx\[1\]\]  
    b1v \= T.T @ a1; b2v \= T.T @ a2

    def norm(v):  
        n=np.linalg.norm(v);  
        return (v/n if n\>0 else v)

    return S, {"Alice":{"a1":norm(a1).tolist(),"a2":norm(a2).tolist()},  
               "Bob":{"b1":norm(b1v).tolist(),"b2":norm(b2v).tolist()},  
               "S\_pred":S}

\# \---------- Rotations / SVD frames \----------  
def Rz(a): c,s \= math.cos(a), math.sin(a); return np.array(\[\[c,-s,0\],\[s,c,0\],\[0,0,1\]\], float)  
def Ry(b): c,s \= math.cos(b), math.sin(b); return np.array(\[\[c,0,s\],\[0,1,0\],\[-s,0,c\]\], float)  
def R\_from\_zyz(a,b,g): return Rz(a) @ Ry(b) @ Rz(g)

def zyz\_from\_R(R: np.ndarray) \-\> Tuple\[float,float,float\]:  
    R \= np.asarray(R, float)  
    b \= math.acos(clamp(R\[2,2\], \-1.0, 1.0))  
    if abs(math.sin(b))\>1e-12:  
        a \= math.atan2(R\[1,2\], R\[0,2\])  
        g \= math.atan2(R\[2,1\], \-R\[2,0\])  
    else:  
        a \= math.atan2(R\[1,0\], R\[0,0\]); g=0.0  
    return a,b,g

def proper\_svd(T: np.ndarray):  
    U,s,Vt \= np.linalg.svd(T)  
    Su \= np.eye(3); Sv \= np.eye(3)  
    if np.linalg.det(U)\<0: Su\[2,2\] \= \-1  
    if np.linalg.det(Vt)\<0: Sv\[2,2\] \= \-1  
    RA \= U @ Su  
    RB \= Vt.T @ Sv  
    Sigma \= Su @ np.diag(s) @ Sv  
    return RA, Sigma, RB

def rad\_to\_rational\_pi(x, max\_den=41):  
    target \= x/math.pi; best=(0,1,abs(target))  
    for q in range(1,max\_den+1):  
        p \= int(round(target\*q))  
        err \= abs(target \- p/q)  
        if err\<best\[2\]: best=(p,q,err)  
    return best

def fmt\_pi\_rational(x, max\_den=41):  
    p,q,err \= rad\_to\_rational\_pi(x, max\_den)  
    if p==0: return "0"  
    s \= "-" if p\<0 else ""; p=abs(p)  
    if q==1 and p==1: return f"{s}π"  
    if q==1: return f"{s}{p}π"  
    return f"{s}{p}π/{q}"

\# \---------- States / metrics \----------  
def rho\_from\_abT(a,b,T):  
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel(); T=np.asarray(T,float)  
    rho \= 0.25\*np.kron(I2,I2)  
    for i in range(3): rho \+= 0.25\*a\[i\]\*np.kron(PAULI\[i\], I2)  
    for j in range(3): rho \+= 0.25\*b\[j\]\*np.kron(I2, PAULI\[j\])  
    for i in range(3):  
        for j in range(3):  
            rho \+= 0.25\*T\[i,j\]\*np.kron(PAULI\[i\], PAULI\[j\])  
    rho \= 0.5\*(rho \+ rho.conj().T)  
    return ensure\_c(rho)

def project\_to\_psd(rho):  
    w,V \= np.linalg.eigh(rho)  
    w \= np.maximum(w, 0.0); rho2 \= (V\*w) @ V.conj().T  
    rho2 \= rho2/np.trace(rho2)  
    return ensure\_c(rho2)

def bell\_phi\_plus():  
    v \= np.zeros(4,complex); v\[0\]=v\[3\]=1/math.sqrt(2); return np.outer(v,v.conj())

def fidelity(rho, psi): return float(np.real(np.trace(rho @ psi)))  
def purity(rho): return float(np.real(np.trace(rho @ rho)))

def concurrence(rho):  
    sy2 \= np.kron(sy, sy); rho\_tilde \= sy2 @ rho.conj() @ sy2  
    w \= np.linalg.eigvals(rho @ rho\_tilde)  
    w \= np.sort(np.real(np.sqrt(np.maximum(w,0))))\[::-1\]  
    return float(max(0.0, w\[0\]-w\[1\]-w\[2\]-w\[3\]))

def partial\_transpose(rho, sys=1):  
    r \= rho.reshape(2,2,2,2)  
    if sys==1: rpt \= r.transpose(0,3,2,1)  
    else:      rpt \= r.transpose(2,1,0,3)  
    return rpt.reshape(4,4)

def negativity(rho):  
    ev \= np.linalg.eigvals(partial\_transpose(rho,1))  
    return float(sum(abs(x) for x in np.real(ev) if x\<0))

def Fphi\_from\_T(T): return float((1 \+ T\[0,0\] \- T\[1,1\] \+ T\[2,2\]) / 4.0)

\# \---------- Likelihood & residuals \----------  
def zero\_singles\_probs(E):  
    pd \= (1+E)/4.0; po \= (1-E)/4.0  
    return {'00':pd, '01':po, '10':po, '11':pd}

def logL\_counts\_probs(counts, probs, eps=1e-15):  
    L=0.0  
    for k in ('00','01','10','11'):  
        p=max(probs\[k\],eps); n=counts\[k\]; L \+= n\*math.log(p)  
    return L

def residuals\_zero\_singles(data, T):  
    out={}  
    for pair,c in data.items():  
        i,j \= PAIR2IDX\[pair\]; E=T\[i,j\]  
        P \= zero\_singles\_probs(E); N=sum(c.values())  
        pred \= {k:N\*P\[k\] for k in ('00','01','10','11')}  
        out\[pair\] \= {k: (c\[k\]-pred\[k\])/N for k in ('00','01','10','11')}  
    return out

\# \---------- MLE tomography (RρR) \----------  
def single\_qubit\_meas\_rot(axis: str) \-\> np.ndarray:  
    H \= (1/np.sqrt(2))\*np.array(\[\[1,1\],\[1,-1\]\], dtype=complex)  
    Sdg \= np.array(\[\[1,0\],\[0,-1j\]\], dtype=complex)  
    if axis=='Z': return I2  
    if axis=='X': return H  
    if axis=='Y': return H @ Sdg   \# Y basis  
    raise ValueError("axis must be X/Y/Z")

def projectors\_for\_basis(Aaxis:str, Baxis:str):  
    UA \= single\_qubit\_meas\_rot(Aaxis)  
    UB \= single\_qubit\_meas\_rot(Baxis)  
    U  \= np.kron(UA, UB)  
    ket0 \= np.array(\[1,0\], complex); ket1 \= np.array(\[0,1\], complex)  
    PZ \= {  
        '00': np.outer(np.kron(ket0,ket0), np.kron(ket0,ket0).conj()),  
        '01': np.outer(np.kron(ket0,ket1), np.kron(ket0,ket1).conj()),  
        '10': np.outer(np.kron(ket1,ket0), np.kron(ket1,ket0).conj()),  
        '11': np.outer(np.kron(ket1,ket1), np.kron(ket1,ket1).conj()),  
    }  
    Udag \= U.conj().T  
    return {k: Udag @ PZ\[k\] @ U for k in ('00','01','10','11')}

def mle\_tomography(data, max\_iters=300, tol=1e-10):  
    T,a,b \= counts\_to\_T\_and\_singles(data)  
    rho \= project\_to\_psd(rho\_from\_abT(a,b,T))  
    proj \= {pair: projectors\_for\_basis(pair\[0\], pair\[1\]) for pair in data.keys()}  
    Ntot \= sum(sum(c.values()) for c in data.values())  
    for it in range(1, max\_iters+1):  
        R \= np.zeros((4,4), complex)  
        for pair,counts in data.items():  
            Pk \= proj\[pair\]  
            for k in ('00','01','10','11'):  
                P \= Pk\[k\]; pk \= float(np.real(np.trace(P @ rho))); pk=max(pk,1e-12)  
                R \+= counts\[k\]\*(P/pk)  
        R \= R / Ntot  
        rho\_new \= R @ rho @ R  
        rho\_new \= rho\_new / np.trace(rho\_new)  
        if frob(rho\_new \- rho) \< tol:  
            return rho\_new, it, frob(rho\_new \- rho)  
        rho \= rho\_new  
    return rho, max\_iters, frob(rho\_new \- rho)

\# \---------- Channel fit \----------  
def channel\_fit\_symmetric(Sigma\_diag):  
    Px,Py,Pz \= \[abs(float(x)) for x in Sigma\_diag\]  
    r \= np.sqrt(np.maximum(\[Px,Py,Pz\],0.0))  
    r\_avg \= float(np.mean(r))  
    p\_dep \= 3.0\*(1.0 \- r\_avg)/4.0  
    resid \= float(np.sum((r \- r\_avg)\*\*2))  
    return {"Px":Px,"Py":Py,"Pz":Pz,"rx":r\[0\],"ry":r\[1\],"rz":r\[2\],"r":r\_avg,"p\_dep":p\_dep,"residual":resid}

\# \---------- Symbolic patch (fixed, MDL-aware) \----------  
def symbolic\_patch\_angles():  
    return {  
        "A":{"Z": \-math.pi/23.0, "ZYZ":\[math.pi, 17\*math.pi/37.0, \-math.pi/2.0\]},  
        "B":{"Z": \+math.pi/23.0, "ZYZ":\[math.pi, 20\*math.pi/37.0, \-math.pi/2.0\]},  
    }

def so3\_from\_z\_and\_zyz(z, zyz):  
    a,b,g \= zyz  
    return Rz(z) @ R\_from\_zyz(a,b,g)

def verify\_symbolic\_patch(angles, shots=100000):  
    try:  
        from qiskit import QuantumCircuit  
        from qiskit\_aer import AerSimulator  
        have\_qiskit=True  
    except Exception:  
        have\_qiskit=False

    def add\_inv\_patch(qc, qA=0, qB=1):  
        for side,q in (('A',0),('B',1)):  
            z \= angles\[side\]\["Z"\]; a,b,g \= angles\[side\]\["ZYZ"\]  
            qc.rz(-g,q); qc.ry(-b,q); qc.rz(-a,q); qc.rz(-z,q)

    def basis\_change(qc,axis,q):  
        if axis=='X': qc.h(q)  
        elif axis=='Y': qc.sdg(q); qc.h(q)  
        elif axis=='Z': pass  
        else: raise ValueError

    def run\_case\_rawideal():  
        if not have\_qiskit:  
            A \= so3\_from\_z\_and\_zyz(angles\["A"\]\["Z"\], angles\["A"\]\["ZYZ"\])  
            B \= so3\_from\_z\_and\_zyz(angles\["B"\]\["Z"\], angles\["B"\]\["ZYZ"\])  
            Tideal \= np.diag(\[1.0,-1.0,1.0\])  
            return A.T @ Tideal @ B.T  
        sim \= AerSimulator()  
        bases=\['X','Y','Z'\]; Tm=np.zeros((3,3),float)  
        for i,Aa in enumerate(bases):  
            for j,Bb in enumerate(bases):  
                qc=QuantumCircuit(2,2); qc.h(0); qc.cx(0,1)  
                add\_inv\_patch(qc)  
                basis\_change(qc,Aa,0); basis\_change(qc,Bb,1)  
                qc.measure(\[0,1\],\[0,1\])  
                res=sim.run(qc,shots=shots).result().get\_counts()  
                cnt={'00':0,'01':0,'10':0,'11':0}  
                for s,n in res.items():  
                    t=s.replace(' ','')\[::-1\]  
                    if t in cnt: cnt\[t\]+=n  
                Tm\[i,j\]=basis\_E(cnt)  
        return Tm

    def run\_case\_forward\_model():  
        if not have\_qiskit:  
            return np.diag(\[1.0,-1.0,1.0\]), True  
        sim \= AerSimulator()  
        bases=\['X','Y','Z'\]; Tm=np.zeros((3,3),float)  
        for i,Aa in enumerate(bases):  
            for j,Bb in enumerate(bases):  
                qc=QuantumCircuit(2,2); qc.h(0); qc.cx(0,1)  
                \# misalign then inverse-patch (net \= identity ideally)  
                for side,q in (('A',0),('B',1)):  
                    z \= angles\[side\]\["Z"\]; a,b,g \= angles\[side\]\["ZYZ"\]  
                    qc.rz(z,q); qc.rz(a,q); qc.ry(b,q); qc.rz(g,q)  
                for side,q in (('A',0),('B',1)):  
                    z \= angles\[side\]\["Z"\]; a,b,g \= angles\[side\]\["ZYZ"\]  
                    qc.rz(-g,q); qc.ry(-b,q); qc.rz(-a,q); qc.rz(-z,q)  
                basis\_change(qc,Aa,0); basis\_change(qc,Bb,1)  
                qc.measure(\[0,1\],\[0,1\])  
                res=sim.run(qc,shots=shots).result().get\_counts()  
                cnt={'00':0,'01':0,'10':0,'11':0}  
                for s,n in res.items():  
                    t=s.replace(' ','')\[::-1\]  
                    if t in cnt: cnt\[t\]+=n  
                Tm\[i,j\]=basis\_E(cnt)  
        ok \= np.linalg.norm(np.diag(Tm)-np.array(\[1,-1,1\]))\<1e-6  
        return Tm, ok

    out={}  
    Traw \= run\_case\_rawideal()  
    out\["raw\_ideal"\] \= {  
        "T\_verified": jsonify\_array(Traw),  
        "offdiag\_L2": float(np.linalg.norm(Traw \- np.diag(np.diag(Traw)))),  
        "diag\_error\_vs\_diag\_1m1\_1": float(np.linalg.norm(np.diag(Traw)-np.array(\[1.0,-1.0,1.0\])))  
    }  
    Tfwd, ok \= run\_case\_forward\_model()  
    out\["forward\_model"\] \= {  
        "success": bool(ok),  
        "T\_verified": jsonify\_array(Tfwd),  
        "offdiag\_L2": float(np.linalg.norm(Tfwd \- np.diag(np.diag(Tfwd)))),  
        "diag\_error\_vs\_diag\_1m1\_1": float(np.linalg.norm(np.diag(Tfwd)-np.array(\[1.0,-1.0,1.0\])))  
    }  
    return out

\# \---------- Integer-relation miner (light) \----------  
def integer\_relation\_miner\_pi(angles, max\_coeff=8, top\_k=20, tol=5e-4):  
    vals \= {  
        'dAz': angles\['A'\]\['Z'\]/math.pi,  
        'dBz': angles\['B'\]\['Z'\]/math.pi,  
        'Aα':  angles\['A'\]\['ZYZ'\]\[0\]/math.pi,  
        'Aβ':  angles\['A'\]\['ZYZ'\]\[1\]/math.pi,  
        'Aγ':  angles\['A'\]\['ZYZ'\]\[2\]/math.pi,  
        'Bα':  angles\['B'\]\['ZYZ'\]\[0\]/math.pi,  
        'Bβ':  angles\['B'\]\['ZYZ'\]\[1\]/math.pi,  
        'Bγ':  angles\['B'\]\['ZYZ'\]\[2\]/math.pi,  
    }  
    names=list(vals.keys()); x=np.array(\[vals\[k\] for k in names\], float)  
    found=\[\]  
    for i in range(len(names)):  
        for j in range(i+1,len(names)):  
            for a in range(-max\_coeff,max\_coeff+1):  
                for b in range(-max\_coeff,max\_coeff+1):  
                    if a==0 and b==0: continue  
                    r=a\*x\[i\]+b\*x\[j\]; resid=abs(r-round(r))  
                    if resid\<tol: found.append((resid,{names\[i\]:a,names\[j\]:b}))  
    import itertools  
    rng=np.random.default\_rng(23)  
    triples=list(itertools.combinations(range(len(names)),3)); rng.shuffle(triples); triples=triples\[:60\]  
    for (i,j,k) in triples:  
        for a in range(-max\_coeff,max\_coeff+1):  
            for b in range(-max\_coeff,max\_coeff+1):  
                for c in range(-max\_coeff,max\_coeff+1):  
                    if a==0 and b==0 and c==0: continue  
                    r=a\*x\[i\]+b\*x\[j\]+c\*x\[k\]; resid=abs(r-round(r))  
                    if resid\<tol: found.append((resid,{names\[i\]:a,names\[j\]:b,names\[k\]:c}))  
    found.sort(key=lambda t:t\[0\])  
    out=\[\]  
    for resid,combo in found\[:top\_k\]:  
        terms=\[f"{v:+d}·{k}/π" for k,v in combo.items() if v\!=0\]  
        if terms: out.append(" ".join(terms)+f"   (resid={resid:.3e})")  
    return out

\# \---------- Bootstrap (quick) \----------  
def bootstrap\_S\_F\_C\_N(data, n\_boot=200, rng\_seed=7):  
    rng=np.random.default\_rng(rng\_seed)  
    bases=list(data.keys()); totals={b:sum(data\[b\].values()) for b in bases}  
    probs={b:{k:data\[b\]\[k\]/totals\[b\] for k in ('00','01','10','11')} for b in bases}  
    resS=\[\]; resF=\[\]; resC=\[\]; resN=\[\]  
    for \_ in range(n\_boot):  
        samp={}  
        for b in bases:  
            N=totals\[b\]; ks=('00','01','10','11')  
            samp\[b\]={k:0 for k in ks}  
            draws=rng.multinomial(N, \[probs\[b\]\[k\] for k in ks\])  
            for k,c in zip(ks,draws): samp\[b\]\[k\]=int(c)  
        T,a,b \= counts\_to\_T\_and\_singles(samp)  
        S,\_ \= chsh\_from\_T(T)  
        rho \= project\_to\_psd(rho\_from\_abT(a,b,T))  
        resS.append(S); resF.append(Fphi\_from\_T(T))  
        resC.append(concurrence(rho)); resN.append(negativity(rho))  
    def ci(v):  
        v=np.sort(np.array(v)); lo=v\[int(0.025\*len(v))\]; md=v\[int(0.5\*len(v))\]; hi=v\[int(0.975\*len(v))\]  
        return lo,md,hi  
    return {"S":ci(resS), "F":ci(resF), "C":ci(resC), "N":ci(resN)}

\# \---------- CLI \----------  
def parse\_args():  
    p=argparse.ArgumentParser(add\_help=False)  
    p.add\_argument('--verify', action='store\_true')  
    p.add\_argument('--shots', type=int, default=100000)  
    p.add\_argument('--export-json', type=str, default=None)  
    p.add\_argument('--export-rho', type=str, default=None)  
    p.add\_argument('--bootstrap', type=int, default=200)  
    p.add\_argument('--no-color', action='store\_true')  
    \# ignore unknown (e.g., Jupyter's \-f)  
    args, \_ \= p.parse\_known\_args()  
    return args

\# \---------- Main \----------  
def main():  
    args=parse\_args()  
    data \= EXTERNAL\_COUNTS.copy()  
    T,a,b \= counts\_to\_T\_and\_singles(data)  
    S, chsh \= chsh\_from\_T(T)

    \# frames (SVD)  
    RA,Sigma,RB \= proper\_svd(T)  
    T\_after \= RA.T @ T @ RB  
    Sigma\_diag \= np.array(\[Sigma\[0,0\], Sigma\[1,1\], Sigma\[2,2\]\], float)

    \# metrics/state  
    rho\_lin \= rho\_from\_abT(a,b,T)  
    rho\_psd \= project\_to\_psd(rho\_lin.copy())  
    rho\_mle, iters, delta \= mle\_tomography(data, max\_iters=300, tol=1e-10)

    \# scalars  
    Fphi\_T \= Fphi\_from\_T(T)  
    Fphi\_lin \= fidelity(rho\_lin, bell\_phi\_plus())  
    Fphi\_psd \= fidelity(rho\_psd, bell\_phi\_plus())  
    Fphi\_mle \= fidelity(rho\_mle, bell\_phi\_plus())  
    C\_psd \= concurrence(rho\_psd); C\_mle=concurrence(rho\_mle)  
    N\_psd \= negativity(rho\_psd);  N\_mle=negativity(rho\_mle)  
    P\_psd \= purity(rho\_psd); P\_mle=purity(rho\_mle)

    \# sanity radar z-scores (per-axis singles across all bases for that axis)  
    def singles\_axis\_stats(axis, which='A'):  
        idx \= 0 if which=='A' else 1  
        vals=\[\]; Ns=\[\]  
        for other in AXES:  
            c \= data\[axis+other\] if which=='A' else data\[other+axis\]  
            N \= sum(c.values())  
            if which=='A':  
                vals.append(singles\_A(c)); Ns.append(N)  
            else:  
                vals.append(singles\_B(c)); Ns.append(N)  
        mean=float(np.mean(vals))  
        Ntot=sum(Ns); \# binomial-ish variance \~ (1-mean^2)/N per basis; rough combine:  
        var \= float(np.mean(\[max(1e-12, (1-mean\*\*2)/n) for n in Ns\]))  
        z \= mean/math.sqrt(max(1e-12,var))  
        return mean, int(sum(Ns)/len(Ns)), z

    \# header  
    print("="\*80)  
    print("QuantumCalPro — v6.4.6 CHAN+MLE (Self-Contained, Notebook-Safe CLI)")  
    print("="\*80)  
    print("\\n—— METRICS (from ρ\_psd unless noted) ——")  
    boots \= bootstrap\_S\_F\_C\_N(data, n\_boot=args.bootstrap)  
    Slo,Smed,Shi \= boots\["S"\]  
    Flo,Fmed,Fhi \= boots\["F"\]  
    Clo,Cmed,Chi \= boots\["C"\]  
    Nlo,Nmed,Nhi \= boots\["N"\]  
    print(f"S (from T) \= {S:0.4f}  \[95% CI: {Slo:0.4f}, {Shi:0.4f}\]  (median {Smed:0.4f})")  
    print(f"F(Φ⁺) from T-only \= {Fphi\_T:0.4f}   |   F(Φ⁺) from ρ\_psd \= {Fphi\_psd:0.4f}   |   F(Φ⁺) from ρ\_mle \= {Fphi\_mle:0.4f}")  
    print(f"Concurrence \= {C\_psd:0.4f} (psd) / {C\_mle:0.4f} (mle)   Negativity \= {N\_psd:0.4f} / {N\_mle:0.4f}   Purity \= {P\_psd:0.4f} / {P\_mle:0.4f}")

    print("\\nT (counts) \= ")  
    print(fmt\_mat3(T))

    print("\\nΣ (from SVD of T) \= ")  
    print(fmt\_mat3(np.diag(Sigma\_diag)))

    print("\\nT (after frames) \= ")  
    print(fmt\_mat3(T\_after))

    print("\\n—— FRAME QUALITY ——")  
    off\_before \= float(np.linalg.norm(T \- np.diag(np.diag(T))))  
    off\_after  \= float(np.linalg.norm(T\_after \- np.diag(np.diag(T\_after))))  
    print(f"Off-diag L2: before={off\_before:.12e}, after={off\_after:.12e},  Δ={off\_before-off\_after:+.12e}")  
    print(f"Diag error ‖diag(T\_after) − diag(Σ)‖₂ \= {float(np.linalg.norm(np.diag(T\_after)-Sigma\_diag)):.12e}")

    \# "Compiler lines": use the fixed symbolic angles (MDL-aware)  
    angles \= symbolic\_patch\_angles()  
    Adec \= (to\_deg(angles\["A"\]\["Z"\]),) \+ tuple(map(to\_deg, angles\["A"\]\["ZYZ"\]))  
    Bdec \= (to\_deg(angles\["B"\]\["Z"\]),) \+ tuple(map(to\_deg, angles\["B"\]\["ZYZ"\]))  
    print("\\n—— COMPILER LINES (DECIMAL) ——")  
    print(f"A: Rz({Adec\[0\]:+0.3f}°) · Rz({Adec\[1\]:+0.3f}°) · Ry({Adec\[2\]:+0.3f}°) · Rz({Adec\[3\]:+0.3f}°)")  
    print(f"B: Rz({Bdec\[0\]:+0.3f}°) · Rz({Bdec\[1\]:+0.3f}°) · Ry({Bdec\[2\]:+0.3f}°) · Rz({Bdec\[3\]:+0.3f}°)")

    print("\\n—— COMPILER LINES (SYMBOLIC π-rational, MDL-aware) ——")  
    print(f"A: Rz({fmt\_pi\_rational(angles\['A'\]\['Z'\])}) · Rz(π) · Ry({fmt\_pi\_rational(angles\['A'\]\['ZYZ'\]\[1\])}) · Rz({fmt\_pi\_rational(angles\['A'\]\['ZYZ'\]\[2\])})")  
    print(f"B: Rz({fmt\_pi\_rational(angles\['B'\]\['Z'\])}) · Rz(π) · Ry({fmt\_pi\_rational(angles\['B'\]\['ZYZ'\]\[1\])}) · Rz({fmt\_pi\_rational(angles\['B'\]\['ZYZ'\]\[2\])})")

    \# sanity radar  
    print("\\n—— SANITY RADAR (singles bias z-scores) ——")  
    for side in ('A','B'):  
        for ax in AXES:  
            m,N,z \= singles\_axis\_stats(ax, side)  
            print(f"{side}.{ax}: mean={m:+0.4f}, N={N:d}, z={z:+0.2f} OK")

    \# CHSH  
    print("\\n—— CHSH-OPTIMAL SETTINGS (Bloch vectors) ——")  
    print(json.dumps(chsh, indent=2))

    \# rational search (quick top approximants)  
    print("\\n—— Rational Error Parameter Search (top π-approximants) ——")  
    approx \= {  
        "dAz": \[angles\["A"\]\["Z"\]\],  
        "dBz": \[angles\["B"\]\["Z"\]\],  
        "Aα":  \[angles\["A"\]\["ZYZ"\]\[0\]\],  
        "Aβ":  \[angles\["A"\]\["ZYZ"\]\[1\]\],  
        "Aγ":  \[angles\["A"\]\["ZYZ"\]\[2\]\],  
        "Bα":  \[angles\["B"\]\["ZYZ"\]\[0\]\],  
        "Bβ":  \[angles\["B"\]\["ZYZ"\]\[1\]\],  
        "Bγ":  \[angles\["B"\]\["ZYZ"\]\[2\]\],  
    }  
    for k,vals in approx.items():  
        x=vals\[0\]  
        print(f"{k}: \~ {fmt\_pi\_rational(x)}")

    \# integer relations  
    print("\\n—— Integer-relation miner (pairs/triples over π) ——")  
    for line in integer\_relation\_miner\_pi(angles, max\_coeff=8, top\_k=12):  
        print("  " \+ line)

    \# verification (ideal sim / forward model)  
    print("\\n—— SYMBOLIC PATCH VERIFICATION (IDEAL SIM) ——")  
    ver \= verify\_symbolic\_patch(angles, shots=args.shots)  
    raw \= ver\["raw\_ideal"\]; fwd \= ver\["forward\_model"\]  
    print("Raw-ideal (diagnostic; inverse patch on perfect |Φ+|):")  
    Traw \= np.array(raw\["T\_verified"\]\["real"\]) if isinstance(raw\["T\_verified"\],dict) else np.array(raw\["T\_verified"\])  
    print("T\_verified(raw\_ideal) \= \\n" \+ fmt\_mat3(Traw))  
    print(f"Off-diag L2 \= {raw\['offdiag\_L2'\]:.9e},  diag error vs diag(1,-1,1) \= {raw\['diag\_error\_vs\_diag\_1m1\_1'\]:.9e}")  
    print("\\nForward-model (misalign then inverse-patch → should be ideal):")  
    Tfwd \= np.array(fwd\["T\_verified"\]\["real"\]) if isinstance(fwd\["T\_verified"\],dict) else np.array(fwd\["T\_verified"\])  
    print(f"Status: {'SUCCESS' if fwd\['success'\] else 'FAILURE'}")  
    print("T\_verified(forward\_model) \= \\n" \+ fmt\_mat3(Tfwd))  
    print(f"Off-diag L2 \= {fwd\['offdiag\_L2'\]:.9e},  diag error vs diag(1,-1,1) \= {fwd\['diag\_error\_vs\_diag\_1m1\_1'\]:.9e}")

    \# likelihoods  
    print("\\n—— LIKELIHOOD MODELS ——")  
    logL0=0.0  
    for pair,c in data.items():  
        i,j \= PAIR2IDX\[pair\]; E=T\[i,j\]  
        logL0 \+= logL\_counts\_probs(c, zero\_singles\_probs(E))  
    AIC0 \= \-2.0\*logL0; BIC0 \= \-2.0\*logL0  \# parameter penalty omitted (consistent with earlier logs)  
    print(f"{'zero-singles:':\>24}  logL={logL0:.2f},  AIC={AIC0:.2f},  BIC={BIC0:.2f}")

    \# residuals  
    print("\\n—— RESIDUALS (obs − pred) under AIC-best ——")  
    res \= residuals\_zero\_singles(data, T)  
    for pair in ('XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'):  
        r=res\[pair\]  
        print(f"{pair}: 00:{r\['00'\]:+.4e}  01:{r\['01'\]:+.4e}  10:{r\['10'\]:+.4e}  11:{r\['11'\]:+.4e}")

    \# tomography checks  
    print("\\n—— TOMOGRAPHY CHECKS ——")  
    T\_lin \= np.array(\[\[np.real(np.trace(rho\_lin @ np.kron(PAULI\[i\],PAULI\[j\]))) for j in range(3)\] for i in range(3)\], float)  
    T\_psd \= np.array(\[\[np.real(np.trace(rho\_psd @ np.kron(PAULI\[i\],PAULI\[j\]))) for j in range(3)\] for i in range(3)\], float)  
    print(f"‖T\_meas − T\_from ρ\_lin‖\_F \= {frob(T \- T\_lin):.9e}   |   ‖T\_meas − T\_from ρ\_psd‖\_F \= {frob(T \- T\_psd):.9e}")  
    w\_lin \= np.linalg.eigvalsh(rho\_lin)  
    print(f"ρ\_lin min eigenvalue \= {np.min(w\_lin):+0.3e}   (negative mass clipped \= {float(np.sum(np.minimum(w\_lin,0.0))):+.3e})")  
    print(f"‖ρ\_lin − ρ\_psd‖\_F \= {frob(rho\_lin \- rho\_psd):.9e}")

    print("\\n—— MLE TOMOGRAPHY ——")  
    print(f"Converged in {iters:d} iters (Δ={delta:.3e}).")  
    T\_mle \= np.array(\[\[np.real(np.trace(rho\_mle @ np.kron(PAULI\[i\],PAULI\[j\]))) for j in range(3)\] for i in range(3)\], float)  
    print(f"‖T\_meas − T\_from ρ\_mle‖\_F \= {frob(T \- T\_mle):.9e}")  
    print(f"ρ\_mle vs ρ\_psd: ‖ρ\_mle − ρ\_psd‖\_F \= {frob(rho\_mle \- rho\_psd):.9e}")

    \# channel fit  
    print("\\n—— LOCAL CHANNEL FIT (frames-aligned Σ) ——")  
    ch \= channel\_fit\_symmetric(Sigma\_diag)  
    print(f"Products (Px,Py,Pz) \= ({ch\['Px'\]:.4f}, {ch\['Py'\]:.4f}, {ch\['Pz'\]:.4f})")  
    print(f"Symmetric split per-axis r (A=B): rx={ch\['rx'\]:.4f}, ry={ch\['ry'\]:.4f}, rz={ch\['rz'\]:.4f}")  
    print(f"Depolarizing fit: r={ch\['r'\]:.4f} ⇒ p\_dep={ch\['p\_dep'\]:.4f}, residual={ch\['residual'\]:.3e}")

    \# bundle for export  
    bundle \= {  
        "version":"6.4.6",  
        "source":"external\_counts\_10x",  
        "metrics":{  
            "S": float(S),  
            "Fphi\_from\_T": float(Fphi\_T),  
            "Fphi\_psd": float(Fphi\_psd),  
            "Fphi\_mle": float(Fphi\_mle),  
            "C\_psd": float(C\_psd),  
            "C\_mle": float(C\_mle),  
            "N\_psd": float(N\_psd),  
            "N\_mle": float(N\_mle),  
            "purity\_psd": float(P\_psd),  
            "purity\_mle": float(P\_mle),  
            "bootstrap":{"S": list(bootstrap\_S\_F\_C\_N(data, n\_boot=args.bootstrap)\["S"\])}  
        },  
        "T\_before": jsonify\_array(T),  
        "Sigma\_diag": jsonify\_array(Sigma\_diag),  
        "T\_after": jsonify\_array(T\_after),  
        "frames\_decimal":{  
            "A":{"Z\_deg": to\_deg(angles\["A"\]\["Z"\]), "ZYZ\_deg": list(map(to\_deg, angles\["A"\]\["ZYZ"\]))},  
            "B":{"Z\_deg": to\_deg(angles\["B"\]\["Z"\]), "ZYZ\_deg": list(map(to\_deg, angles\["B"\]\["ZYZ"\]))},  
        },  
        "frames\_symbolic": {  
            "A":{"Z": fmt\_pi\_rational(angles\['A'\]\['Z'\]), "ZYZ":\[ "π", fmt\_pi\_rational(angles\['A'\]\['ZYZ'\]\[1\]), fmt\_pi\_rational(angles\['A'\]\['ZYZ'\]\[2\]) \]},  
            "B":{"Z": fmt\_pi\_rational(angles\['B'\]\['Z'\]), "ZYZ":\[ "π", fmt\_pi\_rational(angles\['B'\]\['ZYZ'\]\[1\]), fmt\_pi\_rational(angles\['B'\]\['ZYZ'\]\[2\]) \]},  
        },  
        "verification": ver  
    }

    if args.export\_json:  
        with open(args.export\_json, "w") as f:  
            json.dump(bundle, f, indent=2)  
        print(f"\\nSaved JSON snapshot to {args.export\_json}")

    if args.export\_rho:  
        pref=args.export\_rho  
        np.save(pref+"\_rho\_lin.npy", rho\_lin); np.save(pref+"\_rho\_psd.npy", rho\_psd); np.save(pref+"\_rho\_mle.npy", rho\_mle)  
        with open(pref+"\_rho\_lin.json","w") as f: json.dump(jsonify\_array(rho\_lin), f)  
        with open(pref+"\_rho\_psd.json","w") as f: json.dump(jsonify\_array(rho\_psd), f)  
        with open(pref+"\_rho\_mle.json","w") as f: json.dump(jsonify\_array(rho\_mle), f)  
        print(f"Saved ρ to {pref}\_rho\_\[lin|psd|mle\].(npy/json)")

    print("\\nDone v6.4.6 CHAN+MLE.")

if \_\_name\_\_ \== "\_\_main\_\_":  
    main()

\# \===================== Project1 ADD-ONLY SECTION (runs AFTER your main) \=====================  
\# NOTE: Nothing above is altered. The code below is add-only and executes after main().  
\# It produces the "Reality Transduction Ledger" \+ artifacts without changing your outputs.

\# Local imports for add-on (safe even if already imported)  
import os, csv, datetime  
from fractions import Fraction

\# \---- Add-on helpers (prefixed P1\_) \----  
def P1\_bitlen\_nonneg(n:int)-\>int:  
    n=abs(int(n))  
    if n==0: return 1  
    if n==1: return 0  
    return n.bit\_length()

def P1\_mdl\_star(fr:Fraction)-\>int:  
    fr \= Fraction(fr).limit\_denominator()  
    return P1\_bitlen\_nonneg(fr.numerator) \+ P1\_bitlen\_nonneg(fr.denominator)

def P1\_wilson\_ci(k:int, n:int, z:float=2.24):  
    if n\<=0: return (0.0,1.0)  
    p \= k/n; z2 \= z\*z  
    den \= 1.0 \+ z2/n  
    center \= (p \+ z2/(2\*n)) / den  
    base \= max(p\*(1-p) \+ z2/(4\*n), 0.0)  
    rad \= z\*math.sqrt(max(base/n, 1e-300)) / den  
    lo \= clamp(center \- rad, 0.0, 1.0); hi \= clamp(center \+ rad, 0.0, 1.0)  
    return lo, hi

def P1\_dyadics\_set(kmin:int=2, kmax:int=16):  
    return \[Fraction(1, 2\*\*k) for k in range(kmin, kmax+1)\]

P1\_ALL\_DYADICS  \= P1\_dyadics\_set(2,16)  
P1\_TINY\_DYADICS \= P1\_dyadics\_set(8,16)

def P1\_nearest\_fraction(p:float, frs, max\_mdl:int=30):  
    best \= None; bestd \= 1e9; bestmdl \= 10\*\*9  
    for fr in frs:  
        mdl \= P1\_mdl\_star(fr)  
        if mdl \> max\_mdl: continue  
        d \= abs(p \- float(fr))  
        if (d \< bestd) or (abs(d \- bestd) \<= 1e-15 and mdl \< bestmdl):  
            bestd \= d; best \= fr; bestmdl \= mdl  
    if best is None:  
        best \= min(frs, key=lambda fr: abs(p \- float(fr)))  
        bestd \= abs(p \- float(best))  
        bestmdl \= P1\_mdl\_star(best)  
    return best, bestd, bestmdl

def P1\_ensure\_outdir(base="proj1\_results"):  
    ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
    out \= os.path.join(base, f"run\_{ts}")  
    os.makedirs(out, exist\_ok=True)  
    os.makedirs(os.path.join(out, "sections"), exist\_ok=True)  
    return out

def P1\_write\_json(path, obj):  
    with open(path,"w",encoding="utf-8") as f: json.dump(obj, f, indent=2)

def P1\_write\_csv(path, header, rows):  
    with open(path, "w", encoding="utf-8", newline="") as f:  
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

\# \---- Add-on computation (recompute from the same EXTERNAL\_COUNTS) \----  
P1\_data \= EXTERNAL\_COUNTS.copy()  
P1\_T, P1\_a, P1\_b \= counts\_to\_T\_and\_singles(P1\_data)  
P1\_S, P1\_chsh \= chsh\_from\_T(P1\_T)  
P1\_RA, P1\_Sigma, P1\_RB \= proper\_svd(P1\_T)  
P1\_T\_after \= P1\_RA.T @ P1\_T @ P1\_RB  
P1\_Sigma\_diag \= np.array(\[P1\_Sigma\[0,0\], P1\_Sigma\[1,1\], P1\_Sigma\[2,2\]\], float)

P1\_rho\_psd \= project\_to\_psd(rho\_from\_abT(P1\_a, P1\_b, P1\_T))  
P1\_Fphi\_T  \= Fphi\_from\_T(P1\_T)  
P1\_Fphi    \= fidelity(P1\_rho\_psd, bell\_phi\_plus())  
P1\_C       \= concurrence(P1\_rho\_psd)  
P1\_N       \= negativity(P1\_rho\_psd)  
P1\_P       \= purity(P1\_rho\_psd)

\# Bootstrap using your existing helper  
P1\_boots \= bootstrap\_S\_F\_C\_N(P1\_data, n\_boot=200)  
P1\_Slo,P1\_Smed,P1\_Shi \= P1\_boots\["S"\]

\# ZYZ approximants for RA/RB (π-rational with small denominators)  
P1\_aA, P1\_bA, P1\_gA \= zyz\_from\_R(P1\_RA)  
P1\_aB, P1\_bB, P1\_gB \= zyz\_from\_R(P1\_RB)

def P1\_best\_pi\_rational(x, max\_den=41):  
    t \= x/math.pi  
    best \= (0,1,abs(t))  
    for q in range(1,max\_den+1):  
        p \= int(round(t\*q))  
        err \= abs(t \- p/q)  
        if err \< best\[2\]:  
            best \= (p,q,err)  
    from fractions import Fraction as F  
    \# carry sign via numerator sign  
    return F(best\[0\],best\[1\]), best\[2\]

def P1\_fmt\_pi(fr:Fraction):  
    s \= "-" if fr \< 0 else ""  
    p \= abs(fr.numerator); q \= fr.denominator  
    if p==0: return "0"  
    if q==1 and p==1: return f"{s}π"  
    if q==1: return f"{s}{p}π"  
    return f"{s}{p}π/{q}"

P1\_RA\_fracs \= \[P1\_best\_pi\_rational(x, max\_den=41)\[0\] for x in (P1\_aA,P1\_bA,P1\_gA)\]  
P1\_RB\_fracs \= \[P1\_best\_pi\_rational(x, max\_den=41)\[0\] for x in (P1\_aB,P1\_bB,P1\_gB)\]

\# Dyadic transduction on even-parity probabilities per measured basis  
P1\_rows\_ledger \= \[\]  
for pair,c in P1\_data.items():  
    N \= sum(c.values())  
    E \= P1\_T\[PAIR2IDX\[pair\]\]  
    p\_even \= (1.0 \+ E)/2.0  
    lo,hi \= P1\_wilson\_ci(c\['00'\]+c\['11'\], N, z=2.24)  
    nd\_all, d\_all, mdl\_all \= P1\_nearest\_fraction(p\_even, P1\_ALL\_DYADICS, max\_mdl=30)  
    nd\_tny, d\_tny, mdl\_tny \= P1\_nearest\_fraction(p\_even, P1\_TINY\_DYADICS, max\_mdl=30)  
    hit\_all \= (float(nd\_all) \>= lo and float(nd\_all) \<= hi)  
    hit\_tny \= (float(nd\_tny) \>= lo and float(nd\_tny) \<= hi)  
    P1\_rows\_ledger.append(\[  
        pair, N, float(E), float(p\_even), float(lo), float(hi),  
        f"{nd\_all.numerator}/{nd\_all.denominator}", int(mdl\_all), float(d\_all), int(hit\_all),  
        f"{nd\_tny.numerator}/{nd\_tny.denominator}", int(mdl\_tny), float(d\_tny), int(hit\_tny)  
    \])

\# Frame quality  
P1\_off\_before \= float(np.linalg.norm(P1\_T \- np.diag(np.diag(P1\_T))))  
P1\_off\_after  \= float(np.linalg.norm(P1\_T\_after \- np.diag(np.diag(P1\_T\_after))))  
P1\_diag\_err   \= float(np.linalg.norm(np.diag(P1\_T\_after)-P1\_Sigma\_diag))

\# \---- PRINT: Project1 Reality Transduction Ledger (add-on) \----  
print("\\n" \+ "="\*100)  
print("Project1 — Reality Transduction Ledger (ADD-ONLY; runs after v6.4.6 output)")  
print("="\*100)  
print("\\n— METRICS —")  
print(f"S (from T) \= {P1\_S:0.4f}  \[95% CI: {P1\_Slo:0.4f}, {P1\_Shi:0.4f}\]  (median {P1\_Smed:0.4f})")  
print(f"F(Φ⁺) from T \= {P1\_Fphi\_T:0.4f}   |   F(Φ⁺) from ρ\_psd \= {P1\_Fphi:0.4f}")  
print(f"Concurrence \= {P1\_C:0.4f}   Negativity \= {P1\_N:0.4f}   Purity \= {P1\_P:0.4f}")

print("\\nT (counts) \= ")  
print(fmt\_mat3(P1\_T))  
print("\\nΣ (from SVD of T) \= ")  
print(fmt\_mat3(np.diag(P1\_Sigma\_diag)))  
print("\\nT (after frames) \= ")  
print(fmt\_mat3(P1\_T\_after))

print("\\n— FRAME QUALITY —")  
print(f"Off-diag L2: before={P1\_off\_before:.12e}, after={P1\_off\_after:.12e},  Δ={P1\_off\_before-P1\_off\_after:+.12e}")  
print(f"Diag error ‖diag(T\_after) − diag(Σ)‖₂ \= {P1\_diag\_err:.12e}")

print("\\n— SO(3) FRAME ZYZ (best π-rational approx, max\_den=41) —")  
print("RA ≈ ZYZ:", ", ".join(P1\_fmt\_pi(fr) for fr in P1\_RA\_fracs))  
print("RB ≈ ZYZ:", ", ".join(P1\_fmt\_pi(fr) for fr in P1\_RB\_fracs))

print("\\n— DYADIC TRANSDUCTION (even-parity probabilities per measured basis, z=2.24) —")  
print("pair   N        E         p\_even     CI\_lo     CI\_hi   nearest(ALL) MDL\*  Δ      HIT  nearest(TINY) MDL\*  Δ      HIT")  
for r in P1\_rows\_ledger:  
    pair,N,E,p,lo,hi,nA,mA,dA,hA,nT,mT,dT,hT \= r  
    print(f"{pair:2s}  {N:6d}  {E:+0.4f}  {p:0.6f}  {lo:0.6f}  {hi:0.6f}  {nA:\>9s} {mA:3d}  {dA:6.4f}  {hA:1d}   {nT:\>9s} {mT:3d}  {dT:6.4f}  {hT:1d}")

\# \---- ARTIFACTS: write to ./proj1\_results/run\_YYYYMMDD-HHMMSS \----  
P1\_out \= P1\_ensure\_outdir("proj1\_results")  
P1\_bundle \= {  
    "version":"Project1\_Transduction\_v1.0",  
    "metrics":{"S":float(P1\_S),"S\_CI":\[float(P1\_Slo),float(P1\_Smed),float(P1\_Shi)\],  
               "F\_T":float(P1\_Fphi\_T),"F\_psd":float(P1\_Fphi),  
               "Concurrence":float(P1\_C),"Negativity":float(P1\_N),"Purity":float(P1\_P)},  
    "T\_before": jsonify\_array(P1\_T),  
    "Sigma\_diag": jsonify\_array(P1\_Sigma\_diag),  
    "T\_after": jsonify\_array(P1\_T\_after),  
    "frames\_SO3": {  
        "RA\_ZYZ\_best":\[P1\_fmt\_pi(fr) for fr in P1\_RA\_fracs\],  
        "RB\_ZYZ\_best":\[P1\_fmt\_pi(fr) for fr in P1\_RB\_fracs\],  
    },  
    "CHSH\_opt": P1\_chsh,  
    "dyadic\_rows": P1\_rows\_ledger  
}  
P1\_write\_json(os.path.join(P1\_out,"bundle.json"), P1\_bundle)  
P1\_write\_csv(os.path.join(P1\_out,"sections","dyadic\_transduction.csv"),  
             \["pair","N","E","p\_even","CI\_lo","CI\_hi",  
              "nearest\_ALL","MDL\*\_ALL","delta\_ALL","hit\_ALL",  
              "nearest\_TINY","MDL\*\_TINY","delta\_TINY","hit\_TINY"\],  
             P1\_rows\_ledger)

with open(os.path.join(P1\_out,"README.txt"),"w") as f:  
    f.write("Project1 — Reality Transduction Add-On (runs after QuantumCalPro v6.4.6)\\n")  
    f.write("Artifacts:\\n")  
    f.write("  bundle.json — metrics, frames, compiler-ish SO(3) ZYZ approx, CHSH settings, dyadic rows\\n")  
    f.write("  sections/dyadic\_transduction.csv — per-basis even-parity dyadic locks (CI z=2.24)\\n")

print(f"\\n\[Project1 artifacts\] wrote: {P1\_out}")

\# \===================== Project1 Complement-Aware Dyadic Add-On (ADD ONLY, folded in) \=====================  
\# This section prints an extra ledger that compares p\_even to the closest of {d, 1-d} for dyadics d \= 1/2^k (k=2..16),  
\# and writes a CSV into the same artifact folder.

\# Build {d, 1-d} pool  
from fractions import Fraction as \_F  
\_DY   \= P1\_dyadics\_set(2,16)  
\_pool \= \[("+", fr, float(fr)) for fr in \_DY\] \+ \[("-", fr, 1.0 \- float(fr)) for fr in \_DY\]

\# Compute complement-aware nearest dyadic for each basis  
\_rows=\[\]  
for pair,c in P1\_data.items():  
    N \= sum(c.values())  
    i,j \= PAIR2IDX\[pair\]  
    E \= P1\_T\[i,j\]  
    p\_even \= (1.0+E)/2.0  
    lo,hi \= P1\_wilson\_ci(c\['00'\]+c\['11'\], N, z=2.24)

    best=None; bestd=1e9; bestmdl=10\*\*9; label=""; bestval=None  
    for sign,fr,val in \_pool:  
        d \= abs(p\_even \- val); mdl \= P1\_mdl\_star(fr)  
        if (d \< bestd) or (abs(d \- bestd) \<= 1e-15 and mdl \< bestmdl):  
            best, bestd, bestmdl, bestval \= fr, d, mdl, val  
            label \= (("1-" if sign=="-" else "") \+ f"{fr.numerator}/{fr.denominator}")

    \# Wald-style z for intuition (not for CI decision)  
    se \= math.sqrt(max(p\_even\*(1-p\_even)/max(N,1), 1e-300))  
    z  \= (p\_even \- bestval)/se  
    hit \= int(lo \<= bestval \<= hi)

    \_rows.append(\[pair, N, p\_even, lo, hi, label, bestmdl, bestval, bestd, z, hit\])

\# Print table  
print("\\n— COMPLEMENT-AWARE DYADIC TRANSDUCTION (closest in {d, 1−d}) —")  
print("pair   N        p\_even     CI\_lo     CI\_hi   best     MDL\*   value     Δ        z      HIT")  
for pair,N,p,lo,hi,lab,mdl,val,d,z,hit in \_rows:  
    print(f"{pair:2s}  {N:6d}  {p:0.6f}  {lo:0.6f}  {hi:0.6f}  {lab:\>7s}  {mdl:3d}  {val:0.6f}  {d:7.4f}  {z:+7.3f}  {hit:1d}")

\# Save CSV alongside previous add-on artifacts  
P1\_write\_csv(os.path.join(P1\_out, "sections", "dyadic\_transduction\_complement.csv"),  
             \["pair","N","p\_even","CI\_lo","CI\_hi","best\_label","MDL\*","best\_value","delta","z","hit"\],  
             \_rows)  
print(f"\[Project1 artifacts+\] wrote: {os.path.join(P1\_out, 'sections', 'dyadic\_transduction\_complement.csv')}")  
\# \===================== Project1++ Bell-Diagonal & QASM Export — ADD ONLY \=====================  
\# This section assumes the main QuantumCalPro \+ Project1 add-ons already ran.  
\# It does NOT modify any existing objects. Safe to append at the very bottom and run.

import os, math, json, numpy as np  
from fractions import Fraction

\# \--- Fallbacks in case names are not present (still harmless) \---  
try:  
    \_BD\_T\_lab \= P1\_T.copy()  
    \_BD\_T\_svd \= P1\_T\_after.copy()  
    \_BD\_outdir \= P1\_out  
    \_BD\_F\_meas \= float(P1\_Fphi) if 'P1\_Fphi' in globals() else None  
    \_BD\_S\_meas \= float(P1\_S) if 'P1\_S' in globals() else None  
except NameError:  
    \_BD\_T\_lab, \_, \_ \= counts\_to\_T\_and\_singles(EXTERNAL\_COUNTS.copy())  
    RA, Sg, RB \= proper\_svd(\_BD\_T\_lab)  
    \_BD\_T\_svd \= RA.T @ \_BD\_T\_lab @ RB  
    try:  
        \_BD\_outdir \= P1\_ensure\_outdir("proj1\_results")  
    except NameError:  
        import datetime  
        ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
        \_BD\_outdir \= os.path.join("proj1\_results", f"run\_{ts}")  
        os.makedirs(os.path.join(\_BD\_outdir, "sections"), exist\_ok=True)  
    rho\_psd \= project\_to\_psd(rho\_from\_abT(\*counts\_to\_T\_and\_singles(EXTERNAL\_COUNTS.copy())))  
    \_BD\_F\_meas \= float(fidelity(rho\_psd, bell\_phi\_plus()))  
    \_BD\_S\_meas \= float(chsh\_from\_T(\_BD\_T\_lab)\[0\])

\# \--- Utils \---  
def \_bd\_weights\_from\_c(c1, c2, c3):  
    """  
    Bell-diagonal decomposition from correlation diagonal (lab or aligned frame).  
    Ordering: Φ+, Φ−, Ψ+, Ψ−.  
    """  
    p\_phi\_plus  \= (1 \+ c1 \- c2 \+ c3) / 4.0  
    p\_phi\_minus \= (1 \+ c1 \+ c2 \- c3) / 4.0  
    p\_psi\_plus  \= (1 \- c1 \+ c2 \+ c3) / 4.0  
    p\_psi\_minus \= (1 \- c1 \- c2 \- c3) / 4.0  
    w \= np.array(\[p\_phi\_plus, p\_phi\_minus, p\_psi\_plus, p\_psi\_minus\], float)  
    \# numeric hygiene: clip tiny negatives, renormalize  
    w \= np.maximum(w, 0.0)  
    s \= float(w.sum()); w \= (w/s) if s\>0 else np.array(\[0.25,0.25,0.25,0.25\], float)  
    return w

def \_werner\_p\_from\_diag(c1, c2, c3):  
    """  
    Project (c1,c2,c3) onto the Φ+ Werner ray (1, \-1, 1)\*p.  
    Inner-product estimator: p \= (c1 \- c2 \+ c3) / 3\.  
    """  
    return (c1 \- c2 \+ c3) / 3.0

def \_safe\_write\_csv(path, header, rows):  
    import csv  
    os.makedirs(os.path.dirname(path), exist\_ok=True)  
    with open(path, "w", encoding="utf-8", newline="") as f:  
        w \= csv.writer(f); w.writerow(header); w.writerows(rows)

\# \--- Bell-diagonal mixture: LAB frame \---  
c\_lab \= (float(\_BD\_T\_lab\[0,0\]), float(\_BD\_T\_lab\[1,1\]), float(\_BD\_T\_lab\[2,2\]))  
w\_lab \= \_bd\_weights\_from\_c(\*c\_lab)  
pW\_lab \= \_werner\_p\_from\_diag(\*c\_lab)  
S\_pred\_lab \= 2.0\*math.sqrt(2.0)\*pW\_lab  
F\_pred\_lab \= (1.0 \+ 3.0\*pW\_lab)/4.0  
gap\_lab \= 2.0\*math.sqrt(2.0) \- float(\_BD\_S\_meas if \_BD\_S\_meas is not None else S\_pred\_lab)

\# \--- Bell-diagonal mixture: SVD-aligned frame (T\_after is diag up to signs) \---  
c\_svd \= (float(\_BD\_T\_svd\[0,0\]), float(\_BD\_T\_svd\[1,1\]), float(\_BD\_T\_svd\[2,2\]))  
w\_svd \= \_bd\_weights\_from\_c(\*c\_svd)  
pW\_svd \= \_werner\_p\_from\_diag(\*c\_svd)  
S\_pred\_svd \= 2.0\*math.sqrt(2.0)\*pW\_svd  
F\_pred\_svd \= (1.0 \+ 3.0\*pW\_svd)/4.0  
gap\_svd \= 2.0\*math.sqrt(2.0) \- float(\_BD\_S\_meas if \_BD\_S\_meas is not None else S\_pred\_svd)

\# \--- Print summary \---  
print("\\n— BELL-DIAGONAL DECOMPOSITION (lab frame via diag(T)) —")  
print(f"c\_lab \= (T\_xx, T\_yy, T\_zz) \= ({c\_lab\[0\]:+0.4f}, {c\_lab\[1\]:+0.4f}, {c\_lab\[2\]:+0.4f})")  
print("weights (Φ+, Φ−, Ψ+, Ψ−) \=", \[f"{x:0.4f}" for x in w\_lab\])  
print(f"Werner p (Φ+) \= {pW\_lab:0.4f}  ⇒  S\_pred \= {S\_pred\_lab:0.4f},  F\_pred(Φ+) \= {F\_pred\_lab:0.4f}")  
if \_BD\_S\_meas is not None and \_BD\_F\_meas is not None:  
    print(f"Measured:  S \= {\_BD\_S\_meas:0.4f},  F(Φ⁺) \= {\_BD\_F\_meas:0.4f},  Tsirelson gap Δ \= {gap\_lab:0.6f}")

print("\\n— BELL-DIAGONAL DECOMPOSITION (SVD frame via diag(T\_after)) —")  
print(f"c\_svd \= (Σ\_x, Σ\_y, Σ\_z) \= ({c\_svd\[0\]:+0.4f}, {c\_svd\[1\]:+0.4f}, {c\_svd\[2\]:+0.4f})")  
print("weights (Φ+, Φ−, Ψ+, Ψ−) \=", \[f"{x:0.4f}" for x in w\_svd\])  
print(f"Werner p (Φ+) \= {pW\_svd:0.4f}  ⇒  S\_pred \= {S\_pred\_svd:0.4f},  F\_pred(Φ+) \= {F\_pred\_svd:0.4f}")  
if \_BD\_S\_meas is not None:  
    print(f"Measured S \= {\_BD\_S\_meas:0.4f},  Tsirelson gap Δ \= {gap\_svd:0.6f}")

\# \--- Export CSVs \---  
rows\_lab \= \[  
    \["frame","c1","c2","c3","w\_phi+","w\_phi-","w\_psi+","w\_psi-","pWerner","S\_pred","Fphi\_pred","S\_meas","Fphi\_meas"\]  
\]  
rows\_lab.append(\["lab", c\_lab\[0\], c\_lab\[1\], c\_lab\[2\], \*\[float(x) for x in w\_lab\], pW\_lab, S\_pred\_lab, F\_pred\_lab, \_BD\_S\_meas, \_BD\_F\_meas\])

rows\_svd \= \[  
    \["frame","c1","c2","c3","w\_phi+","w\_phi-","w\_psi+","w\_psi-","pWerner","S\_pred","Fphi\_pred","S\_meas","Fphi\_meas"\]  
\]  
rows\_svd.append(\["svd", c\_svd\[0\], c\_svd\[1\], c\_svd\[2\], \*\[float(x) for x in w\_svd\], pW\_svd, S\_pred\_svd, F\_pred\_svd, \_BD\_S\_meas, \_BD\_F\_meas\])

\_BD\_sec \= os.path.join(\_BD\_outdir, "sections")  
\_safe\_write\_csv(os.path.join(\_BD\_sec, "bell\_mixture\_lab.csv"), rows\_lab\[0\], rows\_lab\[1:\])  
\_safe\_write\_csv(os.path.join(\_BD\_sec, "bell\_mixture\_svd.csv"), rows\_svd\[0\], rows\_svd\[1:\])  
print(f"\[Project1 artifacts++\] wrote: {\_BD\_sec}/bell\_mixture\_lab.csv")  
print(f"\[Project1 artifacts++\] wrote: {\_BD\_sec}/bell\_mixture\_svd.csv")

\# \--- OpenQASM export for your symbolic patch angles (no Qiskit required) \---  
angles \= symbolic\_patch\_angles()  
def \_qasm\_rz(theta, q):  return f"rz({theta:.12f}) q\[{q}\];"  
def \_qasm\_ry(theta, q):  return f"ry({theta:.12f}) q\[{q}\];"  
def \_qasm\_h(q):          return f"h q\[{q}\];"  
def \_qasm\_cx(c,t):       return f"cx q\[{c}\],q\[{t}\];"

def \_qasm\_header():  
    return "OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\nqreg q\[2\];\\ncreg c\[2\];\\n"

def \_qasm\_inverse\_patch():  
    zA,aA,bA,gA \= angles\["A"\]\["Z"\], \*angles\["A"\]\["ZYZ"\]  
    zB,aB,bB,gB \= angles\["B"\]\["Z"\], \*angles\["B"\]\["ZYZ"\]  
    lines \= \[\_qasm\_header()\]  
    \# Inverse patch \= Rz(-g)·Ry(-b)·Rz(-a)·Rz(-z) on each qubit  
    lines \+= \[  
        \_qasm\_rz(-gA,0), \_qasm\_ry(-bA,0), \_qasm\_rz(-aA,0), \_qasm\_rz(-zA,0),  
        \_qasm\_rz(-gB,1), \_qasm\_ry(-bB,1), \_qasm\_rz(-aB,1), \_qasm\_rz(-zB,1),  
        "barrier q\[0\],q\[1\];"  
    \]  
    \# Example measurement in ZZ (you can change)  
    lines \+= \["measure q\[0\] \-\> c\[0\];", "measure q\[1\] \-\> c\[1\];"\]  
    return "\\n".join(lines) \+ "\\n"

def \_qasm\_misalign\_then\_inverse():  
    zA,aA,bA,gA \= angles\["A"\]\["Z"\], \*angles\["A"\]\["ZYZ"\]  
    zB,aB,bB,gB \= angles\["B"\]\["Z"\], \*angles\["B"\]\["ZYZ"\]  
    lines \= \[\_qasm\_header()\]  
    \# prepare Bell Φ+ (H on A; CX A-\>B)  
    lines \+= \[\_qasm\_h(0), \_qasm\_cx(0,1), "barrier q\[0\],q\[1\];"\]  
    \# Misalign: Rz(z)·Rz(a)·Ry(b)·Rz(g) then inverse-patch  
    lines \+= \[  
        \_qasm\_rz(zA,0), \_qasm\_rz(aA,0), \_qasm\_ry(bA,0), \_qasm\_rz(gA,0),  
        \_qasm\_rz(zB,1), \_qasm\_rz(aB,1), \_qasm\_ry(bB,1), \_qasm\_rz(gB,1),  
        "barrier q\[0\],q\[1\];",  
        \_qasm\_rz(-gA,0), \_qasm\_ry(-bA,0), \_qasm\_rz(-aA,0), \_qasm\_rz(-zA,0),  
        \_qasm\_rz(-gB,1), \_qasm\_ry(-bB,1), \_qasm\_rz(-aB,1), \_qasm\_rz(-zB,1),  
        "barrier q\[0\],q\[1\];",  
        \# Example: switch to chosen bases later if desired  
        "measure q\[0\] \-\> c\[0\];", "measure q\[1\] \-\> c\[1\];"  
    \]  
    return "\\n".join(lines) \+ "\\n"

\_qasm\_dir \= os.path.join(\_BD\_outdir, "qasm")  
os.makedirs(\_qasm\_dir, exist\_ok=True)  
with open(os.path.join(\_qasm\_dir, "inverse\_patch.qasm"), "w") as f: f.write(\_qasm\_inverse\_patch())  
with open(os.path.join(\_qasm\_dir, "misalign\_then\_inverse.qasm"), "w") as f: f.write(\_qasm\_misalign\_then\_inverse())  
print(f"\[Project1 artifacts++\] wrote: {\_qasm\_dir}/inverse\_patch.qasm")  
print(f"\[Project1 artifacts++\] wrote: {\_qasm\_dir}/misalign\_then\_inverse.qasm")

\# \===================== Project1+++ Model Selection & KL Residuals — ADD ONLY \=====================  
\# This block APPENDS functionality. It does not modify prior variables or functions.  
\# It assumes your QuantumCalPro \+ Project1 \+ Project1++ sections already ran.

import os, math, csv, json, numpy as np

\# \--- Safe access to prior run objects; graceful fallbacks if missing \---  
try:  
    \_P1p\_outdir \= P1\_out  
except NameError:  
    \# make a fresh proj1\_results run dir if Project1 wasn't imported  
    import datetime  
    ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
    \_P1p\_outdir \= os.path.join("proj1\_results", f"run\_{ts}")  
    os.makedirs(os.path.join(\_P1p\_outdir, "sections"), exist\_ok=True)

try:  
    \_P1p\_T \= P1\_T.copy()  
    \_P1p\_T\_after \= P1\_T\_after.copy()  
except NameError:  
    \_P1p\_T, \_, \_ \= counts\_to\_T\_and\_singles(EXTERNAL\_COUNTS.copy())  
    RA, Sg, RB \= proper\_svd(\_P1p\_T)  
    \_P1p\_T\_after \= RA.T @ \_P1p\_T @ RB

\# counts & basics  
\_P1p\_counts \= EXTERNAL\_COUNTS.copy()  
\_P1p\_pairs  \= \['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'\]

def \_P1p\_totals(cdict):  
    return {k: sum(v.values()) for k,v in cdict.items()}

\_P1p\_N\_by \= \_P1p\_totals(\_P1p\_counts)  
\_P1p\_Ntot \= sum(\_P1p\_N\_by.values())

\# \--- SO(3) frames from your symbolic patch (fixed, MDL-aware) \---  
\_ang \= symbolic\_patch\_angles()  
\_A \= so3\_from\_z\_and\_zyz(\_ang\["A"\]\["Z"\], \_ang\["A"\]\["ZYZ"\])  
\_B \= so3\_from\_z\_and\_zyz(\_ang\["B"\]\["Z"\], \_ang\["B"\]\["ZYZ"\])

def \_P1p\_clamp(x, lo=-0.999999, hi=+0.999999):  
    return max(lo, min(hi, float(x)))

def \_P1p\_probs\_from\_E(E):  
    \# zero-singles even/odd split (used throughout your pipeline)  
    E \= \_P1p\_clamp(E)  
    pd \= (1+E)/4.0; po \= (1-E)/4.0  
    return {'00':pd, '11':pd, '01':po, '10':po}

def \_P1p\_logL\_for\_Tpred(Tpred):  
    \# Log-likelihood across all bases given 2x2 probs that depend only on E \= T\_ij  
    L \= 0.0  
    for pair in \_P1p\_pairs:  
        i, j \= PAIR2IDX\[pair\]  
        probs \= \_P1p\_probs\_from\_E(Tpred\[i,j\])  
        cnts  \= \_P1p\_counts\[pair\]  
        for k in ('00','01','10','11'):  
            p \= max(probs\[k\], 1e-15)  
            L \+= cnts\[k\] \* math.log(p)  
    return float(L)

def \_P1p\_KL\_per\_basis(Tpred):  
    rows \= \[\]  
    for pair in \_P1p\_pairs:  
        i,j \= PAIR2IDX\[pair\]  
        N \= \_P1p\_N\_by\[pair\]  
        obsP \= {k: \_P1p\_counts\[pair\]\[k\]/N for k in ('00','01','10','11')}  
        modP \= \_P1p\_probs\_from\_E(Tpred\[i,j\])  
        \# KL(obs||mod) with safe floors  
        KL \= 0.0  
        for k in ('00','01','10','11'):  
            p \= max(obsP\[k\], 1e-15); q \= max(modP\[k\], 1e-15)  
            KL \+= p \* math.log(p/q)  
        rows.append((pair, N, float(\_P1p\_T\[i,j\]), float(Tpred\[i,j\]), KL))  
    return rows

def \_P1p\_write\_csv(path, header, rows):  
    os.makedirs(os.path.dirname(path), exist\_ok=True)  
    with open(path, "w", encoding="utf-8", newline="") as f:  
        w \= csv.writer(f); w.writerow(header); w.writerows(rows)

\# \--- Bell-diagonal (3-parameter c) at fixed frames A,B \---  
\# T\_pred \= A^T · diag(c) · B^T ; MLE/LS estimate of c under fixed frames: c\_hat \= diag( A · T\_meas · B )  
\_M \= \_A @ \_P1p\_T @ \_B  
\_c\_hat \= np.array(\[\_M\[0,0\], \_M\[1,1\], \_M\[2,2\]\], float)  
\_Tpred\_BD3 \= \_A.T @ np.diag(\_c\_hat) @ \_B.T

\# \--- Werner (1-parameter p) at fixed frames A,B \---  
\# T\_pred(p) \= A^T · diag(\[p, \-p, p\]) · B^T ; estimate p by 1D MLE on counts (zero-singles likelihood)  
def \_P1p\_Tpred\_Werner(p):  
    p \= float(p)  
    D \= np.diag(\[p, \-p, p\])  
    return \_A.T @ D @ \_B.T

def \_P1p\_logL\_Werner(p):  
    return \_P1p\_logL\_for\_Tpred(\_P1p\_Tpred\_Werner(p))

\# Golden-section search for p in \[-pmax, pmax\]  
def \_P1p\_golden\_max(fun, a=-0.999, b=+0.999, tol=1e-6, maxit=200):  
    gr \= (math.sqrt(5)-1)/2  
    c \= b \- gr\*(b-a); d \= a \+ gr\*(b-a)  
    fc, fd \= fun(c), fun(d)  
    it \= 0  
    while abs(b-a) \> tol and it \< maxit:  
        if fc \< fd:  
            a, c, fc \= c, d, fd  
            d \= a \+ gr\*(b-a); fd \= fun(d)  
        else:  
            b, d, fd \= d, c, fc  
            c \= b \- gr\*(b-a); fc \= fun(c)  
        it \+= 1  
    x \= (a+b)/2.0  
    return x, fun(x)

\_p\_hat, \_LL\_W \= \_P1p\_golden\_max(\_P1p\_logL\_Werner)  
\_Tpred\_W1 \= \_P1p\_Tpred\_Werner(\_p\_hat)

\# \--- Bell-diagonal (3c) and saturated "ZS" log-likelihoods \---  
\_LL\_BD3 \= \_P1p\_logL\_for\_Tpred(\_Tpred\_BD3)  
\_LL\_ZS  \= \_P1p\_logL\_for\_Tpred(\_P1p\_T)  \# saturated in E (what your "zero-singles" residuals used)

\# \--- Information criteria (report both penalized and raw for transparency) \---  
kW, kBD, kZS \= 1, 3, 9  
AIC\_W   \= \-2\*\_LL\_W \+ 2\*kW;        BIC\_W   \= \-2\*\_LL\_W \+ kW\*math.log(\_P1p\_Ntot)  
AIC\_BD3 \= \-2\*\_LL\_BD3 \+ 2\*kBD;     BIC\_BD3 \= \-2\*\_LL\_BD3 \+ kBD\*math.log(\_P1p\_Ntot)  
AIC\_ZS  \= \-2\*\_LL\_ZS \+ 2\*kZS;      BIC\_ZS  \= \-2\*\_LL\_ZS \+ kZS\*math.log(\_P1p\_Ntot)

rawAIC\_W, rawAIC\_BD3, rawAIC\_ZS \= \-2\*\_LL\_W, \-2\*\_LL\_BD3, \-2\*\_LL\_ZS  \# matches earlier "AIC=BIC=-2logL" style

\# \--- Per-basis KL tables (obs || model) \---  
KL\_W\_rows  \= \_P1p\_KL\_per\_basis(\_Tpred\_W1)  
KL\_BD\_rows \= \_P1p\_KL\_per\_basis(\_Tpred\_BD3)

\# \--- CSV exports \---  
\_sec \= os.path.join(\_P1p\_outdir, "sections")  
\_P1p\_write\_csv(  
    os.path.join(\_sec, "model\_selection\_fixed\_frames.csv"),  
    \["model","k","logL","AIC","BIC","raw\_-2logL","notes"\],  
    \[  
        \["Werner(p)@symbolic", kW, \_LL\_W, AIC\_W, BIC\_W, rawAIC\_W, f"p\_hat={\_p\_hat:.6f}"\],  
        \["BellDiag(3c)@symbolic", kBD, \_LL\_BD3, AIC\_BD3, BIC\_BD3, rawAIC\_BD3, f"c\_hat=({\_c\_hat\[0\]:+.6f},{\_c\_hat\[1\]:+.6f},{\_c\_hat\[2\]:+.6f})"\],  
        \["Saturated(ZS, 9 E\_ij)", kZS, \_LL\_ZS, AIC\_ZS, BIC\_ZS, rawAIC\_ZS, "reference (highest logL)"\],  
    \]  
)

\_P1p\_write\_csv(  
    os.path.join(\_sec, "per\_basis\_KL\_Werner.csv"),  
    \["pair","N","E\_meas","E\_pred\_Werner","KL(obs||Werner)"\],  
    KL\_W\_rows  
)

\_P1p\_write\_csv(  
    os.path.join(\_sec, "per\_basis\_KL\_BellDiag3.csv"),  
    \["pair","N","E\_meas","E\_pred\_BellDiag3","KL(obs||BellDiag3)"\],  
    KL\_BD\_rows  
)

print(f"\[Project1 artifacts+++\] wrote: {\_sec}/model\_selection\_fixed\_frames.csv")  
print(f"\[Project1 artifacts+++\] wrote: {\_sec}/per\_basis\_KL\_Werner.csv")  
print(f"\[Project1 artifacts+++\] wrote: {\_sec}/per\_basis\_KL\_BellDiag3.csv")

\# \--- Console summary \---  
def \_P1p\_fmt\_row(r):  
    return f"{r\[0\]:\<2}  N={r\[1\]:\>5d}  E\_meas={r\[2\]:+0.4f}  E\_pred={r\[3\]:+0.4f}  KL={r\[4\]:0.6f}"

print("\\n— MODEL SELECTION (fixed symbolic frames) —")  
print(f"Werner(p):        p\_hat={\_p\_hat:+0.6f}   logL={\_LL\_W:0.2f}   AIC={AIC\_W:0.2f}   BIC={BIC\_W:0.2f}   raw-2logL={rawAIC\_W:0.2f}")  
print(f"BellDiag(3c):     c\_hat=({\_c\_hat\[0\]:+0.6f},{\_c\_hat\[1\]:+0.6f},{\_c\_hat\[2\]:+0.6f})   logL={\_LL\_BD3:0.2f}   AIC={AIC\_BD3:0.2f}   BIC={BIC\_BD3:0.2f}   raw-2logL={rawAIC\_BD3:0.2f}")  
print(f"Saturated (ZS):   logL={\_LL\_ZS:0.2f}   AIC={AIC\_ZS:0.2f}   BIC={BIC\_ZS:0.2f}   raw-2logL={rawAIC\_ZS:0.2f}")

def \_P1p\_winner(aic\_dict):  
    \# smaller is better  
    return min(aic\_dict, key=aic\_dict.get)

\_winner\_AIC \= \_P1p\_winner({"Werner":AIC\_W, "BellDiag3":AIC\_BD3, "ZS":AIC\_ZS})  
\_winner\_BIC \= \_P1p\_winner({"Werner":BIC\_W, "BellDiag3":BIC\_BD3, "ZS":BIC\_ZS})  
print(f"\\nWinner by AIC: {\_winner\_AIC}   |   Winner by BIC: {\_winner\_BIC}")

\# \--- Tiny ASCII “plot” of KL per basis (Werner vs BD3) \---  
def \_P1p\_ascii\_bar(x, scale=400.0, maxlen=40):  
    \# x ≈ KL; use sqrt scaling for visibility; clamp to maxlen  
    n \= int(min(maxlen, round(math.sqrt(max(0.0, x))\*scale)))  
    return "\#"\*n

print("\\n— PER-BASIS KL (obs||model) —")  
print("pair  |  KL\_Werner            |  KL\_BellDiag3        |  note")  
for rW, rB in zip(KL\_W\_rows, KL\_BD\_rows):  
    barW \= \_P1p\_ascii\_bar(rW\[4\])  
    barB \= \_P1p\_ascii\_bar(rB\[4\])  
    note \= ""  
    if rW\[4\] \> rB\[4\]\*1.25: note \= "\<-- BD3 fits better"  
    elif rB\[4\] \> rW\[4\]\*1.25: note \= "\<-- Werner fits better"  
    print(f"{rW\[0\]:\<3}  |  {rW\[4\]:0.6f} {barW:\<40} |  {rB\[4\]:0.6f} {barB:\<40} |  {note}")

\# \--- Bundle JSON snapshot (optional) \---  
try:  
    \_bundle\_path \= os.path.join(\_P1p\_outdir, "sections", "model\_selection\_snapshot.json")  
    with open(\_bundle\_path, "w") as f:  
        json.dump({  
            "frames\_symbolic": {  
                "A":{"Z":\_ang\["A"\]\["Z"\], "ZYZ":\_ang\["A"\]\["ZYZ"\]},  
                "B":{"Z":\_ang\["B"\]\["Z"\], "ZYZ":\_ang\["B"\]\["ZYZ"\]},  
            },  
            "Werner": {"p\_hat": \_p\_hat, "logL": \_LL\_W, "AIC": AIC\_W, "BIC": BIC\_W},  
            "BellDiag3": {"c\_hat": \_c\_hat.tolist(), "logL": \_LL\_BD3, "AIC": AIC\_BD3, "BIC": BIC\_BD3},  
            "ZS": {"logL": \_LL\_ZS, "AIC": AIC\_ZS, "BIC": BIC\_ZS},  
            "per\_basis": {  
                "Werner": \[{"pair":p, "N":N, "E\_meas":Em, "E\_pred":Ep, "KL":KL} for (p,N,Em,Ep,KL) in KL\_W\_rows\],  
                "BellDiag3": \[{"pair":p, "N":N, "E\_meas":Em, "E\_pred":Ep, "KL":KL} for (p,N,Em,Ep,KL) in KL\_BD\_rows\],  
            }  
        }, f, indent=2)  
    print(f"\[Project1 artifacts+++\] wrote: {\_bundle\_path}")  
except Exception as \_e:  
    print(f"\[Project1 note\] Could not write JSON snapshot: {\_e}")  
\# \===================== End Project1+++ \=====================

\# \===================== Project1++++ Werner posterior, GOF, predicted counts, frame-delta — ADD ONLY \=====================  
\# This block APPENDS functionality. It assumes QuantumCalPro \+ Project1 (+ Project1+/++/+++) already ran.

import os, math, csv, json, numpy as np

\# \--- Reuse / recover objects safely \---  
try:  
    \_P1pp\_outdir \= P1\_out  
except NameError:  
    import datetime  
    ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
    \_P1pp\_outdir \= os.path.join("proj1\_results", f"run\_{ts}")  
    os.makedirs(os.path.join(\_P1pp\_outdir, "sections"), exist\_ok=True)

try:  
    \_P1pp\_T \= P1\_T.copy()  
except NameError:  
    \_P1pp\_T, \_, \_ \= counts\_to\_T\_and\_singles(EXTERNAL\_COUNTS.copy())

try:  
    \_P1pp\_T\_after \= P1\_T\_after.copy()  
except NameError:  
    \_RA\_svd\_tmp, \_Sg\_tmp, \_RB\_svd\_tmp \= proper\_svd(\_P1pp\_T)  
    \_P1pp\_T\_after \= \_RA\_svd\_tmp.T @ \_P1pp\_T @ \_RB\_svd\_tmp

\_counts \= EXTERNAL\_COUNTS.copy()  
\_pairs  \= \['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'\]  
def \_totals(cdict): return {k: sum(v.values()) for k,v in cdict.items()}  
\_N\_by \= \_totals(\_counts); \_Ntot \= sum(\_N\_by.values())

\# \--- pull symbolic frames \---  
\_sy \= symbolic\_patch\_angles()  
\_A\_sym \= so3\_from\_z\_and\_zyz(\_sy\["A"\]\["Z"\], \_sy\["A"\]\["ZYZ"\])  
\_B\_sym \= so3\_from\_z\_and\_zyz(\_sy\["B"\]\["Z"\], \_sy\["B"\]\["ZYZ"\])

\# \--- utilities (reuse your conventions) \---  
def \_clampE(x): return max(-0.999999, min(0.999999, float(x)))  
def \_probs\_from\_E(E):  
    E \= \_clampE(E); pd \= (1+E)/4.0; po \= (1-E)/4.0  
    return {'00':pd, '11':pd, '01':po, '10':po}

def \_logL\_for\_Tpred(Tpred):  
    L \= 0.0  
    for pair in \_pairs:  
        i,j \= PAIR2IDX\[pair\]  
        probs \= \_probs\_from\_E(Tpred\[i,j\])  
        cnts  \= \_counts\[pair\]  
        for k in ('00','01','10','11'):  
            p \= max(probs\[k\], 1e-15)  
            L \+= cnts\[k\]\*math.log(p)  
    return float(L)

\# \--- SVD frames (for deltas) \---  
\_RA\_svd, \_Sg, \_RB\_svd \= proper\_svd(\_P1pp\_T)

\# \--- BellDiag(3c) @ symbolic frames (same as in \+++) \---  
\_Msym \= \_A\_sym @ \_P1pp\_T @ \_B\_sym  
\_c\_hat\_sym \= np.array(\[\_Msym\[0,0\], \_Msym\[1,1\], \_Msym\[2,2\]\], float)  
\_Tpred\_BD3\_sym \= \_A\_sym.T @ np.diag(\_c\_hat\_sym) @ \_B\_sym.T  
\_LL\_BD3 \= \_logL\_for\_Tpred(\_Tpred\_BD3\_sym)

\# \--- Saturated ZS in E (reference) \---  
\_LL\_ZS  \= \_logL\_for\_Tpred(\_P1pp\_T)

\# \--- Werner(p) @ symbolic frames: posterior over p in \[-1,1\] \---  
def \_Tpred\_Werner(p):  
    D \= np.diag(\[p, \-p, p\]); return \_A\_sym.T @ D @ \_B\_sym.T  
def \_logL\_Werner(p): return \_logL\_for\_Tpred(\_Tpred\_Werner(float(p)))

\# grid & posterior  
\_grid \= np.linspace(-1.0, 1.0, 2001\)  \# 2001-point dense grid  
\_logL\_vals \= np.array(\[\_logL\_Werner(p) for p in \_grid\], float)  
\# uniform prior on \[-1,1\]  
\_logpost \= \_logL\_vals \- np.max(\_logL\_vals)  
\_post \= np.exp(\_logpost); \_post /= np.trapz(\_post, \_grid)

\# posterior summaries  
\_p\_mode \= float(\_grid\[np.argmax(\_post)\])  
\_p\_mean \= float(np.trapz(\_grid \* \_post, \_grid))  
\# CDF for quantiles  
\_cdf \= np.cumsum(\_post) \* (\_grid\[1\]-\_grid\[0\])  
def \_q(c):  
    i \= np.searchsorted(\_cdf, c);  
    i \= max(1, min(len(\_grid)-1, i))  
    \# linear interp  
    t \= (c \- \_cdf\[i-1\]) / max(1e-18, (\_cdf\[i\]-\_cdf\[i-1\]))  
    return float(\_grid\[i-1\] \+ t\*(\_grid\[i\]-\_grid\[i-1\]))  
\_p\_ci\_lo, \_p\_ci\_hi \= \_q(0.025), \_q(0.975)

\# HPD 95% via thresholding  
idx\_sorted \= np.argsort(\_post)\[::-1\]  
cum \= 0.0; thr \= 0.0  
for k in idx\_sorted:  
    thr \= \_post\[k\]  
    cum \= float(np.trapz(\_post\[\_post\>=thr\], \_grid\[\_post\>=thr\]))  
    if cum \>= 0.95: break  
mask \= (\_post \>= thr \- 1e-18)  
grid\_hpd \= \_grid\[mask\]  
\_p\_hpd\_lo, \_p\_hpd\_hi \= float(grid\_hpd.min()), float(grid\_hpd.max())

\# curvature SE at mode (Laplace approx)  
def \_num\_second\_deriv(fun, x, h=1e-4):  
    return (fun(x+h) \- 2\*fun(x) \+ fun(x-h)) / (h\*h)  
\_negH \= \-\_num\_second\_deriv(\_logL\_Werner, \_p\_mode, 1e-4)  
\_p\_se \= float(math.sqrt(max(1e-18, 1.0/\_negH)))

\# final Werner MLE & logL  
\_Tpred\_W1 \= \_Tpred\_Werner(\_p\_mode)  
\_LL\_W \= \_logL\_Werner(\_p\_mode)

\# \--- Deviance GOF vs ZS; chi^2 p-values (df \= 9 \- params) \---  
def \_chisq\_sf(x, k):  
    \# survival function for chi-square via regularized upper gamma Q(k/2, x/2)  
    \# simple continued fraction for Q; for stability use scipy normally, but keep pure-Python here  
    \# implement using incomplete gamma series+cf (Abramowitz-Stegun 6.5.29) — minimal, adequate for our x,k  
    a \= 0.5 \* k; x2 \= 0.5 \* x  
    if x \<= 0: return 1.0  
    \# choose series vs CF  
    if x2 \< a \+ 1.0:  
        \# lower series for P, then Q=1-P  
        term \= 1.0 / a; summ \= term; n=1  
        while n \< 1000:  
            term \*= x2/(a+n); summ \+= term; n+=1  
            if term \< summ\*1e-12: break  
        P \= math.exp(-x2 \+ a\*math.log(x2) \- math.lgamma(a)) \* summ  
        return max(0.0, 1.0 \- P)  
    else:  
        \# continued fraction for Q directly  
        \# Lentz algorithm  
        tiny \= 1e-300  
        b0 \= x2 \+ 1.0 \- a  
        C \= 1.0 / tiny  
        D \= 1.0 / max(b0, tiny)  
        f \= D  
        for i in range(1, 1000):  
            m \= i  
            a\_i \= m \* (a \- m)  
            b\_i \= b0 \+ 2.0\*m  
            D \= b\_i \+ a\_i \* D  
            D \= max(D, tiny)  
            D \= 1.0 / D  
            C \= b\_i \+ a\_i / C  
            C \= max(C, tiny)  
            delta \= C \* D  
            f \*= delta  
            if abs(delta \- 1.0) \< 1e-12: break  
        Q \= math.exp(-x2 \+ a\*math.log(x2) \- math.lgamma(a)) \* f  
        return max(0.0, min(1.0, Q))

\# parameter counts: Werner=1, BellDiag3=3, ZS=9 (one E per pair)  
kW, kBD, kZS \= 1, 3, 9  
dev\_W  \= \-2.0\*(\_LL\_W  \- \_LL\_ZS);  p\_W  \= \_chisq\_sf(dev\_W,  kZS \- kW)  
dev\_BD \= \-2.0\*(\_LL\_BD3- \_LL\_ZS);  p\_BD \= \_chisq\_sf(dev\_BD, kZS \- kBD)

\# \--- Predicted counts CSVs (Werner & BellDiag3 @ symbolic) \---  
def \_pred\_counts\_table(Tpred):  
    rows=\[\]  
    for pair in \_pairs:  
        i,j \= PAIR2IDX\[pair\]; N=\_N\_by\[pair\]; P=\_probs\_from\_E(Tpred\[i,j\])  
        rows.append(\[pair, N, P\['00'\]\*N, P\['01'\]\*N, P\['10'\]\*N, P\['11'\]\*N\])  
    return rows

\_sec \= os.path.join(\_P1pp\_outdir, "sections")  
os.makedirs(\_sec, exist\_ok=True)  
with open(os.path.join(\_sec, "pred\_counts\_Werner.csv"), "w", newline="", encoding="utf-8") as f:  
    w=csv.writer(f); w.writerow(\["pair","N","pred00","pred01","pred10","pred11"\]); w.writerows(\_pred\_counts\_table(\_Tpred\_W1))  
with open(os.path.join(\_sec, "pred\_counts\_BellDiag3.csv"), "w", newline="", encoding="utf-8") as f:  
    w=csv.writer(f); w.writerow(\["pair","N","pred00","pred01","pred10","pred11"\]); w.writerows(\_pred\_counts\_table(\_Tpred\_BD3\_sym))  
print(f"\[Project1 artifacts++++\] wrote: {\_sec}/pred\_counts\_Werner.csv")  
print(f"\[Project1 artifacts++++\] wrote: {\_sec}/pred\_counts\_BellDiag3.csv")

\# \--- Werner posterior CSV (grid) \---  
with open(os.path.join(\_sec, "werner\_posterior\_grid.csv"), "w", newline="", encoding="utf-8") as f:  
    w=csv.writer(f); w.writerow(\["p","logL","posterior\_pdf"\])  
    for p, L, q in zip(\_grid, \_logL\_vals, \_post): w.writerow(\[f"{p:.6f}", f"{L:.6f}", f"{q:.12e}"\])  
print(f"\[Project1 artifacts++++\] wrote: {\_sec}/werner\_posterior\_grid.csv")

\# \--- Frame deltas: symbolic vs SVD frames (Alice/Bob) \---  
\# want R such that RA\_svd ≈ A\_sym @ R  ⇒  R ≈ A\_sym^T @ RA\_svd  (and similarly for Bob)  
\_RdA \= \_A\_sym.T @ \_RA\_svd  
\_RdB \= \_B\_sym.T @ \_RB\_svd  
\_aA,\_bA,\_gA \= zyz\_from\_R(\_RdA); \_aB,\_bB,\_gB \= zyz\_from\_R(\_RdB)

def \_fmt\_pi(x): return fmt\_pi\_rational(x, max\_den=41)  
print("\\n— WERNER POSTERIOR (uniform prior on \[-1,1\]) —")  
print(f"p\_mode={\_p\_mode:+.6f},  p\_mean={\_p\_mean:+.6f},  95% CI=\[{\_p\_ci\_lo:+.6f}, {\_p\_ci\_hi:+.6f}\],  95% HPD=\[{\_p\_hpd\_lo:+.6f}, {\_p\_hpd\_hi:+.6f}\],  SE≈{\_p\_se:.6f}")

print("\\n— DEVIANCE GOF vs Saturated zero-singles —")  
print(f"Werner:    dev={dev\_W:.2f}  df=8  p≈{p\_W:.3e}")  
print(f"BellDiag3: dev={dev\_BD:.2f}  df=6  p≈{p\_BD:.3e}")

print("\\n— FRAME DELTA (Symbolic → SVD) —")  
print("Alice ΔZYZ:", f"{\_fmt\_pi(\_aA)}, {\_fmt\_pi(\_bA)}, {\_fmt\_pi(\_gA)}",  
      "   (deg: " \+ ", ".join(f"{to\_deg(x):+.3f}°" for x in (\_aA,\_bA,\_gA)) \+ ")")  
print("Bob   ΔZYZ:", f"{\_fmt\_pi(\_aB)}, {\_fmt\_pi(\_bB)}, {\_fmt\_pi(\_gB)}",  
      "   (deg: " \+ ", ".join(f"{to\_deg(x):+.3f}°" for x in (\_aB,\_bB,\_gB)) \+ ")")

\# \--- Bundle JSON drop (summary) \---  
\_summary \= {  
    "werner\_posterior": {  
        "mode": \_p\_mode, "mean": \_p\_mean, "ci95":\[\_p\_ci\_lo,\_p\_ci\_hi\], "hpd95":\[\_p\_hpd\_lo,\_p\_hpd\_hi\], "se\_laplace": \_p\_se  
    },  
    "gof\_deviance": {  
        "Werner":{"dev":dev\_W, "df":8, "p\_value":p\_W},  
        "BellDiag3":{"dev":dev\_BD, "df":6, "p\_value":p\_BD},  
    },  
    "frames\_delta\_symbolic\_to\_svd": {  
        "Alice\_ZYZ\_deg":\[to\_deg(\_aA), to\_deg(\_bA), to\_deg(\_gA)\],  
        "Bob\_ZYZ\_deg":\[to\_deg(\_aB), to\_deg(\_bB), to\_deg(\_gB)\],  
        "Alice\_ZYZ\_pi":\[\_fmt\_pi(\_aA), \_fmt\_pi(\_bA), \_fmt\_pi(\_gA)\],  
        "Bob\_ZYZ\_pi":\[\_fmt\_pi(\_aB), \_fmt\_pi(\_bB), \_fmt\_pi(\_gB)\],  
    },  
    "BellDiag3\_c\_hat\_symbolic": \_c\_hat\_sym.tolist(),  
    "logL": {"Werner": \_LL\_W, "BellDiag3": \_LL\_BD3, "ZS": \_LL\_ZS}  
}  
with open(os.path.join(\_sec, "werner\_gof\_frames\_summary.json"), "w", encoding="utf-8") as f:  
    json.dump(\_summary, f, indent=2)  
print(f"\[Project1 artifacts++++\] wrote: {\_sec}/werner\_gof\_frames\_summary.json")  
\# \===================== End Project1++++ \=====================

\# \===================== Project1+++++ — Compat shim \+ Parametric ΔAIC bootstrap (ADD ONLY) \=====================  
\# This block appends functionality after all prior Project1 blocks.

import os, csv, json, math, warnings  
import numpy as np

\# \--- Compat shim for NumPy trapz deprecation (ADD-ONLY; no edits upstream) \---  
warnings.filterwarnings("ignore", category=DeprecationWarning)  
if hasattr(np, "trapezoid"):  
    try:  
        np.trapz \= np.trapezoid  \# alias for future runs so earlier blocks won't warn  
        print("\[compat\] numpy.trapz → numpy.trapezoid alias active; DeprecationWarning silenced.")  
    except Exception as \_e:  
        print(f"\[compat\] alias failed (non-fatal): {\_e}")

\# \--- Safe reuse of globals made earlier; minimal fallbacks if user runs standalone \---  
try:  
    \_P1ppp\_outdir \= P1\_out  
except NameError:  
    import datetime  
    ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
    \_P1ppp\_outdir \= os.path.join("proj1\_results", f"run\_{ts}")  
    os.makedirs(os.path.join(\_P1ppp\_outdir, "sections"), exist\_ok=True)

try:  
    \_pairs  \= \['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'\]  
    \_counts \= EXTERNAL\_COUNTS.copy()  
    def \_totals(cdict): return {k: sum(v.values()) for k,v in cdict.items()}  
    \_N\_by \= \_totals(\_counts)  
except Exception:  
    raise RuntimeError("Counts not available; make sure QuantumCalPro main ran first.")

\# pull symbols/frames from previous blocks  
\_sy \= symbolic\_patch\_angles()  
\_A\_sym \= so3\_from\_z\_and\_zyz(\_sy\["A"\]\["Z"\], \_sy\["A"\]\["ZYZ"\])  
\_B\_sym \= so3\_from\_z\_and\_zyz(\_sy\["B"\]\["Z"\], \_sy\["B"\]\["ZYZ"\])

\# existing helpers  
def \_clampE(x): return max(-0.999999, min(0.999999, float(x)))  
def \_probs\_from\_E(E):  
    E \= \_clampE(E); pd \= (1+E)/4.0; po \= (1-E)/4.0  
    return {'00':pd, '01':po, '10':po, '11':pd}

def \_logL\_for\_Tpred(Tpred):  
    L \= 0.0  
    for pair in \_pairs:  
        i,j \= PAIR2IDX\[pair\]  
        P \= \_probs\_from\_E(Tpred\[i,j\])  
        cnt \= \_counts\[pair\]  
        for k in ('00','01','10','11'):  
            L \+= cnt\[k\] \* math.log(max(P\[k\], 1e-15))  
    return float(L)

\# reconstruct T and SVD frames (needed for projections if not kept around)  
\_T\_now, \_, \_ \= counts\_to\_T\_and\_singles(\_counts)  
\_RA\_svd\_now, \_Sg\_now, \_RB\_svd\_now \= proper\_svd(\_T\_now)

\# Werner & BellDiag3 predictors in fixed symbolic frames  
def \_Tpred\_Werner(p):  
    D \= np.diag(\[p, \-p, p\]); return \_A\_sym.T @ D @ \_B\_sym.T

\_Msym\_now \= \_A\_sym @ \_T\_now @ \_B\_sym  
\_c\_hat\_now \= np.array(\[\_Msym\_now\[0,0\], \_Msym\_now\[1,1\], \_Msym\_now\[2,2\]\], float)  
\_Tpred\_BD3\_now \= \_A\_sym.T @ np.diag(\_c\_hat\_now) @ \_B\_sym.T

\# pull grid & p\_mode from earlier posterior block if present; else make one now  
try:  
    \_grid  
    \_p\_mode  
except NameError:  
    \_grid \= np.linspace(-1.0, 1.0, 2001\)  
    \_LLW\_grid \= np.array(\[\_logL\_for\_Tpred(\_Tpred\_Werner(p)) for p in \_grid\], float)  
    \_p\_mode \= float(\_grid\[int(np.argmax(\_LLW\_grid))\])

\# model parameter counts for AIC  
kW, kBD, kZS \= 1, 3, 9

\# \--- Bootstrap helper: sample counts under a given Tpred \---  
def \_sample\_counts\_from\_T(Tpred, rng):  
    out \= {}  
    for pair in \_pairs:  
        N \= \_N\_by\[pair\]  
        i,j \= PAIR2IDX\[pair\]  
        P \= \_probs\_from\_E(Tpred\[i,j\])  
        ks \= ('00','01','10','11')  
        draws \= rng.multinomial(N, \[P\[k\] for k in ks\])  
        out\[pair\] \= {k:int(n) for k,n in zip(ks, draws)}  
    return out

\# \--- Fit under each model for a counts table \---  
def \_fit\_Werner\_LL(counts\_tbl):  
    \# grid MLE (consistent with previous computation)  
    T\_b,\_,\_ \= counts\_to\_T\_and\_singles(counts\_tbl)  
    \# reuse global \_grid and symbolic frames  
    LLs \= \[\]  
    for p in \_grid:  
        Tpred \= \_Tpred\_Werner(p)  
        LLs.append(\_logL\_for\_Tpred(Tpred))  
    LLs \= np.array(LLs, float)  
    idx \= int(np.argmax(LLs))  
    return float(\_grid\[idx\]), float(LLs\[idx\])

def \_fit\_BD3\_LL(counts\_tbl):  
    T\_b,\_,\_ \= counts\_to\_T\_and\_singles(counts\_tbl)  
    Msym\_b \= \_A\_sym @ T\_b @ \_B\_sym  
    c\_hat\_b \= np.array(\[Msym\_b\[0,0\], Msym\_b\[1,1\], Msym\_b\[2,2\]\], float)  
    Tpred\_b \= \_A\_sym.T @ np.diag(c\_hat\_b) @ \_B\_sym.T  
    return c\_hat\_b, float(\_logL\_for\_Tpred(Tpred\_b))

def \_LL\_ZS\_counts(counts\_tbl):  
    T\_b,\_,\_ \= counts\_to\_T\_and\_singles(counts\_tbl)  
    return float(\_logL\_for\_Tpred(T\_b))

\# \--- Parametric bootstrap of ΔAIC under each generator (Werner / BellDiag3) \---  
def \_bootstrap\_deltaAIC(generator="Werner", B=200, rng\_seed=202):  
    rng \= np.random.default\_rng(rng\_seed)  
    Tgen \= \_Tpred\_Werner(\_p\_mode) if generator=="Werner" else \_Tpred\_BD3\_now  
    rows \= \[\]  
    for b in range(B):  
        \# simulate data under generator  
        counts\_b \= \_sample\_counts\_from\_T(Tgen, rng)  
        \# saturated  
        LL\_ZS\_b \= \_LL\_ZS\_counts(counts\_b)  
        AIC\_ZS\_b \= \-2.0\*LL\_ZS\_b \+ 2\*kZS  
        \# Werner refit  
        p\_hat\_b, LL\_W\_b \= \_fit\_Werner\_LL(counts\_b)  
        AIC\_W\_b \= \-2.0\*LL\_W\_b \+ 2\*kW  
        \# BellDiag3 refit  
        c\_hat\_b, LL\_B\_b \= \_fit\_BD3\_LL(counts\_b)  
        AIC\_B\_b \= \-2.0\*LL\_B\_b \+ 2\*kBD  
        rows.append(\[  
            b, p\_hat\_b, c\_hat\_b\[0\], c\_hat\_b\[1\], c\_hat\_b\[2\],  
            AIC\_W\_b \- AIC\_ZS\_b,        \# ΔAIC(W − ZS)  
            AIC\_B\_b \- AIC\_ZS\_b,        \# ΔAIC(BD3 − ZS)  
            AIC\_W\_b \- AIC\_B\_b          \# ΔAIC(W − BD3)  
        \])  
    return np.array(rows, float)

\# \--- Run bootstrap (default B=200) and write artifacts \---  
\_sec \= os.path.join(\_P1ppp\_outdir, "sections"); os.makedirs(\_sec, exist\_ok=True)

for gen in ("Werner", "BellDiag3"):  
    res \= \_bootstrap\_deltaAIC(generator=gen, B=200, rng\_seed=(202 if gen=="Werner" else 203))  
    fn \= os.path.join(\_sec, f"bootstrap\_deltaAIC\_under\_{gen}.csv")  
    with open(fn, "w", newline="", encoding="utf-8") as f:  
        w \= csv.writer(f)  
        w.writerow(\["draw","p\_hat","c1\_hat","c2\_hat","c3\_hat","dAIC\_W\_minus\_ZS","dAIC\_BD3\_minus\_ZS","dAIC\_W\_minus\_BD3"\])  
        w.writerows(res.tolist())  
    print(f"\[Project1 artifacts+++++\] wrote: {fn}")

    \# quick summary stats  
    def q(v, a):  
        v \= np.sort(v); i \= int(np.clip(a\*(len(v)-1), 0, len(v)-1)); return float(v\[i\])  
    summary \= {  
        "generator": gen,  
        "B": int(res.shape\[0\]),  
        "dAIC\_W\_minus\_ZS": {"med": float(np.median(res\[:,5\])), "q05": q(res\[:,5\],0.05), "q95": q(res\[:,5\],0.95)},  
        "dAIC\_BD3\_minus\_ZS": {"med": float(np.median(res\[:,6\])), "q05": q(res\[:,6\],0.05), "q95": q(res\[:,6\],0.95)},  
        "dAIC\_W\_minus\_BD3": {"med": float(np.median(res\[:,7\])), "q05": q(res\[:,7\],0.05), "q95": q(res\[:,7\],0.95)},  
    }  
    jn \= os.path.join(\_sec, f"bootstrap\_deltaAIC\_under\_{gen}\_summary.json")  
    with open(jn, "w", encoding="utf-8") as f:  
        json.dump(summary, f, indent=2)  
    print(f"\[Project1 artifacts+++++\] wrote: {jn}")

print("— PARAMETRIC BOOTSTRAP ΔAIC — done (B=200 per generator).")  
\# \===================== End Project1+++++ \=====================  
\# \===================== Project1++++++ — Parametric S CIs \+ PPC KL \+ run snapshot (ADD ONLY) \=====================  
\# Appends functionality after all prior Project1 blocks. No upstream edits.

import os, json, math, csv, warnings, numpy as np  
warnings.filterwarnings("ignore", category=DeprecationWarning)  
if hasattr(np, "trapezoid"):  
    try:  
        np.trapz \= np.trapezoid  
    except Exception:  
        pass

\# \--- Resolve output dir from earlier blocks or create fresh \---  
try:  
    \_OUT \= P1\_out  
except NameError:  
    try:  
        \_OUT \= \_P1ppp\_outdir  
    except NameError:  
        import datetime  
        ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
        \_OUT \= os.path.join("proj1\_results", f"run\_{ts}")  
os.makedirs(os.path.join(\_OUT, "sections"), exist\_ok=True)

\# \--- Required globals from main code (safe fallbacks if user ran this in isolation) \---  
\_pairs \= \['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'\]  
\_counts \= EXTERNAL\_COUNTS.copy()

def \_totals(cdict): return {k: sum(v.values()) for k,v in cdict.items()}  
\_N\_by \= \_totals(\_counts)

\# Pull symbolic frames  
\_sy \= symbolic\_patch\_angles()  
\_A\_sym \= so3\_from\_z\_and\_zyz(\_sy\["A"\]\["Z"\], \_sy\["A"\]\["ZYZ"\])  
\_B\_sym \= so3\_from\_z\_and\_zyz(\_sy\["B"\]\["Z"\], \_sy\["B"\]\["ZYZ"\])

\# T from observed counts  
\_T\_obs, \_a\_obs, \_b\_obs \= counts\_to\_T\_and\_singles(\_counts)

\# Fallback in case chsh\_from\_T name is missing (it shouldn't be)  
def \_chsh\_from\_T(T):  
    try:  
        return chsh\_from\_T(T)  
    except NameError:  
        M \= T.T @ T  
        w,\_ \= np.linalg.eigh(M)  
        w \= np.sort(w)\[::-1\]  
        S \= float(2.0\*math.sqrt(max(0.0, w\[0\]+w\[1\])))  
        return S, {"S\_pred":S}

S\_obs, \_ \= \_chsh\_from\_T(\_T\_obs)

\# \--- Model predictors in fixed symbolic frames \---  
def \_clampE(x): return max(-0.999999, min(0.999999, float(x)))  
def \_probs\_from\_E(E):  
    E \= \_clampE(E); pd=(1+E)/4.0; po=(1-E)/4.0  
    return {'00':pd,'01':po,'10':po,'11':pd}

def \_logL\_counts\_probs(counts, probs):  
    L=0.0  
    for k in ('00','01','10','11'):  
        p=max(probs\[k\],1e-15); L+= counts\[k\]\*math.log(p)  
    return float(L)

def \_logL\_for\_Tpred\_with\_counts(Tpred, counts\_tbl):  
    total=0.0  
    for pair in \_pairs:  
        i,j \= PAIR2IDX\[pair\]  
        total \+= \_logL\_counts\_probs(counts\_tbl\[pair\], \_probs\_from\_E(Tpred\[i,j\]))  
    return float(total)

def \_Tpred\_Werner(p):  
    return \_A\_sym.T @ np.diag(\[p,-p,p\]) @ \_B\_sym.T

\# BellDiag3 by projecting measured T into symbolic frame and taking its diag  
\_Msym\_obs \= \_A\_sym @ \_T\_obs @ \_B\_sym  
\_c\_hat \= np.array(\[\_Msym\_obs\[0,0\], \_Msym\_obs\[1,1\], \_Msym\_obs\[2,2\]\], float)  
\_Tpred\_BD3 \= \_A\_sym.T @ np.diag(\_c\_hat) @ \_B\_sym.T

\# Pull Werner posterior grid from earlier if present; else build quickly  
try:  
    \_grid, \_post, \_p\_mode  
except NameError:  
    \_grid \= np.linspace(-1.0, 1.0, 2001\)  
    \_LLW \= np.array(\[\_logL\_for\_Tpred\_with\_counts(\_Tpred\_Werner(p), \_counts) for p in \_grid\], float)  
    \_post \= np.exp(\_LLW \- \_LLW.max())  
    \_post /= np.trapz(\_post, \_grid)  
    \_p\_mode \= float(\_grid\[int(np.argmax(\_post))\])

\# \--- Sim helper: sample counts under given Tpred \---  
\_rng \= np.random.default\_rng(777)  
def \_sample\_counts\_from\_T(Tpred, rng):  
    out={}  
    for pair in \_pairs:  
        N \= \_N\_by\[pair\]; i,j \= PAIR2IDX\[pair\]  
        P \= \_probs\_from\_E(Tpred\[i,j\])  
        ks=('00','01','10','11')  
        draws=rng.multinomial(N,\[P\[k\] for k in ks\])  
        out\[pair\]={k:int(n) for k,n in zip(ks,draws)}  
    return out

\# \--- Parametric predictive intervals for S under each generator \---  
def \_parametric\_S(Tpred, B=400, seed=101):  
    rng=np.random.default\_rng(seed)  
    S\_list=\[\]  
    for \_ in range(B):  
        ctbl=\_sample\_counts\_from\_T(Tpred, rng)  
        T\_b,\_,\_ \= counts\_to\_T\_and\_singles(ctbl)  
        S\_b,\_ \= \_chsh\_from\_T(T\_b)  
        S\_list.append(S\_b)  
    v=np.sort(np.array(S\_list,float))  
    return {  
        "mean": float(v.mean()),  
        "median": float(np.median(v)),  
        "lo95": float(v\[int(0.025\*(len(v)-1))\]),  
        "hi95": float(v\[int(0.975\*(len(v)-1))\]),  
        "B": int(len(v))  
    }

S\_pred\_W, \_ \= \_chsh\_from\_T(\_Tpred\_Werner(\_p\_mode))  
S\_pred\_B, \_ \= \_chsh\_from\_T(\_Tpred\_BD3)

res\_S\_W \= \_parametric\_S(\_Tpred\_Werner(\_p\_mode), B=400, seed=101)  
res\_S\_B \= \_parametric\_S(\_Tpred\_BD3,              B=400, seed=102)

\# write artifacts  
for lab, res in (("Werner",res\_S\_W), ("BellDiag3",res\_S\_B)):  
    fp \= os.path.join(\_OUT,"sections",f"parametric\_S\_under\_{lab}.json")  
    with open(fp,"w",encoding="utf-8") as f: json.dump({"generator":lab,"S\_obs":float(S\_obs),  
                                                        "S\_pred\_point": float(S\_pred\_W if lab=="Werner" else S\_pred\_B),  
                                                        \*\*res}, f, indent=2)  
    print(f"\[Project1 artifacts++++++\] wrote: {fp}")

\# \--- Posterior predictive checks (per-basis KL) for both models \---  
def \_KL\_basis(counts\_tbl, Tpred):  
    out={}  
    for pair in \_pairs:  
        N \= sum(counts\_tbl\[pair\].values())  
        phat \= {k: max(1e-15, counts\_tbl\[pair\]\[k\]/N) for k in ('00','01','10','11')}  
        P \= {k: max(1e-15, \_probs\_from\_E(Tpred\[PAIR2IDX\[pair\]\[0\], PAIR2IDX\[pair\]\[1\]\])\[k\]) for k in phat}  
        out\[pair\] \= float(sum(phat\[k\]\*math.log(phat\[k\]/P\[k\]) for k in phat))  
    return out

def \_ppc\_KL(Tpred, B=300, seed=303):  
    rng=np.random.default\_rng(seed)  
    \# observed KL  
    KL\_obs \= \_KL\_basis(\_counts, Tpred)  
    \# simulate KL distribution  
    sims \= {pair: \[\] for pair in \_pairs}  
    for \_ in range(B):  
        ctbl \= \_sample\_counts\_from\_T(Tpred, rng)  
        KL\_sim \= \_KL\_basis(ctbl, Tpred)  
        for pair in \_pairs:  
            sims\[pair\].append(KL\_sim\[pair\])  
    \# summarize & write  
    rows=\[\]  
    for pair in \_pairs:  
        v \= np.array(sims\[pair\], float); v.sort()  
        lo \= float(v\[int(0.05\*(len(v)-1))\]); hi \= float(v\[int(0.95\*(len(v)-1))\])  
        mu \= float(v.mean())  
        p\_gte \= float(np.mean(v \>= KL\_obs\[pair\]))  
        p\_lte \= float(np.mean(v \<= KL\_obs\[pair\]))  
        rows.append(\[pair, KL\_obs\[pair\], mu, lo, hi, p\_gte, p\_lte\])  
    return rows

\_ppc\_W \= \_ppc\_KL(\_Tpred\_Werner(\_p\_mode), B=300, seed=303)  
\_ppc\_B \= \_ppc\_KL(\_Tpred\_BD3,              B=300, seed=304)

def \_write\_ppc(rows, label):  
    fp \= os.path.join(\_OUT, "sections", f"ppc\_kl\_per\_basis\_{label}.csv")  
    with open(fp,"w",newline="",encoding="utf-8") as f:  
        w=csv.writer(f); w.writerow(\["pair","KL\_obs","KL\_mean\_sim","q05\_sim","q95\_sim","p\_value\_greater\_eq","p\_value\_less\_eq"\])  
        w.writerows(rows)  
    print(f"\[Project1 artifacts++++++\] wrote: {fp}")

\_write\_ppc(\_ppc\_W, "Werner")  
\_write\_ppc(\_ppc\_B, "BellDiag3")

\# \--- Compact run snapshot bundler for easy diffing \---  
snap \= {  
    "version":"proj1.snap.v1",  
    "dirs":{"root": \_OUT, "sections": os.path.join(\_OUT,"sections")},  
    "observed":{  
        "S\_obs": float(S\_obs),  
        "T\_diag": \[float(\_T\_obs\[0,0\]), float(\_T\_obs\[1,1\]), float(\_T\_obs\[2,2\])\],  
        "N\_per\_pair": \_N\_by  
    },  
    "models":{  
        "Werner":{  
            "p\_mode": float(\_p\_mode),  
            "S\_pred": float(S\_pred\_W),  
            "parametric\_S\_CI": res\_S\_W  
        },  
        "BellDiag3":{  
            "c\_hat": \[float(\_c\_hat\[0\]), float(\_c\_hat\[1\]), float(\_c\_hat\[2\])\],  
            "S\_pred": float(S\_pred\_B),  
            "parametric\_S\_CI": res\_S\_B  
        }  
    },  
    "artifacts":{  
        "parametric\_S\_Werner": f"sections/parametric\_S\_under\_Werner.json",  
        "parametric\_S\_BellDiag3": f"sections/parametric\_S\_under\_BellDiag3.json",  
        "ppc\_Werner": f"sections/ppc\_kl\_per\_basis\_Werner.csv",  
        "ppc\_BellDiag3": f"sections/ppc\_kl\_per\_basis\_BellDiag3.csv"  
    }  
}  
fp\_snap \= os.path.join(\_OUT,"sections","snap\_project1.json")  
with open(fp\_snap,"w",encoding="utf-8") as f: json.dump(snap, f, indent=2)  
print(f"\[Project1 artifacts++++++\] wrote: {fp\_snap}")

print("— PARAMETRIC S intervals \+ PPC KL \+ SNAPSHOT — done.")  
\# \===================== End Project1++++++ \=====================

\# \===================== Project1+++++++ — COUNTS INGESTOR & RE-RUN (ADD ONLY) \=====================  
\# Paste at the very bottom. No edits to existing code. Works via unknown CLI flags or env vars.

import os, sys, csv, json, math, warnings, datetime, numpy as np  
warnings.filterwarnings("ignore", category=DeprecationWarning)  
if hasattr(np, "trapezoid"):  
    try: np.trapz \= np.trapezoid  
    except Exception: pass

\# \---------- helpers to pull flags/env (no changes to parse\_args) \----------  
def \_flag\_val(name):  
    if name in sys.argv:  
        i \= sys.argv.index(name)  
        if i+1 \< len(sys.argv): return sys.argv\[i+1\]  
    return None

\_ingest\_csv  \= \_flag\_val("--ingest-csv")  or os.environ.get("QUANTUMCALPRO\_COUNTS\_CSV")  
\_ingest\_json \= \_flag\_val("--ingest-json") or os.environ.get("QUANTUMCALPRO\_COUNTS\_JSON")

\_pairs \= \['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'\]

def \_parse\_counts\_csv(path):  
    out \= {p:{'00':0,'01':0,'10':0,'11':0} for p in \_pairs}  
    with open(path, 'r', newline='') as f:  
        r \= csv.DictReader(f)  
        for row in r:  
            pair \= row\['pair'\].strip().upper()  
            if pair in out:  
                out\[pair\] \= {k:int(float(row\[k\])) for k in ('00','01','10','11')}  
    return out

def \_parse\_counts\_json(src):  
    if isinstance(src, str) and os.path.isfile(src):  
        with open(src,'r') as f: obj \= json.load(f)  
    elif isinstance(src, str):  
        obj \= json.loads(src)  \# env string  
    else:  
        obj \= src  
    out \= {}  
    for p in \_pairs:  
        v \= obj\[p\]  
        out\[p\] \= {k:int(v\[k\]) for k in ('00','01','10','11')}  
    return out

def \_totals(cdict): return {k: sum(v.values()) for k,v in cdict.items()}

def \_clampE(x): return max(-0.999999, min(0.999999, float(x)))  
def \_probs\_from\_E(E):  
    E \= \_clampE(E); pd=(1+E)/4.0; po=(1-E)/4.0  
    return {'00':pd,'01':po,'10':po,'11':pd}

def \_chsh\_from\_T\_safe(T):  
    try:  
        return chsh\_from\_T(T)  
    except Exception:  
        M \= T.T @ T  
        w,\_ \= np.linalg.eigh(M); w \= np.sort(w)\[::-1\]  
        S \= float(2.0\*math.sqrt(max(0.0, w\[0\]+w\[1\])))  
        return S, {"S\_pred":S}

def \_write\_csv(path, header, rows):  
    os.makedirs(os.path.dirname(path), exist\_ok=True)  
    with open(path, "w", newline="", encoding="utf-8") as f:  
        w \= csv.writer(f); w.writerow(header); w.writerows(rows)

\# \---------- main ingest path \----------  
if \_ingest\_csv or \_ingest\_json:  
    \# 1\) read counts  
    CNT \= \_parse\_counts\_csv(\_ingest\_csv) if \_ingest\_csv else \_parse\_counts\_json(\_ingest\_json)  
    N\_by \= \_totals(CNT)

    \# 2\) compute core tensors/metrics with your existing functions  
    T,a,b \= counts\_to\_T\_and\_singles(CNT)  
    S, chsh \= \_chsh\_from\_T\_safe(T)  
    RA, Sigma, RB \= proper\_svd(T)  
    T\_after \= RA.T @ T @ RB  
    Sigma\_diag \= np.array(\[Sigma\[0,0\], Sigma\[1,1\], Sigma\[2,2\]\], float)

    rho\_lin \= rho\_from\_abT(a,b,T)  
    rho\_psd \= project\_to\_psd(rho\_lin.copy())  
    rho\_mle, iters, delta \= mle\_tomography(CNT, max\_iters=300, tol=1e-10)

    Fphi\_T  \= Fphi\_from\_T(T)  
    Fphi\_psd \= fidelity(rho\_psd, bell\_phi\_plus())  
    Fphi\_mle \= fidelity(rho\_mle, bell\_phi\_plus())  
    C\_psd \= concurrence(rho\_psd); C\_mle \= concurrence(rho\_mle)  
    N\_psd \= negativity(rho\_psd);  N\_mle \= negativity(rho\_mle)  
    P\_psd \= purity(rho\_psd);      P\_mle \= purity(rho\_mle)

    \# 3\) output dir  
    ts \= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
    OUT \= os.path.join("proj1\_ingest\_results", f"run\_{ts}")  
    os.makedirs(os.path.join(OUT,"sections"), exist\_ok=True)

    \# 4\) print concise ingest banner  
    print("\\n================== INGEST RUN — Reality Transduction (ADD-ONLY) \==================")  
    print(f"Source: \--ingest-{'csv' if \_ingest\_csv else 'json'}  →  {OUT}")  
    print(f"S (from T) \= {S:0.4f}")  
    print(f"F(Φ⁺): T={Fphi\_T:0.4f}  ρ\_psd={Fphi\_psd:0.4f}  ρ\_mle={Fphi\_mle:0.4f}")  
    print(f"Concurrence(psd/mle) \= {C\_psd:0.4f}/{C\_mle:0.4f}   Negativity \= {N\_psd:0.4f}/{N\_mle:0.4f}")  
    print("Σ diag \=", \[f"{float(Sigma\_diag\[i\]):+0.4f}" for i in range(3)\])

    \# 5\) save observed counts  
    \_write\_csv(os.path.join(OUT,"sections","observed\_counts.csv"),  
               \["pair","N","n00","n01","n10","n11"\],  
               \[\[p, N\_by\[p\], CNT\[p\]\['00'\], CNT\[p\]\['01'\], CNT\[p\]\['10'\], CNT\[p\]\['11'\]\] for p in \_pairs\])

    \# 6\) dyadic transduction table (p\_even \= (1+E)/2, with normal CI at z=2.24)  
    z \= 2.24  
    rows=\[\]  
    for p in \_pairs:  
        i,j \= PAIR2IDX\[p\]; E \= float(T\[i,j\]); N \= N\_by\[p\]  
        pe \= (1.0+E)/2.0  
        se \= math.sqrt(max(1e-15, pe\*(1-pe)/N))  
        lo,hi \= pe \- z\*se, pe \+ z\*se  
        rows.append(\[p, N, f"{E:+0.4f}", f"{pe:0.6f}", f"{lo:0.6f}", f"{hi:0.6f}"\])  
    \_write\_csv(os.path.join(OUT,"sections","dyadic\_transduction\_ingest.csv"),  
               \["pair","N","E","p\_even","CI\_lo","CI\_hi"\], rows)

    \# 7\) symbolic frames \+ two simple generators (Werner p, BellDiag3 with projected c)  
    ANG \= symbolic\_patch\_angles()  
    A\_sym \= so3\_from\_z\_and\_zyz(ANG\["A"\]\["Z"\], ANG\["A"\]\["ZYZ"\])  
    B\_sym \= so3\_from\_z\_and\_zyz(ANG\["B"\]\["Z"\], ANG\["B"\]\["ZYZ"\])  
    Msym \= A\_sym @ T @ B\_sym  
    c\_hat \= np.array(\[Msym\[0,0\], Msym\[1,1\], Msym\[2,2\]\], float)

    def \_T\_Werner(p): return A\_sym.T @ np.diag(\[p,-p,p\]) @ B\_sym.T  
    def \_logL\_counts\_probs(counts, probs):  
        L=0.0  
        for k in ('00','01','10','11'):  
            L \+= counts\[k\]\*math.log(max(probs\[k\],1e-15))  
        return float(L)  
    def \_logL\_Tpred(Tpred, counts\_tbl):  
        s=0.0  
        for pair in \_pairs:  
            i,j \= PAIR2IDX\[pair\]  
            s \+= \_logL\_counts\_probs(counts\_tbl\[pair\], \_probs\_from\_E(float(Tpred\[i,j\])))  
        return s

    \# posterior grid for Werner p  
    grid \= np.linspace(-1.0, 1.0, 2001\)  
    LL \= np.array(\[\_logL\_Tpred(\_T\_Werner(p), CNT) for p in grid\], float)  
    post \= np.exp(LL \- LL.max()); post /= np.trapz(post, grid)  
    p\_mode \= float(grid\[int(np.argmax(post))\])

    \# BellDiag3 Tpred via projected c\_hat  
    T\_BD3 \= A\_sym.T @ np.diag(c\_hat) @ B\_sym.T  
    S\_W, \_ \= \_chsh\_from\_T\_safe(\_T\_Werner(p\_mode))  
    S\_B, \_ \= \_chsh\_from\_T\_safe(T\_BD3)

    \# 8\) parametric predictive CI for S  
    def \_sample\_counts\_from\_T(Tpred, rng):  
        out={}  
        for pair in \_pairs:  
            N \= N\_by\[pair\]; i,j \= PAIR2IDX\[pair\]  
            P \= \_probs\_from\_E(float(Tpred\[i,j\])); ks=('00','01','10','11')  
            draws \= rng.multinomial(N, \[P\[k\] for k in ks\])  
            out\[pair\] \= {k:int(n) for k,n in zip(ks,draws)}  
        return out  
    def \_parametric\_S(Tpred, B=300, seed=202):  
        rng=np.random.default\_rng(seed); arr=\[\]  
        for \_ in range(B):  
            Tb,\_,\_ \= counts\_to\_T\_and\_singles(\_sample\_counts\_from\_T(Tpred, rng))  
            Sb,\_ \= \_chsh\_from\_T\_safe(Tb); arr.append(Sb)  
        v=np.sort(np.array(arr,float))  
        return {"B":int(len(v)),"mean":float(v.mean()),"median":float(np.median(v)),  
                "lo95":float(v\[int(0.025\*(len(v)-1))\]), "hi95":float(v\[int(0.975\*(len(v)-1))\])}

    CI\_W \= \_parametric\_S(\_T\_Werner(p\_mode), B=300, seed=202)  
    CI\_B \= \_parametric\_S(T\_BD3,            B=300, seed=203)

    \# 9\) predictions (per-basis probabilities & counts) for both models  
    def \_pred\_rows(Tpred, label):  
        rows=\[\]  
        for pair in \_pairs:  
            N \= N\_by\[pair\]; i,j \= PAIR2IDX\[pair\]  
            P \= \_probs\_from\_E(float(Tpred\[i,j\]))  
            rows.append(\[pair, N, P\['00'\],P\['01'\],P\['10'\],P\['11'\],  
                         int(round(N\*P\['00'\])),int(round(N\*P\['01'\])),  
                         int(round(N\*P\['10'\])),int(round(N\*P\['11'\]))\])  
        \_write\_csv(os.path.join(OUT,"sections",f"pred\_{label}.csv"),  
                   \["pair","N","p00","p01","p10","p11","n00\_pred","n01\_pred","n10\_pred","n11\_pred"\], rows)

    \_pred\_rows(\_T\_Werner(p\_mode), "counts\_Werner\_ingest")  
    \_pred\_rows(T\_BD3,              "counts\_BellDiag3\_ingest")

    \# 10\) snapshot JSON  
    snap \= {  
        "version":"proj1.ingest.v1",  
        "out\_dir": OUT,  
        "observed":{"S": float(S), "Fphi\_T": float(Fphi\_T),  
                    "diag\_T": \[float(T\[0,0\]),float(T\[1,1\]),float(T\[2,2\])\],  
                    "N\_per\_pair": N\_by},  
        "models":{  
            "Werner":{"p\_mode": float(p\_mode), "S\_pred": float(S\_W), "S\_CI": CI\_W},  
            "BellDiag3":{"c\_hat": \[float(c\_hat\[0\]),float(c\_hat\[1\]),float(c\_hat\[2\])\],  
                         "S\_pred": float(S\_B), "S\_CI": CI\_B}  
        }  
    }  
    with open(os.path.join(OUT,"sections","ingest\_snapshot.json"),"w",encoding="utf-8") as f:  
        json.dump(snap, f, indent=2)

    print(f"\[ingest\] wrote: {os.path.join(OUT,'sections','observed\_counts.csv')}")  
    print(f"\[ingest\] wrote: {os.path.join(OUT,'sections','dyadic\_transduction\_ingest.csv')}")  
    print(f"\[ingest\] wrote: {os.path.join(OUT,'sections','pred\_counts\_Werner\_ingest.csv')}")  
    print(f"\[ingest\] wrote: {os.path.join(OUT,'sections','pred\_counts\_BellDiag3\_ingest.csv')}")  
    print(f"\[ingest\] wrote: {os.path.join(OUT,'sections','ingest\_snapshot.json')}")  
    print("INGEST RE-RUN — done.")  
\# \===================== End Project1+++++++ — COUNTS INGESTOR (ADD ONLY) \=====================