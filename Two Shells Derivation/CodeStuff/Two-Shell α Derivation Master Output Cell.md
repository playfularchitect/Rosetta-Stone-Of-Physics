\==============================================================================================================  
Two-Shell α — Master Cell (angles → denominator → Pauli integral → α^{-1} prediction)  
\==============================================================================================================  
|S49|=54  |S50|=84  |S|=138  (expected 54, 84, 138\)

Row-sum witnesses (∑\_{NB} cosθ per source-type) \~ 1.0:  
  49\_007   : \+0.999999999999998  
  49\_236   : \+1.000000000000001  
  50\_017   : \+1.000000000000000  
  50\_055   : \+1.000000000000001  
  50\_345   : \+1.000000000000006

Denominator sum\_NB\_cos2  (rounded) \= 6210  (raw=6209.999999999871)

Computing Pauli one-corner continuum integrals per angle class ...  
  table 49\_007: 18 angle classes  
  table 49\_236: 90 angle classes  
  table 50\_017: 68 angle classes  
  table 50\_055: 30 angle classes  
  table 50\_345: 98 angle classes

\==================== SUMMARY \====================  
Continuum estimate for c\_Pauli: \+0.0015308706  
α^-1 (numerical, continuum)  ≈ 137 \+ c/137 \= 137.0000111742

Rigorous lattice–continuum bracket (global sharp factor \[1, π^2/4\]):  
 c\_Pauli ∈ \[0.0015308706, 0.0037772717\]  
 α^-1  ∈ \[137.0000111742, 137.0000275713\]  
\=================================================

Artifacts written to: ./two\_shell\_artifacts\_20250909-210645  
 \- angles\_49\_007.csv, angles\_49\_236.csv, angles\_50\_017.csv, angles\_50\_055.csv, angles\_50\_345.csv  
 \- denominator.json  
 \- pauli\_integrals.json  
 \- alpha\_prediction.txt

Preview of a few distinct angle-class integrals I\_cont(cosθ):  
  cosθ=-0.989949493661  I\_cont=-1.88508845e+00  
  cosθ=-0.980000000000  I\_cont=-1.86893192e+00  
  cosθ=-0.979591836735  I\_cont=-1.86826770e+00  
  cosθ=-0.969746442770  I\_cont=-1.85221164e+00  
  cosθ=-0.960000000000  I\_cont=-1.83625232e+00

Tips:  
  • For speed/repeatability, keep QUAD\_KIND='gl'. Clenshaw–Curtis nodes are available via QUAD\_KIND='cc'.  
  • Increase NKAPPA/NPHI and MP\_DPS to tighten numerical stability; results are cached per unique cosθ.  
  • The printed bracket is rigorous (global \[1, π^2/4\] lattice factor). Sectorwise lattice bounds can tighten it later.

\==================== TILE-WISE CERTIFIED (CONTINUUM) \====================  
Tiles: K=48, P/half=48  ⇒  c\_Pauli ∈ \[-0.0645410168, 0.0660898001\]  
α^-1 continuum ∈ \[136.9995288977, 137.0004824073\]

(Heuristic tightening, assumes monotonicity of sinc and J1(x)/x on \[0,π\])  
c\_Pauli (heur) ∈ \[-0.0102491491, 0.0133076196\]  
α^-1   (heur) ∈ \[136.9999251887, 137.0000971359\]

\==================== TILE-WISE CERTIFIED (LATTICE via \[1,π²/4\]) \====================  
c\_Pauli ∈ \[-0.1592485759, 0.1630700454\]  
α^-1   ∈ \[136.9988376016, 137.0011902923\]

(Heuristic lattice tightening)  
c\_Pauli (heur) ∈ \[-0.0252887617, 0.0328352352\]  
α^-1   (heur) ∈ \[136.9998154105, 137.0002396732\]

Saved: pauli\_tilewise\_bounds.json  (per-class tile intervals \+ global brackets)
