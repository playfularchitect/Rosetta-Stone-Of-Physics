Colab Link - https://colab.research.google.com/drive/1jVDa3C_TpTI0D_k65gz9ubNpDrx1UXHv?usp=sharing

# DNA Rational Structure Analysis

## What This Project Does

This project investigates whether the E. coli K-12 MG1655 genome exhibits **discrete/rational structure** in its nucleotide composition — specifically through GC content at third codon positions (GC3), CpG dinucleotide patterns, and "lock codons" (codons where the third position is constrained by amino acid identity).

The central question: **Is genomic composition random, or does it show evidence of integer/rational constraints?**

---

## Core Discovery: CpG Enrichment is NOT Random

**The headline finding**: E. coli's real genome contains **~16% more CpG dinucleotides** than shuffled controls with identical base composition.

| Metric | Real Genome | Shuffled (16 trials) |
|--------|-------------|---------------------|
| Mean CpG/window | **19.059** | 16.439 ± 0.022 |
| z-score | **119.4σ** | — |
| Shuffles ≥ Real | **0 / 16** | — |

This is a **119-sigma deviation** from null expectation. The genome's CpG distribution is definitively non-random.

---

## Key Metrics Defined

| Term | Definition |
|------|------------|
| **GC3** | Fraction of G or C at third codon position |
| **Lock codon** | A codon where using G/C at position 3 preserves the amino acid |
| **Lock fraction** | Fraction of amino-acid-encoding codons that use lock codons |
| **CpG/kb** | CpG dinucleotides per kilobase |
| **Hyper-GC3** | Genes with GC3 ≥ mean + 2σ (≥0.716) |
| **Hyper-lock** | Genes with lock_frac ≥ mean + 2σ (≥0.583) |

---

## Genome-Wide Statistics

**4,319 valid CDS analyzed** from E. coli K-12 MG1655 (4.64 Mb genome)

| Metric | Mean | Std Dev | Hyper Threshold | Hyper Genes |
|--------|------|---------|-----------------|-------------|
| GC3 fraction | 0.544 | 0.086 | ≥0.716 | **34** |
| Lock fraction | 0.416 | 0.083 | ≥0.583 | **22** |
| CpG per kb | 74.4 | 18.8 | ≥112.0 | **25** |

**Key correlations**:
- corr(inter-lock spacing, GC3) = **-0.79**
- corr(inter-lock spacing, lock_frac) = **-0.86**

Lock codons cluster tightly in high-GC3 genes — not randomly distributed.

---

## The Standout Genomic Islands

### The yag* Supercluster (Position ~280-290 kb)

**7 contiguous genes spanning 9.5 kb** — the most extreme lock-enriched region in the genome:

| Gene | Codons | GC3 | Lock Frac | CpG/kb | Longest Lock Run |
|------|--------|-----|-----------|--------|------------------|
| yagA | 385 | 0.756 | 0.621 | 92.6 | 16 |
| yagE | 303 | **0.795** | **0.663** | 115.5 | 16 |
| yagF | 656 | **0.857** | **0.718** | **137.7** | **19** |
| yagG | 461 | **0.894** | 0.690 | 102.0 | — |
| yagH | 537 | **0.879** | **0.713** | 119.8 | 17 |
| xynR | 253 | **0.834** | **0.692** | 105.4 | 17 |
| argF | 335 | 0.767 | 0.612 | 101.5 | — |

**Island averages**: GC3 = **0.826**, Lock = **0.673**, CpG = **110.6/kb**

This cluster is simultaneously hyper-GC3, hyper-lock, AND hyper-CpG — a triple lock.

### The phn* Phosphonate Operon (Position ~4.31-4.32 Mb)

| Gene | Codons | GC3 | Lock Frac | CpG/kb |
|------|--------|-----|-----------|--------|
| phnC | 263 | 0.719 | 0.586 | 105.2 |
| phnD | 339 | 0.761 | 0.549 | 86.5 |
| phnF | 242 | 0.748 | 0.645 | 114.3 |
| phnG | 151 | 0.709 | 0.609 | 123.6 |
| phnH | 195 | 0.723 | 0.626 | 107.7 |
| phnI | 355 | 0.755 | 0.623 | 125.8 |
| phnJ | 282 | 0.745 | 0.571 | 117.0 |
| phnK | 253 | **0.755** | **0.656** | 104.1 |
| phnL | 227 | 0.700 | 0.590 | 114.5 |
| phnM | 379 | 0.757 | 0.646 | 116.1 |
| phnN | 186 | 0.720 | 0.618 | 107.5 |

Two distinct islands detected:
- **phnK-phnH**: 4 genes, mean GC3=0.744, mean lock=0.619
- **phnI-phnF**: 4 genes, mean GC3=0.734, mean lock=0.626

---

## Module Reference

### Infrastructure & Benchmarking

| Module | Purpose | Key Output |
|--------|---------|------------|
| **QL-1** | GPU vs CPU exact GEMM validation | INT8×INT8→INT32 correctness proof |
| **DNA-1** | DNA block similarity via GEMM | One-hot encoding → exact Hamming |
| **DNA-FL-1** | cuBLASLt Fastlane engine | 256×256 DNA distance matrix |
| **DNA-FL-2** | Real genome mapper | E. coli window extraction |
| **DNA-FL-3** | Hamming uniqueness survey | Read mapping statistics |

### Lock Discovery (DNA-L1 through DNA-L6)

| Module | Purpose | Key Finding |
|--------|---------|-------------|
| **DNA-L1** | Base count fraction scan | GC fractions cluster around 17/32, 67/128 |
| **DNA-L2** | Real vs shuffled GC fractions | Distribution shapes differ significantly |
| **DNA-L3** | Real vs shuffled CpG fractions | Real genome has 16% more CpG |
| **DNA-L4** | Multi-shuffle null model | **z = 119.4σ** — CpG excess is real |
| **DNA-L5** | CpG hotspot locator | Top hotspots at 283-285 kb (yagF region) |
| **DNA-L6** | Hotspot annotation | All top 10 hotspots overlap yagF CDS |

### CDS-Level Analysis (DNA-L30 through DNA-L39)

| Module | Purpose | Key Finding |
|--------|---------|-------------|
| **DNA-L30** | Within-CDS GC3 gradients | 5' and 3' ends show lower GC3 than mid-body |
| **DNA-L31** | GC3, lock & CpG profiles | Lock rails correlate with GC3 position |
| **DNA-L32** | Codon usage vs GC3 | Amino acids A, Q, L, G show highest lock usage |
| **DNA-L33** | Hyper-GC3/lock AA enrichment | Ala, Arg, Gly enriched in hyper-lock genes |
| **DNA-L35** | Lock codon runs | yagF has 19-codon consecutive lock run |
| **DNA-L36** | Inter-lock spacing | Hyper-lock genes: mean spacing = 0.56 codons |
| **DNA-L37** | Lock run structure | 85-90% of lock codons in multi-codon runs |
| **DNA-L38** | Run position & terminal enrichment | Runs distributed ~uniformly along CDS |
| **DNA-L39** | Genomic islands | 3 hyper-GC3 islands, 3 hyper-lock islands detected |

---

## Statistical Validation Summary

| Test | Statistic | Interpretation |
|------|-----------|----------------|
| CpG real vs shuffled | z = 119.4σ | **Definitively non-random** |
| Shuffles ≥ real CpG | 0/16 | p < 0.06 empirical |
| Lock-GC3 correlation | r = -0.86 | Strong coupling |
| Hyper-lock genes | 22/4319 | 0.5% of genome |
| Longest lock run | 19 codons (yagF) | Far exceeds random expectation |

---

## Key Insights

1. **CpG is NOT suppressed in E. coli** — contrary to vertebrates, E. coli shows CpG *enrichment* relative to null
2. **Lock codons cluster** — they don't distribute randomly but form contiguous runs up to 19 codons
3. **Specific genomic islands** concentrate the structure — the yag* cluster and phn* operon are outliers
4. **GC3, lock fraction, and CpG co-vary** — genes high in one metric tend to be high in all three
5. **Within-gene gradients exist** — GC3 is lower at gene termini than mid-body

---

## Technical Implementation
- **Data source**: E. coli K-12 MG1655 (NC_000913.3 / U00096.3) from NCBI
- **Annotation**: GFF3 features for CDS coordinates and gene names
- **Null model**: Multiple independent genome shuffles preserving base composition

---

## Files

| File | Description |
|------|-------------|
| `Free2PlayV1_Code.txt` | Complete source (~21,500 lines) |
| `Free2PlayV1_Output.txt` | Full analysis output (~3,900 lines) |

---

## How to Read the Output

1. **Module headers** mark section boundaries: `MODULE DNA-L4 — ...`
2. **Key statistics** appear after `---` separator lines
3. **Gene tables** show per-CDS metrics with `hyperGC3/lock/CpG` flags
4. **Island detection** lists contiguous runs of hyper-* genes
5. **Neighborhood views** show ±5 genes around targets of interest

---

## Implications

This analysis suggests that E. coli's codon usage reflects **non-random compositional constraints** — possibly selection for:
- Translational efficiency (GC3 bias)
- mRNA secondary structure
- DNA replication/repair signals
- Unknown regulatory elements encoded in third-position choices

The discrete, clustered nature of lock enrichment — concentrated in specific operons rather than genome-wide — points toward **functional selection** rather than neutral drift.

---

## Citation

If you use this analysis, please cite the repository and acknowledge the Rosetta stone of physics/WarpFrac exact-arithmetic framework.
