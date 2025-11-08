Neutrino Cracker — Plain-English 

What this project is:
A series of quick, print-first studies (“v38–v62”) exploring how DUNE-like long-baseline data could respond to different modeling assumptions. We focus on how much of the neutrino/antineutrino spectral difference (driven by the CP phase δ) survives once you include realistic systematics and correlations.

TL;DR (for non-neutrino folks)

We checked a simple hypothesis (“correlation-only”) and proved it can’t explain the observed deltas.
If you assume all detector/flux systematics are perfectly correlated in energy, the best you can ever reach is a ceiling we call Δ_on ≈ 15.17. All our target summaries (v38–v42) are well below that ceiling, so correlation-only is mathematically ruled out.

You need extra “damping.”
We model that with a factor β (think: additional smearing/multi-scale nuisances) and a robust fraction α (fraction of bins effectively down-weighted by outliers/heavy-tailed noise).
With a single baseline damping (β_base ≈ 0.829) and tag-specific α’s, we exactly close v38–v42.

One α for everything doesn’t work.
A single global α would force all tags to the same Δχ² value. Fitting the four distinct targets then leaves sizeable residuals. We quantified the minimal “capacity drift” (how much you’d have to let the overall scale float) and it’s ~±56%—too big to be a neat one-knob fix.

Splitting neutrino vs antineutrino response is fine.
As long as the total damping stays fixed, you can slide strength between ν and ν̄ over a wide range and still match the targets. Same story per channel (crust/mantle × NO/IO): there’s a healthy Δβ window where everything remains consistent.

δ scans (phase sweeps) land at a minimum around δ ≈ 195° with mild curvature; testing δ = 192° or 198° gives ~1σ–2σ-ish toy separations depending on how robustly you treat outliers/correlations. (These are relative toy sensitivities, not discovery claims.)

Robustness checks:
Swapping in robust (Gaussian+Laplace) priors and sprinkling small outliers keeps the closure intact; higher robust weight generally increases stability and slightly boosts Δχ².

What the knobs mean (no equations needed)

Δ_on / Δ_off: Best-case vs worst-case sensitivity if systematics line up across energy bins (on) or don’t (off).

β (beta): Extra damping beyond those correlations (e.g., finite resolution, multi-scale effects, robust losses).

α (alpha): Fraction of the spectrum that behaves “robustly” (down-weights weird/outlier bins).

Δβ = β_ν − β_ν̄: How differently ν and ν̄ are damped.

r / sys / fν: Exposure scale, per-bin systematics size, and ν:ν̄ running split.

δ (delta): The CP-phase knob; small changes near the minimum lead to small, curved changes in Δχ².

What we did show (strong takeaways)

No-Go: Correlation-only is insufficient (provable upper bound).

Closure: A simple Reality Bridge (β_base plus per-target α) matches all summary targets (v38–v42) to rounding.

Flexibility: ν/ν̄ and per-channel splits have broad feasible ranges while preserving totals.

Single-α Impossibility: You can’t fit all four tags with one α unless you allow large capacity drift (~±56%).

Robustness: Results persist under robust priors and modest outlier stress.

What we did not claim

No discovery of CP violation and no absolute sensitivity forecast.

Flux/cross-section/efficiency shapes were treated in toy form for leverage studies, not as collaboration-final models.

Numbers are relative diagnostics, not publication-grade DUNE projections.

How to read the outputs

JSON summaries (e.g., summary_v**.json): machine-readable snapshots of the key numbers (best fits, deltas, α, β).

PNGs: “print-first” plots—channel scans, combined bars, feasibility/heatmaps, etc.

CSVs: simple tables of δ-scans or parameter grids so you can re-plot in your favorite tool.

If you only skim one:

For the structural claim: see v46 (No-Go bound).

For the constructive fit: see v48–v52 (Reality Bridge closures).

For the limits of a single α: v60–v62.

Practical next steps (if you want to extend)

Swap in collaboration flux×σ×ε models to turn the toy “shape-only” proxies into rate-level predictions.

Tighten the robust mix (tune the Gaussian/Laplace blend) against real residuals.

Stress with realistic systematics (detector blocks, calibration pulls, migration matrices) and verify the Δβ windows remain feasible.

Embed into Asimov+toys with your experiment’s full likelihood for publication-grade numbers.

One-liner Takeaway!

Even under generous correlations, a correlation-only picture can’t reach the observed Δχ² targets. You need extra damping (β) and a robust fraction (α) to bridge toy δ-curvature to the reported summaries; that bridge closes cleanly and robustly across ν/ν̄ and per-channel splits, while a single global α is mathematically incompatible without large capacity drift.
