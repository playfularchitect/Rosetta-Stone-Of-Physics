OG Colab  -  https://colab.research.google.com/drive/1jslpb0Sf5OGFiki2n_bSOHBhW1Apdw4y?usp=sharing

What we did (big picture)

stopped arguing about models. we worked only with observables that reality already fences:
mass (mχ), warmth (free-streaming cutoff), direct-detect cross-sections (σ_SI with nuclei / σ_e if we needed), annihilation at recombination (⟨σv⟩_rec via CMB/dwarfs), and self-interaction (σ/m).

built a giant yes/no map (“survivor set”). we gridded the observable space and marked each point as alive (consistent with data fences) or dead (ruled out). no model intermediates, just data-grade cuts.

made it fast.

Used my custom GPU/ kernels (A100-ready) for the physics slices and diagnostics.

CPU AVX2/OpenMP backends to crunch masks exactly and cross-check bit-for-bit.
speed isn’t fluff here, I absolutely despise floating point so use exact math by using my kernels — it let us iterate, sweep, and verify without cutting corners.

iterated the constraints → survivors → “strike plans”. we tried progressively stronger, still-defensible fences (structure, SIDM, CMB/dwarfs, DD) and watched what survived. then we hardened until nothing did — and recorded the exact numbers where a resurrection would first occur.

the simple, load-bearing facts we ended on

date: November 6th, 2025
grid: (121 × 51 × 81 × 61 × 41) over (mass, warmth, σ_SI, ⟨σv⟩_rec, σ/m)

hard fences we enforced (data-native)

warmth (free-streaming): m_WDM ≥ 10 keV
(i.e., no warm DM that erases small-scale power below that)

self-interaction: σ/m ≤ 0.03 cm²/g
(keeps cluster mergers clean; mild SIDM only)

“kill switch” caps at the last bin that died

⟨σv⟩_rec cap: ≤ 2.506×10⁻³¹ cm³/s

σ_SI cap: ≤ 2.435×10⁻⁴⁶ cm²

that mass bin: m ≈ 0.0416 GeV

local resurrection threshold (that specific mass bin)

equal-relax factor x* (scale both caps by the same factor): x* ≈ 3.9904
→ below this, that bin stays dead; at/above this, it comes back.

strict global red-line (keep the whole box sealed)

if you scale both caps everywhere by ×5, survivors reappear.

our sanity retest (equal relax):

×1 → 0 survivors

×3 → 0 survivors

×5 → survivors appear (first count ≈ 107,085)

×6 → more survivors (≈ 217,800)

so: the sealed box (no survivors) is guaranteed if you keep warmth ≥10 keV, σ/m ≤0.03, and don’t relax both caps beyond ×5 globally.

what that proves (in our “reality-first” frame)

given those empirical fences, the entire observable space we gridded is empty (no viable points).

the first place the box breaks open is known: m ≈ 0.0416 GeV with the caps relaxed by ×3.9904 (locally), or by ×5 if you relax globally.

this is a receipt, not a model: any proposed microphysics must map into this observable space. if it lands outside our sealed box, it’s inconsistent with the fences; if it lands inside, it only survives if you also accept a relax beyond our red-line (which we’ve flagged explicitly).

how we know we didn’t cheat

dual implementations (GPU + AVX2) agree.

progressive strikes (v1→v5) show monotonic removal with printed deltas and per-mass tallies.

we produced the Public Proof Deck + a Belt & Suspenders Freeze: paths, hashes, and a minimal meta json so anyone can re-run and verify the same numbers.

where the “blame” lies if someone disagrees

we didn’t assume a model; we assumed data fences. to disagree, you must move one of:

the warmth floor (from structure/lensing data),

the SIDM ceiling (from mergers),

the CMB/dwarfs annihilation envelope (⟨σv⟩_rec),

the direct-detection σ_SI envelope.

we showed exactly how far each would need to move to resurrect points (local ×3.99, global ×5). that’s the measurable gap reality must “give back” before dark corners reopen.

practical “triangulate reality” takeaways

to keep it sealed:
keep m_WDM ≥ 10 keV, σ/m ≤ 0.03 cm²/g, and don’t relax both caps beyond ×5 globally.

to try to resurrect (on purpose, for tests):
aim experiments that would tighten or relax the critical caps by order-one factors near sub-GeV mass: this is where the first revivals show up (we listed the first resurrected bins).

to pressure the border without models:

push σ_SI reach (DD) in the sub-GeV to few-GeV window,

push ⟨σv⟩_rec (CMB/21-cm/dwarfs combos) at low masses,

keep small-scale structure analyses honest (they set the 10 keV floor everyone is leaning on),

watch merger analyses; stronger σ/m ceilings make sealing even easier.

limitations (straight talk)

the warmth floor (10 keV) and exact caps are as-implemented envelopes. stronger or weaker literature choices shift the borders; our framework shows by how much (the × factors), not just “pass/fail”.

annihilation was treated in a channel-agnostic, s-wave flavored cap via p_ann; truly exotic invisible/velocity-suppressed sectors can dodge that — but then you must show their observable image still obeys the other fences.

this is a finite grid (dense, but finite). if someone needs even finer bins, we can densify — our backends are built for it.

what to hand people (receipts)

Public Proof Deck (numbers only + retests + red-line).

Freeze meta with the SHA-256 for the TXT and the exact values we claim.

Receipts: strike logs, resurrection margins, and the last-pin audit.

if anyone wants to fight a number, they can swap a fence and re-run; the machinery tells you how much it must move to matter.
