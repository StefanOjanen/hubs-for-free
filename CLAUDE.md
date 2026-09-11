# CLAUDE.md

## What this project is now

This folder began as the Emergent Coordinator Hypothesis (ECH). That framing
is retired: the headline results were unsupported (published symbolic law
R^2 = 0.982 absent from the project's own PySR logs, archived best fit
R^2 = 0.014; hub/eigengap/clustering signatures reproduce on random matrices;
depth-wise "crystallization" follows from attention locality). The active
deliverable is paper.md, "Hubs for Free: What Commutator-Based Analyses of
Attention Discover in Random Matrices". Old ECH documents are historical
case-study material; do not treat their claims as established.

## Ground rules for all work in this repo

1. Every number in any document must be regenerable from a script and
   traceable to a saved artifact. This includes numbers Claude produces:
   write the artifact first, then cite it.
2. Null models before interpretation: random matrices from the constraint
   set, marginal-matched surrogates, untrained same-architecture model,
   convention changes (sign flips, relabeling, index permutation).
3. Report adaptive fits against the best fit the identical search achieves
   on matched noise; label searched exhibits as searched.
4. No physics vocabulary (gauge, Lie algebra, phase transition) unless the
   mathematical property is verified and a null is shown.
5. Single-input results characterize (model, input); report distributions.

## Artifact status

experiments.py, results.json, and figures/fig1 to fig6 were reimplemented
from paper.md's specifications on 2026-09-02 (the August originals were
never committed); the paper's synthetic numbers are the regenerated values
and CI (.github/workflows/regenerate.yml) checks they regenerate. Still
author-held and missing from the repo: the local pilot (gptmini.py,
train_chunk.py, analyze_local.py, local_results.json, fig7). Section 7's
pilot numbers therefore have no committed artifact; remind the user to
upload them or mark the pilot as unreproduced.

Compute since 2026-09-10 is the author's M1 Max (32 GB) through Apple MPS;
the device and precision policy lives in hubsfree/adapters.py (float32
whenever the weights fit, bfloat16 above about 4B parameters) and the
measured fidelity against the CPU float32 protocol is in
alignment_study/platform_fidelity.json and RUNLOG.md. Every model run
writes results per model as it completes.

Rounds and their artifacts (all preregistered before execution, evaluation
scripts committed before results): 1 (gram_theorem.json,
robustness_addendum.json), 2 (heldout_round.json), 3
(scale_round_results.json, scale_partial/), 5 dynamics (dynamics/,
dynamics_results.json), 6 rerun (rerun/, rerun_results.json), 7
head-merging (merge/, merge_results.json; both primary clauses failed), 8
derivation (derivation/, derivation_results.json; gate failed at R^2
0.55, sink-set post hoc 0.75). Audit base results: audits/clark2019/,
audits/chai/, audits/dewage2026/, audits/retrieval_heads/; criteria in
audits/PREREGISTRATION4_DRAFT.md, batteries gated on public registration.
Registered failures are reported as failures in NOTE.md, README.md and
manuscript.md; never reframe them.

## Style

Direct, measured, no em-dashes, no hype. Claims sized to evidence. The paper
critiques methods, not people.
