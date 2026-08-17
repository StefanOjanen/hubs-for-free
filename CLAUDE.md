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

## Missing artifacts still in the claude.ai chat

experiments.py, results.json, figures/ (fig1-fig7 PNGs), gptmini.py,
train_chunk.py, analyze_local.py, local_results.json. These regenerate the
synthetic null battery and the local pilot cited in paper.md. Remind the user
to download them from the chat before submission.

## Style

Direct, measured, no em-dashes, no hype. Claims sized to evidence. The paper
critiques methods, not people.
