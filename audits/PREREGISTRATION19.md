# Preregistration 19: vision transformers' attention distance and artifact tokens

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory run.

## Question

Dosovitskiy et al. (ICLR 2021, Section 4.5 and Appendix D.7, Figure 11 on
ViT-L/16) report that attention distance, "computed for 128 example images by
averaging the distance between the query pixel and all other pixels,
weighted by the attention weight", grows with depth, and that in the second
half of the network most heads attend widely. Darcet et al. (ICLR 2024) found
that large vision transformers develop a few high-norm artifact patch tokens
that absorb attention, and that register tokens remove them. If a late-layer
query sends much of its attention to an artifact at an arbitrary place in the
image, its attention distance looks wide whatever the rest of the row does.
How much of the depth increase is that?

## Instrument (`audits/vit/vit_distance.py`)

128 images (Imagenette, a fixed-seed sample of johnowhitaker/imagenette2-320),
each model's own preprocessing configuration applied with PIL and numpy.
Over patch queries and patch keys, positions at patch centers, distances in
pixels: D_all, the attention-weighted mean distance with weights
renormalized over patch keys (the source's definition); D_raw without
renormalization; D_ex, D_all with the layer's artifact columns removed and
the rest renormalized. Artifact columns of an image and layer: patch
columns receiving at least ten times the uniform share of the patch queries'
attention, averaged over heads, at most eight. Depth increase: mean over the
second half of the layers minus the first half. Artifact share of the
increase: (Delta_all - Delta_ex) / Delta_all.

## Development (labeled, `audits/vit/dev/`, 32 images)

DINOv2-S with and without registers: no column passes the artifact rule;
the depth increase is 12.3 and 4.9 pixels. DINOv2-B: two to three artifact
columns per image in the last three layers, holding 0.14 of the
second-half attention; the depth increase is 17.5 pixels and 0.47 of it
disappears when the artifact columns are removed. DINOv2-B with registers:
no late-layer artifacts, increase 4.8 pixels, share -0.02. Darcet et al.
report stronger artifacts in larger models, so the confirmatory test uses
the large models; the thresholds below come from the base models.

## Registered predictions

Models: facebook/dinov2-large, facebook/dinov2-with-registers-large,
google/vit-large-patch16-224 (the source's architecture), and
google/vit-base-patch16-224.
- V1: in DINOv2-L, at least 0.4 of the depth increase of D_all disappears
  when the artifact columns are removed.
- V2: in DINOv2-L, the artifact columns hold at least 0.1 of the second-half
  attention of patch queries.
- V3: in DINOv2-L with registers, the artifact share is at most 0.1 and the
  artifact columns hold less than 0.02 of the second-half attention.
- V4: registers shrink the depth increase: Delta D_all with registers is
  below that of the model without.
- V5: the source's claim reproduces on ViT-L/16: D_all increases from the
  first to the second half and at least half of the heads have a positive
  correlation of D_all with depth.
- Reported without thresholds: the artifact share, artifact mass and CLS
  mass in the supervised ViTs, whose artifacts are not characterized in
  advance; D_raw; per-layer values; distances relative to the uniform
  distance.

## Falsification and reporting

V1 and V2 failing means artifact tokens do not drive the late-layer
globality of the large self-supervised model; V3 and V4 failing means
registers do not change it. Every clause and value is reported whichever way
it falls (`audits/vit/eval_vit.py` writes `audits/vit_results.json`).
