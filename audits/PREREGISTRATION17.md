# Preregistration 17: two audits under the sources' own readings

Date: 2026-10-13. Frozen by commit and pushed before the confirmatory runs.
The criteria of preregistration 4 (`audits/PREREGISTRATION4.md`, frozen at
ee4e267) are unchanged. This file adds two readings that the audit reports,
the manuscript and the letters to the authors name as caveats.

## Why

Target 1 (Clark et al. 2019) ran on bert-base-uncased over 38 WikiText-103
windows of up to 64 tokens with one [SEP]. Clark et al. used BERT-base,
cased in the main analysis (their Section 3.2 reports the uncased model as a
check), on 1000 Wikipedia segments, most of them 128 tokens long, presented
as [CLS] A [SEP] B [SEP] following BERT's pre-training setup, which their
repository's `preprocess_unlabeled.py` implements.

Target 3 (CHAI 2024) read the cross-head correlation from each head's
attention row of the last token. The paper does not say which rows. A
maintainer confirmed Pearson correlation (facebookresearch/chai issue 6),
and the maintainer's dump code (issue 4), which feeds the released
clustering script, writes the rows of every query position.

## Part K: Target 1 in the source's input format

Instrument: `audits/readings/clark_source_format.py` with
`audits/readings/readings_common.py`. Segments are rebuilt with the logic of
Clark et al.'s preprocessing (sentence units, chunks up to 125 wordpieces, a
random split into A and B, pair truncation, `random.seed(0)`, then their
shuffle) on the Wikipedia articles of WikiText-103 validation, which was
Target 1's text source, detokenized with the usual WikiText detokenizer.
Model bert-base-cased with segment ids. 1000 segments, mean 127 tokens, 97.9
percent at 128 (a construction check; no model was run on them). The
statistic is Target 1's: D, the mean Jensen-Shannon divergence between heads
in different layers minus that within a layer, and the share of heads whose
nearest neighbor is in their own layer. Nulls: (a) random bidirectional
softmax maps; (b) each row permuted off the diagonal; (c) the same with the
[CLS] column and both [SEP] columns fixed; (d) five untrained
initializations. 20 draws per random family: Target 1's null distributions
were narrow (the separator null's 5th to 95th percentile spanned 0.08068 to
0.08075).

Instrument check (labeled, `audits/readings/check_clark.json`): on Target 1's
own inputs the GPU implementation reproduces the registered contrast to
1e-8, the nearest-neighbor share exactly, and the marginal and separator
null medians to 1e-5, with the same outcome label.

## Part C: Target 3 under the every-row reading

Instrument: `audits/readings/chai_every_row.py`. Per layer, the Pearson
correlation across heads of each query row over its causal support,
averaged over 64 rows evenly spaced from position 16 to 1023, then the
registered summaries per document (mean off-diagonal correlation, largest
complete-linkage cluster at 0.95 and 0.90) averaged over documents, as in
Target 3's battery. Same models (Mistral-7B, OPT-6.7B), documents (32 C4
documents of 1,024 tokens), precision and null families as Target 3, applied
to every row read; 30 draws per random family. Reproduced share = (null
median - random median) / (real - random median). The last-row statistic
is recomputed on the same pass and checked against Target 3's battery.

Development (labeled, `audits/readings/dev/`): Qwen2.5-0.5B, 8 documents, 10
draws, 2 untrained initializations. The sink-set null reproduces a median
0.86 of the every-row correlation (0.18 to 0.98 over layers), the marginal
null 0.04, the untrained models 0.00.

## Registered predictions

Thresholds were written after the instrument check and the development run;
K1, K2 and K4 come from Target 1's registered result, C1 to C4 from Target
3's and the development model.
- K0: in the source format the contrast D is positive and the same-layer
  nearest-neighbor share is at least twice chance (0.154).
- K1: the separator null reproduces between 0.49 and 0.79 of D, within 0.15
  of Target 1's 0.64.
- K2: the marginal null reproduces less than 0.25 of D.
- K3: the random and untrained nulls reproduce less than 0.05 of D in
  absolute value.
- K4: preregistration 4's outcome rule gives Target 1's label, "shrinks".
- C1: on Mistral-7B, the sink-set null reproduces at least half of the
  every-row correlation in the median layer.
- C2: the same on OPT-6.7B.
- C3: the marginal null reproduces at most 0.2 in the median layer of each
  model.
- C4: the untrained models reproduce at most 0.1 in the median layer of
  each model.
- Reported without thresholds: per-layer shares; the depth trend and
  majority clusters under the every-row reading; the cosine variant (the
  released clustering's distance); Target 3's last-row values beside them;
  the nulls' nearest-neighbor shares in Part K.

## Falsification and reporting

K1 failing means the input format moves the separator share by more than
0.15; K4 failing means it changes the verdict. C1 or C2 failing means the
sink account of CHAI's redundancy depends on which rows are read. Every
clause is reported whichever way it falls (`audits/readings/eval_readings.py`
writes `audits/readings/readings_results.json`), and the manuscript, the
README and the letters are updated with the result.
