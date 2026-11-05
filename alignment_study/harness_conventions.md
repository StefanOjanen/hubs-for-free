# Which windows carry the beginning-of-sequence token: the conventions of five common tools

Date: 2026-11-05. Read from the tools' code at the commits named below; no
compute. Motivation: bos_ppl_check.py, pair_start_diag.py and
bos_quant_check.py found that a conversion's or a rounding's cost measured on
windows without the beginning-of-sequence token carries a sink-start share, the
part of the measured cost that disappears when the token is prepended: the
fusion's 21 percent on Mistral-7B-Instruct and 3 percent on Mistral-7B-v0.1
(bos_ppl/, Table 12c), a per-channel 3-bit rounding's 23 and 63 percent and a
grouped rounding's none (bos_quant/, Table 12e). Whether a published number
carries that share depends on how the tool that produced it builds its
windows. This note records what five tools do. It describes the code in words
and names the file, function and commit so that each statement can be checked
against the source.

## The conventions

GPTQ reference implementation (IST-DASLab/gptq, commit 2d65066, 2023-07-11).
datautils.py, get_wikitext2: the train split is joined with blank lines and
tokenized once; each calibration sample is a crop of seqlen tokens starting at
a uniformly random position of that stream. get_c4: one document is tokenized
at a time and the crop starts at a random position inside it. The test split
is joined and tokenized once; llama.py, llama_eval, cuts the stream into
consecutive chunks of seqlen tokens (the number of chunks is the integer
quotient of the token count by seqlen) and scores every position of every
chunk. The tokenizer is called without add_special_tokens, so it adds its
default; for Llama tokenizers that is one beginning-of-sequence token at the
start of the stream. Calibration windows therefore start with the token only
when the random start is 0, and the evaluation's first chunk is the only one
that carries it.

AutoAWQ (casper-hansen/AutoAWQ, commit 88e4c76, 2025-05-11).
awq/utils/calib_data.py, get_calib_dataset: each text line is encoded
separately, the encodings are concatenated, and the concatenation is cut into
consecutive blocks of max_seq_len tokens; the remainder is dropped. Lines
encoded with a Llama tokenizer each begin with the token, so the token sits
inside the stream at every line boundary, and a block starts with it only
where a block boundary happens to fall on a line boundary. Evaluation is
external.

The transformers perplexity guide (huggingface.co/docs/transformers/perplexity,
read 2026-11-05). The test split is joined and tokenized once; windows of
max_length are taken at a stride, and in each window the positions before the
last stride are masked from the loss. The first window is the only one that can
carry the token (the guide's GPT-2 tokenizer adds none). With a stride of 512
the scored positions of every window after the first sit at positions 512 and
beyond of a window whose first token is an ordinary token.

lm-evaluation-harness (EleutherAI/lm-evaluation-harness, commit d6de816,
2026-09-14). lm_eval/models/huggingface.py, prefix_token_id: a custom token if
one is set, else the tokenizer's beginning-of-sequence token, else its
end-of-sequence token. loglikelihood_rolling builds rolling windows of
max_length with one token of context, where max_length is the model
configuration's maximum (n_positions, max_position_embeddings or n_ctx; 32,768
for Mistral-7B; 2,048 when the configuration gives none, or the value passed).
lm_eval/utils.py, get_rolling_token_windows: the first window's input is the
prefix token followed by the first max_length - 1 tokens; every later window's
input is a slice of the token stream starting one token before its first scored
token, and no prefix is added to it. So the harness's word perplexity scores
the first window with the token and every later window without it, and on a
7B model with a 32k maximum the WikiText-2 test set is about ten windows, nine
of them token-less and 32k tokens long.

llama.cpp (ggml-org/llama.cpp, commit 016c1d9, 2026-11-05).
tools/perplexity/perplexity.cpp, perplexity(): the text is tokenized once with
special tokens, cut into chunks of the context size, and, when the vocabulary
asks for a beginning-of-sequence token, the first token of every chunk is
overwritten with it before the chunk's first batch is decoded and restored
afterwards; that position is not scored. The function's header comment says
that the token is added for each chunk. Every chunk carries it.

## What follows

A WikiText-2 perplexity from GPTQ's script, from the transformers recipe or
from the harness is, past the first window, the token-less form that our "as
is" windows measure; a perplexity from llama.cpp is the with-token form. The
two differ by the method's sink-start share, and a method whose damage falls on
the sink-serving heads (the fusion's layer-0 pairs; the per-channel rounding's
outlier columns) has a share that the tools' numbers cannot be compared
across.

The share by position band, from bos_ppl/Mistral-7B-Instruct-v0.2.json and
bos_ppl/Mistral-7B-v0.1.json (the per-band share is one minus the ratio of the
log excess with the token to the log excess without it; Table 12c's band rows,
make_paper_tables.py):

| model, fusion at half the heads | positions 0 to 31 | 32 to 127 | 128 on |
|---|---|---|---|
| Mistral-7B-Instruct: excess ratio without, with the token; share | 6.22, 1.30; 85 percent | 3.38, 2.09; 39 percent | 2.96, 2.43; 18 percent |
| Mistral-7B-v0.1: excess ratio without, with the token; share | 4.67, 1.89; 59 percent | 4.57, 3.76; 13 percent | 6.00, 5.80; 2 percent |

The effect decays with position but does not vanish: on the Instruct a fifth of
the fusion's excess remains past position 128, so long windows (the harness's
32k) and strided scoring (the transformers recipe scores positions past the
stride) keep that fifth; on the base model it is gone by position 128, so there
the window length decides whether a number carries it. The per-channel
rounding's shares (Table 12e) were measured on 2,048-token windows only and
have no band split.

Calibration-based methods fit the token-less statistics: GPTQ's and AutoAWQ's
calibration windows start with the token only by chance. Prepending it to the
calibration windows was tested as a correction and failed for the fusion
(token_first_calib_dev/) and for GPTQ (gptq_bos_dev/, a parallel session's
check), so the convention of the calibration data is not what sets the share;
the share comes from where a method's damage falls and from whether the
evaluation's windows carry the token.

The recommendation stands as written in NOTE.md: evaluate under the deployed
prompt format or report both forms, name the tool and its window convention
beside the number, and state the rounding layout with it. This note adds the
convention of each tool so that a reader can tell which form a published
number is.
