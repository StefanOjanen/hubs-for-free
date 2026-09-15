# Preregistration 4: addenda

Dated corrections and clarifications to `PREREGISTRATION4.md`, which is not
edited after its freeze (commit `ee4e267`, pushed 2026-09-14 23:24 UTC;
SHA-256 of the file at that commit
`f078f7e43ea166f4f3877084bb424b547b064fdb3107c775b6ca07d2ee702b70`;
permalink
https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md).
Nothing below changes a statistic, a null family, a threshold or an
outcome label. Each addendum was committed before the run it concerns.

## Addendum 1 (2026-09-15): title line

The first line of the frozen file still reads "(DRAFT, not yet frozen)".
The Status line directly below it, written at the freeze, governs. The
title was left as it is so that the file's digest stays the one recorded in
`alignment_study/RUNLOG.md`.

## Addendum 2 (2026-09-15): registration of record

The header says the file is "frozen by commit and pushed to the public
repository, then registered on OSF". The author deferred the OSF
registration on 2026-09-15 (a draft was prepared on OSF and left
unregistered). The registration of record is therefore the public freeze
commit, the same discipline as preregistrations 1 to 3 and 5 to 10 of this
project. The battery scripts' `--registered=` flag takes the freeze
permalink and writes it into each result file under the key `registered`.
No battery ran on real data before this addendum was committed.

## Addendum 3 (2026-09-15): precision of the untrained model in Target 5

The retrieval battery instantiates the config-initialized Qwen2.5-7B in the
same dtype as the trained model under the platform policy
(`hubsfree/adapters.py`: bfloat16 for a 7B on the 32 GB Apple MPS working
set). The committed script had omitted the dtype argument, which would have
requested float32 (about 30 GB of weights) and failed on this machine. The
frozen text fixes no precision for the untrained model, and the base result
of Target 5 was reproduced in bfloat16 as well. Changed before the run.
