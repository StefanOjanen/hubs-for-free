# Clean rerun of the exploratory follow-up's full-model conversions (not preregistered). The follow-up
# (llama2_caga_followup.py, run by a parallel session with its second moments kept on disk) reported perplexity
# above 9,000 for CAGA's construction and for the weighted fusion on pairs matched by CAGA's residual, while the
# per-layer diagnostic (llama2_followup_diag.py) found those pairs as good as round 28's in the fusion's residual
# and in single-layer conversions. This repeats the two full conversions on the same pairs exactly as round 28's
# instrument does (second moments in memory, llama2_round.convert, round 28's evaluation windows).
#   python alignment_study/llama2_followup_rerun.py
import json
import shutil
import sys
import tempfile
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

argv0 = sys.argv
sys.argv = [argv0[0], "--dev=NousResearch/Llama-2-7b-hf"]            # import the frozen round's functions unchanged
import importlib.util
spec = importlib.util.spec_from_file_location("R", "alignment_study/llama2_round.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
sys.argv = argv0
NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_rerun.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    groups = json.load(open("alignment_study/llama2_followup.json"))["groups"]
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    S, ntok = K.second_moments(model, kv, C.windows_c4(tok, 128, 2048), DEV)
    log(f"second moments from {ntok} tokens")
    win = {T: Z.wikitext2_windows(tok, T=T) for T in (2048, 4096)}
    snapdir = tempfile.mkdtemp(prefix="llama2_rerun_snapshot_")
    for l in range(kv.L):
        torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")

    def restore():
        for l in range(kv.L):
            for p, s_ in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
                p.copy_(s_.to(p.device))
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "groups_from": "alignment_study/llama2_followup.json", "evals": {}}
    for construction in ("fuse", "caga"):
        t = time.time(); R.convert(kv, S, groups, construction)
        r = {f"ppl_{T}": Z.perplexity(model, win[T], DEV, bs=max(1, 4096 // T)) for T in (2048, 4096)}
        res["evals"][f"{construction}/caga_matched"] = r
        log(f"{construction} on pairs matched by CAGA's residual (clean rerun): ppl 2048 {r['ppl_2048']:.2f}, 4096 {r['ppl_4096']:.2f} ({time.time() - t:.0f}s)")
        restore()
        json.dump(res, open(OUT, "w"), indent=1)
    res["restore_check_ppl_2048"] = Z.perplexity(model, win[2048], DEV, bs=2)
    json.dump(res, open(OUT, "w"), indent=1)
    shutil.rmtree(snapdir, ignore_errors=True)
    log(f"restored model: ppl 2048 {res['restore_check_ppl_2048']:.3f}")
    log("RERUN_DONE")
