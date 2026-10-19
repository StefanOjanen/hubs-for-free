# Dry run of round 27's instrument on one 7B model, reporting memory at each phase (run under mem_watchdog.sh).
import sys, time, os, resource, tempfile, shutil, subprocess
sys.path.insert(0, "alignment_study")
import torch
import gqa_common as C, fuse_kv as K, zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
name = sys.argv[1]
t0 = time.time()
def mem():
    free = subprocess.run("memory_pressure | awk -F': ' '/free percentage/ {print $2}'", shell=True, capture_output=True, text=True).stdout.strip()
    return f"GPU driver {torch.mps.driver_allocated_memory() / 1e9:.1f} GB, process peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB, system free {free}"
log = lambda s: print(f"[{time.time()-t0:5.0f}s] {s} | {mem()}", flush=True)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
kv = K.KV(model); log(f"loaded {name}: n {kv.n} nkv {kv.nkv} hd {kv.hd} r {kv.r}")
cal = C.windows_c4(tok, 8, 1024); X = Z.wikitext2_windows(tok)[:8]
x0 = cal[:1, :256].to(DEV); ref = model(input_ids=x0).logits.float()
S, n = K.second_moments(model, kv, cal, DEV); log(f"second moments from {n} tokens")
snapdir = tempfile.mkdtemp(prefix="fusekv_snapshot_")
for l in range(kv.L):
    torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")
def restore():
    for l in range(kv.L):
        for p, s in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
            p.copy_(s.to(p.device))
log("snapshot on disk")
t = time.time(); K.convert(kv, S, [[[i] for i in range(kv.nkv)] for _ in range(kv.L)], "fuse")
log(f"groups of one in {time.time()-t:.0f}s: max logit difference {float((model(input_ids=x0).logits.float() - ref).abs().max()):.3f} (scale {float(ref.abs().max()):.1f})"); restore()
base = Z.perplexity(model, X, DEV); log(f"base ppl (8 windows) {base:.3f}")
for m in ("mean", "fuse_u", "fuse"):
    t = time.time(); K.convert(kv, S, [[list(range(s, s + 2)) for s in range(0, kv.nkv, 2)] for _ in range(kv.L)], m)
    log(f"{m}: converted in {time.time()-t:.0f}s, ppl {Z.perplexity(model, X, DEV):.3f}"); restore()
tasks = Z.load_tasks(subset=30); reqs = Z.build_requests(tok, tasks)
t = time.time(); r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV); log(f"harness on 30 per task: {len(reqs)} requests in {time.time()-t:.0f}s, avg {r['average']:.3f}")
shutil.rmtree(snapdir, ignore_errors=True); log("DRY_RUN_DONE")
