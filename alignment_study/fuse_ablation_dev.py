# Development (exploratory, labelled): the single-weighting ablation of the closed-form fusion. Round 26 compared
# the fusion weighted by both the queries and the output projections with the unweighted fit. This adds the two
# single weightings, queries only and outputs only, on round 24's simulated-merge groups, the same five models,
# both group sizes, the same calibration and evaluation (fuse_round.run_model with its arm table replaced; every
# example of each task, as round 26). The base, both-weighted and unweighted evaluations are copied from round 26's
# results (alignment_study/fuse/) so that only the two new arms are computed.
#   python alignment_study/fuse_ablation_dev.py [--only=<model>]
import json
import os
import sys
import time

argv0 = sys.argv
sys.argv = [argv0[0], "--dev"]                       # fuse_round reads its mode at import; everything is overridden below
sys.path.insert(0, "alignment_study")
import fuse_round as R
sys.argv = argv0
R.DEV_RUN, R.REGISTERED = False, None                 # full evaluation as in round 26; not a registered run
R.MODELS = ["facebook/opt-1.3b", "EleutherAI/pythia-1.4b", "gpt2-large", "allenai/OLMo-1B-hf", "microsoft/phi-2"]
R.OUT, R.PARTS = "alignment_study/fuse_ablation_dev", "alignment_study/gqa"
R.ARMS = {"fuse_q/sim_align": ("sim_align", True, False), "fuse_o/sim_align": ("sim_align", False, True)}
COPY = ["base"] + [f"g{g}/{a}" for g in R.G_SIZES for a in ("fuse/sim_align", "fuse_u/sim_align")]
if "--tasks=1" not in argv0:                          # default: perplexity only (the copied rows carry the accuracies)
    R.Z.load_tasks = lambda subset=None: {}
    R.Z.build_requests = lambda tok, tasks: {}
    R.Z.evaluate_tasks = lambda model, tok, tasks, reqs, dev: {"average": float("nan"), "tasks_skipped": True}

if __name__ == "__main__":
    os.makedirs(R.OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in R.MODELS:
        if only and name not in only:
            continue
        short = name.split("/")[-1]
        path = f"{R.OUT}/{short}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        partial = path + ".partial"
        if not os.path.exists(partial):
            r26 = json.load(open(f"alignment_study/fuse/{short}.json"))
            json.dump({"evals": {k: r26["evals"][k] for k in COPY}, "copied_from_round26": COPY}, open(partial, "w"))
        res = R.run_model(name, log, partial)
        res.update({"exploratory": True, "registered": None, "dev": True, "copied_from_round26": COPY})
        json.dump(res, open(path, "w"), indent=1)
        os.remove(partial)
    log("FUSE_ABLATION_DONE")
