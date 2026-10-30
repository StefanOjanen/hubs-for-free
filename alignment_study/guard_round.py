# PREREGISTRATION29 (draft): spurious massive activations after training-free key-value head merging, and a check
# on one window. The exploratory follow-up to round 28 found that on Llama-2-7B one merged pair of key-value heads
# in layer 1, harmless alone and as good as any other pair by every per-layer measure, switches the super-weight
# neuron of layer 1's MLP on at ordinary tokens once layer 0 is converted, and the model collapses (perplexity in
# the thousands). This round asks how often a merge in the early layers does that in other models, and whether one
# calibration window catches it before any evaluation.
#
# Every member of a model's population is a full conversion to half the key-value heads with the weighted fusion
# (fuse_kv.py, the unweighted model's second moments, round 28's calibration): every layer on the reference
# matching except one early layer (0 to EARLY-1) on another matching. Reference matching: the minimum-cost perfect
# matching under the fusion's own residual (kvfuse.pair_costs; models whose projections carry biases: neighbouring
# pairs). Other matchings of a layer: with 4 key-value heads, the two other perfect matchings; with more, the
# minimum-cost matching forced to contain one given pair, for every pair (8 key-value heads) or a seeded sample of
# pairs (more than 8).
#
# The check, on one C4 window that is neither calibration nor evaluation text: the median over positions after the
# first of each position's largest absolute residual entry after each of layers 0 to CHECK-1, divided by the
# unconverted model's; flagged when the largest ratio exceeds TAU. Ground truth: WikiText-2 perplexity on the
# first N_EVAL windows of 2,048 tokens, as a multiple of the reference conversion's (severe: at least 5 times;
# benign: at most 2 times; eval_guard.py). Repair, for a flagged member:
# in the layers up to the first flagged one, the pair whose restoration to its unmerged heads lowers the check's
# largest ratio most is forbidden in that layer's matching (cost set to infinity) and the layer re-matched and
# re-fused; at most REPAIR_STEPS times.
#   python alignment_study/guard_round.py --registered=<freeze permalink> [--only=<model>] | --dev=<model> [--pairs=N]
import gc
import json
import os
import random
import sys
import time
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION29.md>; use --dev=<model>")
MODELS = [DEV_MODEL] if DEV_MODEL else ["mistralai/Mistral-7B-Instruct-v0.2", "openlm-research/open_llama_3b_v2",
                                       "allenai/OLMo-1B-hf", "Qwen/Qwen2.5-7B"]
OUT = "alignment_study/guard_dev" if DEV_MODEL else "alignment_study/guard"
BF16_ABOVE = 4e9                                   # parameters; hubsfree policy: float32 whenever the weights fit
TAU_FROZEN = 2.35                                  # the development survey (Llama-2-7B): benign at most 1.86, severe at least 2.98
EARLY, CHECK = int(arg("early", 5)), 6
if REGISTERED and (TAU_FROZEN is None or arg("tau") or arg("early") or arg("pairs") or arg("extra")):
    sys.exit("the registered run uses the frozen threshold and settings; no --tau, --early, --pairs or --extra")
TAU = TAU_FROZEN if REGISTERED else float(arg("tau", 3.0))
N_SAMPLE = int(arg("pairs", 24))                   # sampled forced pairs per layer when a layer has more than 8 KV heads
N_EVAL, REPAIR_STEPS, SEED = 4, 3, 29
N_CAL, T_CAL = 128, 2048
EXTRA = json.loads(arg("extra", "[]"))             # development only: [[layer, i, j], ...] forced pairs added to the sample
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def guard_window(tok):
    """The first C4 validation document after the calibration documents with at least 2,048 tokens."""
    return C.windows_c4(tok, N_CAL + 1, T_CAL)[-1:]


def matchings_for(kv, cost, ref, l, rng):
    """Other matchings of layer l: (label, groups)."""
    n, out, seen = kv.nkv, [], {tuple(map(tuple, ref))}
    if n == 4:
        cands = [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]]
        out = [("matching " + str(m), m) for m in cands]
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if n > 8:
            pairs = rng.sample(pairs, N_SAMPLE) + [(i, j) for (ll, i, j) in EXTRA if ll == l]
        assert cost is not None, "more than 4 KV heads needs the fusion's pair costs (no biases)"
        for i, j in pairs:
            rest = [h for h in range(n) if h not in (i, j)]
            m = sorted([sorted([i, j])] + [sorted([rest[a], rest[b]]) for a, b in kvfuse.min_cost_pairs(cost[rest][:, rest])])
            out.append((f"forced {i}-{j}", m))
    res = []
    for label, m in out:
        key = tuple(map(tuple, m))
        if key not in seen:
            seen.add(key); res.append((label, m))
    assert all(tuple(map(tuple, m)) != tuple(map(tuple, ref)) for _, m in res), "the reference matching is not a member"
    return res


def run_model(name, log, path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    nparam = sum(p.numel() for p in model.parameters())
    if nparam < BF16_ABOVE:
        model = model.float()
    model = model.eval().to(DEV)
    kv = K.KV(model)
    bias = kv.has_bias(0)
    log(f"{name}: {nparam / 1e9:.2f}B parameters, {next(model.parameters()).dtype}, layers {kv.L}, query heads {kv.n}, "
        f"KV heads {kv.nkv}, biases {bias}")
    body = model.model
    orig = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in range(kv.L)}   # attention weights (host)

    def put(l, ts):
        for p, t in zip(kv.params(l), ts):
            p.copy_(t.to(p.device))
    gw = guard_window(tok).to(DEV)

    def check(dims=False):
        """Median over positions after the first of each position's largest |entry|, after layers 0..CHECK-1; with
        dims, also each layer's per-dimension median |entry| over those positions and the first position's largest
        entry's dimension."""
        H = {}
        def mk(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0]
                if l == CHECK - 1:
                    raise Stop
            return hook
        hooks = [kv.blocks[l].register_forward_hook(mk(l)) for l in range(CHECK)]
        try:
            body(input_ids=gw, use_cache=False)
        except Stop:
            pass
        for h in hooks:
            h.remove()
        stat = [float(H[l][1:].abs().amax(1).float().median()) for l in range(CHECK)]
        if not dims:
            return stat
        return (stat, [H[l][1:].abs().float().median(0).values.cpu() for l in range(CHECK)], [int(H[l][0].abs().argmax()) for l in range(CHECK)],
                [float(H[l][0].abs().max()) for l in range(CHECK)])
    base_check, base_dim_median, base_first_dim, base_first_max = check(dims=True)
    # G5: the layer where the first position's massive activation forms (its largest entry first exceeds 20 times the
    # layer-0 value), the MLP neuron of that layer that writes the activation's dimension most strongly, and each KV
    # head's write strength onto that neuron's gate input (from the weights: |O_h^T (gate_j * norm weight)| over the
    # query heads h that read KV head i)
    me = next((l for l in range(1, CHECK) if base_first_max[l] > 20 * base_first_max[0]), None)
    writers = None
    assert kv.n % kv.nkv == 0, "equal query groups (write strengths sum over a KV head's query heads)"
    if me is not None:
        blk_me = kv.blocks[me]
        dstar = base_first_dim[me]
        col = blk_me.mlp.down_proj.weight[dstar].float().abs()
        jstar = int(col.argmax())
        top_neurons = [[int(j), float(col[j])] for j in col.topk(3).indices]
        lnw = getattr(blk_me.post_attention_layernorm, "weight", None)
        gvec = blk_me.mlp.gate_proj.weight[jstar].float() * (lnw.float() if lnw is not None else 1.0)   # not gw: the check's window
        Wo = kv.attn(me).o_proj.weight.float()
        strength = [float(torch.cat([Wo[:, h * kv.hd:(h + 1) * kv.hd].T @ gvec for h in range(i * kv.rep, (i + 1) * kv.rep)]).norm())
                    for i in range(kv.nkv)]
        med = sorted(strength)[len(strength) // 2]
        writers = {"layer": me, "dim": dstar, "neuron": jstar, "top_neurons": top_neurons, "strength": strength, "median": med,
                   "set": [i for i in range(kv.nkv) if strength[i] >= 5 * med],
                   "top3": sorted(range(kv.nkv), key=lambda i: -strength[i])[:3]}
    win = Z.wikitext2_windows(tok, T=2048)[:N_EVAL]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    base_ppl = ppl()
    S, ntok = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
    log(f"  unconverted: ppl {base_ppl:.3f}; check {[round(v, 3) for v in base_check]}; second moments from {ntok} tokens ({time.time() - t0:.0f}s)")
    costs = {l: (kvfuse.pair_costs(kv, l, S[l].double()) if not bias and kv.nkv > 2 else None) for l in range(EARLY)}
    if bias or kv.nkv <= 2:
        ref = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)]
        ref_kind = "neighbour"
    else:
        ref = [kvfuse.min_cost_pairs(costs[l] if l < EARLY else kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
        ref_kind = "fusion-residual matching"
    fit = {}

    def fuse(l, groups):
        if l not in fit:
            S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); fit[l] = (S64, roots, roots[0] @ roots[0])
        S64, roots, Lsq = fit[l]
        put(l, orig[l])
        for G in groups:
            K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
    for l in range(kv.L):
        fuse(l, ref[l])
        if l >= EARLY:                      # later layers stay on the reference: free what only their fusion needed
            fit.pop(l, None); orig.pop(l); S.pop(l)
    refw = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in range(EARLY)}
    ratio = lambda c: [a / b for a, b in zip(c, base_check)]
    ref_check = check(); ref_ppl = ppl()
    log(f"  reference ({ref_kind}): ppl {ref_ppl:.3f}; check ratios {[round(v, 2) for v in ratio(ref_check)]} ({time.time() - t0:.0f}s)")
    res = {"model": name, "params": nparam, "dtype": str(next(model.parameters()).dtype), "layers": kv.L, "kv_heads": kv.nkv,
           "query_heads": kv.n, "biases": bias, "reference": ref_kind, "reference_groups": ref, "base_ppl": base_ppl,
           "base_check": base_check, "base_first_position_dim": base_first_dim, "base_first_position_max": base_first_max,
           "massive_activation_layer": me, "writers": writers, "reference_ppl": ref_ppl, "reference_check": ref_check, "tau": TAU, "check_layers": CHECK,
           "early_layers": EARLY, "eval_windows": N_EVAL, "members": []}
    rng = random.Random(SEED)

    def unmerge_scores(cur):
        """For each layer up to the first flagged one and each pair of its matching: the check's largest ratio with
        that pair's heads restored to their unmerged weights."""
        r = ratio(check()); first = next(i for i, v in enumerate(r) if v > TAU)
        out = []
        for l in range(min(first, EARLY - 1) + 1):
            for G in cur[l]:
                saved = [p.detach().clone() for p in kv.params(l)]
                a, hd = kv.attn(l), kv.hd
                q_heads = [h for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)]
                for mod, rows in ((a.k_proj, G), (a.v_proj, G), (a.q_proj, q_heads)):
                    idx = mod_index(mod, a)
                    for i in rows:
                        mod.weight[i * hd:(i + 1) * hd] = orig[l][idx][i * hd:(i + 1) * hd].to(mod.weight.device)
                for h in q_heads:
                    a.o_proj.weight[:, h * hd:(h + 1) * hd] = orig[l][mod_index(a.o_proj, a)][:, h * hd:(h + 1) * hd].to(a.o_proj.weight.device)
                if bias:
                    for mod, rows in ((a.q_proj, q_heads), (a.k_proj, G), (a.v_proj, G)):
                        bi = bias_index(mod, a)
                        for i in rows:
                            mod.bias[i * hd:(i + 1) * hd] = orig[l][bi][i * hd:(i + 1) * hd].to(mod.bias.device)
                out.append((max(ratio(check())), l, list(G)))
                for p, s_ in zip(kv.params(l), saved):
                    p.copy_(s_)
        return sorted(out)

    def mod_index(mod, a):
        """Position of a projection's weight in kv.params(l) (weights and biases of q, k, v, o in that order)."""
        order = [(m.weight, m.bias) for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)]
        flat = [t for w, b in order for t in (w, b) if t is not None]
        return next(i for i, t in enumerate(flat) if t is mod.weight)

    def bias_index(mod, a):
        order = [(m.weight, m.bias) for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)]
        flat = [t for w, b in order for t in (w, b) if t is not None]
        return next(i for i, t in enumerate(flat) if t is mod.bias)

    def repair(cur):
        cur = [list(map(list, g)) for g in cur]
        forbidden, steps = [], []
        for step in range(REPAIR_STEPS):
            r = ratio(check())
            if max(r) <= TAU:
                break
            worst = unmerge_scores(cur)[0]
            _, l, G = worst
            forbidden.append([l, G])
            c = costs[l].clone() if costs[l] is not None else None
            if c is None:                       # 4 KV heads without costs: the remaining matching without that pair
                assert kv.nkv == 4, "repair without pair costs is defined for 4 KV heads only"
                cands = [m for m in ([[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]])
                         if not any(sorted(g) in [sorted(f[1]) for f in forbidden if f[0] == l] for g in m)]
                if not cands:
                    steps.append({"forbid_layer": l, "forbid_pair": G, "ratio_after_unmerge": worst[0], "new_groups": None,
                                  "failure": "no matching of the layer avoids every forbidden pair"})
                    break
                new = cands[0]
            else:
                for fl, fg in forbidden:
                    if fl == l:
                        c[fg[0], fg[1]] = c[fg[1], fg[0]] = 1e9
                new = kvfuse.min_cost_pairs(c)
            cur[l] = new; fuse(l, new)
            steps.append({"forbid_layer": l, "forbid_pair": G, "ratio_after_unmerge": worst[0], "new_groups": new})
        return cur, steps

    for l in range(EARLY):
        for label, m in matchings_for(kv, costs[l], ref[l], l, rng):
            t = time.time()
            fuse(l, m)
            tc = time.time(); c, dm, _, _ = check(dims=True); tc = time.time() - tc
            r = ratio(c)
            tp = time.time(); p = ppl(); tp = time.time() - tp
            top = max(range(CHECK), key=lambda i: r[i])  # G4: at the layer of the largest ratio, flagged or not
            mem = {"layer": l, "label": label, "groups": m, "check": c, "ratios": r, "max_ratio": max(r), "flag": max(r) > TAU,
                   "ppl": p, "ppl_over_reference": p / ref_ppl, "check_seconds": round(tc, 2), "ppl_seconds": round(tp, 2),
                   "max_ratio_layer": top, "spread_dim": int((dm[top] - base_dim_median[top]).argmax()),
                   "base_first_position_dim": base_first_dim[top]}
            new_pairs = [G for G in m if sorted(G) not in [sorted(H) for H in ref[l]]]
            mem["new_pairs"] = new_pairs
            mem["touches_writer"] = bool(writers) and l == writers["layer"] and any(set(G) & set(writers["set"]) for G in new_pairs)
            mem["strongest_writer_new_partner"] = bool(writers) and l == writers["layer"] and any(writers["top3"][0] in G for G in new_pairs)
            if mem["flag"]:
                cur = [ref[k] if k != l else m for k in range(kv.L)]
                tr = time.time(); cur, steps = repair(cur); mem["repair_seconds"] = round(time.time() - tr, 1)
                mem.update({"repair_steps": steps, "repaired_ratios": ratio(check()), "repaired_ppl": ppl()})
                mem["repaired_over_reference"] = mem["repaired_ppl"] / ref_ppl
                for k in range(EARLY):
                    put(k, refw[k])
            put(l, refw[l])
            res["members"].append(mem)
            json.dump(res, open(path + ".partial", "w"))
            log(f"  layer {l} {label}: ppl {p:.2f} ({p / ref_ppl:.2f}x reference), largest check ratio {max(r):.2f}"
                f"{' FLAGGED' if mem['flag'] else ''}" + (f", repaired ppl {mem['repaired_ppl']:.2f}" if mem["flag"] else "") +
                f" ({time.time() - t:.0f}s)")
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)" if not s.startswith("  ") else "", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = arg("only")
    for name in MODELS:
        if only and name != only:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            log(f"{name}: exists, skipped"); continue
        res = run_model(name, log, path)
        res["registered"] = REGISTERED; res["dev"] = bool(DEV_MODEL)
        json.dump(res, open(path, "w"), indent=1)
        if os.path.exists(path + ".partial"):
            os.remove(path + ".partial")
        del res
        gc.collect()                         # the model and its views live in run_model; collect before the next one
        if DEV == "mps":
            torch.mps.empty_cache()
    log("GUARD_ROUND_DONE")
