# Renders the README figures and illustrations from committed artifacts.
# Charts read results.json, alignment_study/*.json and scale_partial/*.json;
# nothing is typed in by hand. Illustrations are conceptual diagrams drawn
# from small synthetic arrays and carry no reported numbers.
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- design tokens (reference palette, light surface) ----
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12,
    "axes.titlecolor": INK, "axes.titleweight": "normal", "axes.titlesize": 15,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 11, "lines.linewidth": 2.6,
})
OUT = "figures/readme/"

def style(ax, title=None, sub=None):
    if title: ax.set_title(title, loc="left", pad=20 if sub else 10)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=11, va="bottom")
    ax.tick_params(length=0)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)

def endlabel(ax, x, y, text, color=INK2, dx=0.6, **kw):
    ax.text(x + dx, y, text, color=color, fontsize=11, va="center", **kw)

def dot(ax, x, y, c, s=60, z=3):
    ax.scatter(x, y, s=s, color=c, edgecolors=SURF, linewidths=1.8, zorder=z)

# ---- data ----
R = json.load(open("results.json"))
QM = json.load(open("alignment_study/qwen_multilayer.json"))["per_layer"]
HO = json.load(open("alignment_study/heldout_round.json"))
GT = json.load(open("alignment_study/gram_theorem.json"))["per_layer"]
TOY = json.load(open("alignment_study/tier2_toy.json"))["toy_curves"]
SCALE = [json.load(open(f)) for f in sorted(glob.glob("alignment_study/scale_partial/*_capture.json"))]

# ============ Figure 1: the coordinator that random matrices produce ============
N, T = 14, 26
def ens_gaussian(rng, n=N, T_=T):
    tau = np.exp(rng.normal(0, 0.5, n))
    lo = rng.normal(size=(n, T_, T_)) / tau[:, None, None]
    lo = np.where(np.tril(np.ones((T_, T_), bool)), lo, -1e9)
    ex = np.exp(lo - lo.max(-1, keepdims=True))
    return ex / ex.sum(-1, keepdims=True)
seed = R["section42"]["searched_exhibit"]["first_match"]["seed"]
A = ens_gaussian(np.random.default_rng(seed))
G = (A - A.transpose(0, 2, 1)) / 2
P = np.einsum("aij,bjk->abik", G, G); C = np.sqrt(((P - P.transpose(1, 0, 2, 3)) ** 2).sum((2, 3)))
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
gap = ev[0] / abs(ev[1]); hub = int(C.sum(1).argmax())
assert abs(gap - R["section42"]["searched_exhibit"]["first_match"]["gap"]) < 1e-6
fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.9), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3, "top": 0.78})
fig.text(0.02, 0.965, "The coordinator that random matrices produce", fontsize=15, color=INK, va="top")
fig.text(0.02, 0.905, f"Fourteen Gaussian-logit causal softmax heads, seed {seed}. No training anywhere.", fontsize=11, color=INK2, va="top")
ax[0].grid(False); im = ax[0].imshow(C, cmap=CMAP); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].add_patch(plt.Rectangle((-0.5, hub - 0.5), N, 1, fill=False, ec=S2, lw=1.6))
ax[0].text(0, N - 0.1, f"outlined row: head {hub}, the hub\nthat couples to everyone", color=INK2, fontsize=10.5, va="top", transform=ax[0].transData)
ax[0].set_title("coupling matrix C_ij", fontsize=12, loc="left", color=INK2)
ax[1].bar(range(N), ev, width=0.55, color=S1, zorder=3)
ax[1].set_xticks(range(N)); ax[1].set_xticklabels([str(i) for i in range(N)], fontsize=9.5)
ax[1].set_xlabel("eigenvalue index"); ax[1].axhline(0, color=GRID, lw=0.8)
ax[1].annotate(f"gap = {gap:.0f}x", xy=(0, ev[0]), xytext=(2.4, ev[0] * 0.9), color=INK, fontsize=12,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax[1].set_title("its eigenvalues: one versus the rest", fontsize=12, loc="left", color=INK2); style(ax[1])
fig.savefig(OUT + "fig1_coordinator_from_noise.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 2: false-flag rate of the 2-sigma rule ============
s42 = R["section42"]; ns, ps = s42["n"], [100 * p for p in s42["p_any_flag"]]
fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(ns, ps, color=S1, lw=2.6, solid_capstyle="round", zorder=2); dot(ax, ns, ps, S1)
for n_, p_ in zip(ns, ps):
    if n_ in (14, 32): ax.text(n_, p_ + 4, f"{p_:.0f}%", color=INK, fontsize=12.5, ha="center")
ax.set_ylim(0, 105); ax.set_xticks(ns); ax.set_xticklabels([str(n_) if n_ != 16 else "" for n_ in ns]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100%"]); ax.set_xlabel("number of heads n")
style(ax, "How often noise produces a 'special' head", "2-sigma rule on generator norms or coupling row sums, 300 draws per n")
fig.savefig(OUT + "fig2_false_flag_rate.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 3: depth profile, real vs surrogates (Qwen2.5-0.5B) ============
L = [r["layer"] for r in QM]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.12})
series = [("plain surrogate (row sharpness, self-mass)", [r["plain_r1"] for r in QM], S2),
          ("real attention", [r["real_r1"] for r in QM], S1),
          ("sink-column surrogate (plus one number per row)", [r["sinkfix_r1"] for r in QM], S3)]
for name, ys, c in series:
    a1.plot(L, ys, color=c, lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=2, label=name)
    dot(a1, L, ys, c, s=34)
a1.axhline(0, color=GRID, lw=0.8); a1.set_ylim(-0.8, 1.1); a1.set_yticks([-0.5, 0, 0.5, 1.0])
a1.set_ylabel("rank-1 correlation r1"); a1.legend(loc="lower left", ncol=1)
endlabel(a1, L[-1], series[0][1][-1], "noise level", dx=0.5)
style(a1, "One number per row reproduces the interaction geometry", "Qwen2.5-0.5B, 24 layers, 12 windows: the sink-column surrogate tracks the real curve layer by layer")
m = [r["mean_col0"] for r in QM]
a2.fill_between(L, 0, m, color=S1, alpha=0.10); a2.plot(L, m, color=S1, lw=2.6); dot(a2, L, m, S1, s=28)
a2.set_ylim(0, 1); a2.set_yticks([0, 0.5, 1.0]); a2.set_ylabel("sink mass"); a2.set_xlabel("layer"); a2.set_xticks(range(0, 24, 3))
style(a2)
fig.savefig(OUT + "fig3_depth_profile.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ pooled high-sink layers ============
rows = []
for mname, rs in HO["models"].items():
    if isinstance(rs, list): rows += [dict(r, model=mname, round="sub-2B held-out") for r in rs]
for d in SCALE: rows += [dict(r, model=d["model"], round="3B to 7B") for r in d["per_layer"]]
for r in GT: rows.append({"model": "Qwen/Qwen2.5-0.5B", "round": "development", "smass": r["mean_col0"],
                          "cosS": r["cos_S_sink_median"], "sharedE": r["mean_shared_energy"], "r1_real": r["r1_real_median"]})
hs = [r for r in rows if r["smass"] > 0.4]

# ============ Figure 4: the shared operator is the sink ============
fig, ax = plt.subplots(figsize=(8.8, 5.2))
dot(ax, [r["smass"] for r in hs], [r["cosS"] for r in hs], S1, s=40)
ax.axhline(0.7, color=MUTED, lw=1); ax.text(0.905, 0.712, "0.70 registered threshold", color=INK2, fontsize=11, ha="right")
ax.set_xlim(0.38, 0.92); ax.set_ylim(0.6, 1.02); ax.set_xlabel("sink mass of the layer"); ax.set_ylabel("cosine of shared component to the sink operator")
n_hs = len(hs); frac87 = np.mean([r["cosS"] > 0.87 for r in hs]) * 100
ax.text(0.39, 0.622, f"{n_hs} high-sink layers, {len(set(r['model'] for r in hs))} models:\nall above 0.70, {frac87:.0f}% above 0.87", color=INK, fontsize=12)
style(ax, "The shared component of a layer's heads is the sink operator", "Every high-sink layer of every model measured, 0.1B to 7B")
fig.savefig(OUT + "fig4_shared_operator_is_sink.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 5: the law that transfers ============
def spearman(x, y):
    def rank(v):
        v = np.array(v); idx = np.argsort(v); rk = np.empty(len(v)); rk[idx] = np.arange(len(v)); return rk
    return float(np.corrcoef(rank(x), rank(y))[0, 1])
pools = [("sub-2B held-out (5 models)", S1, [r for r in hs if r["round"] == "sub-2B held-out"]),
         ("3B to 7B (5 models)", S2, [r for r in hs if r["round"] == "3B to 7B"])]
fig, ax = plt.subplots(figsize=(8.8, 5.6))
toy = sorted(TOY["aligned"], key=lambda d: d["sharedE"])
ax.plot([d["sharedE"] for d in toy], [d["r1"] for d in toy], color=S3, lw=2.8, zorder=2, label="toy ensemble: noise plus one shared column")
labels = []
for name, c, pool in pools:
    sp = spearman([r["sharedE"] for r in pool], [r["r1_real"] for r in pool])
    dot(ax, [r["sharedE"] for r in pool], [r["r1_real"] for r in pool], c, s=38)
    ax.scatter([], [], color=c, s=38, label=f"{name}, Spearman {sp:.2f}")
ax.axvspan(0.85, 0.93, color=S3, alpha=0.08, lw=0); ax.text(0.89, 0.93, "toy regime flip", color=INK2, fontsize=11, ha="center")
ax.axhline(0, color=GRID, lw=0.8); ax.set_xlim(0.48, 1.0); ax.set_ylim(-0.8, 1.02)
ax.set_xlabel("shared-energy fraction of the layer"); ax.set_ylabel("rank-1 correlation r1"); ax.legend(loc="lower left")
style(ax, "One number per layer predicts its interaction geometry", "High-sink layers of ten held-out models; the toy curve was fixed before any model was run")
fig.savefig(OUT + "fig5_alignment_law.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 6: alignment versus concentration (toy worlds) ============
fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.8, 4.4), gridspec_kw={"wspace": 0.32})
for world, c, lab in (("aligned", S1, "same sink column for all heads"), ("misaligned", S2, "different sink column per head")):
    rows_ = TOY[world]; x = [d["boost"] for d in rows_]
    b1.plot(x, [d["sharedE"] for d in rows_], color=c, lw=2.6, label=lab); dot(b1, x, [d["sharedE"] for d in rows_], c, s=30)
    b2.plot(x, [d["r1"] for d in rows_], color=c, lw=2.6, label=lab); dot(b2, x, [d["r1"] for d in rows_], c, s=30)
b1.set_ylim(0, 1.05); b1.set_xlabel("sink strength (logit boost)"); b1.set_ylabel("shared-energy fraction")
b2.set_ylim(-1.05, 1.05); b2.axhline(0, color=GRID, lw=0.8); b2.set_xlabel("sink strength (logit boost)"); b2.set_ylabel("rank-1 correlation r1")
hs, ls = b1.get_legend_handles_labels(); fig.legend(hs, ls, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=11.5, handlelength=2.4, columnspacing=2.4)
style(b1, "Shared mode", "Same per-head concentration in both worlds"); style(b2, "Interaction geometry", "Only the shared column collapses it")
fig.savefig(OUT + "fig6_alignment_not_concentration.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============ Figure 7: training dynamics (preregistration 5) ============
STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
DYN = {}
for f in glob.glob("alignment_study/dynamics/*.json"):
    c = json.load(open(f)); DYN.setdefault(c["model"], {})[c["step"]] = c["rows"]
PH = json.load(open("alignment_study/posthoc_dynamics.json"))
DR = json.load(open("alignment_study/dynamics_results.json"))
step_colors = [CMAP(v) for v in np.linspace(0.08, 1.0, len(STEPS))]
cells = [(k, r) for m in DYN for k, s in enumerate(STEPS) for r in DYN[m][s]]
fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.9), gridspec_kw={"wspace": 0.3, "top": 0.86, "bottom": 0.24})
for k, r in cells:
    ax[0].scatter(r["smass"], r["cosS"], s=34, color=step_colors[k], edgecolors=SURF, linewidths=1.0, zorder=3)
ax[0].axvline(0.4, color=GRID, lw=1.2, zorder=1); ax[0].axhline(0.7, color=GRID, lw=1.2, zorder=1)
ax[0].text(0.41, 0.03, "high-sink\nthreshold", color=MUTED, fontsize=9.5, va="bottom")
ax[0].annotate("step 0: the uniform\ncausal operator", xy=(0.11, 0.3), xytext=(0.32, 0.36), color=INK2, fontsize=10, va="center",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkA=0, shrinkB=6))
ax[0].text(0.83, 0.66, "sink formed: the shared\nmode is the sink operator", color=INK2, fontsize=10, va="top", ha="right")
ax[0].set_xlim(0, 0.85); ax[0].set_ylim(0, 1.03); ax[0].set_xlabel("sink mass of the layer"); ax[0].set_ylabel("cos(shared component, sink operator)")
style(ax[0], "The shared mode follows the sink", f"all 360 cells, Spearman {DR['D']['D1']['spearman']:.2f}")
tc = sorted((r["sharedE"], r["r1"]) for r in TOY["aligned"])
ax[1].plot([e for e, _ in tc], [v for _, v in tc], color=S3, lw=2.6, zorder=2, label="toy curve")
for k, r in cells:
    if r["smass"] > 0.4:
        ax[1].scatter(r["sharedE"], r["r1_real"], s=34, color=step_colors[k], edgecolors=SURF, linewidths=1.0, zorder=3)
cs = PH["crossing_summary"]
ax[1].axvspan(cs["min"], cs["max"], color=S2, alpha=0.10, lw=0, zorder=1)
ax[1].text(0.455, -0.36, f"r1 changes sign along training\nat shared energy {cs['min']:.2f} to {cs['max']:.2f}\n({cs['n']} crossings, median {cs['median']:.2f};\ntoy crossing {PH['toy_r1_zero_crossing_sharedE']:.2f})", color=S2, fontsize=9.5, ha="left", va="top")
ax[1].set_xlim(0.44, 1.0); ax[1].set_ylim(-1.0, 1.05); ax[1].set_xlabel("shared-energy fraction of the layer"); ax[1].set_ylabel("rank-1 correlation r1")
ax[1].legend(loc="lower left", fontsize=9.5, handlelength=1.6)
style(ax[1], "The law holds along training", f"{DR['D']['D2']['n']} high-sink cells, Spearman {DR['D']['D2']['spearman']:.2f}")
from matplotlib.colors import ListedColormap, BoundaryNorm
cax = fig.add_axes([0.3, 0.075, 0.4, 0.028])
cb = matplotlib.colorbar.ColorbarBase(cax, cmap=ListedColormap(step_colors), norm=BoundaryNorm(np.arange(len(STEPS) + 1) - 0.5, len(STEPS)), orientation="horizontal", ticks=[0, 1, 3, 5, 7, 9])
cb.ax.set_xticklabels(["0", "512", "2k", "8k", "32k", "143k"], fontsize=10); cb.outline.set_visible(False); cb.ax.tick_params(length=0)
cb.set_label("training step", color=INK2, fontsize=11)
fig.savefig(OUT + "fig7_training_dynamics.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============ Figure 8: dissociation under the fixed control (preregistration 6) ============
RR = {}
for f in glob.glob("alignment_study/rerun/*.json"):
    c = json.load(open(f)); RR[c["model"]] = c
CALIB = {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"}
HEADS = {"gpt2": 12, "gpt2-medium": 16, "EleutherAI/pythia-160m": 12, "EleutherAI/pythia-410m": 16, "TinyLlama/TinyLlama_v1.1": 32,
         "Qwen/Qwen2.5-1.5B": 12, "Qwen/Qwen2.5-1.5B-Instruct": 12, "microsoft/Phi-3-mini-4k-instruct": 32,
         "mistralai/Mistral-7B-v0.1": 32, "Qwen/Qwen2.5-7B": 28, "allenai/OLMo-2-1124-7B": 32}
order = sorted([m for m in RR if m not in CALIB], key=lambda m: (HEADS.get(m, 0), m))
fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.9), gridspec_kw={"wspace": 0.3, "top": 0.86, "bottom": 0.3, "width_ratios": [1.25, 1]})
rng8 = np.random.default_rng(0)
for i, m in enumerate(order):
    rows = [r for r in RR[m]["protocol_B"]["rows"] if r["smass"] > 0.4]
    col = S2 if HEADS.get(m, 0) >= 32 else S1
    y = [r["rho_wrapped"] / r["rho_real"] for r in rows]
    ax[0].scatter(i + rng8.uniform(-0.18, 0.18, len(y)), y, s=26, color=col, edgecolors=SURF, linewidths=0.9, zorder=3)
ax[0].axhline(0.5, color=INK2, lw=1.2, zorder=2); ax[0].text(len(order) - 0.6, 0.53, "registered threshold 0.5", color=INK2, fontsize=10, ha="right", va="bottom")
ax[0].set_ylim(0, 0.6); ax[0].set_xlim(-0.6, len(order) - 0.4)
short = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2 M", "EleutherAI/pythia-160m": "Pythia 160m", "EleutherAI/pythia-410m": "Pythia 410m", "TinyLlama/TinyLlama_v1.1": "TinyLlama",
         "Qwen/Qwen2.5-1.5B": "Qwen 1.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 1.5B-I", "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini",
         "mistralai/Mistral-7B-v0.1": "Mistral 7B", "Qwen/Qwen2.5-7B": "Qwen 7B", "allenai/OLMo-2-1124-7B": "OLMo-2 7B"}
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([f"{short[m]}\n{HEADS[m]} heads" for m in order], fontsize=8.5, rotation=90)
ax[0].set_ylabel("shared mode under the control, as a fraction of real"); ax[0].grid(axis="x", visible=False)
style(ax[0], "The shared mode collapses", "182 high-sink layers, eleven models, T = 256")
for m in order:
    rows = [r for r in RR[m]["protocol_B"]["rows"] if r["smass"] > 0.4]
    col = S2 if HEADS.get(m, 0) >= 32 else S1
    ax[1].scatter([r["zmed_real"] for r in rows], [r["zmed_wrapped"] for r in rows], s=26, color=col, edgecolors=SURF, linewidths=0.9, zorder=3)
lim = max(max(r["zmed_wrapped"] for m in order for r in RR[m]["protocol_B"]["rows"] if r["smass"] > 0.4), 10) * 1.05
lo = min(min(r["zmed_real"] for m in order for r in RR[m]["protocol_B"]["rows"] if r["smass"] > 0.4), 0) - 10
ax[1].plot([lo, lim], [lo, lim], color=INK2, lw=1.2, zorder=2); ax[1].text(150, 150 + 14, "equal", color=INK2, fontsize=10, rotation=21, ha="right", va="bottom")
ax[1].set_xlim(lo, 175); ax[1].set_ylim(lo, lim)
ax[1].set_xlabel("per-pair z of the real layer"); ax[1].set_ylabel("per-pair z under the control")
ax[1].scatter([], [], color=S1, s=40, label="12 to 28 heads"); ax[1].scatter([], [], color=S2, s=40, label="32 heads")
ax[1].legend(loc="upper left", fontsize=10)
style(ax[1], "Commutators grow", "per-pair z, control against real")
fig.savefig(OUT + "fig8_dissociation_fixed.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============ Figure 9: shared energy derived from per-head sink profiles (preregistration 8) ============
DER = [json.load(open(f)) for f in sorted(glob.glob("alignment_study/derivation/*.json"))]
if DER:
    DRS = json.load(open("alignment_study/derivation_results.json")) if glob.glob("alignment_study/derivation_results.json") else None
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.7), gridspec_kw={"wspace": 0.3, "top": 0.86, "bottom": 0.16, "width_ratios": [1, 1]})
    hs_x, hs_y, lo_x, lo_y, lo_d = [], [], [], [], []
    for c in DER:
        for r in c["rows"]:
            (hs_x if r["smass"] > 0.4 else lo_x).append(r["derived_ideal"]); (hs_y if r["smass"] > 0.4 else lo_y).append(r["measured"])
            if r["smass"] <= 0.4: lo_d.append(r["derived_dict"])
    ax[0].plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    ax[0].scatter(lo_x, lo_y, s=22, color=SEQ[1], edgecolors=SURF, linewidths=0.8, zorder=2, label="low-sink layers")
    ax[0].scatter(hs_x, hs_y, s=30, color=S1, edgecolors=SURF, linewidths=0.9, zorder=3, label="high-sink layers")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].set_xlabel("derived from per-head sink mass and sharpness"); ax[0].set_ylabel("measured shared-energy fraction")
    ax[0].legend(loc="upper left", fontsize=10)
    ax[0].text(0.98, 0.04, "diagonal: derived equals measured", color=INK2, fontsize=9.5, ha="right", transform=ax[0].transAxes)
    sub0 = f"{len(DER)} models, {DRS['G']['G1']['n']} high-sink layers, linear R² {DRS['G']['G1']['pearson_r2_linear']:.2f}" if DRS else f"{len(DER)} models"
    style(ax[0], "Sink-only derivation", sub0)
    ax[1].plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    ax[1].scatter(lo_d, lo_y, s=22, color=SEQ[1], edgecolors=SURF, linewidths=0.8, zorder=2, label="low-sink layers")
    ax[1].scatter([r["derived_dict"] for c in DER for r in c["rows"] if r["smass"] > 0.4], hs_y, s=30, color=S1, edgecolors=SURF, linewidths=0.9, zorder=3, label="high-sink layers")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_xlabel("derived: sink plus uniform causal operator"); ax[1].set_ylabel("measured shared-energy fraction")
    sub1 = f"all layers: median error {DRS['G']['G4']['dict_median_abs_err_all']:.3f}" if DRS else ""
    style(ax[1], "With the untrained operator added", sub1)
    fig.savefig(OUT + "fig9_derived_shared_energy.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============ Figure 10: head-merging tolerance (preregistration 7) ============
MER = [json.load(open(f)) for f in sorted(glob.glob("alignment_study/merge/*.json"))]
if MER:
    MRS = json.load(open("alignment_study/merge_results.json")) if glob.glob("alignment_study/merge_results.json") else None
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.7), gridspec_kw={"wspace": 0.32, "top": 0.86, "bottom": 0.16})
    cols = [S1, S2, S3]
    floor = 1e-4
    for k, c in enumerate(MER):
        rows = c["rows"]; lab = c["model"].split("/")[-1].replace("Qwen2.5-", "Qwen2.5 ")
        xb = [max(r["dloss_bottom"], floor) for r in rows]; yt = [max(r["dloss_top"], floor) for r in rows]
        ax[0].scatter(xb, yt, s=30, color=cols[k % 3], edgecolors=SURF, linewidths=0.9, zorder=3, label=lab)
        ax[1].scatter([r["sharedE"] for r in rows], yt, s=30, color=cols[k % 3], edgecolors=SURF, linewidths=0.9, zorder=3, label=lab)
    lim = [floor, 0.5]
    ax[0].plot(lim, lim, color=GRID, lw=1.2, zorder=1); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("loss increase, bottom-cosine pair merged (nats/token)"); ax[0].set_ylabel("loss increase, top-cosine pair merged")
    ax[0].text(0.03, 0.97, "below the diagonal: the cosine\npicked the cheaper merge", color=INK2, fontsize=9.5, va="top", transform=ax[0].transAxes)
    ax[0].legend(loc="lower right", fontsize=9.5)
    sub0 = f"top cheaper than bottom in {100*np.mean([pm['frac_top_below_bottom'] for m, pm in MRS['per_model'].items()]):.0f}% of layers" if MRS else ""
    style(ax[0], "Which pair to merge", sub0)
    ax[1].set_yscale("log"); ax[1].set_ylim(lim); ax[1].set_xlim(0.3, 1.0)
    ax[1].set_xlabel("shared-energy fraction of the layer"); ax[1].set_ylabel("loss increase, top-cosine pair merged")
    sub1 = f"pooled Spearman {MRS['M']['M2']['spearman_pooled']:.2f} over {MRS['M']['M2']['n_layers']} layers" if MRS else ""
    style(ax[1], "Does shared energy predict tolerance?", sub1)
    fig.savefig(OUT + "fig10_head_merging.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============ Figure 11: shared energy and sink mass across depth, thirteen models (rerun) ============
RR2 = {json.load(open(f))["model"]: json.load(open(f)) for f in sorted(glob.glob("alignment_study/rerun/*.json"))}
if RR2:
    order11 = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
               "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
    names11 = {"gpt2": "GPT-2 124M", "gpt2-medium": "GPT-2 355M", "EleutherAI/pythia-160m": "Pythia 160M", "EleutherAI/pythia-410m": "Pythia 410M", "TinyLlama/TinyLlama_v1.1": "TinyLlama 1.1B",
               "Qwen/Qwen2.5-0.5B": "Qwen2.5 0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5 1.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B Instruct", "Qwen/Qwen2.5-3B": "Qwen2.5 3B",
               "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini 3.8B", "mistralai/Mistral-7B-v0.1": "Mistral 7B", "Qwen/Qwen2.5-7B": "Qwen2.5 7B", "allenai/OLMo-2-1124-7B": "OLMo-2 7B"}
    order11 = [m for m in order11 if m in RR2]
    fig, axes = plt.subplots(3, 5, figsize=(8.8, 5.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.12, "hspace": 0.5, "top": 0.86, "bottom": 0.1, "left": 0.07, "right": 0.99})
    axes = axes.ravel()
    for k, m in enumerate(order11):
        ax = axes[k]; rows = RR2[m]["protocol_A"]["rows"]; L = len(rows); x = [(l + 0.5) / L for l in range(L)]
        ax.fill_between(x, 0, [r["smass"] for r in rows], color=SEQ[1], alpha=0.55, lw=0, label="sink mass")
        ax.plot(x, [r["sharedE"] for r in rows], color=S1, lw=2.0, label="shared energy")
        ax.axhline(0.4, color=GRID, lw=0.8)
        ax.set_title(names11[m], fontsize=10, loc="left", pad=3); ax.set_ylim(0, 1); ax.set_xlim(0, 1); ax.tick_params(length=0, labelsize=9)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "depth", "1"]); ax.set_yticks([0, 0.5, 1])
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    for k in range(len(order11), len(axes)):
        axes[k].axis("off")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="center", ncol=1, frameon=False, bbox_to_anchor=(0.8, 0.2), fontsize=11)
    fig.text(0.02, 0.965, "One shared operator, thirteen models", fontsize=15, color=INK, va="top")
    fig.text(0.02, 0.91, "Shared-energy fraction and sink mass by relative depth, 48 windows per model at T = 64; the line at 0.4 is the high-sink threshold", fontsize=10.5, color=INK2, va="top")
    fig.savefig(OUT + "fig11_depth_profiles.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Illustrations (conceptual, drawn at display size, 2x PNG) ============
CARD, CARD_EDGE, SHADOW = "#f3f2ee", "#dedcd6", "#000000"
plt.rcParams["axes.grid"] = False

def canvas(w, h):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def card(ax, x, y, w, h, fc=CARD, ec=CARD_EDGE, r=0.14, accent=None):
    ax.add_patch(FancyBboxPatch((x + 0.03, y - 0.04), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=SHADOW, ec="none", alpha=0.06, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=accent or ec, lw=2.2 if accent else 1.2, zorder=2))

def mini(ax, M, x, y, w, h, vmax=None, outline=None, frame=True):
    """Draw a small attention matrix inside data coordinates."""
    ia = ax.inset_axes([x, y, w, h], transform=ax.transData)
    ia.imshow(M, cmap=CMAP, vmin=0, vmax=vmax or M.max(), aspect="auto", interpolation="nearest")
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values(): sp.set_visible(frame); sp.set_color(CARD_EDGE)
    if outline is not None:
        ia.add_patch(plt.Rectangle((-0.5, outline - 0.5), M.shape[1], 1, fill=False, ec=S2, lw=2.2))
    return ia

def chevron(ax, x0, x1, y, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, color=color, lw=2.4, zorder=3))

def causal(rng, sink=0.0, diag=0.0, n=9, noise=1.0):
    lo = rng.normal(size=(n, n)) * noise; lo[:, 0] += sink; lo[np.arange(n), np.arange(n)] += diag
    lo = np.where(np.tril(np.ones((n, n), bool)), lo, -1e9); ex = np.exp(lo - lo.max(-1, keepdims=True)); return ex / ex.sum(-1, keepdims=True)

def shuffle_rows(rng, M, keep_cols=()):
    S_ = M.copy(); n = M.shape[0]
    for i in range(1, n):
        cols = [c for c in range(i) if c not in keep_cols]
        if len(cols) > 1: S_[i, cols] = S_[i, np.array(cols)[rng.permutation(len(cols))]]
    return S_

# ---- I4: the toolkit, as a pipeline of real objects ----
rng = np.random.default_rng(11)
fig, ax = canvas(8.8, 4.05)
xs = [0.15, 2.35, 4.55, 6.75]; W, H, Y = 1.9, 3.1, 0.75
titles = ["Attention maps", "Your statistic", "Four null families", "Percentile report"]
for x, t in zip(xs, titles):
    card(ax, x, Y, W, H); ax.text(x + W / 2, Y + H - 0.28, t, ha="center", va="center", fontsize=13.5, color=INK, zorder=4)
# card 1: fanned attention maps
real = causal(rng, sink=3.0, diag=1.0)
for k, (dx, dy) in enumerate([(0.42, 0.12), (0.27, 0.27), (0.12, 0.42)]):
    mini(ax, causal(rng, sink=3.0, diag=1.0) if k < 2 else real, xs[0] + 0.2 + dx, Y + 0.95 + dy, 1.05, 1.05, vmax=real.max())
ax.text(xs[0] + W / 2, Y + 0.45, "any model,\nany layer", ha="center", va="center", fontsize=11, color=INK2, zorder=4)
# card 2: coupling matrix glyph with hub row and a spectrum glyph
Aq = ens_gaussian(np.random.default_rng(seed)); Gq = (Aq - Aq.transpose(0, 2, 1)) / 2
Pq = np.einsum("aij,bjk->abik", Gq, Gq); Cq = np.sqrt(((Pq - Pq.transpose(1, 0, 2, 3)) ** 2).sum((2, 3)))
mini(ax, Cq, xs[1] + 0.25, Y + 1.55, 1.0, 1.0, outline=int(Cq.sum(1).argmax()))
ia = ax.inset_axes([xs[1] + 0.35, Y + 0.72, 1.2, 0.62], transform=ax.transData)
evq = np.sort(np.linalg.eigvalsh(Cq))[::-1]; ia.bar(range(len(evq)), evq, color=S1, width=0.6); ia.axis("off")
ax.text(xs[1] + W / 2, Y + 0.4, "clusters, hubs,\neigengaps, similarity,\nimportance scores", ha="center", va="center", fontsize=10, color=INK2, zorder=4)
# card 3: 2x2 nulls
nulls = [(causal(rng), "random"), (shuffle_rows(rng, real), "marginal"), (shuffle_rows(rng, real, (0,)), "sink column"), (causal(rng, noise=0.4), "untrained")]
for k, (M, lab) in enumerate(nulls):
    cx = xs[2] + 0.22 + (k % 2) * 0.82; cy = Y + 1.72 - (k // 2) * 1.05
    mini(ax, M, cx, cy, 0.66, 0.66, vmax=real.max()); ax.text(cx + 0.33, cy - 0.16, lab, ha="center", va="center", fontsize=9.5, color=INK2, zorder=4)
# card 4: null distribution with the real statistic in the tail
ia = ax.inset_axes([xs[3] + 0.18, Y + 0.95, 1.5, 1.35], transform=ax.transData)
xx = np.linspace(-3, 4.5, 200); ia.fill_between(xx, 0, np.exp(-xx ** 2 / 2), color=S1, alpha=0.18); ia.plot(xx, np.exp(-xx ** 2 / 2), color=S1, lw=2)
ia.axvline(3.1, color=S2, lw=3); ia.set_ylim(0, 1.15); ia.axis("off")
ia.text(3.25, 1.08, "real", ha="left", color=INK, fontsize=10.5); ia.text(-0.4, 1.08, "null draws", ha="center", color=INK2, fontsize=10.5)
ax.text(xs[3] + W / 2, Y + 0.45, "where the real value\nfalls among the nulls,\nfamily by family", ha="center", va="center", fontsize=10, color=INK2, zorder=4)
for x0, x1 in zip(xs[:-1], xs[1:]): chevron(ax, x0 + W + 0.05, x1 - 0.05, Y + H / 2)
ax.add_patch(FancyBboxPatch((2.75, 0.12), 3.3, 0.42, boxstyle="round,pad=0,rounding_size=0.1", fc="#e9e8e3", ec="none", zorder=2))
ax.text(4.4, 0.33, "hubsfree audit config.yaml", ha="center", va="center", fontsize=12, color=INK, family="DejaVu Sans Mono", zorder=4)
fig.savefig(OUT + "illus4_toolkit.png", dpi=200); plt.close(fig)

# ---- I1: one shared operator plus a residual ----
rng = np.random.default_rng(3)
fig, ax = canvas(8.8, 4.4)
heads = [causal(rng, sink=3.0) for _ in range(3)]
Ssink = np.zeros((9, 9)); Ssink[1:, 0] = 1.0
coef = [0.9, 0.6, 0.75]
for k in range(3):
    x = 0.25 + k * 1.25
    mini(ax, heads[k], x, 2.35, 1.05, 1.05, vmax=heads[k].max())
    ax.text(x + 0.52, 3.58, f"head {k + 1}", ha="center", fontsize=12, color=INK)
    ax.add_patch(plt.Rectangle((x + 0.12, 1.95), 0.8 * coef[k], 0.16, color=S1, zorder=3)); ax.add_patch(plt.Rectangle((x + 0.12, 1.95), 0.8, 0.16, fill=False, ec=CARD_EDGE, lw=1))
    ax.text(x + 0.52, 1.75, f"a{k + 1}", ha="center", fontsize=10.5, color=INK2)
ax.text(2.1, 1.38, "three heads of one layer, each with\nits own weight a_h on the shared operator", ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.4)
ax.text(4.25, 2.9, "=", ha="center", va="center", fontsize=26, color=INK)
ax.text(4.25, 2.35, "a_h ×", ha="center", va="center", fontsize=13, color=INK)
mini(ax, Ssink, 4.7, 2.1, 1.55, 1.55, vmax=1)
ax.text(5.47, 3.85, "shared operator S", ha="center", fontsize=12.5, color=INK); ax.text(5.47, 1.78, "the sink column,\none per layer", ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.4)
ax.text(6.55, 2.9, "+", ha="center", va="center", fontsize=26, color=INK)
for k in range(3):
    Ek = heads[k].copy(); Ek[:, 0] = Ek[:, 1:].mean(); Ek = Ek * (np.tril(np.ones((9, 9))) > 0)
    mini(ax, Ek, 6.85 + k * 0.5, 2.4 + (2 - k) * 0.12, 0.85, 0.85, vmax=Ek.max() * 1.3)
ax.text(7.7, 3.85, "residuals E_h", ha="center", fontsize=12.5, color=INK); ax.text(7.7, 1.78, "what makes\neach head itself", ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.4)
ax.add_patch(FancyBboxPatch((0.3, 0.25), 8.2, 0.78, boxstyle="round,pad=0,rounding_size=0.12", fc="#e6f3ef", ec="none", zorder=2))
ax.text(4.4, 0.64, "G_h = a_h S + E_h   In every commutator [G_i, G_j] the S-with-S term cancels identically.\nWhat the interaction statistics measure is how each head deviates from the one operator they share.",
        ha="center", va="center", fontsize=11, color=INK, zorder=4, linespacing=1.5)
fig.savefig(OUT + "illus1_shared_operator.png", dpi=200); plt.close(fig)

# ---- I2: the null families ----
rng = np.random.default_rng(4)
real = causal(rng, sink=3.0, diag=1.0)
panels = [(real, "Real map", "the object\nunder test", True),
          (causal(rng), "Random softmax", "keeps only the\ncausal shape", False),
          (shuffle_rows(rng, real), "Marginal-matched", "keeps each row's\nsharpness and\nself-mass", False),
          (shuffle_rows(rng, real, (0,)), "Sink-column", "keeps the above\nplus one number\nper row", False),
          (causal(rng, noise=0.4), "Untrained model", "keeps the\narchitecture,\nnot the training", False)]
fig, ax = canvas(8.8, 4.1)
W, H, Y = 1.6, 3.1, 0.7
for k, (M, t, cap, is_real) in enumerate(panels):
    x = 0.15 + k * 1.72
    card(ax, x, Y, W, H, accent=S2 if is_real else None)
    ax.text(x + W / 2, Y + H - 0.3, t, ha="center", va="center", fontsize=12, color=INK, zorder=4)
    mini(ax, M, x + 0.25, Y + 1.15, 1.1, 1.1, vmax=real.max())
    ax.text(x + W / 2, Y + 0.6, cap, ha="center", va="center", fontsize=10, color=INK2, zorder=4, linespacing=1.35)
ax.text(4.4, 0.24, "A statistic earns its meaning by how far the real map sits from each null.\nThe toolkit reports that distance as a percentile.", ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.4)
fig.savefig(OUT + "illus2_null_families.png", dpi=200); plt.close(fig)

# ---- I3: same signature, two mechanisms ----
fig, ax = canvas(8.8, 5.1)
card(ax, 0.25, 2.85, 3.6, 2.0); card(ax, 0.25, 0.65, 3.6, 2.0); card(ax, 5.35, 1.55, 3.2, 2.4)
ax.text(2.05, 4.58, "Random matrices", ha="center", fontsize=13, color=INK, zorder=4)
ax.text(2.05, 2.38, "Trained model", ha="center", fontsize=13, color=INK, zorder=4)
ax.text(6.95, 3.7, "The same signature", ha="center", fontsize=13, color=INK, zorder=4)
norms = np.array([0.55, 0.5, 1.7, 0.45, 0.6, 0.52, 0.58])
ia = ax.inset_axes([0.5, 3.05, 1.55, 1.25], transform=ax.transData); ia.bar(range(7), norms, color=[S1] * 2 + [S2] + [S1] * 4, width=0.62); ia.axis("off")
ax.text(2.98, 3.75, "one head happens to\nhave a larger norm", ha="center", va="center", fontsize=10.5, color=INK2, zorder=4)
blk = np.full((7, 9), 0.18) + np.random.default_rng(1).uniform(-0.06, 0.1, (7, 9)); blk[:, 0] = 1.0
mini(ax, np.clip(blk, 0, 1), 0.5, 0.88, 1.55, 1.25, vmax=1)
ax.text(2.98, 1.55, "every head shares\none column: the sink", ha="center", va="center", fontsize=10.5, color=INK2, zorder=4)
Cn = np.outer(norms, norms); np.fill_diagonal(Cn, 0)
mini(ax, Cn, 5.6, 1.95, 1.25, 1.25, outline=int(Cn.sum(1).argmax()))
ia = ax.inset_axes([7.05, 1.95, 1.3, 1.25], transform=ax.transData); ia.bar(range(7), np.sort(np.linalg.eigvalsh(Cn))[::-1], color=S1, width=0.6); ia.axis("off")
ax.text(6.95, 1.75, "a hub, an eigengap, a special head", ha="center", fontsize=10.5, color=INK2, zorder=4)
for y0 in (3.85, 1.65):
    ax.add_patch(FancyArrowPatch((3.95, y0), (5.25, 2.75), arrowstyle="-|>", mutation_scale=22, color=MUTED, lw=2.4, connectionstyle="arc3,rad=0.0", zorder=3))
ax.text(4.4, 0.2, "Bare statistics cannot tell the two apart.\nMarginal-matched surrogates reproduce the first; only a sink-column surrogate reproduces the second.", ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.4)
fig.savefig(OUT + "illus3_two_mechanisms.png", dpi=200); plt.close(fig)
print("FIGURES_DONE", sorted(glob.glob(OUT + "*")))
