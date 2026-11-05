# Committed from the parallel session's private/drafts on 2026-11-05 (exploratory, CPU, 512-token windows); ROOT made repository-relative, nothing else changed.
# Development (fork, CPU): cost-aware allocation of KV heads across layers at a fixed cache budget.  Each layer may stay
# unmerged, go to pairs or to groups of four; per-layer costs are measured by converting that layer alone (held-out C4);
# an exact DP picks the allocation at the budget of uniform pairs; evaluated on WikiText and held-out C4 against uniform
# pairs and random allocations at the same head count.  Pairs/quads from the residual's matching.
import os, sys, json, time, itertools, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root (this file sits in alignment_study/alloc_small_dev/); sys.path.insert(0, ROOT+"/alignment_study"); os.chdir(ROOT)
import gqa_common as C, fuse_common as F, fuse_kv as K, kvfuse, zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.set_num_threads(6)
NAME=sys.argv[1]; N_CAL,T_CAL,N_EVAL,T_EVAL=[int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else (32,512,24,512)
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "alloc_dev_"+NAME.split("/")[-1]+".json")
t0=time.time(); log=lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True)
tok=AutoTokenizer.from_pretrained(NAME); cal=C.windows_c4(tok,N_CAL,T_CAL); X=Z.wikitext2_windows(tok,T=T_EVAL)[:N_EVAL]
Xh=C.windows_c4(tok,N_CAL+8,T_CAL)[-8:]
def nll(m,W):
    tot=[]
    for w in range(W.shape[0]):
        x=W[w:w+1]; lg=m(input_ids=x,use_cache=False).logits[0,:-1].float()
        tot.append(torch.nn.functional.cross_entropy(lg,x[0,1:],reduction="none").numpy())
    return float(np.concatenate(tot).mean())
def load():
    m=AutoModelForCausalLM.from_pretrained(NAME,dtype=torch.float32,attn_implementation="sdpa").eval(); return m,K.KV(m)
m,kv=load(); L,H=kv.L,kv.nkv
S,_=K.second_moments(m,kv,cal,"cpu")
G={2:[kvfuse.min_cost_pairs(kvfuse.pair_costs(kv,l,S[l].double())) for l in range(L)], 4:[kvfuse.quad_groups(kv,l,S[l].double()) for l in range(L)]}
base={"wikitext":nll(m,X),"c4":nll(m,Xh)}; del m,kv
def convert(levels):
    m,kv=load()
    for l,g in enumerate(levels):
        if g==1: continue
        S64=S[l].double(); roots=F.layer_roots(S64,kv.d); Lsq=roots[0]@roots[0]
        for grp in G[g][l]: K.fuse_kv_group(kv,l,list(grp),S64,roots,qw=True,ow=True,Lsq=Lsq)
    r={"wikitext":nll(m,X),"c4":nll(m,Xh)}; del m,kv; return r
cost={}
for l in range(L):
    for g in (2,4):
        r=convert([g if i==l else 1 for i in range(L)]); cost[(l,g)]=r["c4"]-base["c4"]
    log(f"layer {l}: pairs {cost[(l,2)]:.4f} quads {cost[(l,4)]:.4f} (held-out C4 nats)")
heads=lambda levels: sum(H//g for g in levels)
B=heads([2]*L)
# exact DP over layers: minimise summed cost at total heads <= B
best={0:(0.0,[])}
for l in range(L):
    nb={}
    for h,(c,lv) in best.items():
        for g,cg in ((1,0.0),(2,cost[(l,2)]),(4,cost[(l,4)])):
            hh=h+H//g
            if hh<=B and (hh not in nb or c+cg<nb[hh][0]): nb[hh]=(c+cg,lv+[g])
    best=nb
dp=min(best.values(),key=lambda t:t[0])[1]
res={"model":NAME,"exploratory":True,"layers":L,"kv_heads":H,"budget_heads":B,"base_nll":base,"per_layer_cost":{f"{l}_{g}":cost[(l,g)] for (l,g) in cost},"arms":{}}
rng=np.random.default_rng(0)
arms={"uniform_pairs":[2]*L,"dp":dp}
for s in range(3):
    while True:
        lv=[int(rng.choice([1,2,4])) for _ in range(L)]
        if heads(lv)==B: break
    arms[f"random_{s}"]=lv
for name,lv in arms.items():
    r=convert(lv); r["levels"]=lv; r["heads"]=heads(lv); r["predicted"]=sum(cost[(l,g)] for l,g in enumerate(lv) if g>1)
    r["ppl_wikitext"]=float(np.exp(r["wikitext"])); r["ppl_c4"]=float(np.exp(r["c4"])); res["arms"][name]=r
    log(f"{name} levels {lv} heads {r['heads']}: wikitext {r['ppl_wikitext']:.2f} c4 {r['ppl_c4']:.2f} (predicted {r['predicted']:.3f}, actual c4 {r['c4']-base['c4']:.3f})")
json.dump(res,open(OUT,"w"),indent=1); log("ALLOC_DEV_DONE "+OUT)
