# Committed from the parallel session's private/drafts on 2026-11-05 (exploratory, CPU, 512-token windows); ROOT made repository-relative, nothing else changed.
# Development (fork, CPU): can the fusion's free closed-form residual replace the measured single-layer costs as the
# allocation's selector?  Per layer: residual at pairs = sum of matched pair costs; at quads = sum of group costs.
import os, sys, json, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root (this file sits in alignment_study/alloc_small_dev/); sys.path.insert(0, ROOT+"/alignment_study"); os.chdir(ROOT)
import gqa_common as C, fuse_common as F, fuse_kv as K, kvfuse, zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.set_num_threads(6)
NAME=sys.argv[1]; N_CAL,T_CAL,N_EVAL,T_EVAL=32,512,24,512
prev=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"alloc_dev_"+NAME.split("/")[-1]+".json")))
tok=AutoTokenizer.from_pretrained(NAME); cal=C.windows_c4(tok,N_CAL,T_CAL); X=Z.wikitext2_windows(tok,T=T_EVAL)[:N_EVAL]; Xh=C.windows_c4(tok,N_CAL+8,T_CAL)[-8:]
def nll(m,W):
    tot=[]
    for w in range(W.shape[0]):
        x=W[w:w+1]; lg=m(input_ids=x,use_cache=False).logits[0,:-1].float(); tot.append(torch.nn.functional.cross_entropy(lg,x[0,1:],reduction="none").numpy())
    return float(np.concatenate(tot).mean())
def load():
    m=AutoModelForCausalLM.from_pretrained(NAME,dtype=torch.float32,attn_implementation="sdpa").eval(); return m,K.KV(m)
m,kv=load(); L,H=kv.L,kv.nkv; S,_=K.second_moments(m,kv,cal,"cpu")
G={2:[],4:[]}; res_cost={}
for l in range(L):
    S64=S[l].double(); st=kvfuse.layer_stats(kv,l,S64); pairs=kvfuse.min_cost_pairs(kvfuse.pair_costs(kv,l,S64)); quads=kvfuse.quad_groups(kv,l,S64)
    G[2].append(pairs); G[4].append(quads)
    res_cost[(l,2)]=sum(kvfuse.group_cost(st,p) for p in pairs); res_cost[(l,4)]=sum(kvfuse.group_cost(st,q) for q in quads)
del m,kv
meas={(int(k.split("_")[0]),int(k.split("_")[1])):v for k,v in prev["per_layer_cost"].items()}
from scipy.stats import spearmanr
for g in (2,4): print(f"level {g}: spearman(residual, measured) over layers {spearmanr([res_cost[(l,g)] for l in range(L)],[meas[(l,g)] for l in range(L)]).correlation:+.2f}", flush=True)
B=sum(H//2 for _ in range(L))
def dp(cost):
    best={0:(0.0,[])}
    for l in range(L):
        nb={}
        for h,(c,lv) in best.items():
            for g,cg in ((1,0.0),(2,cost[(l,2)]),(4,cost[(l,4)])):
                hh=h+H//g
                if hh<=B and (hh not in nb or c+cg<nb[hh][0]): nb[hh]=(c+cg,lv+[g])
        best=nb
    return min(best.values(),key=lambda t:t[0])[1]
lv=dp(res_cost); print("residual DP levels", lv, flush=True)
print("measured DP levels", prev["arms"]["dp"]["levels"])
m,kv=load()
for l,g in enumerate(lv):
    if g==1: continue
    S64=S[l].double(); roots=F.layer_roots(S64,kv.d); Lsq=roots[0]@roots[0]
    for grp in G[g][l]: K.fuse_kv_group(kv,l,list(grp),S64,roots,qw=True,ow=True,Lsq=Lsq)
r={"levels":lv,"heads":sum(H//g for g in lv),"ppl_wikitext":float(np.exp(nll(m,X))),"ppl_c4":float(np.exp(nll(m,Xh)))}
print("dp_residual", r, flush=True)
prev["arms"]["dp_residual"]=r; prev["residual_cost"]={f"{l}_{g}":res_cost[(l,g)] for (l,g) in res_cost}
json.dump(prev,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"alloc_dev_"+NAME.split("/")[-1]+".json"),"w"),indent=1); print("DONE")
