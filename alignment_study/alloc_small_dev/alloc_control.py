# Committed from the parallel session's private/drafts on 2026-11-05 (exploratory, CPU, 512-token windows): the layer-0-whole control: layer 0 whole and the rest at pairs with the cheapest layers quartered until the head count matches uniform pairs, plus layer 0 whole with the dynamic programme on the remaining layers; reads the measured costs from the sweep's artifact; ROOT made repository-relative, nothing else changed.
# Development (fork, CPU): the control the allocation must beat: layer 0 whole and the rest uniform at matched head
# count (the cheapest layers quartered to pay for layer 0), against the DP (which may also keep layer 0) and against
# layer 0 whole plus the DP on the remaining layers.  Uses the measured single-layer costs from alloc_dev_<tag>.json.
import os, sys, json, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root (this file sits in alignment_study/alloc_small_dev/); sys.path.insert(0, ROOT+"/alignment_study"); os.chdir(ROOT)
import gqa_common as C, fuse_common as F, fuse_kv as K, kvfuse, zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.set_num_threads(6)
NAME, TAG = sys.argv[1], sys.argv[2]; N_CAL,T_CAL,N_EVAL,T_EVAL=32,512,24,512; N_HELD=8
D=os.path.dirname(os.path.abspath(__file__)); prev=json.load(open(f"{D}/alloc_dev_{TAG}.json"))
tok=AutoTokenizer.from_pretrained(NAME); cal=C.windows_c4(tok,N_CAL,T_CAL); X=Z.wikitext2_windows(tok,T=T_EVAL)[:N_EVAL]; Xh=C.windows_c4(tok,N_CAL+N_HELD,T_CAL)[-N_HELD:]
def nll(m,W):
    tot=[]
    for w in range(W.shape[0]):
        x=W[w:w+1]; lg=m(input_ids=x,use_cache=False).logits[0,:-1].float(); tot.append(torch.nn.functional.cross_entropy(lg,x[0,1:],reduction="none").numpy())
    return float(np.concatenate(tot).mean())
def load():
    m=AutoModelForCausalLM.from_pretrained(NAME,dtype=torch.float32,attn_implementation="sdpa").eval(); return m,K.KV(m)
m,kv=load(); L,H=kv.L,kv.nkv; S,_=K.second_moments(m,kv,cal,"cpu")
G={2:[kvfuse.min_cost_pairs(kvfuse.pair_costs(kv,l,S[l].double())) for l in range(L)],4:[kvfuse.quad_groups(kv,l,S[l].double()) for l in range(L)]}
del m,kv
cost={(int(k.split("_")[0]),int(k.split("_")[1])):v for k,v in prev["per_layer_cost"].items()}
B=L*(H//2); heads=lambda lv: sum(H//g for g in lv)
def dp(cost,layers,budget):
    best={0:(0.0,[])}
    for l in layers:
        nb={}
        for h,(c,lv) in best.items():
            for g,cg in ((1,0.0),(2,cost[(l,2)]),(4,cost[(l,4)])):
                hh=h+H//g
                if hh<=budget and (hh not in nb or c+cg<nb[hh][0]): nb[hh]=(c+cg,lv+[g])
        best=nb
    return min(best.values(),key=lambda t:t[0])[1]
# control: layer 0 whole, rest at pairs, then quarter the cheapest layers (by quad-minus-pair cost) until heads <= B
lv=[1]+[2]*(L-1); order=sorted(range(1,L),key=lambda l: cost[(l,4)]-cost[(l,2)])
for l in order:
    if heads(lv)<=B: break
    lv[l]=4
control=lv
rest_dp=[1]+dp(cost,list(range(1,L)),B-H)
arms={"uniform_pairs":[2]*L,"dp":prev["arms"]["dp"]["levels"],"layer0_whole_rest_uniform":control,"layer0_whole_rest_dp":rest_dp}
res={"model":NAME,"exploratory":True,"budget_heads":B,"arms":{}}
for name,levels in arms.items():
    m,kv=load()
    for l,g in enumerate(levels):
        if g==1: continue
        S64=S[l].double(); roots=F.layer_roots(S64,kv.d); Lsq=roots[0]@roots[0]
        for grp in G[g][l]: K.fuse_kv_group(kv,l,list(grp),S64,roots,qw=True,ow=True,Lsq=Lsq)
    r={"levels":levels,"heads":heads(levels),"ppl_wikitext":float(np.exp(nll(m,X))),"ppl_c4":float(np.exp(nll(m,Xh)))}
    res["arms"][name]=r; print(name,levels,r["heads"],"wikitext",round(r["ppl_wikitext"],2),"c4",round(r["ppl_c4"],2),flush=True); del m,kv
json.dump(res,open(f"{D}/alloc_control_{TAG}.json","w"),indent=1); print("DONE")
