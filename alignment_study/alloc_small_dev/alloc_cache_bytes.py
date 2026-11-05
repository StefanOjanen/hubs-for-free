# Committed from the parallel session's private/drafts on 2026-11-05 (exploratory, CPU, 512-token windows); ROOT made repository-relative, nothing else changed.
# Development (fork, CPU): below two bits per cached value, is merging plus cache rounding the frontier, and does the
# cost-aware allocation move it?  Matched cache bytes on TinyLlama: budgets 1/8 and 1/16 of the 16-bit all-heads cache.
# Cache rounding as cachebudget_dev.py's KIVI layout (keys per channel over 32-token groups, values per token).
import os, sys, json, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root (this file sits in alignment_study/alloc_small_dev/); sys.path.insert(0, ROOT+"/alignment_study"); os.chdir(ROOT)
import gqa_common as C, fuse_common as F, fuse_kv as K, kvfuse, zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.set_num_threads(6)
NAME="TinyLlama/TinyLlama_v1.1"; N_CAL,T_CAL,N_EVAL,T_EVAL=32,512,24,512
prev=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"alloc_dev_TinyLlama_v1.1.json")))
DP=prev["arms"]["dp"]["levels"]
def rtn_tokens(y,hd,bits):
    s=y.shape; z=y.reshape(*s[:-1],-1,hd).float(); lo,hi=z.amin(-1,keepdim=True),z.amax(-1,keepdim=True)
    sc=((hi-lo)/(2**bits-1)).clamp_min(1e-8); return (torch.round((z-lo)/sc)*sc+lo).reshape(s).to(y.dtype)
def rtn_channels(y,bits,g=32):
    B_,T,D=y.shape; pad=(-T)%g; z=torch.nn.functional.pad(y.float(),(0,0,0,pad),mode="replicate").reshape(B_,-1,g,D)
    lo,hi=z.amin(2,keepdim=True),z.amax(2,keepdim=True); sc=((hi-lo)/(2**bits-1)).clamp_min(1e-8)
    return (torch.round((z-lo)/sc)*sc+lo).reshape(B_,-1,D)[:,:T].to(y.dtype)
tok=AutoTokenizer.from_pretrained(NAME); cal=C.windows_c4(tok,N_CAL,T_CAL); X=Z.wikitext2_windows(tok,T=T_EVAL)[:N_EVAL]
def ppl(m):
    tot=[]
    for w in range(X.shape[0]):
        x=X[w:w+1]; lg=m(input_ids=x,use_cache=False).logits[0,:-1].float(); tot.append(torch.nn.functional.cross_entropy(lg,x[0,1:],reduction="none").numpy())
    return float(np.exp(np.concatenate(tot).mean()))
def load():
    m=AutoModelForCausalLM.from_pretrained(NAME,dtype=torch.float32,attn_implementation="sdpa").eval(); return m,K.KV(m)
m,kv=load(); L,H=kv.L,kv.nkv; S,_=K.second_moments(m,kv,cal,"cpu")
G={2:[kvfuse.min_cost_pairs(kvfuse.pair_costs(kv,l,S[l].double())) for l in range(L)],4:[kvfuse.quad_groups(kv,l,S[l].double()) for l in range(L)]}
del m,kv
def run(levels,bits):
    m,kv=load()
    for l,g in enumerate(levels):
        if g==1: continue
        S64=S[l].double(); roots=F.layer_roots(S64,kv.d); Lsq=roots[0]@roots[0]
        for grp in G[g][l]: K.fuse_kv_group(kv,l,list(grp),S64,roots,qw=True,ow=True,Lsq=Lsq)
    hooks=[]
    if bits<16:
        for l in range(L):
            a=kv.attn(l); hooks.append(a.k_proj.register_forward_hook(lambda mod,i,o,b=bits: rtn_channels(o,b))); hooks.append(a.v_proj.register_forward_hook(lambda mod,i,o,b=bits: rtn_tokens(o,kv.hd,b)))
    r=ppl(m); heads=sum(H//g for g in levels); frac=heads*bits/(L*H*16)
    for h in hooks: h.remove()
    del m,kv; return {"ppl":r,"heads":heads,"bits":bits,"cache_fraction":frac}
res={"model":NAME,"exploratory":True,"quant":"kivi","dp_levels":DP,"arms":{}}
arms={"all_16":([1]*L,16),"all_2":([1]*L,2),"half_uniform_4":([2]*L,4),"half_dp_4":(DP,4),"quarter_8":([4]*L,8),
      "all_1":([1]*L,1),"half_uniform_2":([2]*L,2),"half_dp_2":(DP,2),"quarter_4":([4]*L,4)}
for name,(lv,b) in arms.items():
    res["arms"][name]=run(lv,b); print(name,{k:(round(v,3) if isinstance(v,float) else v) for k,v in res["arms"][name].items()},flush=True)
    json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"alloc_cache_TinyLlama_v1.1.json"),"w"),indent=1)
print("DONE")
