"""PK K=50 precision floor against CONVERGED chains (200k draws)."""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
from scipy.linalg import sqrtm
sys.path.insert(0, os.path.expanduser("~/amortix_exp"))
sys.path.insert(0, os.path.expanduser("~/amortix_exp/examples"))
OUT = os.environ.get("OUT", os.path.expanduser("~/amortix_exp/paper2/sat"))
from amortix import FlowPosterior
from amortix.mcmc import metropolis
from amortix.problems.design_zoo import PharmacoKineticsDesign, pk_logpost_factory

prob = PharmacoKineticsDesign()
lo = prob.prior.low.numpy().astype(np.float64); hi = prob.prior.high.numpy().astype(np.float64)
rng = hi - lo; B, K = 32, 50

def fd2(A,R):
    s=R.std(0)+1e-12; A,R=A/s,R/s; S1,S2=np.cov(A.T),np.cov(R.T)
    cm=sqrtm(S2@S1); cm=cm.real if np.iscomplexobj(cm) else cm
    return float(((A.mean(0)-R.mean(0))**2).sum()+np.trace(S1+S2-2*cm))
def chain(a):
    tt,y,seed=a; torch.set_num_threads(1)
    lp=pk_logpost_factory(tt,y,dose=prob.DOSE,logsd=prob.LOGSD)
    c,_=metropolis(lp,0.5*(lo+hi),n_samples=200000,prior_low=lo,prior_high=hi,seed=seed)
    c=np.asarray(c); return c[np.linspace(0,len(c)-1,4000).astype(int)]

gen=torch.Generator().manual_seed(2000+K)
m_true=prob.prior.sample(B,gen); raw=prob.trajectories(m_true,generator=gen)
kmax=prob.observer.k_max
tokens=torch.zeros(B,kmax,6); mask=torch.zeros(B,kmax,dtype=torch.bool); jobs=[]
for i in range(B):
    tidx,cidx=prob.sample_design(gen,K); tidx=torch.unique(tidx); cidx=torch.zeros_like(tidx)
    tk=prob.tokens_for(raw[i],tidx,cidx,gen); tokens[i,:tk.shape[0]]=tk; mask[i,:tk.shape[0]]=True
    y=tk[:,1].numpy().astype(np.float64); tt=tidx.numpy()*prob.observer.dt_sim
    jobs.append((tt,y,100+i))
post=FlowPosterior(prob); post.load_state_dict(torch.load(f"{OUT}/ckpt17_pk_small_120000_s0.pt",map_location="cpu"))
smp=post.sample_batch(tokens,n=4000,seed=0,mask=mask).numpy()
with ProcessPoolExecutor(max_workers=32) as ex: refs=list(ex.map(chain,jobs))
bias=np.median([np.abs(smp[i].mean(0)-refs[i].mean(0))/rng for i in range(B)],0)
sd=np.median([refs[i].std(0)/rng for i in range(B)],0)
fid=np.median([fd2(smp[i],refs[i]) for i in range(B)])
print("PK K=50 converged, model = 120k (best):")
for n,b,s in zip(prob.prior.names,bias,sd):
    print(f"  {n}: sd/range {s:.4f}  err/range {b:.4f}  err/sd {b/s:.3f}")
print(f"  FID median {fid:.4f}")
print("JOB_DONE", flush=True)
