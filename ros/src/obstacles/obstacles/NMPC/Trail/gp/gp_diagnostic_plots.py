import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.seterr(all='ignore')

d = np.load('obstacle_model.npz', allow_pickle=True)
FEAT = ['x','v','theta','omega','tau']
Z=d['gp_Z']; xm=d['gp_x_mean']; xs=d['gp_x_std']
ellv=d['gp_ell_v']; ellw=d['gp_ell_w']; sf2v=float(d['gp_sf2_v']); sf2w=float(d['gp_sf2_w'])
av=d['gp_alpha_v']; aw=d['gp_alpha_w']; ymv=float(d['gp_y_mean_v']); ymw=float(d['gp_y_mean_w'])
X=d['gp_buf_X']; Y=d['gp_buf_Y']

def ard(A,B,ell,sf2):
    Aw,Bw=A/ell,B/ell
    d2=(Aw**2).sum(1)[:,None]+(Bw**2).sum(1)[None,:]-2.0*Aw@Bw.T
    r=np.sqrt(np.maximum(d2,1e-12)); c=np.sqrt(3.0)*r
    return sf2*(1.0+c)*np.exp(-c)
# inducing-support (DTC) variance for light uncertainty bands
Kzz_v=ard(Z,Z,ellv,sf2v)+1e-6*np.eye(50); Kzzi_v=np.linalg.pinv(Kzz_v)
Kzz_w=ard(Z,Z,ellw,sf2w)+1e-6*np.eye(50); Kzzi_w=np.linalg.pinv(Kzz_w)
def predv(Xraw):
    Xst=(np.atleast_2d(Xraw)-xm)/xs; K=ard(Z,Xst,ellv,sf2v)
    m=K.T@av+ymv; var=np.maximum(sf2v-np.sum(K*(Kzzi_v@K),0),1e-9); return m,np.sqrt(var)
def predw(Xraw):
    Xst=(np.atleast_2d(Xraw)-xm)/xs; K=ard(Z,Xst,ellw,sf2w)
    m=K.T@aw+ymw; var=np.maximum(sf2w-np.sum(K*(Kzzi_w@K),0),1e-9); return m,np.sqrt(var)

pv,_=predv(X); pw,_=predw(X)
def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
R2v,R2w=r2(Y[:,0],pv),r2(Y[:,1],pw)

def binned(xv, yv, nb=22):
    lo,hi=np.percentile(xv,1),np.percentile(xv,99)
    edges=np.linspace(lo,hi,nb+1); cen=0.5*(edges[:-1]+edges[1:])
    mean=np.full(nb,np.nan); sd=np.full(nb,np.nan)
    for i in range(nb):
        m=(xv>=edges[i])&(xv<edges[i+1])
        if m.sum()>=8: mean[i]=yv[m].mean(); sd[i]=yv[m].std()
    return cen,mean,sd

med=np.median(X,0)
def slice_along(col, pred):
    g=np.linspace(np.percentile(X[:,col],1),np.percentile(X[:,col],99),120)
    P=np.tile(med,(120,1)); P[:,col]=g
    m,s=pred(P); return g,m,s

# ============ FIGURE 1: 2x3 diagnostics ============
fig,ax=plt.subplots(2,3,figsize=(16,9))
sub=np.random.RandomState(0).choice(len(Y),min(2500,len(Y)),replace=False)

# parity v
a=ax[0,0]; a.scatter(Y[sub,0],pv[sub],s=6,alpha=.3,color='#2b6cb0')
lim=[np.percentile(Y[:,0],1),np.percentile(Y[:,0],99)]; a.plot(lim,lim,'k--',lw=1)
a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel(r'measured residual $\Delta\dot v$'); a.set_ylabel('GP prediction')
a.set_title(f'(A) Parity  $\\Delta\\dot v$   R$^2$={R2v:.3f}',fontweight='bold')
a.text(.05,.92,'flat cloud = weak fit',transform=a.transAxes,fontsize=9,color='#c53030')

# parity w
a=ax[0,1]; a.scatter(Y[sub,1],pw[sub],s=6,alpha=.3,color='#2f855a')
lim=[np.percentile(Y[:,1],1),np.percentile(Y[:,1],99)]; a.plot(lim,lim,'k--',lw=1)
a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel(r'measured residual $\Delta\dot\omega$'); a.set_ylabel('GP prediction')
a.set_title(f'(B) Parity  $\\Delta\\dot\\omega$   R$^2$={R2w:.3f}',fontweight='bold')

# ARD relevance
a=ax[0,2]; xpos=np.arange(5); w=0.38
a.bar(xpos-w/2, 1/ellv, w, label=r'$\Delta\dot v$ channel', color='#2b6cb0')
a.bar(xpos+w/2, 1/ellw, w, label=r'$\Delta\dot\omega$ channel', color='#2f855a')
a.set_xticks(xpos); a.set_xticklabels(FEAT); a.set_ylabel('relevance  1/lengthscale')
a.set_title('(C) Which inputs the GP uses (ARD)',fontweight='bold'); a.legend()
a.text(.5,.78,'x and θ ≈ ignored\nω, τ, v dominate',transform=a.transAxes,fontsize=10,color='#c53030',ha='center')

# residual vs x  (SPATIAL test) for v
a=ax[1,0]
cen,mn,sd=binned(X[:,0],Y[:,0]); a.errorbar(cen,mn,yerr=sd,fmt='o',ms=4,color='#2b6cb0',alpha=.7,label='measured (binned)')
g,m,s=slice_along(0,predv); a.plot(g,m,'-',color='#c53030',lw=2,label='GP mean (others=median)')
a.fill_between(g,m-2*s,m+2*s,color='#c53030',alpha=.12)
a.set_xlabel('x  (position)'); a.set_ylabel(r'$\Delta\dot v$ residual'); a.legend(fontsize=8)
a.set_title('(D) $\\Delta\\dot v$ vs POSITION x',fontweight='bold')
a.text(.05,.05,'GP mean ~flat in x\n(x is ignored)',transform=a.transAxes,fontsize=9,color='#c53030')

# residual vs omega for v  (the feature it DID use)
a=ax[1,1]
cen,mn,sd=binned(X[:,3],Y[:,0]); a.errorbar(cen,mn,yerr=sd,fmt='o',ms=4,color='#2b6cb0',alpha=.7,label='measured (binned)')
g,m,s=slice_along(3,predv); a.plot(g,m,'-',color='#c53030',lw=2,label='GP mean')
a.fill_between(g,m-2*s,m+2*s,color='#c53030',alpha=.12)
a.set_xlabel(r'$\omega$  (pitch rate)'); a.set_ylabel(r'$\Delta\dot v$ residual'); a.legend(fontsize=8)
a.set_title(r'(E) $\Delta\dot v$ vs $\omega$ (the input it uses)',fontweight='bold')

# residual vs tau for w
a=ax[1,2]
cen,mn,sd=binned(X[:,4],Y[:,1]); a.errorbar(cen,mn,yerr=sd,fmt='o',ms=4,color='#2f855a',alpha=.7,label='measured (binned)')
g,m,s=slice_along(4,predw); a.plot(g,m,'-',color='#c53030',lw=2,label='GP mean')
a.fill_between(g,m-2*s,m+2*s,color='#c53030',alpha=.12)
a.set_xlabel(r'$\tau$  (torque)'); a.set_ylabel(r'$\Delta\dot\omega$ residual'); a.legend(fontsize=8)
a.set_title(r'(F) $\Delta\dot\omega$ vs $\tau$ (the input it uses)',fontweight='bold')

plt.suptitle('GP residual model diagnostics — input z=[x,v,θ,ω,τ] → [Δv̇, Δω̇]',fontsize=14,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('gp_diagnostics.png',dpi=130,bbox_inches='tight'); plt.close()
print('saved gp_diagnostics.png   R2v=%.3f R2w=%.3f'%(R2v,R2w))

# ============ FIGURE 2: learning over episodes (from CSV pre-update preds) ============
cols=['episode','r_v_dot','r_omega_dot','gp_v_dot_pred_pre','gp_omega_dot_pred_pre','gp_ready']
df=pd.read_csv('obstacle_mujoco.csv', usecols=cols)
df=df[df['gp_ready']==1]
eps=sorted(df['episode'].unique())
rmse_v=[]; rms_v=[]; rmse_w=[]; rms_w=[]
for e in eps:
    g=df[df['episode']==e]
    rmse_v.append(np.sqrt(np.mean((g['gp_v_dot_pred_pre']-g['r_v_dot'])**2)))
    rms_v.append(np.sqrt(np.mean(g['r_v_dot']**2)))
    rmse_w.append(np.sqrt(np.mean((g['gp_omega_dot_pred_pre']-g['r_omega_dot'])**2)))
    rms_w.append(np.sqrt(np.mean(g['r_omega_dot']**2)))

fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(eps,rms_v,'o-',color='#999',label='residual RMS (signal to beat)')
ax[0].plot(eps,rmse_v,'o-',color='#2b6cb0',label='GP one-step pred error')
ax[0].set_xlabel('episode'); ax[0].set_ylabel(r'$\Delta\dot v$  RMS'); ax[0].legend(); ax[0].set_title(r'(A) $\Delta\dot v$: does prediction error fall below the signal?',fontweight='bold')
ax[1].plot(eps,rms_w,'o-',color='#999',label='residual RMS (signal to beat)')
ax[1].plot(eps,rmse_w,'o-',color='#2f855a',label='GP one-step pred error')
ax[1].set_xlabel('episode'); ax[1].set_ylabel(r'$\Delta\dot\omega$  RMS'); ax[1].legend(); ax[1].set_title(r'(B) $\Delta\dot\omega$: prediction error vs signal over episodes',fontweight='bold')
plt.suptitle('Out-of-sample learning over episodes (pre-update GP prediction vs measured residual)',fontsize=13,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('gp_learning_over_episodes.png',dpi=130,bbox_inches='tight'); plt.close()
print('saved gp_learning_over_episodes.png   episodes=%d'%len(eps))

# print the spatial-signal check
cen,mn,sd=binned(X[:,0],Y[:,0])
print('v-residual binned by x: span of bin-means = %.3f (vs residual std %.3f) -> spatial signal %s'%(
    np.nanmax(mn)-np.nanmin(mn), Y[:,0].std(), 'WEAK' if (np.nanmax(mn)-np.nanmin(mn))<Y[:,0].std() else 'present'))
