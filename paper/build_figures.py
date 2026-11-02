from pathlib import Path
import os,json,statistics,hashlib
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'.cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(Path(__file__).resolve().parents[1]/'.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
BASE=Path(__file__).resolve().parents[1];S=BASE/'results/final';O=Path(__file__).resolve().parent/'figures'
F=json.loads((S/'final_results.json').read_text());D=json.loads((S/'counterfactual_schema_diagnostic.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
C={'B0':'#777777','B1':'#315D85','N1':'#C47A39','P1':'#3D8177'};names={'B0':'Base','B1':'Text','N1':'Numeric prefix','P1':'Learned prefix'}
figure_data={}
def save(fig,name,values):
    for ext in ('pdf','png'):fig.savefig(O/(name+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.08)
    figure_data[name]=values;plt.close(fig)
def box(ax,xy,w,h,text,color='#F1F4F7',size=10):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.012',facecolor=color,edgecolor='#5C6975',linewidth=.9,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.4,zorder=3)
def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#5C6975',linewidth=1,shrinkA=5,shrinkB=5,zorder=1))
fig,ax=plt.subplots(figsize=(8.5,3.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_position([.015,.06,.97,.90])
box(ax,(.01,.56),.27,.29,'Narrative, table, question\nIdentical text across models',size=10)
box(ax,(.01,.14),.27,.29,'Structured table cells\nValue, row, period, unit',color='#FAF2E9',size=10)
box(ax,(.36,.56),.27,.29,'Text token embeddings\nPinned Qwen3-8B tokenizer',size=10)
box(ax,(.36,.14),.27,.29,'Numerical encoder (N1)\nFour learned pooling queries\n688,256 parameters',color='#FAF2E9',size=10)
box(ax,(.73,.14),.26,.71,'Qwen3-8B + LoRA\n\nOptional prefix\n+ text embeddings\n\nJSON answer + scale',size=10.5)
for a,b in [((.28,.705),(.36,.705)),((.28,.285),(.36,.285)),((.63,.705),(.73,.705)),((.63,.285),(.73,.285))]:arrow(ax,a,b)
ax.text(.5,.015,'B1: no prefix     |     N1: four cell-conditioned tokens     |     P1: four input-independent tokens',ha='center',fontsize=9)
save(fig,'architecture',{'kind':'Schematic; no performance values','prefix_tokens':4,'encoder_parameters':688256})
fig,ax=plt.subplots(figsize=(8.5,2.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_position([.01,.05,.98,.9])
for x,text in [(.005,'Pinned source data\n256 train + 128 dev\nFull selected contexts'),(.265,'Three training seeds\nB1 / N1 / P1\n9 adapted checkpoints'),(.525,'Frozen final evaluation\n64 FinQA + 64 TAT-QA\n128 counterfactual cases'),(.785,'All 10 conditions\n2,560 predictions\nLocal re-scoring')]:box(ax,(x,.34),.21,.50,text,size=9)
for x in [.215,.475,.735]:arrow(ax,(x,.59),(x+.05,.59))
ax.text(.5,.11,'P1 specified after development, before its training and all final predictions; all predetermined seeds retained.',ha='center',fontsize=9)
save(fig,'study_design',{'training':256,'development':128,'final_public':128,'final_synthetic':128,'conditions':10,'predictions':2560,'seeds':[42,11,23]})
# Each seed uses a fixed row offset and distinct marker, rather than unexplained jitter.
from matplotlib.lines import Line2D
fig,axes=plt.subplots(1,2,figsize=(8.5,3.65),sharey=True);fig.subplots_adjust(left=.15,right=.985,bottom=.27,top=.85,wspace=.12);v={}
markers=['o','s','^'];offsets=[-.16,0,.16]
for ax,source,title in zip(axes,['finqa','tatqa'],['FinQA raw reference agreement','TAT-QA arithmetic exact match']):
    v[source]={}
    for i,m in enumerate(C):
        labs=['B0'] if m=='B0' else [f'{m}_seed{s}' for s in (42,11,23)]
        vals=[100*F['summaries'][l]['datasets'][source]['exact_match'] for l in labs];avg=statistics.mean(vals);v[source][m]=vals
        if m=='B0':
            ax.scatter(vals,[i],color=C[m],s=38,marker='D',zorder=4,clip_on=False)
        else:
            ax.hlines(i,min(vals),max(vals),color=C[m],lw=1,alpha=.6,zorder=2)
            for value,off,marker in zip(vals,offsets,markers):
                ax.scatter(value,i+off,s=32,marker=marker,facecolors='white',edgecolors=C[m],linewidth=1.2,zorder=4)
            ax.scatter(avg,i,s=24,marker='D',color=C[m],zorder=5)
        ax.text(99,i,f'{avg:.2f}%',ha='right',va='center',fontsize=9,color=C[m],bbox={'facecolor':'white','edgecolor':'none','pad':1})
    ax.set(xlim=(-2,102),xticks=[0,20,40,60,80,100],xlabel='Matching answers (%)',title=title,ylim=(3.5,-.5))
    ax.set_yticks(range(4),['Base (B0)','Text (B1)','Numeric prefix (N1)','Learned prefix (P1)']);ax.grid(axis='x',color='#DDE2E6',linewidth=.6);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
handles=[Line2D([],[],color='#35434F',marker=marker,linestyle='none',markerfacecolor='white',markersize=5,label=f'Seed {seed}') for marker,seed in zip(markers,[42,11,23])]
handles.append(Line2D([],[],color='#35434F',marker='D',linestyle='none',markersize=4,label='Mean / single base run'))
fig.legend(handles=handles,frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.5,.015),fontsize=9)
save(fig,'benchmark_scores',v)
fig,ax=plt.subplots(figsize=(8.2,2.65));fig.subplots_adjust(left=.24,right=.70,bottom=.26,top=.94);intervals={}
for y,(source,title) in enumerate([('synthetic_cf','Synthetic strict labels'),('tatqa','TAT-QA exact match'),('finqa','FinQA raw reference')]):
    x=F['mean_seed_fusion_minus_text'][source];mean=100*x['mean_paired_delta'];lo,hi=[100*z for z in x['descriptive_seed_and_context_bootstrap_95_interval']]
    ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=C['N1'],markersize=6);ax.text(1.035,y,f'{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]',transform=ax.get_yaxis_transform(),va='center',fontsize=10)
    intervals[source]={'delta_pp':mean,'lower_pp':lo,'upper_pp':hi}
ax.axvline(0,color='#606060',ls='--',lw=.9);ax.set_yticks(range(3),['Synthetic strict labels','TAT-QA exact match','FinQA raw reference']);ax.set(xlim=(-25,95),xticks=[-20,0,20,40,60,80],xlabel='Numerical prefix minus text (percentage points)',ylim=(-.55,2.55));ax.grid(axis='x',alpha=.2)
save(fig,'paired_intervals',intervals)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.55),layout='constrained');v={}
for ax,key,title in zip(axes,['label_only_case_accuracy','all_four_correct_rate'],['Single-label accuracy','All-four-correct scenarios']):
    v[key]={}
    for i,m in enumerate(C):
        labs=['B0'] if m=='B0' else [f'{m}_seed{s}' for s in (42,11,23)]
        strict=[100*(F['summaries'][l]['datasets']['synthetic_cf']['exact_match'] if key=='label_only_case_accuracy' else F['counterfactuals'][l]['all_four_correct_rate']) for l in labs]
        diag=[100*D['summaries'][l][key] for l in labs]
        ax.bar(i-.18,statistics.mean(strict),width=.32,color=C['B1'],label='Frozen strict' if i==0 else None)
        ax.bar(i+.18,statistics.mean(diag),width=.32,color='#D5AA55',label='Post-hoc: ignore scale' if i==0 else None,hatch='//',edgecolor='#8A6F39',linewidth=.4)
        v[key][m]={'strict':strict,'post_hoc':diag}
    ax.set(ylim=(0,105),ylabel='Correct (%)',title=title);ax.set_xticks(range(4),['Base','Text','Numeric\nprefix','Learned\nprefix']);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,ncol=2,loc='outside lower center',fontsize=10)
save(fig,'schema_diagnostic',v)
fig,ax=plt.subplots(figsize=(8.5,3.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_position([.015,.03,.97,.94])
for xy,text in [((.02,.58),'Original: margin rises\nClaim: margin improved\nSUPPORTED'),((.57,.58),'Claim flip: margin rises\nClaim: margin declined\nCONTRADICTED'),((.02,.08),'Numeric flip: margin falls\nClaim: margin improved\nCONTRADICTED'),((.57,.08),'Both flips: margin falls\nClaim: margin declined\nSUPPORTED')]:box(ax,xy,.41,.28,text,size=10)
for a,b in [((.43,.72),(.57,.72)),((.43,.22),(.57,.22)),((.225,.58),(.225,.36)),((.775,.58),(.775,.36))]:arrow(ax,a,b)
ax.text(.5,.94,'Four related variants per scenario; labels use exact displayed values',ha='center',fontsize=10)
ax.text(.5,.79,'Claim flip',ha='center',fontsize=8.5,color='#52616A')
ax.text(.5,.29,'Claim flip',ha='center',fontsize=8.5,color='#52616A')
ax.text(.225,.47,'Numeric flip',ha='center',fontsize=8.5,color='#52616A',bbox={'facecolor':'white','edgecolor':'none','pad':3})
ax.text(.775,.47,'Numeric flip',ha='center',fontsize=8.5,color='#52616A',bbox={'facecolor':'white','edgecolor':'none','pad':3})
save(fig,'counterfactual_design',{'kind':'Illustrative design only; no empirical example or performance claim','cases_per_scenario':4,'balanced_labels':True})
fig,axes=plt.subplots(1,2,figsize=(8.2,3.45),layout='constrained');v={}
dev=BASE/'results/development'
for m,label in [('B1','Text'),('N1','Numeric prefix')]:
    logs=json.loads((dev/(m+'_training.json')).read_text());steps=[x['step'] for x in logs]
    axes[0].plot(steps,[x['loss'] for x in logs],color=C[m],label=label,linewidth=1.2)
    axes[1].plot(steps,[x['grad_norm'] for x in logs],color=C[m],label=label,linewidth=1.2)
    v[m]=[{'step':x['step'],'loss':x['loss'],'pre_clip_gradient_norm':x['grad_norm']} for x in logs]
axes[0].set(title='Recorded training loss',xlabel='Optimizer update',ylabel='Loss');axes[1].set(title='Gradient norm before clipping',xlabel='Optimizer update',ylabel='Global norm (log scale)',yscale='log');axes[1].axhline(1,color='#606060',ls='--',lw=.8)
for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10)
save(fig,'training_diagnostics',v)
(O/'figure_data.json').write_text(json.dumps(figure_data,indent=2))
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps({'final_results_sha256':hashlib.sha256((S/'final_results.json').read_bytes()).hexdigest(),'post_hoc_results_sha256':hashlib.sha256((S/'counterfactual_schema_diagnostic.json').read_bytes()).hexdigest(),'figures':list(figure_data),'all_seeds_retained':True},indent=2))
print(json.dumps({'figures':len(figure_data),'directory':str(O)},indent=2))
