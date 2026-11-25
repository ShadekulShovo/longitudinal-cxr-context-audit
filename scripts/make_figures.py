"""Render numerical audit figures from the shared aggregate extract."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

GROUPS=[('ranking',['gain'],'AUROC difference',0),('events',['onset','resolution'],'Difference in event performance',0),('within',['M4','B2'],'Within-prior share of ranking gain',.5),('reliance',['M4','B2'],'Image-swap share',.5),('decisions',['override','rescue'],'Rate among all resolution events',0)]
def build(data,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for number,(group,keys,title,reference) in enumerate(GROUPS,1):
        rows=[r for r in data['rows'] if r['table']==group]
        fig,ax=plt.subplots(figsize=(7.5,max(3.5,len(rows)*.46+1)))
        for offset,key in enumerate(keys):
            for j,row in enumerate(rows):
                value=row.get(key)
                if value is None or value.get('point') is None:continue
                point=value['point'];ci=value.get('ci95');y=j+(offset-(len(keys)-1)/2)*.18
                color=['#175b8c','#c35422'][offset%2]
                ax.plot(point,y,'o',color=color,label=key if j==0 else None,markersize=5)
                if ci and all(x is not None for x in ci):ax.hlines(y,*ci,color=color,lw=1.6)
        ax.set_yticks(np.arange(len(rows)),[r['label'] for r in rows]);ax.invert_yaxis()
        ax.axvline(reference,color='.55',ls='--',lw=1);ax.set_xlabel(title);ax.grid(axis='x',color='.9')
        if len(keys)>1:ax.legend(loc='best',frameon=False)
        fig.tight_layout()
        fig.savefig(out/f'fig{number}.pdf',metadata={'Creator':'','Producer':'','Author':''})
        fig.savefig(out/f'fig{number}.png',dpi=300,metadata={'Software':''})
        fig.savefig(out/f'fig{number}.eps',metadata={'Creator':''})
        fig.savefig(out/f'fig{number}.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.6));ax.axis('off')
    labels=['Select serial cohorts\nand patient-level splits','Build strictly earlier\npatient histories','Freeze representations;\nfit and select heads','Score confirmation once;\nreplicate externally','Audit ranking, events,\nthresholds and reliance','Audit saved linear logits;\nretain numerical stops']
    for j,label in enumerate(labels):
        col=j%3;row=j//3;x=.17+col*.33;y=.75-row*.48
        ax.add_patch(FancyBboxPatch((x-.145,y-.11),.29,.22,boxstyle='round,pad=.008',facecolor='#f0f5f8',edgecolor='#175b8c',transform=ax.transAxes,zorder=1))
        ax.text(x,y,label,ha='center',va='center',transform=ax.transAxes,fontsize=9,zorder=10,color='black',clip_on=False)
        if col<2:ax.annotate('',xy=(x+.175,y),xytext=(x+.153,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#175b8c'))
    ax.plot([.83,.83,.17],[.632,.51,.51],color='#175b8c',transform=ax.transAxes,lw=1)
    ax.annotate('',xy=(.17,.39),xytext=(.17,.51),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#175b8c'))
    fig.tight_layout();fig.savefig(out/'pipeline.pdf',metadata={'Creator':'','Producer':'','Author':''});fig.savefig(out/'pipeline.png',dpi=300,metadata={'Software':''});fig.savefig(out/'pipeline.eps',metadata={'Creator':''});fig.savefig(out/'pipeline.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',type=Path,default=Path('results/metrics.json'));p.add_argument('--output',type=Path,default=Path('figures'));a=p.parse_args()
    build(json.loads(a.metrics.read_text()),a.output)
