"""Extract only numerical aggregate endpoints from supplied analysis summaries.

No study-level inputs are read. Output contains no source file locations.
"""
import argparse
import json
import math
from pathlib import Path

def numeric(value):
    if value is None or isinstance(value,bool):return value
    if isinstance(value,(float,int)):return value if math.isfinite(value) else None
    if isinstance(value,list) and all(x is None or isinstance(x,(int,float,bool,list)) for x in value):return [numeric(x) for x in value]
    if isinstance(value,dict):
        allowed=('point','ci95','valid_replicates','reliable','studies','patients','replicates','B','seed','positives','onset_events','resolution_events')
        return {k:numeric(value[k]) for k in allowed if k in value}
    raise ValueError('Non-numerical endpoint')

def metric(point,ci=None,valid=None):
    if isinstance(point,dict):return numeric(point)
    result={'point':point,'ci95':ci}
    if valid is not None:result['valid_replicates']=valid
    return result

def extract(summary,decisions,nih_upgrade,nih_sex):
    s=summary
    rows=[]
    def add(table,label,**cells):rows.append(dict(table=table,label=label,**cells))
    def standardized(r):return metric(r['macro_difference'],r['macro_ci95'],r.get('valid_replicates'))
    add('ranking','TAIX: M4 minus M0',gain=standardized(s['confirmation']['primary']))
    for key,label in [('S1_all_C','TAIX: all follow-ups'),('S2_change','TAIX: changed studies'),('S3_first','TAIX: no history')]:
        add('ranking',label,gain=standardized(s['confirmation']['secondary'][key]))
    n=s['NIH_M4']
    add('ranking','NIH: M4 minus M1',gain=metric(n['replication']['M4_minus_M1'],n['replication']['ci95']))
    pc=s['PadChest']['cohorts']
    rex=s['ReXGradient']['analyses']
    cohorts=[('PadChest',pc['official'],pc['biovilt']),('ReXGradient primary',rex['primary_uncertain_negative']['official'],rex['primary_uncertain_negative']['biovilt']),('ReXGradient uncertain-positive',rex['sensitivity_uncertain_positive']['official'],rex['sensitivity_uncertain_positive']['biovilt'])]
    for label,m,b in cohorts:
        add('ranking',label+': M4 minus M1',gain=metric(m['metrics']['contrasts/M4_minus_M1/macro']))
    add('ranking','PadChest physician-labelled',gain=metric(pc['official']['metrics']['subsets/method:Physician/macro_gain']))
    for label,m,b in [('NIH',None,s['NIH_BioViLT']['analysis'])]+cohorts:
        add('ranking',label+': B2 minus B1',gain=metric(b['metrics']['contrasts/B2_minus_B1/macro']))
    t=s['TAIX_context_P']
    add('events','TAIX M4 minus M0',onset=metric(t['A8']['onset_sensitivity']['difference']),resolution=metric(t['A8']['resolution_specificity']['difference']))
    nr=n['A8']['M4_vs_M1']
    add('events','NIH M4 minus M1',**{k:metric(nr[v]['M4']-nr[v]['ref'],nr[v]['ci95_diff']) for k,v in [('onset','onset_sensitivity'),('resolution','resolution_specificity')]})
    def event_row(table,label,report,prefix):
        add(table,label,**{k:metric(report['metrics'][prefix+'/'+v+'_pooled']) for k,v in [('onset','onset_sensitivity'),('resolution','resolution_specificity')]})
    nb=s['NIH_BioViLT']['analysis']
    event_row('events','NIH B2 minus B1',nb,'threshold_contrasts/ST')
    for label,m,b in cohorts:
        event_row('events',label+' M4 minus M1',m,'threshold_contrasts/ST')
        event_row('events',label+' B2 minus B1',b,'threshold_contrasts/ST')
        event_row('thresholds',label+' M4 PST minus M1 ST',m,'threshold_contrasts/PST_vs_reference_ST')
    event_row('thresholds','NIH B2 PST minus B1 ST',nb,'threshold_contrasts/PST_vs_reference_ST')
    event_row('thresholds','NIH B2 PST minus B1 PST',nb,'threshold_contrasts/PST_vs_reference_PST')
    add('within','TAIX',M4=metric(s['TAIX_U2']['intervals']['macro.change_share']))
    add('within','NIH',M4=metric(nih_upgrade['analysis']['metrics']['U2/macro/change_share']),B2=metric(nb['metrics']['U2/macro/change_share']))
    for label,m,b in cohorts:add('within',label,M4=metric(m['metrics']['U2/macro/change_share']),B2=metric(b['metrics']['U2/macro/change_share']))
    add('reliance','TAIX',M4=metric(t['A7']['image_share']))
    add('reliance','NIH',M4=metric(n['A7']['image_share']),B2=metric(s['NIH_BioViLT_A7']['analysis']['metrics']['A7/image_share/macro']))
    for label,m,b in cohorts:
        if label=='PadChest':a=s['PadChest_BioViLT_A7']['analysis']
        else:a=s['ReXGradient_encoder_A7']['analyses']['primary_uncertain_negative' if label.endswith('primary') else 'sensitivity_uncertain_positive']
        add('reliance',label,M4=metric(m['metrics']['A7/image_share/macro']),B2=metric(a['metrics']['A7/image_share/macro']))
    for label,x in [('PadChest',decisions['PadChest']['released/official']),('ReXGradient primary',decisions['ReX']['primary_uncertain_negative/official']),('ReXGradient uncertain-positive',decisions['ReX']['sensitivity_uncertain_positive/official'])]:
        a=x['analysis']['metrics']
        add('decisions',label,override=metric(a['X2/resolution/override/rate/macro']),rescue=metric(a['X2/resolution/rescue/rate/macro']),**{event:metric(a['X2/'+event+'/net_rescues_per1000_studies/macro']) for event in ['resolution','onset','persistent_positive','persistent_negative']})
        add('attribution',label,**{event:metric(a['X1/train_mean/'+event+'/group_share/image/macro']) for event in ['all','onset','resolution']})
    for label,m,b in cohorts[1:]:
        for model,r in [('M4',m),('B2',b)]:
            add('language',label+' '+model,**{key:metric(r['metrics']['U10/'+key+'/macro_gain']) for key in ['comparative','noncomparative','comparative_minus_noncomparative']})
    for model,r in decisions['PadChest_U10'].items():
        a=r.get('analysis',r)
        add('language','PadChest '+('M4' if model=='official' else 'B2'),**{key:metric(a['metrics']['U10/'+key+'/macro_gain']) for key in ['comparative','noncomparative','comparative_minus_noncomparative']})
    counts={'TAIX':numeric(s['confirmation']['primary']),'NIH':numeric(s['NIH_BioViLT_A7']['counts'])}
    for label,m,b in cohorts:counts[label]={'M4':numeric(m['counts']),'B2':numeric(b['counts'])}
    # Only explicit numerical metric containers are retained for the supplement.
    full={}
    for label,m,b in [('NIH',nih_upgrade, s['NIH_BioViLT']['analysis'])]+cohorts:
        for model,r in [('M4',m),('B2',b)]:
            a=r.get('analysis',r)
            full[label+' '+model]={k:numeric(v) for k,v in a['metrics'].items()}
    for dataset in ['PadChest','ReX']:
        for model,r in decisions[dataset].items():full[dataset+' '+model+' linear']={k:numeric(v) for k,v in r['analysis']['metrics'].items()}
    full['NIH sex']={k:numeric(v) for k,v in nih_sex['analysis']['metrics'].items()}
    return dict(rows=rows,cohort_counts=counts,supplement_metrics=full)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['summary','decisions','nih-upgrade','nih-sex','output']:p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    inputs=[json.loads(getattr(a,k).read_text(encoding='utf-8')) for k in ['summary','decisions','nih_upgrade','nih_sex']]
    result=extract(*inputs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
