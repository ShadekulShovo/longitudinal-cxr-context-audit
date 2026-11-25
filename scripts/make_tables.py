"""Rebuild all numerical tables from aggregate metrics."""
import argparse,json
from pathlib import Path

def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def fmt(v,places=4):
    if v is None:return 'Not available'
    if not isinstance(v,dict):return esc(v)
    point=v.get('point')
    if point is None:return 'Undefined'
    t=f'{point:.{places}f}'
    ci=v.get('ci95')
    if ci and all(x is not None for x in ci):t+=f' [{ci[0]:.{places}f}, {ci[1]:.{places}f}]'
    if v.get('reliable') is False:t+=r'\textsuperscript{*}'
    return t

CONFIG=[
 ('ranking','Ranking criteria pass in TAIX, NIH and overall PadChest, but fail in ReXGradient and the physician-labelled subset. Positive estimates alone do not establish replication.','tab:ranking',['gain']),
 ('within','Most measured ranking gain separates prior states. Within-prior ratios are not probabilities and may fall outside the unit interval.','tab:within',['M4','B2']),
 ('events','Primary-rule onset sensitivity falls; resolution losses have a BioViL-T NIH exception. Primary and uncertainty-positive rules are shown separately.','tab:events',['onset','resolution']),
 ('thresholds','Prior-state threshold results depend on the comparator and replication outcome. The noninferiority margin is -0.03; a method claim additionally requires replication.','tab:thresholds',['onset','resolution']),
 ('reliance','Current-image replacements move aggregate predictions more than history replacements. NIH M4 has a point estimate only. Donor rules and model bases differ.','tab:reliance',['M4','B2']),
 ('decisions','Resolution overrides outnumber rescues while persistent-state gains coexist. Override and rescue rates use all resolution events, macro over findings. Net rates use all follow-ups, mean per finding, per 1,000 studies.','tab:decisions',['override','rescue','resolution','onset','persistent_positive','persistent_negative']),
 ('attribution','M4 image contributions dominate overall but decrease at resolution. Shares use the training-mean reference and are not causal disease attributions.','tab:attribution',['all','onset','resolution']),
 ('language','Comparative-report concentration depends on the label rule and is inconclusive in the PadChest companion. PadChest is post hoc. ReXGradient primary and sensitivity readings have equal prominence.','tab:language',['comparative','noncomparative','comparative_minus_noncomparative'])]

def build(data,out):
    out.mkdir(parents=True,exist_ok=True)
    for i,(group,caption,label,keys) in enumerate(CONFIG,1):
        records=[r for r in data['rows'] if r['table']==group]
        if group=='decisions':
            heads=['PadChest','ReX primary','ReX uncertain-positive']
            lines=[r'\begin{table}[p]\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrr}\toprule',r'Endpoint & '+' & '.join(heads)+r' \\ \midrule']
            for key in keys:lines.append(esc(key.replace('_',' '))+' & '+' & '.join(fmt(row[key]) for row in records)+r' \\')
            lines += [r'\bottomrule\end{tabular}}\end{table}']
            text='\n'.join(lines).replace(r'\resizebox{\textwidth}{!}{%','').replace(r'\end{tabular}}',r'\end{tabular}')
            text=text.replace(r'\begin{tabular}{lrrr}',r'\begin{tabularx}{\textwidth}{@{}p{.25\textwidth}XXX@{}}').replace(r'\end{tabular}',r'\end{tabularx}')
            (out/f'table{i}.tex').write_text(text+'\n',encoding='utf-8')
            continue
        lines=[r'\begin{table}[p]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{l'+'r'*len(keys)+'}',r'\toprule','Comparison & '+' & '.join(esc(k.replace('_',' ')) for k in keys)+r' \\',r'\midrule']
        for row in records:
            label=esc(row['label'])
            if group=='ranking':
                v=row['gain'];lower=v['ci95'][0]
                if 'B2' in row['label']:reading='descriptive'
                elif row['label']=='TAIX: no history':reading='met' if lower>-.005 else 'failed'
                elif row['label']=='TAIX: changed studies':reading='met' if lower>-.010 else 'failed'
                elif row['label']=='TAIX: all follow-ups':reading='met' if lower>0 else 'failed'
                else:reading='met' if v['point']>=.010 and lower>0 else 'failed'
                label+=' ('+reading+')'
                if reading=='failed':label=r'\textbf{'+label+'}'
            lines.append(label+' & '+' & '.join(fmt(row.get(k)) for k in keys)+r' \\')
        if group=='ranking':lines.append(r'TAIX graph-model secondary & Not tested: earlier gate failed \\')
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
        text='\n'.join(lines).replace(r'\resizebox{\textwidth}{!}{%','').replace(r'\end{tabular}}',r'\end{tabular}')
        spec='@{}p{.35\\textwidth}'+'X'*len(keys)+'@{}'
        text=text.replace(r'\begin{tabular}{l'+'r'*len(keys)+'}',r'\begin{tabularx}{\textwidth}{'+spec+'}').replace(r'\end{tabular}',r'\end{tabularx}')
        (out/f'table{i}.tex').write_text(text+'\n',encoding='utf-8')
    lines=[r'\begin{table}[p]\centering\caption{Analysis cohort counts. TAIX uses different-reader follow-ups; external datasets use eligible follow-ups.}\label{tab:cohorts}',r'\begin{tabular}{llrr}\toprule Dataset & Arm & Studies & Patients \\ \midrule']
    for name,c in data['cohort_counts'].items():
        for arm,v in (c.items() if 'M4' in c else [('M4',c)]):
            lines.append(esc(name)+' & '+arm+f" & {v['studies']:,} & {v['patients']:,}"+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out/'cohorts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # A complete numerical appendix retains nulls, failed and unreliable endpoints.
    lines=[r'\scriptsize',r'\begin{longtable}{p{0.60\textwidth}p{0.22\textwidth}r}',r'\caption{Extracted aggregate endpoint inventory. Each analysis is reported separately; no pooled effect is inferred. An asterisk marks a recorded unreliable endpoint. The numerical extract also retains finding-level endpoints.}\\',r'\toprule Endpoint & Estimate [95\% CI] & Valid draws \\ \midrule\endfirsthead',r'\toprule Endpoint & Estimate [95\% CI] & Valid draws \\ \midrule\endhead']
    for group,metrics in data['supplement_metrics'].items():
        for key,value in metrics.items():
            if not isinstance(value,dict) or 'point' not in value:continue
            if not (key.endswith('/macro') or key.endswith('_pooled')):continue
            endpoint=esc(group+' / '+key).replace('/',r'/\allowbreak{}')
            lines.append(endpoint+' & '+fmt(value)+' & '+str(value.get('valid_replicates','--'))+r' \\')
    lines += [r'\bottomrule\end{longtable}',r'\normalsize']
    (out/'endpoint_inventory.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',type=Path,default=Path('results/metrics.json'));p.add_argument('--output',type=Path,default=Path('tables'));a=p.parse_args()
    build(json.loads(a.metrics.read_text()),a.output)
