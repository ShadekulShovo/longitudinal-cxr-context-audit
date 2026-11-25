"""Evaluate legitimately obtained, locally prepared arrays with fixed heads."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import models
from report_language import RexStatistics

def run(input_file,output,candidate='M4',reference='M1',replicates=2000,resume=False):
    with np.load(input_file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    names=[k[len('scores_'):] for k in a if k.startswith('scores_')]
    if candidate not in names or reference not in names:raise ValueError('Both paired models required')
    if len(a['y'])!=len(a['patients']) or not np.isin(a['y'],[0,1]).all():raise ValueError('Invalid target alignment')
    scores={k:a['scores_'+k] for k in names}
    cuts={k:{'single':a['single_'+k],'stratified':a['stratified_'+k]} for k in names}
    groups={}
    if 'comparative' in a:groups={'language:comparative':a['comparative'].astype(bool),'language:noncomparative':~a['comparative'].astype(bool)}
    cls=RexStatistics if groups else models.Statistics
    swaps={k:a[k] for k in ['history_swap','image_swap'] if k in a}
    if swaps and len(swaps)!=2:raise ValueError('Both replacement arrays required')
    stat=cls(a['y'],a['prior'],a['patients'],scores,cuts,candidate,reference,groups,swaps=swaps or None)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result_path=output/'aggregate_results.json'
    if result_path.exists():raise FileExistsError('Completed results are never overwritten')
    def status(stage,**values):models.write_json(output/'pipeline_status.json',dict(stage=stage,**values),mutable=True)
    status('running')
    result=models.bootstrap_report(stat,output/'bootstrap',replicates,status,resume=resume)
    models.write_json(result_path,result)
    models.write_json(output/'completion.json',dict(status='completed',replicates=replicates))
    status('completed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',default='M4');p.add_argument('--reference',default='M1');p.add_argument('--replicates',type=int,default=2000);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(a.input,a.output,a.candidate,a.reference,a.replicates,a.resume)
