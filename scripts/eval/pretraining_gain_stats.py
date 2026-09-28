"""Patient-clustered uncertainty from the locally produced per-subject scores; no model runs.

Point estimates keep equal weight per evaluation unit within each cohort. PD
states stay together when resampled. Tests use one mean difference per patient.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/lib'))
from guarded_write import guarded_dump
from evaluation_contract import safe_result_path, sha256
from patient_statistics import cluster_means, groups, paired_test, interval
COHORTS=['cp','normal','vdk_stroke','bmclab_pd']
NBOOT=5000

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/evaluation/kinetics_cv5.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/evaluation/pretraining_gain.json')
    args=parser.parse_args();path=safe_result_path(args.out,ROOT)
    data=json.loads(args.input.read_text())
    pre=data['pretrain/CV5']['per_subject_overall'];scr=data['scratch/CV5']['per_subject_overall']
    for c in COHORTS:
        if not pre[c] or set(pre[c])!=set(scr[c]):raise ValueError(f'Unmatched paired population: {c}')
    rng=np.random.default_rng(42)
    bootstrap={c:cluster_means(c,pre[c],rng,NBOOT) for c in COHORTS}
    out={'n_boot':NBOOT,'seed':42,'input_sha256':sha256(args.input),
         'bootstrap_unit':'patient within cohort; PD ON/OFF kept together',
         'point_weighting':'equal cohorts; equal evaluation units within each cohort',
         'CV5_pretrain':{
             'headline_point':round(float(np.mean([np.mean(list(pre[c].values())) for c in COHORTS])),4),
             'headline_CI95':interval(np.mean(list(bootstrap.values()),axis=0)),
             'per_cohort':{c:{'point':float(np.mean(list(pre[c].values()))),'CI95':interval(bootstrap[c]),
                 'n_evaluation_units':len(pre[c]),'n_patients':len(groups(c,pre[c]))} for c in COHORTS}}}
    differences={c:{s:pre[c][s]-scr[c][s] for s in pre[c]} for c in COHORTS}
    delta_boot={c:cluster_means(c,differences[c],rng,NBOOT) for c in COHORTS}
    paired=paired_test(differences)
    paired['macro_mean_delta']=float(np.mean([np.mean(list(v.values())) for v in differences.values()]))
    paired['macro_delta_CI95']=interval(np.mean(list(delta_boot.values()),axis=0))
    paired['pooled_evaluation_unit_mean_delta']=float(np.mean([v for d in differences.values() for v in d.values()]))
    out['paired_pretrain_vs_scratch_CV5']=paired
    rows=[];channels=data['pretrain/CV5']['per_subject_channel']
    for c in COHORTS:
        for ch in sorted({ch for v in channels[c].values() for ch in v}):
            values={s:v[ch] for s,v in channels[c].items() if ch in v}
            test=paired_test({c:values})
            rows.append({'cohort':c,'channel':ch,'n_evaluation_units':len(values),
                         'n_patients':len(groups(c,values)),'mean_pcc':float(np.mean(list(values.values()))),
                         'CI95':interval(cluster_means(c,values,rng,NBOOT)),
                         'p_vs0_patient_level':test['wilcoxon_p_two_sided']})
    order=np.argsort([r['p_vs0_patient_level'] for r in rows])
    ranked=np.array([rows[i]['p_vs0_patient_level'] for i in order])
    adjusted=np.minimum.accumulate((ranked*len(rows)/np.arange(1,len(rows)+1))[::-1])[::-1]
    for i,q in zip(order,adjusted):rows[i]['q_BH']=min(float(q),1.0)
    out['per_channel_table_with_FDR']=rows
    errors={}
    for metric in ['RMSE','nRMSE_pct']:
        table=data['pretrain/CV5']['per_subject_'+metric];errors[metric]={}
        for c in COHORTS:
            errors[metric][c]={}
            for ch in sorted({ch for row in table[c].values() for ch in row}):
                vals={s:row[ch] for s,row in table[c].items() if ch in row}
                errors[metric][c][ch]={'mean':float(np.mean(list(vals.values()))),
                    'CI95':interval(cluster_means(c,vals,rng,NBOOT)),
                    'n_evaluation_units':len(vals),'n_patients':len(groups(c,vals))}
    out['physical_error_patient_clustered']=errors
    out['reference_note']='Retains archived clipped reference; clustering does not change preprocessing.'
    path.parent.mkdir(parents=True,exist_ok=True);guarded_dump(out,path,'pretraining_gain_stats.py')
    print('wrote',path);print(json.dumps(out['CV5_pretrain'],indent=2))

if __name__=='__main__':main()
