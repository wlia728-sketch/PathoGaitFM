"""Patient-cluster resampling, retaining recording-weighted cohort point estimates.

PD medication states remain separate evaluation units. Their patient ID is the
cluster for uncertainty; paired tests use one mean paired difference per patient.
"""
import re
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon


def patient_key(cohort, stem):
    if cohort == 'bmclab_pd':
        match = re.fullmatch(r'(SUB\d+)_(on|off)', stem, flags=re.I)
        if not match:raise ValueError(f'Unknown PD recording identity: {stem}')
        return match.group(1).upper()
    return stem


def groups(cohort, stems):
    result=defaultdict(list)
    for s in sorted(stems):result[patient_key(cohort,s)].append(s)
    return list(result.values())


def resample_units(cohort, stems, rng):
    clusters=groups(cohort,stems)
    if not clusters:raise ValueError(f'Empty cohort {cohort}')
    draws=rng.choice(len(clusters),size=len(clusters),replace=True)
    return [s for i in draws for s in clusters[i]]


def cluster_means(cohort, values, rng, n_boot=5000):
    clusters=groups(cohort,values)
    if not clusters:raise ValueError(f'Empty cohort {cohort}')
    sums=np.array([sum(values[s] for s in g) for g in clusters],float)
    sizes=np.array([len(g) for g in clusters])
    draws=rng.choice(len(clusters),size=(n_boot,len(clusters)),replace=True)
    return sums[draws].sum(1)/sizes[draws].sum(1)


def patient_differences(differences):
    return np.array([np.mean([values[s] for s in g])
        for c,values in sorted(differences.items()) for g in groups(c,values)],float)


def paired_test(differences):
    values=patient_differences(differences)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid paired differences')
    p=float(wilcoxon(values,alternative='two-sided').pvalue) if np.any(values) else 1.0
    return {'n_patients_paired':len(values),'n_evaluation_units_paired':sum(map(len,differences.values())),
            'n_patients_favouring_first':int((values>0).sum()),
            'median_patient_difference':float(np.median(values)),
            'wilcoxon_p_two_sided':p,'test_unit':'patient; mean paired difference across PD states',
            'test_weighting':'patients pooled across cohorts; differs from cohort-equal macro interval'}


def interval(samples, digits=4):
    return [round(float(x),digits) for x in np.percentile(samples,[2.5,97.5])]
