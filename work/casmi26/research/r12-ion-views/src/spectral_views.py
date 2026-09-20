"""Ion-consistent spectral views; preserves metadata when spectra are filtered.

No model, answer structures or learned parameters are used by these helpers.
They are an experimental alternative to a pinned public baseline, not a claim
that its pretrained merged head improves after a different input transform.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Callable, Iterable, Mapping
import math
import numpy as np


def _identity(row):
    precursor=float(row['precursor_mz'])
    mode=row['ionization_mode'];adduct=row['adduct'];instrument=str(row.get('instrument_type') or 'other')
    if not math.isfinite(precursor) or precursor<=0 or mode not in ('positive','negative'):
        raise ValueError('Invalid precursor or polarity')
    if not isinstance(adduct,str) or not adduct:
        raise ValueError('Missing ion species')
    if adduct.endswith('+') and mode!='positive' or adduct.endswith('-') and mode!='negative':
        raise ValueError('Adduct/polarity contradiction')
    return mode,adduct,instrument


def _peaks(row):
    mz=np.asarray(row['ms2_mzs'],dtype=np.float64)
    intensity=np.asarray(row['ms2_normalized_intensities'],dtype=np.float64)
    if mz.ndim!=1 or intensity.shape!=mz.shape or not np.isfinite(mz).all() or not np.isfinite(intensity).all() or np.any(mz<=0) or np.any(intensity<0):
        raise ValueError('Invalid or misaligned peak arrays')
    return mz,intensity


def prepare_rows(rows: Iterable[Mapping], prepare: Callable):
    """Keep the original row beside its prepared peaks, never rows[:n_kept]."""
    accepted=[];rejected=[]
    for index,row in enumerate(rows):
        _identity(row);mz,intensity=_peaks(row)
        a,b=prepare(mz,intensity,float(row['precursor_mz']))
        a=np.asarray(a);b=np.asarray(b)
        if a.ndim!=1 or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid prepared peaks')
        if not len(a):rejected.append(index);continue
        accepted.append({'source_index':index,'row':row,'mz':a,'intensity':b})
    return accepted,rejected


def compatible_groups(rows: Iterable[Mapping], *, ppm=50., da=.02):
    """No pooling across adducts, polarities, instruments or discordant precursor."""
    if not np.isfinite([ppm,da]).all() or ppm<0 or da<0:
        raise ValueError('Invalid precursor tolerance')
    by_identity=defaultdict(list)
    for row in rows:by_identity[_identity(row)].append(row)
    result=[]
    for signature in sorted(by_identity):
        bucket=sorted(by_identity[signature],key=lambda r:(float(r['precursor_mz']),str(r.get('spectrum_id',''))))
        group=[];anchor=None
        for row in bucket:
            p=float(row['precursor_mz'])
            if anchor is None or p-anchor>max(da,abs(anchor)*ppm*1e-6):
                if group:result.append(group)
                group=[];anchor=p
            group.append(row)
        if group:result.append(group)
    return result


def merge_peaks(rows: Iterable[Mapping], *, tolerance=.005):
    """Per-spectrum base-peak normalization; max-intensity peak in anchored bins.

The first mass anchors each cluster, preventing unbounded single-link chaining.
Ties choose the lower m/z. Unlike filtering neighbouring original rows, every
cluster has exactly one survivor, including rising/falling three-peak chains.
"""
    rows=list(rows)
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid merge tolerance')
    if not rows:return np.empty(0),np.empty(0)
    if len(compatible_groups(rows))!=1:raise ValueError('Only compatible ion measurements can be merged')
    pairs=[]
    for row in rows:
        mz,it=_peaks(row)
        if len(it) and it.max()>0:
            keep=it>0;pairs.extend(zip(mz[keep].tolist(),(it[keep]/it.max()).tolist()))
    pairs.sort(key=lambda p:(p[0],-p[1]))
    if not pairs:return np.empty(0),np.empty(0)
    masses=[];intensities=[];i=0
    while i<len(pairs):
        anchor=pairs[i][0];winner=pairs[i];j=i+1
        while j<len(pairs) and pairs[j][0]-anchor<tolerance:
            if pairs[j][1]>winner[1]:winner=pairs[j]
            j+=1
        masses.append(winner[0]);intensities.append(winner[1]);i=j
    return np.asarray(masses),np.asarray(intensities)


def annotate_groups(groups: Mapping, prepare: Callable):
    totals=Counter();details=[]
    for key,rows in sorted(groups.items()):
        rows=list(rows);good,bad=prepare_rows(rows,prepare)
        precursor=np.array([r['precursor_mz'] for r in rows],dtype=float)
        adducts={r['adduct'] for r in rows};modes={r['ionization_mode'] for r in rows}
        groups_=compatible_groups(rows)
        shifted=[v['source_index'] for pos,v in enumerate(good) if v['source_index']!=pos]
        inconsistent=bool(len(precursor) and np.ptp(precursor)>max(.02,float(np.median(precursor))*50e-6))
        detail={'id':str(key),'spectra':len(rows),'adducts':sorted(adducts),'modes':sorted(modes),
            'compatible_views':len(groups_),'dropped_input_spectra':len(bad),'metadata_misaligned_rows':len(shifted),
            'mixed_precursor':inconsistent,'mixed_polarity':len(modes)>1,
            'zero_mean_polarity':bool(rows and sum(1 if r['ionization_mode']=='positive' else -1 for r in rows)==0),
            'merged_precursor_range_da':float(np.ptp(precursor)) if len(precursor) else 0.}
        totals.update(molecules=1,spectra=len(rows),mixed_adduct_groups=int(len(adducts)>1),mixed_polarity_groups=int(len(modes)>1),
            mixed_precursor_groups=int(inconsistent),zero_mean_polarity_groups=int(detail['zero_mean_polarity']),
            dropped_input_spectra=len(bad),metadata_misaligned_groups=int(bool(shifted)),metadata_misaligned_rows=len(shifted))
        details.append(detail)
    return {**dict(totals),'details':details,'labels_used':False,'accuracy_established':False}
