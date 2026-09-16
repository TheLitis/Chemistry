"""Mass-window streaming for an explicitly versioned public structure catalog."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter
import csv
import io
import math
from pathlib import Path
import zipfile


class MassWindows:
    def __init__(self, masses, ppm=20., da=.005, padding=.001):
        values=sorted(float(v) for v in masses)
        if not values or any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('Need finite positive query masses')
        if any(not math.isfinite(v) or v<0 for v in (ppm,da,padding)) or ppm+da==0:
            raise ValueError('Invalid mass tolerance')
        intervals=sorted((v-max(da,v*ppm*1e-6)-padding,v+max(da,v*ppm*1e-6)+padding) for v in values)
        merged=[]
        for lo,hi in intervals:
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        self.lower=[r[0] for r in merged];self.upper=[r[1] for r in merged]

    def contains(self,mass):
        if not math.isfinite(mass):return False
        i=bisect_right(self.lower,mass+1e-10)-1
        return i>=0 and mass<=self.upper[i]+1e-10


def candidate_rows(archive: Path, windows: MassWindows, counters: Counter):
    required={'identifier','canonical_smiles','exact_molecular_weight'}
    csv.field_size_limit(64*1024*1024)
    with zipfile.ZipFile(archive) as z:
        members=[i for i in z.infolist() if i.filename.lower().endswith('.csv')]
        if len(members)!=1:raise ValueError('Expected exactly one snapshot CSV')
        with io.TextIOWrapper(z.open(members[0]),encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):raise ValueError('Public catalog schema is unsupported')
            for row in reader:
                counters['csv_rows']+=1
                try:
                    mass=float(row['exact_molecular_weight'])
                    if not math.isfinite(mass) or mass<=0:raise ValueError('invalid mass')
                except (ValueError,TypeError):
                    counters['invalid_mass']+=1;continue
                if not windows.contains(mass):continue
                if not row['canonical_smiles'].strip() or not row['identifier'].strip():
                    counters['missing_structure_or_id']+=1;continue
                counters['mass_selected']+=1
                yield row['identifier'],row['canonical_smiles'],mass


def distinct_candidates(records,observed):
    """Choose one representative per evaluation key without consulting the answer."""
    if not math.isfinite(observed) or observed<=0:raise ValueError('Invalid observed mass')
    chosen={}
    for row in records:
        if len(row)<4 or not row[2] or not math.isfinite(row[3]):raise ValueError('Invalid molecular record')
        key=row[2]
        if key not in chosen or (abs(row[3]-observed),str(row[0]))<(abs(chosen[key][3]-observed),str(chosen[key][0])):
            chosen[key]=row
    return list(chosen.values())
