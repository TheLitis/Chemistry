"""Versioned fractional-mass inputs and complementary structure targets.

Moments add information but do NOT preserve the entire peak list. All targets
are computed from training/candidate structures, never unknown test answers.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from .production import batch_features, BINS, FEATURES

FEATURE_DIM = 8200
HEAD_SIZES = (2048, 4096, 8192)
HEAD_NAMES = ('morgan2', 'morgan3', 'atom_pair')
FEATURE_VERSION = 'fractional-mass-v3.1'


def highres_features(mz, intensity, offsets, precursor, adducts, energies):
    base, valid, neutral = batch_features(mz, intensity, offsets, precursor, adducts, energies)
    if BINS != 2048 or FEATURES != 4104:
        raise ValueError('The v1 feature contract changed')
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    precursor = np.asarray(precursor, dtype=np.float64)
    row = np.repeat(np.arange(len(precursor)), np.diff(offsets))
    keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (mz > 0)
    keep &= (mz < precursor[row] - .01) & valid[row]
    rr, weight, mm = row[keep], np.sqrt(intensity[keep]), mz[keep]
    extra = np.zeros((len(precursor), 2 * BINS), dtype=np.float32)
    for channel, values in enumerate((mm, precursor[rr] - mm)):
        bins = np.floor(values).astype(np.int64)
        yes = (bins >= 0) & (bins < BINS)
        indices = (rr[yes], bins[yes])
        raw = np.zeros((len(precursor), BINS), dtype=np.float32)
        np.add.at(raw, indices, weight[yes])
        moments = extra[:, channel*BINS:(channel+1)*BINS]
        np.add.at(moments, indices, weight[yes] * (values[yes] - bins[yes]))
        moments /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    extra[~valid] = 0
    return np.concatenate((base, extra), axis=1), valid, neutral


def arrow_highres(batch):
    mz, inten = batch.column('ms2_mzs'), batch.column('ms2_normalized_intensities')
    mo, io = mz.offsets.to_numpy(zero_copy_only=False), inten.offsets.to_numpy(zero_copy_only=False)
    if not np.array_equal(mo-mo[0], io-io[0]):
        raise ValueError('Peak/intensity lengths differ')
    return highres_features(mz.values.to_numpy(zero_copy_only=False)[mo[0]:mo[-1]],
        inten.values.to_numpy(zero_copy_only=False)[io[0]:io[-1]], mo-mo[0],
        batch.column('precursor_mz').to_numpy(zero_copy_only=False),
        batch.column('adduct').to_pylist(), batch.column('collision_energy_ev').to_pylist())


@lru_cache(maxsize=1)
def generators():
    from rdkit.Chem import rdFingerprintGenerator as g
    from rdkit.Chem.MolStandardize import rdMolStandardize
    return rdMolStandardize.TautomerEnumerator(), (
        g.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False),
        g.GetMorganGenerator(radius=3, fpSize=4096, includeChirality=False),
        g.GetAtomPairGenerator(fpSize=8192, includeChirality=False, countSimulation=True))


def fingerprint_targets(smiles):
    from rdkit import Chem, rdBase
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError('Invalid candidate structure')
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(mol)
        enum, gens = generators()
        mol = enum.Canonicalize(mol)
        return np.packbits(np.concatenate([g.GetFingerprintAsNumPy(mol) for g in gens]))
