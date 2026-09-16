"""Graph identity, explicit ion conventions and bounded structural proposals."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

PROTON = 1.007276466621
ELECTRON = 0.000548579909
WATER = 18.01056468403
FORMIC_ACID = 46.0054793036
# (molecular multiplicity, signed charge, total mass added before dividing by |z|)
ADDUCTS = {
    '[M+H]+': (1, 1, PROTON), '[M-H]-': (1, -1, -PROTON),
    '[M+Na]+': (1, 1, 22.989220702), '[M+K]+': (1, 1, 38.963157906),
    '[M+NH4]+': (1, 1, 18.033825553), '[M+Cl]-': (1, -1, 34.969401262),
    '[M+2H]2+': (1, 2, 2*PROTON), '[M-2H]2-': (1, -2, -2*PROTON),
    '[2M+H]+': (2, 1, PROTON), '[2M-H]-': (2, -1, -PROTON),
    '[M-H2O+H]+': (1, 1, PROTON-WATER),
    '[M-2H2O+H]+': (1, 1, PROTON-2*WATER),
    '[M-H2O-H]-': (1, -1, -PROTON-WATER),
    '[M+CH2O2-H]-': (1, -1, FORMIC_ACID-PROTON),
    '[M]+': (1, 1, -ELECTRON), '[M]-': (1, -1, ELECTRON),
}


def adduct_spec(adduct: str) -> tuple[int, int, float]:
    key = str(adduct).replace(' ', '').replace('−', '-')
    try:
        return ADDUCTS[key]
    except KeyError as exc:
        raise ValueError(f'Unsupported adduct {adduct!r}; specify a verified ion convention.') from exc


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    n, z, shift = adduct_spec(adduct)
    mass = (float(precursor_mz) * abs(z) - shift) / n
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Invalid precursor/neutral mass')
    return mass


@lru_cache(maxsize=8192)
def canonical(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Missing SMILES')
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid SMILES: {smiles[:100]!r}')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    # Keep isotope information. isomericSmiles=False would also erase isotopes.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@dataclass(frozen=True)
class MoleculeInfo:
    smiles: str
    mass: float
    formula: str
    charge: int


@lru_cache(maxsize=8192)
def info(smiles: str) -> MoleculeInfo:
    smi = canonical(smiles)
    mol = Chem.MolFromSmiles(smi)
    return MoleculeInfo(smi, rdMolDescriptors.CalcExactMolWt(mol),
                        rdMolDescriptors.CalcMolFormula(mol), Chem.GetFormalCharge(mol))


def exact_match(prediction: str, target: str) -> bool:
    """Local graph diagnostic; use metric.structure_key for competition matching."""
    return canonical(prediction) == canonical(target)


def proposals(smiles: str, limit: int = 64):
    """Deterministic, bounded formula-preserving rewiring; not exhaustive de novo."""
    if limit <= 0:
        return
    source = info(smiles)
    mol = Chem.MolFromSmiles(source.smiles)
    seen = {source.smiles}
    attempts = 0
    # Move one end of a non-aromatic bond. This changes connectivity, not labels.
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for anchor, old in ((a, b), (b, a)):
            for new in range(mol.GetNumAtoms()):
                if new in (anchor, old) or mol.GetBondBetweenAtoms(anchor, new):
                    continue
                attempts += 1
                if attempts > max(500, limit * 80):
                    return
                rw = Chem.RWMol(mol)
                rw.RemoveBond(anchor, old)
                rw.AddBond(anchor, new, bond.GetBondType())
                with rdBase.BlockLogs():
                    try:
                        candidate = rw.GetMol()
                        Chem.SanitizeMol(candidate)
                        if len(Chem.GetMolFrags(candidate)) != 1:
                            continue
                        c = info(Chem.MolToSmiles(candidate))
                    except (ValueError, RuntimeError):
                        continue
                if c.formula != source.formula or c.charge != source.charge or c.smiles in seen:
                    continue
                seen.add(c.smiles)
                yield c.smiles
                if len(seen) - 1 >= limit:
                    return


@lru_cache(maxsize=1024)
def fragment_masses(smiles: str) -> tuple[float, ...]:
    """One-/two-bond-cut component masses; intentionally a simple heuristic.

    No claim of simulating ion chemistry, rearrangements or fragment intensities.
    Hydrogen shifts are considered separately by the scoring function.
    """
    mol = Chem.MolFromSmiles(canonical(smiles))
    pt = Chem.GetPeriodicTable()
    atom_mass = [
        (pt.GetMassForIsotope(a.GetAtomicNum(), a.GetIsotope()) if a.GetIsotope()
         else pt.GetMostCommonIsotopeMass(a.GetAtomicNum()))
        + a.GetTotalNumHs() * 1.00782503223 for a in mol.GetAtoms()
    ]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    # Cap graph work explicitly; this is not an exact fragmentation calculation.
    cuts = [(i,) for i in range(len(bonds))]
    cuts += list(combinations(range(min(len(bonds), 32)), 2))
    masses = set()
    for cut in cuts:
        adj = [[] for _ in atom_mass]
        for i, (a, b) in enumerate(bonds):
            if i not in cut:
                adj[a].append(b)
                adj[b].append(a)
        remaining = set(range(len(atom_mass)))
        while remaining:
            stack = [remaining.pop()]
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for other in adj[node]:
                    if other in remaining:
                        remaining.remove(other)
                        stack.append(other)
            if len(component) < len(atom_mass):
                masses.add(round(sum(atom_mass[i] for i in component), 8))
    return tuple(sorted(masses))


def fragment_explanation(smiles: str, spectrum, ppm: float = 20, da: float = .01) -> float:
    n, z, shift = adduct_spec(spectrum.adduct)
    if n != 1 or abs(z) != 1:
        return 0.0
    base = np.asarray(fragment_masses(smiles))
    if not len(base):
        return 0.0
    theoretical = np.unique(np.concatenate([base + shift + h * 1.00782503223 for h in (-1, 0, 1)]))
    theoretical = theoretical[(theoretical > 0) & (theoretical < spectrum.precursor_mz - da)]
    observed = spectrum.peaks[spectrum.peaks[:, 0] < spectrum.precursor_mz - da]
    if not len(theoretical) or not len(observed):
        return 0.0
    explained = 0.0
    for mz, intensity in observed:
        if np.any(np.abs(theoretical - mz) <= max(da, mz * ppm * 1e-6)):
            explained += intensity
    # A modest penalty for dense theoretical peak lists, not a probability.
    return float(explained / observed[:, 1].sum() / (1 + .002 * len(theoretical)))
