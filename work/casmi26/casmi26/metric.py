"""Local reproduction of CASMI26's published MRR@25 contract, not Kaggle's scorer."""
from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Sequence
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

OFFICIAL_RDKIT = '2026.03.3'


def require_official_rdkit() -> None:
    if rdBase.rdkitVersion != OFFICIAL_RDKIT:
        raise RuntimeError(f'CASMI26 matching requires RDKit {OFFICIAL_RDKIT}; found {rdBase.rdkitVersion}')


@lru_cache(maxsize=65536)
def structure_key(smiles: str) -> str | None:
    """Apply default RDKit tautomer canonicalization, then use InChIKey14.

    Do not add salt stripping, uncharging, or fingerprint comparisons that the
    published contract does not describe. Invalid guesses occupy a rank but
    never match. Version enforcement is explicit at the competition CLI.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not mol.GetNumAtoms():
                return None
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            key = Chem.MolToInchiKey(mol)
            return key[:14] if len(key) == 27 else None
        except (ValueError, RuntimeError):
            return None


def distinct_guesses(smiles: Sequence[str], k: int = 25) -> list[str]:
    if not 1 <= k <= 25:
        raise ValueError('CASMI26 requires between 1 and 25 guesses')
    seen, result = set(), []
    for smi in smiles:
        key = structure_key(smi)
        if key is None or key in seen:
            continue
        seen.add(key)
        # Keep the selected representation; do not reorder its rank.
        result.append(smi)
        if len(result) == k:
            break
    return result


def mrr_at_25(targets: Mapping[str, str], predictions: Mapping[str, Sequence[str] | str]) -> dict:
    if not targets or set(targets) != set(predictions):
        raise ValueError('Nonempty target and prediction IDs must match exactly')
    reciprocal, ranks = [], {}
    for cid, target in targets.items():
        key = structure_key(target)
        if key is None:
            raise ValueError(f'Invalid target structure for {cid}')
        guesses = predictions[cid]
        guesses = guesses.split(';') if isinstance(guesses, str) else list(guesses)
        if not 1 <= len(guesses) <= 25 or any(not isinstance(s, str) or not s.strip() for s in guesses):
            raise ValueError(f'{cid}: need 1-25 nonempty guesses')
        # No deduplication here: evaluate exactly the ranks actually submitted.
        rank = next((i for i, s in enumerate(guesses, 1) if structure_key(s) == key), None)
        ranks[cid] = rank
        reciprocal.append(0. if rank is None else 1./rank)
    n = len(targets)
    return {'mrr_at_25': sum(reciprocal)/n,
            'top1_accuracy': sum(r == 1 for r in ranks.values())/n,
            'recall_at_25': sum(r is not None for r in ranks.values())/n,
            'molecules': n, 'ranks': ranks, 'rdkit_version': rdBase.rdkitVersion,
            'matching_version_pinned': rdBase.rdkitVersion == OFFICIAL_RDKIT,
            'official_score': None, 'evaluation': 'local_published_metric_reproduction'}
