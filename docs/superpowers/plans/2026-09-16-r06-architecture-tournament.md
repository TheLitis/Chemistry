# R06 architecture tournament

Goal: execute a reproducible broad architecture/loss search without disguising a
finite experiment as exhaustive search, leaking held-out spectra, or probing the
Kaggle public test repeatedly. Preserve V1/V3 models and the two-per-24h cap.

## Fixed design

All new branches start at zero correction on the same frozen, molecule-disjoint
V3 multitarget anchor. Compare low-rank linear, shallow MLP, residual MLP,
fragment/loss dual branch, adduct-conditioned dense experts, convolutional mass
channels, DeepSets, peak self-attention, learned latent cross-attention, a
neutral-loss peak graph, and a feature/peak-attention hybrid. These are compact
implementations, not full pretrained MIST/MS-GPT/DiffMS reproductions.

The custom hybrid combines local/whole-spectrum features, raw peak attention and
metadata gating. Its variants also compare hard mass-neighbor ranking loss and
unweighted logistic loss. All loss functions operate on true training labels.

Common data: deterministic top-three distinct acquisitions per molecule, each
with full high-resolution vector and top-32 raw peak tokens. Random one/three
view training, common splits and initial anchor knowledge. Token truncation and
histogram compression are recorded limitations. Validation mass filtering does
not use the true formula or exact mass. Targets/candidate identities may be
known in this retrieval benchmark; never call it de novo accuracy.

Screen 13 configurations (11 architectures + 2 loss variants) on 32768 unique
training molecules for six epochs and 512 screening molecules. Advance the top
two to six further epochs on 131072 training molecules; replicate the screening
winner with a separate seed. Choose models, fusion and mass calibration on a
separate 512 molecules. Include the no-change anchor as a choice. Lock selection
before evaluating the untouched 2048-molecule audit. Report 1/3/all acquisitions,
coverage, paired bootstrap intervals, param counts, timing and failed trials.

## Execution steps

- [x] Add and observe failing tests for tokens, model families, negative sampling,
  ranking gradients, checkpoint roundtrip, masking and permutation invariance.
- [x] Implement the tested core without changing production.
- [ ] Prepare train-only view/token cache with source and split fingerprints.
- [ ] Implement common training, staged selection, independent audit and exports.
- [ ] Execute on ChemistryPC; verify logs and saved numeric checkpoints.
- [ ] Save results and a runnable candidate only when independent evidence supports
  it; retain prior champion and obey the submission cap.

## Mathematical hypotheses

Averaging before a nonlinear encoder need not equal averaging encoded spectra.
Random view budgets reduce the mismatch between many-spectrum training and
few-spectrum queries. Bernoulli log likelihood ranks candidates by fingerprint
logits (the candidate-independent sum log(1-p) cancels). A same-mass contrastive
term targets ranking errors that bitwise classification alone does not punish.
Gaussian mass likelihood needs calibration rather than a hard-coded width.
Candidate coverage bounds exact-match MRR above: if the correct graph is absent,
no ranker can recover it. None of these statements guarantees an empirical gain.
