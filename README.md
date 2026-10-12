# Simple DMRG

A small, single-site DMRG learning implementation in the matrix product state
(MPS) / matrix product operator (MPO) formulation, by Tong Jiang.

**Originally written in 2017; first published on GitHub in 2026.** The original
development year is recorded from the author's account. This repository was
created in 2026; its commits record the import and subsequent work.

## Files

- [`original/dmrg_2017.py`](original/dmrg_2017.py): the original script, preserved
  byte for byte, including its comments and line endings.
- [`dmrg.py`](dmrg.py): a Python 3 compatibility copy prepared in 2026. It changes
  the machine-specific shebang, normalizes line endings, and removes the NumPy
  wildcard import, which shadows Python's built-in `min` under NumPy 2. The
  numerical algorithm is unchanged.
- [`validation/check_energy.py`](validation/check_energy.py): a seeded check of
  the default example against an independently constructed exact-diagonalization
  Hamiltonian. Written in 2026.

## Model and method

The script treats an open spin-1/2 XXZ chain,

```text
H = sum_i [J/2 (S+_i S-_(i+1) + S-_i S+_(i+1)) + Jz Sz_i Sz_(i+1)]
    - h sum_i Sz_i
```

The supplied parameters are `N=8`, `J=-1`, `Jz=1`, and `h=0`. The code explicitly
constructs the MPO, initializes and canonicalizes an MPS, builds left and right
environments, and optimizes one site at a time with a dense eigensolver. SVD
moves the canonical center between neighboring sites.

## Run

The compatibility copy and verification were checked with Python 3.12, NumPy
2.4.4, and Matplotlib 3.10.8.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validation/check_energy.py
```

To run the historical calculation through the compatibility copy:

```sh
MPLBACKEND=Agg python dmrg.py
```

This prints the local updates and writes `e.pdf` and `ee.pdf` in the current
directory. The verification script instead keeps those outputs in a temporary
directory and compares the contracted MPS state with exact diagonalization.

## Validation and limitations

For the default eight-spin example with random seed 0, the ground-state energy
is approximately **-3.374932598687890**. The verification checks both the energy
error (below `1e-9`) and the eigenvector residual (below `1e-8`).

This remains an early learning implementation:

- The configured `D=100` gives actual internal bond dimensions
  `[2, 4, 8, 16, 8, 4, 2]`, sufficient for the full eight-spin state space.
  This example does not test controlled truncation or large-system accuracy.
- The SVD steps do not truncate singular values.
- The stopping check compares adjacent local-update energies, and the loop
  permits at most three directional sweeps. It is not a complete-sweep
  convergence test.
- The contractions assume real-valued tensors. General complex-valued models
  need appropriate conjugation.

## Related learning project

[`simple_FullCI`](https://github.com/jiangtong1000/simple_FullCI) is my 2019
learning implementation of molecular full configuration interaction, using
determinants and Slater–Condon matrix elements. It is a separate source-reading
example; its required `h1e.npy` and `h2e.npy` inputs are not included there.

## Original-file checksum

SHA-256 of `original/dmrg_2017.py`:

```text
bad80e5a6863798709c025150cada068f53ff198cd720b02e275c8e474e3ea7f
```
