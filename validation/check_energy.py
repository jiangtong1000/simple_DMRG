#!/usr/bin/env python3
"""Check the archived default example against exact diagonalization.

The random seed and validation were added in 2026. This checks one small,
real-valued example, not general DMRG convergence or truncation accuracy.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import tempfile

import numpy as np


def main():
    source = Path(__file__).resolve().parents[1] / "dmrg.py"
    old_cwd = Path.cwd()
    np.random.seed(0)
    with tempfile.TemporaryDirectory(prefix="simple-dmrg-check-") as work:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(Path(work) / "matplotlib")
        import matplotlib
        import matplotlib.pyplot as plt

        try:
            os.chdir(work)
            with contextlib.redirect_stdout(io.StringIO()):
                result = runpy.run_path(str(source), run_name="__main__")
        finally:
            os.chdir(old_cwd)
            plt.close("all")

    n, j, jz, field = (result[key] for key in ("N", "J", "Jz", "h"))
    assert (n, j, jz, field) == (8, -1.0, 1.0, 0.0), "Check is scoped to the default example."
    plus = np.array([[0.0, 1.0], [0.0, 0.0]])
    minus = plus.T
    sz = np.diag([0.5, -0.5])
    bond = j / 2 * (np.kron(plus, minus) + np.kron(minus, plus)) + jz * np.kron(sz, sz)
    hamiltonian = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        hamiltonian += np.kron(np.eye(2**i), np.kron(bond, np.eye(2 ** (n - i - 2))))
    for i in range(n):
        hamiltonian -= field * np.kron(np.eye(2**i), np.kron(sz, np.eye(2 ** (n - i - 1))))
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    # Contract the MPS independently of its internal environment expressions.
    state = result["M"][0][0, :, :]
    for tensor in result["M"][1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    state = state[..., 0].reshape(-1)
    state /= np.linalg.norm(state)
    energy = float(np.vdot(state, hamiltonian @ state).real)
    error = energy - float(eigenvalues[0])
    residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": 0,
        "exact_ground_energy": float(eigenvalues[0]),
        "mps_energy": energy,
        "energy_error": error,
        "eigenvector_residual": residual,
        "ground_state_overlap_squared": float(abs(np.vdot(eigenvectors[:, 0], state)) ** 2),
        "actual_bond_dimensions": [int(t.shape[2]) for t in result["M"][:-1]],
        "local_updates": len(result["E"]),
    }
    print(json.dumps(report, indent=2))
    assert abs(error) < 1e-9, "Ground-state energy mismatch"
    assert residual < 1e-8, "State is not a converged eigenvector"
    print("PASS: default eight-spin example agrees with exact diagonalization.")


if __name__ == "__main__":
    main()
