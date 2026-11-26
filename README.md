# Neural-adjoint β-GLOnet for quantum cascade laser (QCL) design

Code, trained surrogate models and training data for:

> G. Kim and J. Kim, "Neural-adjoint β-GLOnet: Expanding multi-objective global topology optimization framework to quantum-confined nanophotonic devices with enhanced solution diversity," *Scientific Reports* (2026). DOI: [to be added]

The repository optimizes the 24 layer thicknesses of a 4-quantum-well, 7.7 µm mid-IR QCL with a neural-adjoint β-GLOnet. In this framework, a generative network is trained through a frozen, differentiable neural-network surrogate (PBTL) of the in-house physical simulator. A batch-variance term weighted by β promotes structurally diverse designs. The repository also includes the baselines used in the paper: neural-adjoint GLOnet (β-GLOnet without the diversity term), a surrogate-assisted genetic algorithm (GA) and REINFORCE. All four optimizers are evaluated with the same solver-validated protocol.

- Code: this repository. Archived on Zenodo, DOI: [to be added]
- Training data: `QCL_beta_GLOnet_data.zip` on Zenodo, DOI: [to be added]

---

## Repository structure

```
QCL_beta_GLOnet/
├── qcl_simulator.py                 # physical QCL simulator (two-band model, scattering, gain)
├── plot_band_diagram.py             # band diagram + gain spectrum of given structures
├── requirements.txt
├── Algorithms/
│   ├── evaluation_physical.py       # physical FoM, Eqs. (1)-(6), used for pass/fail (FoM >= 4.8)
│   ├── evaluation_surrogate.py      # surrogate FoM (GA fitness, REINFORCE reward)
│   ├── GLOnet/
│   │   ├── train_GLOnet.py          # GLOnet / β-GLOnet model (generator + frozen surrogate)
│   │   ├── run_GLOnet_passK.py      # Section 3.2: β-GLOnet and neural-adjoint GLOnet
│   │   ├── sweep_GLOnet_beta.py     # Fig. 4: β sweep
│   │   ├── sweep_GLOnet_sigma.py    # SI Fig. S4: σ1 / σ2 sensitivity
│   │   └── sweep_surrogate_data_size.py  # SI Fig. S9: surrogate training-set size vs. success
│   ├── GA/
│   │   ├── GA_class.py              # genetic algorithm
│   │   ├── run_GA_passK.py          # Section 3.2: GA
│   │   └── sweep_GA_mutation_rate.py     # SI Fig. S6: mutation rate
│   └── REINFORCE/
│       ├── train_REINFORCE.py       # GRU policy + REINFORCE agent
│       ├── run_REINFORCE_passK.py   # Section 3.2: REINFORCE
│       └── sweep_REINFORCE_lr.py    # SI: learning-rate study
└── Surrogate/
    ├── Deep_Learning_class.py       # early stopping (name referenced by the .pt files; do not rename)
    ├── analyze_surrogate.py         # surrogate accuracy / analysis figures
    ├── 4QW_active/
    │   └── train_surrogate_active.py        # CP: active-region model (8 inputs)
    └── 4QW_total/
        ├── train_surrogate_PBTL.py          # PBTL: physics-based transfer learning (used by all optimizers)
        ├── train_surrogate_total_baseline.py# TP: direct learning baseline (SI)
        ├── train_surrogate_GTL.py           # GTL: general transfer learning (SI)
        └── model/
            ├── surrogate_PBTL_394k_full.pt  # full dataset, used for all optimization results
            ├── surrogate_PBTL_355k.pt       # SI Fig. S9
            ├── surrogate_PBTL_276k.pt       # SI Fig. S9
            └── surrogate_PBTL_197k.pt       # SI Fig. S9
```

Each script starts with a docstring that describes what it does, its settings and its output files.

## Installation

Python 3.9 or later. CPU, CUDA and Apple MPS all work. The device is selected automatically.

```bash
pip install -r requirements.txt
```

`torch` is listed without a CUDA build tag. For GPU support, install the PyTorch build that matches your system from <https://pytorch.org> first.

## Data

Download `QCL_beta_GLOnet_data.zip` from Zenodo (DOI: [to be added]) and unzip it in the repository root. This creates:

```
Data/
├── active_only/structure_4QW_7_7um_valid.csv   # 37,881 structures (8 active layers, fixed injector)
├── active_only/values_4QW_7_7um_valid.csv
├── total/structure_4QW_tot.csv                 # 356,356 structures (24 layers)
└── total/values_4QW_tot.csv
```

All data were generated with `qcl_simulator.py`. `structure_*.csv` contains the layer thicknesses in Å. `values_*.csv` contains five outputs per structure: peak modal gain G43 (cm⁻¹), lasing wavelength (µm), linewidth (meV), and effective injection and extraction energies (meV). Note that the column order of the CSV differs from the surrogate's output order: `train_surrogate_*.py` reorders the columns to (G43, linewidth, wavelength, E_inj, E_ext). Together the two sets contain 394,237 structures.

The data are needed only to retrain the surrogate models. The optimization scripts use the released `.pt` models.

## Reproducing the results

Run every script from any working directory. Paths are resolved relative to each script file. Results are written to `Result/` or `analysis/` next to the script.

| Paper | Script | Notes |
|---|---|---|
| Section 3.2: solver-validated success, Pass@K, distinct high-FoM designs | `Algorithms/GLOnet/run_GLOnet_passK.py`<br>`Algorithms/GA/run_GA_passK.py`<br>`Algorithms/REINFORCE/run_REINFORCE_passK.py` | 50 trials each. Set `USE_PHYSICAL_GUIDANCE = False` for the surrogate-only run |
| Fig. 4: effect of β | `Algorithms/GLOnet/sweep_GLOnet_beta.py` | edit `betas` |
| SI Fig. S4: σ1 / σ2 sensitivity | `Algorithms/GLOnet/sweep_GLOnet_sigma.py` | 5 × 5 grid, 20 trials |
| SI Fig. S6: GA mutation rate | `Algorithms/GA/sweep_GA_mutation_rate.py` | 0.04 to 0.5, 20 trials |
| SI: REINFORCE learning rate | `Algorithms/REINFORCE/sweep_REINFORCE_lr.py` | |
| SI Fig. S9: surrogate data size | `Algorithms/GLOnet/sweep_surrogate_data_size.py` | uses the four `surrogate_PBTL_*.pt` models |
| Section 2.2 / SI Section 2: surrogate models | `Surrogate/4QW_active/train_surrogate_active.py` (CP), then `Surrogate/4QW_total/train_surrogate_PBTL.py` (PBTL) | PBTL needs `4QW_active/model/CP.pt` from the CP step |

Example:

```bash
python Algorithms/GLOnet/run_GLOnet_passK.py
```

## Main settings

| | β-GLOnet | Neural-adjoint GLOnet | GA | REINFORCE |
|---|---|---|---|---|
| Iterations | 500 epochs | 500 epochs | 500 generations | 500 episodes |
| Batch / population | 64 | 64 | 64 | 64 |
| Learning rate / mutation | 1e-3 | 1e-2 | mutation rate 0.1 | 3e-3 |
| Other | β = 1.5, latent dim 64 | latent dim 64 | elitism + top-10 uniform crossover | GRU policy, entropy weight 1e-5 |

Settings shared by all four optimizers:
- **Target:** G43 = 25 cm⁻¹, linewidth = 25 meV, λ = 7.7 µm, E_inj = E_ext = 26 meV.
- **FoM (Eqs. 1–6):** sigmoid terms with slope 0.5 and a Gaussian wavelength term with σ = 0.1 µm. The maximum FoM is 5, and a design passes if its physical FoM is ≥ 4.8. The GLOnet scripts score candidates internally with a wavelength term of σ = 1 µm. This internal score is used for the 4.8 surrogate gate and for ranking.
- **GLOnet FoM loss:** σ1 = 1, σ2 = 0.5.
- **Solver-validated protocol:** the physical model is used only to validate candidates, never in a loss, gradient, fitness or reward. In each iteration, the best not-yet-checked candidate with surrogate FoM ≥ 4.8 is sent to the solver. A trial stops after 2 physical passes, 200 solver calls, or 100 consecutive calls without a pass. After the run, the top-K (K = 1, 4, 8, 16, 32) of the final 64 candidates are also evaluated with the solver.

## Surrogate models

| Model | Inputs | Description |
|---|---|---|
| CP | 8 active-region layers | active-region model; its first layer is transferred to PBTL |
| TP | 24 layers | direct learning, no transfer (baseline) |
| GTL | 24 layers | general transfer learning from CP |
| PBTL | 24 layers | physics-based transfer learning: active branch (from CP) + injector branch |

The released `.pt` files are full PyTorch models, saved with `torch.save(model)`. Load them with `torch.load(path, weights_only=False)` after importing `PBTL` from `train_surrogate_PBTL.py`. That module also registers the class under its former name, which the files require.

## License

[to be added, e.g. MIT]

## Citation

If you use this code or data, please cite the paper above.
