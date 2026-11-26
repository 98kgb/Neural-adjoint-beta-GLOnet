# Neural-adjoint β-GLOnet for quantum cascade laser (QCL) design

Code for:

> G. Kim and J. Kim, "Neural-adjoint β-GLOnet: Expanding multi-objective global topology optimization framework to quantum-confined nanophotonic devices with enhanced solution diversity," *Scientific Reports* (2026). DOI: [to be added]

This repository optimizes the 24 layer thicknesses of a 4-quantum-well, 7.7 µm mid-IR QCL with a neural-adjoint β-GLOnet. A generative network is trained through a frozen, differentiable neural-network surrogate (PBTL) of the physical simulator. A β-weighted diversity term promotes structurally diverse designs. Baselines (neural-adjoint GLOnet, genetic algorithm, REINFORCE) are included and evaluated with the same solver-validated protocol.

## Repository structure

```
QCL_beta_GLOnet/
├── qcl_simulator.py          # physical QCL simulator
├── plot_band_diagram.py      # band diagram and gain spectrum
├── requirements.txt
├── Algorithms/
│   ├── evaluation_physical.py    # physical FoM (pass if FoM >= 4.8)
│   ├── evaluation_surrogate.py   # surrogate FoM
│   ├── GLOnet/                   # β-GLOnet and neural-adjoint GLOnet
│   ├── GA/                       # genetic algorithm
│   └── REINFORCE/                # REINFORCE
└── Surrogate/
    ├── Deep_Learning_class.py
    ├── analyze_surrogate.py
    ├── 4QW_active/               # CP model
    └── 4QW_total/                # PBTL, TP, GTL models
```

Each script begins with a short description of its purpose and outputs.

## Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

For GPU support, install the PyTorch build for your system from <https://pytorch.org> first.

## Data and trained models

The training data (generated with `qcl_simulator.py`) and the trained surrogate models are available from the corresponding author upon reasonable request. Place the data under `Data/active_only/` and `Data/total/` in the repository root.

## Usage

1. Train the surrogate models (CP first, then PBTL):

```bash
python Surrogate/4QW_active/train_surrogate_active.py
python Surrogate/4QW_total/train_surrogate_PBTL.py
```

2. Run the optimizers:

```bash
python Algorithms/GLOnet/run_GLOnet_passK.py
python Algorithms/GA/run_GA_passK.py
python Algorithms/REINFORCE/run_REINFORCE_passK.py
```

Scripts named `sweep_*.py` reproduce the sensitivity studies in the Supplementary Information.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite the paper above.
