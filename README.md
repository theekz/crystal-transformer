# Crystal Transformer: Multi-Fidelity Transfer Learning for Photocatalytic Bandgap Prediction

[![Journal](https://img.shields.io/badge/Journal-Computational%20Materials%20Science-blue)](https://www.journals.elsevier.com/computational-materials-science)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.7.0-green.svg)](https://pyg.org/)
[![Pymatgen](https://img.shields.io/badge/Pymatgen-2025.10.7-orange.svg)](https://pymatgen.org/)
[![FAIR Compliance](https://img.shields.io/badge/FAIR-Compliant-brightgreen)](https://fair-software.eu/)

This repository contains the official codebase and reproduction pipeline for **"Crystal Transformer: Multi-Fidelity Transfer Learning for Photocatalytic Bandgap Prediction"**, submitted to ***Computational Materials Science* (Elsevier)**.

---

## 📖 Abstract
A precise prediction of the bandgap is one of the most important challenges in the high-throughput computational discovery of photocatalytic materials. Typical Density Functional Theory (DFT) with the Perdew-Burke-Ernzerhof (PBE) functional gives large-scale structural information, but systematically underestimates electronic bandgaps by >1.5 eV. The modified Becke-Johnson (mBJ) meta-GGA potential allows for nearly "hybrid" accuracy at a very high computational cost, producing datasets that are too sparse to train complex machine learning architectures from scratch. 

We present **Crystal Transformer**, a novel Transformer-based crystal graph neural network (GNN) that integrates:
1. **Gaussian Distance Smearing** (continuous spatial distance encoding).
2. **Graph Topology Optimization** (utilizing an optimal coordination cutoff radius, $R_c = 2.0$ Å).
3. **Five-Layer Deep Regression Output Head** ($1024 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 1$).

By training within a **multi-fidelity transfer learning framework** (pretraining on 143,259 low-fidelity PBE structures, then fine-tuning on 19,805 high-fidelity mBJ structures), our model achieves an **MAE of 0.3683 eV ($R^2 = 0.8529$)** on the high-fidelity mBJ test set, outperforming scratch-trained models while operating at inference speeds orders of magnitude faster than conventional DFT.

---

## 🛠️ Key Features
* **Physics-Informed Graph Construction**: Crystalline systems are modeled as directed multigraphs under full Periodic Boundary Conditions (PBC) using `pymatgen`.
* **Multi-Fidelity Transfer Protocol**: Retains learned low-fidelity structural representations by initializing weights from a PBE-pretrained model and fine-tuning on mBJ with a restricted learning rate ($1 \times 10^{-5}$) to prevent catastrophic forgetting.
* **Rigorous Validation & Chemistry Stress Tests**:
  * **Metal vs. Insulator Split**: Validates model selectivity on electrical conductors ($E_g = 0$ eV) vs. semiconductors ($E_g > 0$ eV).
  * **Mendeleev Group Substitution Test**: Evaluates chemical periodicity awareness by swapping lattice elements with column neighbors vs. random elements.
* **Architectural Ablation Suite**: Code to programmatically disable individual components (attention, regression depth, or smearing) and trace generalization gap curves.

---

## 📂 Repository Directory Structure
To keep this repository lightweight and adhere to standard Git storage limits (avoiding massive binary assets like raw datasets and heavy model weights), the directory structure is structured as follows. Users should fetch the structures directly from public registries using the database links provided below:

```text
crystal-transformer/
├── README.md               # Repository documentation and benchmark results
├── requirements.txt         # Package dependencies
├── data/
│   ├── raw_pbe/             # Raw GGA-PBE CIF files (Downloaded from source registries)
│   ├── raw_mbj/             # Raw meta-GGA mBJ CIF files (Downloaded from source registries)
│   └── processed/           # Processed PyTorch Geometric graphs (.pt files, locally generated)
├── models/
│   └── crystal_transformer.py
└── scripts/
    ├── prepare_dataset.py   # [Code 01] Graph Generation & Split Preprocessing
    ├── pretrain_pbe.py      # [Code 02] GGA-PBE Base Pretraining & Stress Tests
    ├── finetune_mbj.py      # [Code 03] meta-GGA mBJ fine-tuning & Baselines
    └── run_ablations.py     # [Code 04] Ablation Studies & Generalization curves
```

---

## 📦 Dataset Sourcing & Reconstruction

To reproduce the study, download the raw crystalline structural CIFs and experimental/theoretical bandgaps directly from the primary databases below and place them into the respective folder paths:

1. **GGA-PBE low-fidelity dataset (143,259 structures)**:
   * **Source Registry**: Downloaded from the **DCGAT** public dataset, which represents a joint aggregation of structural databases from the **Materials Project** and **AFLOW** repositories.
   * **Materials Project Portal**: [https://materialsproject.org](https://materialsproject.org)
   * **AFLOW Repository**: [http://aflow.org](http://aflow.org)
   * **Setup**: Place the extracted raw PBE CIF structural files under `data/raw_pbe/` and supply their corresponding ID-to-bandgap metadata registry in the preprocessing script.

2. **meta-GGA mBJ high-fidelity dataset (19,805 structures)**:
   * **Source Registry**: Sourced from the **Joint Automated Repository for Various Integrated Simulations (JARVIS)** database.
   * **JARVIS-DFT (TB-mBJ calculations)**: [https://jarvis.nist.gov/](https://jarvis.nist.gov/) or [JARVIS-DFT Portal](https://jarvis.nist.gov/jarvisdft/).
   * **Setup**: Place the raw high-fidelity TB-mBJ CIF structural files under `data/raw_mbj/` and link their target bandgaps in the preprocessing script.

---

## ⚡ Installation & Setup
To run the training and processing pipelines, establish a local virtual environment with the necessary packages:

```bash
# Clone the repository
git clone https://github.com/your-username/crystal-transformer.git
cd crystal-transformer

# Install core dependencies (offline-ready/pip-compatible)
pip install -r requirements.txt
```

### `requirements.txt` dependencies:
```text
torch>=2.2.0
torch-geometric>=2.7.0
pymatgen>=2025.10.7
numpy>=2.0.2
matplotlib>=3.10.0
tqdm>=4.67.1
```

---

## 🚀 Execution Workflow

### Step 1: Data Preparation & Topology Optimization
Convert your structural CIF files into periodic crystal graphs. To perform the **Coordination Cutoff Radius ($R_c$) Sensitivity Analysis** reported in Section III-C, uncomment the desired radius in `scripts/prepare_dataset.py`:
```python
# Rc = 1.5  # (Å)
Rc = 2.0    # (Å) - OPTIMAL threshold selected for the Crystal Transformer
# Rc = 2.5  # (Å)
```
Run the graph constructor script:
```bash
python scripts/prepare_dataset.py
```
This script generates training, validation, and testing divisions matching the paper configurations (90:10 for PBE; 80:10:10 for mBJ) under `data/processed/`.

### Step 2: Base Model Pretraining & Chemical Stress Tests
Pretrain the baseline Crystal Transformer on PBE structures for 100 epochs and execute the physical selectivity stress tests:
```bash
python scripts/pretrain_pbe.py
```
* **Test 1 (Conductor Selectivity)**: Evaluates pretrain MAE separately on metals vs. semiconductors.
* **Test 2 (Mendeleev Mutation)**: Mutates structural elements with Group neighbors vs. random atoms to inspect chemical awareness.
* Saves validation scatter results as a publication-ready plot: `data/processed/pretrain_val_scatter_plot.png`.

### Step 3: Multi-Fidelity Transfer Fine-Tuning
Fine-tune your pretrained representation weights on target high-fidelity mBJ structures under a constrained learning rate ($1 \times 10^{-5}$):
```bash
python scripts/finetune_mbj.py
```
* Compares **Scratch-Trained mBJ (EXP 1)**, **Zero-Shot Transfer (EXP 2)**, and **Transfer Fine-Tuned (EXP 3)**.
* Employs validation early stopping (patience = 20 epochs).
* Saves comparison scatter charts (matching Fig. 2 in the paper) to: `data/processed/transfer_learning_comparison_plot.png`.

### Step 4: Generalization & Ablation Trajectories
Execute the ablation study to analyze structural regularization and trace learning trajectories:
```bash
python scripts/run_ablations.py
```
* Runs the model with: (A) Single-layer projection head, (B) No Attention (GCNConv), and (C) Raw scalar distances (no Gaussian smearing).
* Generates learning curves comparing training vs. validation MAE trajectories across the baseline and all three ablations, saving them to `data/processed/ablation_trajectories_plot.png`.

---

## 📊 Quantitative Benchmark Results
Below is the test-set performance comparison reproducing the metrics in **Table I** of the manuscript:

| Configuration / Experiment | Architectural Details | MAE (eV) | $R^2$ |
| :--- | :--- | :---: | :---: |
| **PBE Pretraining (Base)** | Pretrained representation on full PBE corpus | **0.2901** | **0.8303** |
| **Scratch-Trained mBJ** | Trained from random initialization on mBJ | 0.4039 | 0.8277 |
| **PBE Pretrained (Zero-Shot)** | Direct inference on mBJ without fine-tuning | 0.6925 | 0.6949 |
| **Transfer Fine-Tuned (PBE → mBJ)** | Weight transfer + $1 \\times 10^{-5}$ fine-tuning | **0.3683** | **0.8529** |
| **Ablation A (Small Head)** | Single-layer linear projection readout | 0.6216 | 0.5398 |
| **Ablation B (No Attention / GCN)** | Isotropic message passing without edge attributes | 0.3011 | 0.8257 |
| **Ablation C (No Gaussian)** | Direct 1D scalar bond distance inputs | 0.2945 | 0.8299 |

### **Key Scientific Insights:**
* **Deep Output Head is Dominant**: Replacing the 5-layer regression MLP with a simple projection readout (**Ablation A**) increases MAE by **114%** (0.2901 eV to 0.6216 eV), proving that structural mapping requires deep, highly non-linear feature transformation.
* **Attention Resolves Overfitting**: Utilizing multi-head attention (**Base**) instead of isotropic aggregation (**Ablation B**) reduces the train-test generalization gap, serving as an implicit regularizer.
* **Gaussian Smearing Smooths Convergence**: Though asymptotic accuracy remains similar, omitting Gaussian smearing (**Ablation C**) causes jagged optimization trajectories and slower convergence.

---

## 🤝 Citation & Publication
If you find this repository or model helpful in your research, please cite our peer-reviewed work:

```bibtex
@article{subasinghe2026crystal,
  title={Crystal Transformer: Multi-Fidelity Transfer Learning for Photocatalytic Bandgap Prediction},
  author={Subasinghe, Thushara and Kavindi, D.M.N. and Perera, E.T.B.},
  journal={Computational Materials Science},
  volume={XXX},
  pages={XXXXXX},
  year={2026},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.commatsci.2026.XXXXXX}
}
```

---

## ⚖️ License & FAIR Sharing Compliance
In compliance with **Elsevier's *Computational Materials Science* FAIR data principles**, all codes are published under an open sharing license.
* **Code License**: [MIT License](LICENSE)
* **External Dataset Citations**: Crystalline structural inputs are referenced directly to their primary online hosting databases (Materials Project, AFLOW, and JARVIS-DFT). Anyone can download and reproduce the dataset using the official source registries referenced in the Sourcing section above.
