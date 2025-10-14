# Quantum Feature Maps (Hybrid Preprocessing for Computer Vision)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-lightblue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-teal.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg)](https://scikit-learn.org/stable/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-yellow.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.12+-9cf.svg)](https://seaborn.pydata.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.37+-purple.svg)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.41+-violet.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![JupyterLab](https://img.shields.io/badge/JupyterLab-4.0+-orange.svg)](https://jupyter.org/)



## Project Overview


## Configure Environment

**Create Conda Environment**
run 
```bash
conda env create --file environment.yml
conda activate qfm-env
```


## Repository Organization
```
quantum-feature-maps/
├── environment.yml # Conda env configuration for qfm-env
├── README.md # This file
├── notebooks/ 
│ ├── data/ 
│ | ├── MNIST            # Local version of MNIST dataset
│ ├── imgs/              # Images used for documentation
│ ├── params/            # Saved trained ansatz parameter files
│ ├── data_prep.ipynb    # Data preprocessing, dimensionality reduction
│ ├── classical_baseline.ipynb   # Classical ML baselines 
│ ├── quantum_pipeline.ipynb     # Full quantum feature map + variational circuit pipeline
│ ├── quantum_feature_map_tests.ipynb   # Tests for PennyLane/Qiskit Feature Maps
│ ├── 05_analysis.ipynb # Post training metrics, plots, performance summary
├── results/            # CSVs, plots
│ ├── metrics/          # Accuracy, loss, and runtime data
│ └── figures/          # Visual outputs and comparisons
├── .gitignore 
└── .env                # Stores IBM Quantum API key
```


## Notebooks 

### `data_prep.ipynb`
**Purpose:** Prepare and visualize the input data.  
**Main steps:**
- Load a small dataset (`MNIST 0 vs 1` subset).
- Normalize and scale features.
- Apply **Principal Component Analysis (PCA)** to reduce dimensionality (≤ 4 features).
- Visualize the reduced features for interpretability.  

**Output:**  
`data/mnist01_pca4.npz` (used by later notebooks)
---

### `classical_baseline.ipynb`
**Purpose:** Establish classical machine learning baselines.  
**Main steps:**
- Train classifiers (Logistic Regression, SVM) on PCA-reduced features.
- Perform cross-validation to assess robustness.
- Plot decision boundaries for the 2D case.
- Save accuracy and runtime metrics.

**Output:**  
`results/classical_baseline.csv` 

---

### `quantum_pipeline.ipynb`
**Purpose:** Implement the **quantum feature map + variational ansatz** pipeline using **PennyLane** and **Qiskit**.  
**Main steps:**
- Define a **ZZFeatureMap** for encoding classical data into quantum states.
- Implement a **variational ansatz** (1–3 layers of parameterized rotations + entanglers).
- Measure expectation values of Pauli operators → quantum feature vectors.
- Train Logistic Regression, SVM on these quantum features.
- Sweep over:
  - Number of qubits (2–4)
  - Circuit depth (repetitions)
  - Number of shots
  - Noise models (optional)
- Log results to CSV.
- Connects to **IBM Quantum devices** using the `IBMQ` provider.

**Output:**  
- Trained parameters → `params/trained_params.npy`  
- Logged metrics → `results/quantum_results.csv`  
- Visualization of circuit topology and training loss.

---


### `quantum_feature_map_tests.ipynb`
**Purpose:** Explores versions of **quantum encoding + quantum feature map* **PennyLane** and **Qiskit**.  
**Main steps:**
- Test **Basis, Angle, and Amplitude Encoding** using both **Qiskit** and **Pennylane**
- Test **ZZFeatureMap** and **Raw Feature Vector** for encoding classical data into quantum states.
- Visualize each methods transformations, circuits. 


### `analysis.ipynb`
**Purpose:** Aggregate and analyze all results.  
**Main steps:**
- Load metrics from both classical and quantum experiments.
- Generate comparison plots:
  - Accuracy vs. number of shots
  - Accuracy vs. circuit depth
  - Runtime vs. number of qubits
- Compute average speedups and resource overhead.
- Interpret findings (when quantum features outperform classical baselines).

**Output:**  
- `results/figures/*.png` (plots)
- `results/metrics_summary.csv`
- Final summary table comparing classical and quantum approaches.



## Key Concepts  

