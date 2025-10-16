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

Quantum feaure maps (QFMs) are hybrid ML frameowrks that explore how quantum circuits can enhance classical preprocessing pipelines, specifically in this case for computer vision datasets such as MNIST. This project investigates the use of QFMs as nonlinear transformations that embed classical data into high dimensional Hilbert space, allowing for the classical models Logistic Regression and SVM to potentially capture more meaningful relationships as compared to standard, classical embeddings. 

QFM integrates classical dimensionality reduction with parameterized quantum circuits (PCQs) implemented in PennyLane (and tested in Qiskit), benchmarking their performance against classical baselines. The goal is to understand: 

* How the choice of quantum feature maps, ansatz, and backend impacts performance. 
* When/if quantum enhanced preprocessing can outperform classical methods when in small-scale classification tasks. 

This repo provides reproducible experiments for classical and hybrid quantum-classical models, along with visualizations, parameter sweeps, and IBM quantum integration. 


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
├── environment.yml      # Conda env configuration for qfm-env
├── README.md            # This file
├── notebooks/ 
│ ├── data/ 
│ | ├── MNIST            # Local version of MNIST dataset
│ ├── imgs/              # Images used for documentation
│ ├── params/            # Saved trained ansatz parameter files
│ ├── data_prep.ipynb    # Data preprocessing, dimensionality reduction
│ ├── classical_baseline.ipynb   # Classical ML baselines 
│ ├── quantum_pipeline.ipynb     # Full quantum feature map + variational circuit pipeline
│ ├── quantum_feature_map_tests.ipynb   # Tests for PennyLane/Qiskit Feature Maps
│ ├── analysis.ipynb # Post training metrics, plots, performance summary
├── results/            # CSVs, plots, models
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

---

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



## Key Features 

### Hybrid Classical Quantum Pipeline

* Combines PCA based feature reduction with quantum embeddings using a PennyLane ZZ feature map. 
* Integrates a variational ansatz to act as a learnable quantum transformation for enhancd feature expressivity. 

### Configurable Experiment Parameters

* Varied number of qubits, circuit depth, number of measurement shots, device backend (simulator vs IBM backend), and noise models. 

### Classical Baseline Comparison

* Uses Logistic Regression and SVMs trained on the same PCA reduced data. 
* Enables easy accuracy and runtime comparison between classical and quantum-enhanced models. 

### Quantum Circuit Visualization 

* Visualized feature maps and ansatz circuits for inspection using PennyLane and Qiskit.

### IBM Quantum Integration

* Access to real IBM Quantum devices via the IBMQ provider, using the least_busy() backend selection for optimal queue management. 

### Logging Results and Analysis

* Automatic metric tracking (accuracy, runtime, number of parameters).
* Generates CSV logs and summary visualizations for classical/quantum experiments. 


---

## Key Concepts

### Quantum Feature Maps

Transform classical data into quantum states through unitary operations, which allows models to implicitly compute inner products in a high-dim Hilbert space (similar to kernel methods but with potentially richer representational capacity). 

The maps used in the quantum_feature_maps_tests.ipynb notebook are: 
* Basis Encoding
* Angle Encoding (encodes data into rotation angles) 
  * ZZ feature map (applies entangling gates to capture data correlations)
* Amplitude Encoding (uses vector amplitudes of quantum states)


### Variational Ansatz 

Parameterized quantum circuit designed to learn transformations that best seperate classes. These parameters are optimzied classicaly, which bridges the gap between classical optimization and quantum hardware.

### Hybrid Training Workflow
* Preprocessing: Classical PCA
* Encoding: Angle encoding QFM
* Measurement: Expectation values produce quantum features
* Classification: Trains classical models on quantum features


### Quantum Hardware and Simulators
This project supports both **local simulators** (default.qubit, aer_simulator) and **IBM Quantum devices** (qiskit.ibmq), allowing for the experiments to scale from local to realistic noisy devices. 

---

## Research Motivation

While quantum computing is still in its early, noise limited stage, hybrid quantum classical learning is one of the more promising pathyways towards achieving a Quantum Advantage in ML.

Classical NNs oftens struggle to represent highly entangled/non linear data relationships. Quantum circuits can encode this data into exponentially larger Hilbert spaces using relatively few qubits, allowing us to potentially reveal complex correlations that are classically impossible. 

This project aims to explore QFMs as trainable preprocessing layers, bridging the gap between classical feature extraction (PCA) and quantum enhanced computation. Through systematic experiments, this repo investigates whether quantum encoded features can provide measurable imporvements in accuracy in small scale datasets. 

---

## Future Work
* Integration with classical deep learning pipelines (hybrid QNN-CNN architectures).
* Noise Resilient Training (evaluation of how noise impacts accuracy, stability, along with noise-adaptive ansatz designs). 
* Trainable QFMs (instead of ZZ feature maps, variational quantum feature maps (VQFMs)). 


