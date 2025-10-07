# Anonymous ICLR 2026 Submission — BACE

This repository contains the implementation of the model and experiments described in our **ICLR 2026** submission.  
All code is anonymized and self-contained for reproducibility.  
No real neural data are shared due to participant privacy and institutional data-use agreements.

---

## Overview

**BACE (Behavior-Adaptive Connectivity Estimation)** is a phase-specific signed-graph learning framework for modeling neural dynamics.  
The codebase supports two main modes of operation:

1. **`main_real.py`** – End-to-end training and evaluation on intracranial neural recordings  
   *(dataset paths provided as placeholders; data not included)*  
2. **`main_synth.py`** – Synthetic experiments reproducing the validation results reported in the paper.  
   This version runs without any external dataset and demonstrates the full model pipeline.

Both modes share the same architecture and loss formulations.

---

## Key Features
- **Phase-specific graph learning** with signed directed adjacency matrices  
- **Temporal modeling** using region-wise GRUs and autoregressive decoding  
- **Regularization suite** for continuity, velocity, curvature, and sparsity control  
- **Built-in visualization** of learned effective connectivity and forecast trajectories  
- **Synthetic benchmark mode** for complete reproducibility without sensitive data  

---

## Repository Structure
