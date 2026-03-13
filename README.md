# Nonlinear Market Dynamics Pipeline

This repository contains a modular research pipeline for nonlinear time-series analysis, regime detection, and dynamical system reconstruction. It allows for the analysis of financial time-series data to discover regime changes and underlying mathematical equations governing the market dynamics.

## Features

- **Phase Space Reconstruction**: Determines optimal time delays and embedding dimensions (Takens' Theorem).
- **Chaos Diagnostics**: Calculates Lyapunov exponents, sample entropy, and correlation dimensions to quantify chaotic signatures.
- **Regime Detection**: Utilizes Hidden Markov Models (HMM) and Changepoint Detection (PELT/Binseg) to identify underlying market states.
- **Dynamics Discovery**: 
  - Uses **SINDy** (Sparse Identification of Nonlinear Dynamics) to discover sparse algebraic equations.
  - Trains **Continuous-depth Neural ODEs** (Ordinary Differential Equations) using `torchdiffeq` to construct predictive neural models of the phase space.
- **Visualization**: Generates 2D/3D attractors, recurrence plots, and regime segmentation timelines automatically.
- **Data Extensibility**: Works out-of-the-box with synthetic data (Lorenz attractor) and easily integrates with real market data (CSV/Parquet).

## Project Structure

- `main_pipeline.py`: Central orchestration script.
- `config.yaml`: Central configuration for hyperparameters (WandB logging, file paths, model dimensions).
- `analysis/`: Phase space metrics, false nearest neighbors, mutual information, and chaos diagnostics.
- `regime_detection/`: State and changepoint detection algorithms.
- `dynamics/` & `training/`: SINDy and Neural ODE model definitions and training loops.
- `utils/`: Data loading and signal denoising (Wavelets, Moving Averages).
- `visualization/`: Plotting and logging helpers.

## Setup & Local Execution

1. **Install Requirements**:
   It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration, ensure you have PyTorch installed with the correct CUDA version for your system from [pytorch.org](https://pytorch.org/get-started/locally/).*

2. **Configure (Optional)**:
   Modify `config.yaml` to adjust hyperparameters, point to your real data files (e.g., `data/raw/market_data.csv`), or enable `wandb` logging.

3. **Run**:
   ```bash
   python main_pipeline.py
   ```
   If no data is present in `data/raw/`, it will automatically generate synthetic noisy Lorenz data for testing! All plots and models will be saved in the `outputs/` folder.

## Collaboration & Editing

This project is open for open-source development. Feel free to fork, create a branch, and submit Pull Requests!
