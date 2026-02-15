# PowerLift: Multi-Agent Wind Turbine Analysis System

A physics-informed machine learning system for wind turbine power prediction and wake flow simulation, developed at **SINTEF Digital**.

## Overview

This project implements a multi-agent system that combines:
- **Real-time weather data** acquisition
- **Expert system** for yaw angle optimization  
- **Gaussian Process regression** for power prediction with uncertainty quantification
- **TT-OpInf (Tensor-Train Operator Inference)** for wake flow field prediction

## Architecture

The system consists of 4 AI agents:

| Agent | Name | Description |
|-------|------|-------------|
| 1 | Weather Station | Fetches real-time wind conditions from Open-Meteo API |
| 2 | Turbine Expert | Consults NREL 5MW reference turbine specifications for optimal yaw |
| 3 | Power Predictor | Gaussian Process model trained on CFD data for power prediction |
| 4 | Wake Flow Simulator | TT-OpInf physics-based model for 3D velocity field prediction |

## Quick Start

### Installation

```bash
pip install numpy scipy matplotlib scikit-learn joblib streamlit pyvista requests
```

### Run the GUI

```bash
streamlit run wind_turbine_gui.py
```

## Models

- **Power Prediction**: Gaussian Process trained on CFD simulation data
- **Wake Flow**: TT-OpInf reduced-order model (180,857 spatial points)

## Acknowledgments

Developed at **SINTEF Digital**, Norway.
