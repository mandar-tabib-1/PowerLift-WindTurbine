# Wind Turbine Multi-Agent AI Analysis System

🌀 **Advanced wind turbine analysis and optimization using multiple AI agents for predictive maintenance and farm optimization**

*Developed by mandar.tabib@sintef.no*

## Overview
This system integrates multiple AI agents and Large Language models for wind turbine operations analysis. It is an exploratory work and has potential for developments.

The system coordinates specialized ML algorithms and Large Language models in an agentic framework to deliver real-time turbine operation recommendations, predictive maintenance, power optimization, and 3D wake flow visualization.

---

## Features

- **Real-time weather integration** (Open-Meteo, yr.no APIs)
- **LLM-powered turbine control recommendations**
- **Wake steering optimization** for power maximization
- **High-fidelity wake flow simulation** (TT-OpInf ROM)
- **Uncertainty-aware power prediction** (Gaussian Process)
- **Predictive maintenance** with RUL estimation
- **Norwegian wind farm integration** with turbine location mapping
- **Comprehensive fault diagnosis** with SHAP-based explainability
- **PdM Chatbot** for maintenance insights
- **What-if analysis mode** for scenario testing

---

## Norwegian Wind Farms Supported

- **Bessaker Wind Farm** (25 turbines, 57.5 MW) - Bessakerfjellet, Åfjord, Trøndelag
- **Smøla Wind Farm** (68 turbines, 150 MW) - One of Norway's oldest and largest wind farms
- **Tonstad Wind Farm** (51 turbines, 208 MW) - One of Norway's largest onshore wind farms
- **Roan Wind Farm** (71 turbines, 255.6 MW) - Part of the Fosen Vind project
- **Raggovidda Wind Farm** (15 turbines, 45 MW) - Norway's northernmost wind farm

---

## Quick Start

### Prerequisites

- **Python 3.11+** (required by scikit-learn 1.8.0)
- **Git** for cloning the repository
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

```bash
# Install uv (if not already installed)
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd PowerLift

# Install all dependencies with uv
uv sync

# Or install with all optional LLM providers (Anthropic, Google)
uv sync --all-extras
```

### Configuration

1. **Create your environment file** from the template:

```bash
cp .env.example .env
```

2. **Edit `.env`** and add your API keys:

```env
# Required for NTNU LLM provider
NTNU_API_KEY=your-ntnu-api-key-here
NTNU_API_BASE=https://llm.hpc.ntnu.no/v1

# Optional providers (uncomment as needed)
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# GOOGLE_API_KEY=your-google-key
```

3. **For Streamlit Cloud deployment**, set secrets in the dashboard:
   - Go to your app's Settings > Secrets
   - Add the same key-value pairs from `.env`

### Important Notes

> **1. NTNU VPN Required**
> To use the LLM model at NTNU, you must be connected to the NTNU VPN.
>
> **2. Alternative LLM Providers**
> You can also use LLM with OpenAI or Anthropic using your own API key.
>
> **3. API Key Privacy**
> Your API keys are stored in your local `.env` file and never shared.

### Run the Application

```bash
# Using uv
uv run streamlit run wind_turbine_gui.py

# Or activate the virtual environment first
# Windows
.venv\Scripts\activate
streamlit run wind_turbine_gui.py

# macOS / Linux
source .venv/bin/activate
streamlit run wind_turbine_gui.py
```

The app will open at `http://localhost:8501` (or use `--server.port 8506` for custom port).

---

## Multi-Agent Architecture

The system consists of **6 specialized AI agents** + an **Expert Reviewer**:

| Agent | Name | Description |
|-------|------|-------------|
| 1 | **Weather Station** | Fetches real-time wind conditions from Open-Meteo API |
| 2 | **Turbine Expert** | NREL 5MW reference turbine specs + LLM-based yaw optimization |
| 2B | **LLM Expert** | Multi-provider LLM for advanced analysis (NTNU/OpenAI/Anthropic/Google/Ollama) |
| 2C/2D | **Pair Selector** | Identifies upstream-downstream turbine pairs for wake analysis |
| 3 | **Wake Steering Optimizer** | Multi-turbine farm optimization for power maximization |
| 4 | **Flow AI Agent** | TT-OpInf (Tensor-Train Operator Inference) reduced-order model for 3D velocity fields |
| 5 | **Power AI** | Gaussian Process model trained on CFD data with uncertainty quantification |
| 6 | **Predictive Maintenance** | Health monitoring with RUL estimation using Autoencoder + GMM + LSTM |
| R | **Expert Reviewer** | LLM-based validation at critical checkpoints (advisory or blocking mode) |

---

## Project Structure

```
PowerLift/
|-- .env                        # API keys (gitignored - create from .env.example)
|-- .env.example                # Template for environment variables
|-- .streamlit/
|   |-- secrets.toml            # Streamlit secrets for deployment (gitignored)
|-- config.yaml                 # LLM and reviewer configuration (gitignored)
|-- config.yaml.template        # Template for config.yaml
|-- pyproject.toml              # UV/Python package configuration
|-- uv.lock                     # Locked dependency versions
|-- requirements.txt            # Legacy pip requirements (kept for compatibility)
|
|-- wind_turbine_gui.py         # Main Streamlit GUI application
|-- wind_turbine_orchestrator.py # Multi-agent orchestrator (CLI mode)
|
|-- rotor_power_agent.py        # Agent 5: GP-based power prediction
|-- tt_opinf_inference_agent.py # Agent 4: TT-OpInf wake flow prediction
|-- wake_flow_prediction_agent.py # Wake flow prediction wrapper
|-- reviewer_agent.py           # Expert LLM reviewer agent
|-- shap_interpreter.py         # SHAP-based model interpretability
|-- wake_animation.py           # Wake flow animation utilities
|-- verify_wake_spacing.py      # Wake spacing verification utilities
|
|-- llm/                        # Multi-provider LLM interface
|   |-- __init__.py
|   |-- base.py                 # Abstract base class + LLMFactory
|   |-- ntnu_llm.py             # NTNU IDUN provider (OpenAI-compatible)
|   |-- openai_llm.py           # OpenAI GPT provider
|   |-- anthropic_llm.py        # Anthropic Claude provider
|   |-- google_llm.py           # Google Gemini provider
|   |-- ollama.py               # Local Ollama provider
|
|-- TT_OpInf/                   # Tensor Train + OpInf ROM module (local)
|   |-- __init__.py             # TT_OpInf module initialization
|   |-- tt_opinf.py             # Main TT-OpInf class with fit/predict methods
|
|-- models/                     # Pre-trained model files
|   |-- tt_opinf_model/
|       |-- metadata.npz        # Model metadata (grid info, normalization)
|       |-- opinf.npz           # OpInf operator matrices
|       |-- tt_decomp.npz       # TT decomposition factors
|
|-- data/
|   |-- Grid_data.vtk           # Computational grid (180,857 spatial points)
|
|-- rotor_power_gp_model.joblib # Trained Gaussian Process model
|-- powerRotor_combined.csv     # Training data (CFD simulation results)
|
|-- RUL/                        # Predictive Maintenance module
|   |-- save_models.py          # PdM model loading utilities
|   |-- wind_turbine_pm_fuhrlander.py # Fuhrlander turbine PdM models
|   |-- wind_turbine_pm_sklearn.py    # Sklearn-based PdM models
|   |-- shap_explainer.py       # SHAP-based model interpretability
|   |-- inference_viz.py        # PdM inference visualization
|   |-- saved_models/           # Pre-trained PdM model files
|       |-- fuhrlander_fl2500_pm_models.joblib
|       |-- metadata.json
|       |-- test_data.npz
|   |-- fuhrlander-master/      # Fuhrlander dataset and examples
|
|-- docs/                       # Documentation
|   |-- user-guides/            # User guides and feature documentation
|   |-- research/               # Research papers and reports
|   |-- setup/                  # Setup and configuration guides
```

---

## Machine Learning Models

- **Wake Flow**: Tensor Train + Operator Inference ROM
- **Power**: Gaussian Process Regressor with uncertainty quantification
- **Health**: Autoencoder + GMM + LSTM for predictive maintenance
- **Fault Diagnosis**: SHAP explainer integration for interpretable predictions

---

## Output Format

The system produces structured outputs at each stage:

### Agent 1 - Weather Data
```json
{
  "wind_speed_ms": 8.5,
  "wind_direction_deg": 275.0,
  "temperature_c": 12.3,
  "location": "Bessaker Wind Farm",
  "timestamp": "2025-01-15T14:30:00"
}
```

### Agent 2 - Expert Recommendations
```json
{
  "suggested_yaw": 278.0,
  "yaw_misalignment": 3.0,
  "operating_region": "Region 2 (Partial Load)",
  "expected_efficiency": 0.95,
  "reasoning": ["Wind speed optimal for partial load operation", ...]
}
```

### Agent 5 - Power Prediction
```json
{
  "yaw_angle": 278,
  "summary": {
    "mean_power_MW": 2.847,
    "max_power_MW": 3.125,
    "min_power_MW": 2.432,
    "mean_uncertainty_MW": 0.089
  },
  "time_series": {"power_mw": [...], "uncertainty_mw": [...]}
}
```

### Agent 4 - Wake Flow Prediction
```json
{
  "status": "success",
  "timesteps": 100,
  "spatial_points": 180857,
  "velocity_magnitude": "numpy array (100, 180857)",
  "yaw_angle": 278
}
```

### Expert Reviewer - Checkpoint Review
```json
{
  "checkpoint": "checkpoint1_agent2",
  "severity": "info",
  "findings": [{"type": "info", "message": "All checks passed", "rule": "all_checks_passed"}],
  "allow_continue": true,
  "reviewer_mode": "advisory",
  "summary": "All checks passed."
}
```

---

## Supported LLM Providers

| Provider | Config Key | Notes |
|----------|-----------|-------|
| NTNU IDUN | `NTNU_API_KEY` | NTNU HPC cluster (OpenAI-compatible API) |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4, etc. |
| Anthropic | `ANTHROPIC_API_KEY` | Claude models (requires `pip install anthropic`) |
| Google | `GOOGLE_API_KEY` | Gemini models (requires `pip install google-genai`) |
| Ollama | -- | Local models, no API key needed |

---

## Predictive Maintenance Features

### Fault Diagnosis Mapping
The system includes comprehensive fault diagnosis with 15+ monitored components:
- **Gearbox Components**: Temperature monitoring, oil pressure, bearing conditions
- **Generator Systems**: Winding temperatures, bearing health, speed variability
- **Main Bearings**: Oil pressure, temperature balance
- **Performance Metrics**: Power efficiency, thermal/bearing stress indices

### SHAP-based Explainability
- Root cause analysis for detected anomalies
- Maintenance action recommendations with cost estimates
- Urgency classification (LOW/MEDIUM/HIGH/CRITICAL)

---

## Deployment (Streamlit Cloud)

**TT_OpInf ROM module now included locally** - No external dependencies needed!

1. Push your code to GitHub (`.env` and `secrets.toml` are gitignored)
2. Connect your repo on [Streamlit Cloud](https://streamlit.io/cloud)
3. In the app dashboard, go to **Settings > Secrets** and add:
   ```toml
   NTNU_API_KEY = "your-key-here"
   NTNU_API_BASE = "https://llm.hpc.ntnu.no/v1"
   ```
4. Deploy - the app reads secrets automatically via `st.secrets`

**Note**: The TT_OpInf (Tensor Train + Operator Inference) module is now included as a local module, eliminating deployment issues related to missing ROM dependencies.

---

## Documentation

Comprehensive documentation is organized by category in the [`docs/`](docs/) folder:

### User Guides ([`docs/user-guides/`](docs/user-guides/))
- [ML Wake Analysis Guide](docs/user-guides/ML_WAKE_ANALYSIS_GUIDE.md) - Advanced wake analysis functions and methodology
- [Predictive Maintenance Features](docs/user-guides/PDM_CHATBOT_FEATURES.md) - PdM chatbot and LLM integration
- [Fault Diagnosis with SHAP](docs/user-guides/FAULT_DIAGNOSIS_SHAP.md) - Intelligent fault diagnosis system
- [LLM Integration Guide](docs/user-guides/README_LLM.md) - Local LLM setup with OpenAI-compatible API
- [Optimization Guide](docs/user-guides/README_Optimization.md) - Wake steering optimization using PyTorch AutoDiff
- [RUL Inference Setup](docs/user-guides/README_INFERENCE.md) - Fuhrlander FL2500 predictive maintenance
- [Predictive Maintenance Framework](docs/user-guides/WIND_TURBINE_PM_DOCUMENTATION.md) - Complete PdM system documentation

### Research Documentation ([`docs/research/`](docs/research/))
- [Research Paper: Predictive Maintenance](docs/research/RESEARCH_PAPER_PM.md) - Academic paper on semi-supervised ML framework
- [Fuhrlander Model Report](docs/research/FUHRLANDER_MODEL_REPORT.md) - Model validation and performance metrics
- [SHAP Analysis Report](docs/research/SHAP_ANALYSIS_REPORT.md) - Explainability analysis with per-sample explanations

### Setup & Configuration ([`docs/setup/`](docs/setup/))
- [Agent 2B Setup](docs/setup/AGENT_2B_SETUP.md) - LLM-based turbine expert configuration
- [Reviewer Agent Setup](docs/setup/REVIEWER_AGENT_SETUP.md) - Expert reviewer agent with validation system

---

## Troubleshooting

### Common Issues

1. **LLM Connection Error**: If you see "Could not connect to LLM. Error: query_local_llm() got an unexpected keyword argument 'temperature'"
   - Ensure your LLM provider configuration is correct in the GUI sidebar
   - Check API keys in `.env` file
   - Try switching to a different LLM provider

2. **Missing Dependencies**: If packages are missing, run:
   ```bash
   uv sync --all-extras
   ```

3. **Port Conflicts**: If port 8501 is busy:
   ```bash
   streamlit run wind_turbine_gui.py --server.port 8506
   ```

---

## Contact

**Developer**: Mandar Tabib
**Email**: mandar.tabib@sintef.no
**Organization**: SINTEF Digital

---

## Acknowledgement

NTNU for providing short-term access to the local LLM at IDUN computational facility for enabling this academic testing and funding from SEP and Northwind FME. Mandar Tabib.

Developed at **SINTEF Digital**, Norway.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{wind_turbine_multi_agent,
  title={Wind Turbine Multi-Agent AI Analysis System},
  author={Tabib, Mandar},
  organization={SINTEF Digital},
  year={2025},
  url={https://github.com/mandar-tabib-1/PowerLift-WindTurbine.git}
}
```
