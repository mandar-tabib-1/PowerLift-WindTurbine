# PowerLift: Multi-Agent Wind Turbine Analysis System

A physics-informed machine learning system for wind turbine power prediction and wake flow simulation, developed at **SINTEF Digital**, Norway.

The system coordinates four specialized AI agents to deliver real-time turbine operation recommendations, power optimization, and 3D wake flow visualization through a Streamlit web interface.

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

The app will open at `http://localhost:8501`.

---

## Architecture

The system consists of **4 AI agents** + an **Expert Reviewer**:

| Agent | Name | Description |
|-------|------|-------------|
| 1 | **Weather Station** | Fetches real-time wind conditions from Open-Meteo API |
| 2 | **Turbine Expert** | NREL 5MW reference turbine specs + LLM-based yaw optimization |
| 2B | **LLM Expert** | Multi-provider LLM for advanced analysis (NTNU/OpenAI/Anthropic/Google/Ollama) |
| 2C/2D | **Pair Selector** | Identifies upstream-downstream turbine pairs for wake analysis |
| 3 | **Power Predictor** | Gaussian Process model trained on CFD data with uncertainty quantification |
| 4 | **Wake Flow Simulator** | TT-OpInf (Tensor-Train Operator Inference) reduced-order model for 3D velocity fields |
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
|-- rotor_power_agent.py        # Agent 3: GP-based power prediction
|-- tt_opinf_inference_agent.py # Agent 4: TT-OpInf wake flow prediction
|-- wake_flow_prediction_agent.py # Wake flow prediction wrapper
|-- reviewer_agent.py           # Expert LLM reviewer agent
|-- shap_interpreter.py         # SHAP-based model interpretability
|-- wake_animation.py           # Wake flow animation utilities
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
```

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

### Agent 3 - Power Prediction
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

## Deployment (Streamlit Cloud)

1. Push your code to GitHub (`.env` and `secrets.toml` are gitignored)
2. Connect your repo on [Streamlit Cloud](https://streamlit.io/cloud)
3. In the app dashboard, go to **Settings > Secrets** and add:
   ```toml
   NTNU_API_KEY = "your-key-here"
   NTNU_API_BASE = "https://llm.hpc.ntnu.no/v1"
   ```
4. Deploy - the app reads secrets automatically via `st.secrets`

---

## Acknowledgments

Developed at **SINTEF Digital**, Norway.
