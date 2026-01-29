# Agent 2B: LLM-based Turbine Expert Setup Guide

## Overview
Agent 2B is an LLM-powered turbine expert that provides intelligent recommendations for yaw and pitch control based on wind conditions. It uses a local LLM server (moonshotai/Kimi-K2.5) with OpenAI-compatible API.

## Prerequisites

### 1. Install Required Packages
```bash
pip install openai pyyaml
```

### 2. Set Up Your Local LLM Server
Ensure your local LLM server is running and accessible. The server should be OpenAI-compatible and support the chat completion endpoint.

## Configuration

Agent 2B uses a `config.yaml` file for all settings. This provides a clean, centralized configuration approach.

### Step 1: Edit config.yaml
The `config.yaml` file is located in the same directory as `wind_turbine_gui.py`. Open it and configure the following settings:

```yaml
# Agent 2B: LLM Configuration

llm:
  # API Configuration
  api_key: "your-api-key-here"  # Replace with your actual API key
  api_base: "http://localhost:8000/v1"  # Base URL of your local LLM server
  
  # Model Configuration
  model: "moonshotai/Kimi-K2.5"  # LLM model name
  
  # Generation Parameters
  temperature: 0.1  # Lower = more deterministic, Higher = more creative
  max_tokens: 200000  # Maximum tokens for output generation
  timeout: 1000.0  # Request timeout in seconds
```

### Configuration Parameters Explained:

- **api_key**: Your API key for authenticating with the LLM server
- **api_base**: Base URL of your local LLM server (e.g., `http://localhost:8000/v1`)
- **model**: Name of the LLM model to use (`moonshotai/Kimi-K2.5`)
- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
  - **0.1** (recommended): More focused and deterministic responses
  - **0.7**: Balanced creativity and coherence
  - **1.0**: More creative but less consistent
- **max_tokens**: Maximum number of tokens the LLM can generate (200000 allows for detailed responses)
- **timeout**: Request timeout in seconds (1000.0 = 16.7 minutes for complex queries)

## How Agent 2B Works

### 1. Input
Agent 2B receives wind conditions from Agent 1:
- Wind Speed (m/s)
- Wind Direction (degrees)

### 2. Processing
1. Constructs a detailed prompt with:
   - Current wind conditions
   - NREL 5 MW turbine specifications
   - Request for yaw/pitch recommendations
2. Sends the prompt to the local LLM via OpenAI-compatible API
3. Receives intelligent recommendations from the LLM

### 3. Output
The LLM provides:
- Optimal yaw angle (nacelle direction)
- Optimal pitch angle
- Expected operating region
- Detailed explanation of the recommendations

## Integration in the GUI

Agent 2B is positioned between Agent 2 and Agent 3:
```
Agent 1 (Weather) → Agent 2 (Rule-based Expert) → Agent 2B (LLM Expert) → Agent 3 (Optimizer)
```

This allows users to:
- Compare rule-based recommendations (Agent 2) with AI-powered recommendations (Agent 2B)
- Get a second opinion from an LLM trained on wind turbine operations
- Benefit from both deterministic and AI-driven decision-making

## Troubleshooting

### Error: "Could not connect to LLM"
**Cause:** LLM server is not running or not accessible.

**Solutions:**
1. Verify your LLM server is running
2. Check the `api_base` URL in `config.yaml` is correct
3. Ensure the `api_key` in `config.yaml` is valid
4. Test the connection manually:
   ```python
   import openai
   import yaml
   
   # Load config
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   llm_config = config['llm']
   openai.api_key = llm_config['api_key']
   openai.api_base = llm_config['api_base']
   
   response = openai.ChatCompletion.create(
       model=llm_config['model'],
       messages=[{"role": "user", "content": "Hello"}],
       temperature=llm_config['temperature'],
       max_tokens=1000,
       request_timeout=llm_config['timeout']
   )
   print(response)
   ```

### Error: "Config file not found"
**Cause:** The `config.yaml` file is missing or in the wrong location.

**Solution:** Ensure `config.yaml` is in the same directory as `wind_turbine_gui.py`. Create it if missing:
```yaml
llm:
  api_key: "your-api-key-here"
  api_base: "http://localhost:8000/v1"
  model: "moonshotai/Kimi-K2.5"
  temperature: 0.1
  max_tokens: 200000
  timeout: 1000.0
```

### Error: "Module 'openai' not found" or "Module 'yaml' not found"
**Solution:** Install the required packages:
```bash
pip install openai pyyaml
```

### LLM Gives Generic Responses
**Cause:** The LLM may not have specific knowledge about NREL 5 MW turbines.

**Solutions:**
1. Fine-tune your LLM on wind turbine data
2. Provide more detailed context in the system message
3. Use a more capable LLM model

## API Reference

### `load_llm_config(config_path=None)`
Loads LLM configuration from YAML file.

**Parameters:**
- `config_path` (str, optional): Path to the config.yaml file. If None, uses default path.

**Returns:**
- `dict`: Configuration dictionary with keys: api_key, api_base, model, temperature, max_tokens, timeout

### `query_local_llm(api_key, api_base, model_name, prompt, system_message, temperature, max_tokens, timeout)`
Queries the local LLM using OpenAI-compatible API.

**Parameters:**
- `api_key` (str): API key for authentication
- `api_base` (str): Base URL of the LLM server
- `model_name` (str): Name of the LLM model (e.g., "moonshotai/Kimi-K2.5")
- `prompt` (str): User query
- `system_message` (str, optional): Context for the LLM
- `temperature` (float): Sampling temperature (default: 0.1)
- `max_tokens` (int): Maximum tokens for output (default: 200000)
- `timeout` (float): Request timeout in seconds (default: 1000.0)

**Returns:**
- `str`: LLM response

### `get_llm_expert_recommendation(wind_speed, wind_direction, config)`
Gets expert recommendations from the LLM for given wind conditions.

**Parameters:**
- `wind_speed` (float): Wind speed in m/s
- `wind_direction` (float): Wind direction in degrees
- `config` (dict, optional): LLM configuration. If None, loads from config.yaml

**Returns:**
- `dict`: Contains 'llm_response', 'wind_speed', 'wind_direction', 'config'

## Example Output

```
🤖 LLM Expert Recommendation

Based on the current wind conditions (Wind Speed: 8.5 m/s, Wind Direction: 275°), 
here are my recommendations for the NREL 5 MW Reference Wind Turbine:

1. Optimal Yaw Angle: 275° (aligned with wind direction)
   - The nacelle should be oriented directly into the wind to maximize power capture
   
2. Optimal Pitch Angle: 0° (fine pitch)
   - At this wind speed (8.5 m/s), the turbine is in the partial load region
   - Blades should be at minimum pitch to maximize rotor speed and power output
   
3. Expected Operating Region: Partial Load
   - Wind speed is between cut-in (3.0 m/s) and rated (11.4 m/s)
   - Turbine will produce approximately 2.5-3.0 MW
   
4. Explanation:
   - In partial load operation, the goal is to extract maximum power from the wind
   - The power output follows the cubic relationship with wind speed
   - Yaw misalignment should be minimized as power loss is proportional to cos³(θ)
   - Pitch control is minimal in this region; blade angle optimization occurs above rated wind speed
```

## Security Notes

1. **Never commit API keys to version control**
   - Add `config.yaml` to your `.gitignore` file
   - Create a `config.yaml.template` with placeholder values for documentation
   
2. **Protect your config.yaml file**
   - Set appropriate file permissions
   - Consider encrypting sensitive data at rest
   
3. **Use secure connections**
   - Ensure your LLM server uses HTTPS in production
   - Validate SSL certificates
   
4. **API Key Management**
   - Rotate API keys regularly
   - Use different keys for development and production
   - Consider using a secrets management system (e.g., Azure Key Vault, AWS Secrets Manager)

## Quick Start Guide

1. **Install dependencies:**
   ```bash
   pip install openai pyyaml
   ```

2. **Edit config.yaml:**
   ```yaml
   llm:
     api_key: "your-actual-api-key"
     api_base: "http://localhost:8000/v1"
     model: "moonshotai/Kimi-K2.5"
     temperature: 0.1
     max_tokens: 200000
     timeout: 1000.0
   ```

3. **Run the application:**
   ```bash
   streamlit run wind_turbine_gui.py --server.port 8506
   ```

4. **Select a wind farm and click "Run Analysis"**
   - Agent 2B will automatically load config.yaml
   - LLM recommendations will appear between Agent 2 and Agent 3

## Contact
For questions or issues with Agent 2B:
- Email: mandar.tabib@sintef.no
