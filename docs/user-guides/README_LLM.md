# Using a Local LLM with OpenAI-Compatible API

This document explains the standard practices for integrating a local LLM (Large Language Model) with an OpenAI-compatible API in your project. It also provides guidance on how to use methods like those in `base.py` and `llm/ntnu_llm.py` with a `config.yaml` file for configuration.

---

## **Overview**

Using a local LLM with an OpenAI-compatible API and a `config.yaml` file is a standard practice for managing API settings, model parameters, and other configurations. This approach ensures modularity, scalability, and maintainability.

---

## **Standard Practices**

### 1. **Configuration Management with `config.yaml`**
- Use a `config.yaml` file to store API keys, base URLs, model names, and other parameters.
- This decouples sensitive information from the codebase and allows easy updates.

Example `config.yaml` structure:
```yaml
llm:
  api_key: "your_api_key_here"
  api_base: "http://localhost:8000/v1"
  model: "moonshotai/Kimi-K2.5"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30.0
```

---

### 2. **Loading Configuration**
- Use a utility function to load the `config.yaml` file. This ensures the configuration is loaded once and reused across the application.

Example:
```python
import yaml
import os

def load_llm_config(config_path: str = "config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("llm", {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
```

---

### 3. **OpenAI-Compatible API Client**
- Use the `openai` Python library to interact with the local LLM server. Configure the `api_base` and `api_key` to point to your local server.

Example (for openai>=1.0.0):
```python
import openai

def query_local_llm(api_key, api_base, model, prompt, system_message=None, temperature=0.7, max_tokens=1000, timeout=None):
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying the LLM: {e}"
```

---

### 4. **System and User Prompts**
- Use a `system_message` to set the context for the LLM (e.g., its role, expertise, or behavior).
- The `prompt` is the user query or task for the LLM.

Example:
```python
system_message = "You are an expert in wind turbine operations and control."
prompt = "What is the optimal yaw angle for a wind turbine at 10 m/s wind speed?"
response = query_local_llm(
    api_key=config["api_key"],
    api_base=config["api_base"],
    model=config["model"],
    prompt=prompt,
    system_message=system_message,
    temperature=config["temperature"],
    max_tokens=config["max_tokens"]
)
print(response)
```

---

### 5. **Error Handling**
- Implement robust error handling to manage issues like connection failures, invalid API keys, or timeouts.

Example:
```python
try:
    response = query_local_llm(...)
except Exception as e:
    print(f"Error: {e}")
```

---

### 6. **Caching Responses (Optional)**
- Use caching (e.g., `functools.lru_cache`) to avoid redundant API calls for the same prompt.

---

## **Implementation in Your Project**

### 1. **Create a `config.yaml` File**
- Place the `config.yaml` file in the root directory of your project.
- Define the LLM configuration as shown above.

---

### 2. **Add Utility Functions**
- Add the `load_llm_config` and `query_local_llm` functions to a utility module (e.g., `llm_utils.py`).

---

### 3. **Integrate with Your Project**
- Replace hardcoded LLM configuration in your code with the `load_llm_config` function.
- Use the `query_local_llm` function to interact with the LLM.

Example integration:
```python
from llm_utils import load_llm_config, query_local_llm

config = load_llm_config()

system_message = "You are an expert in wind turbine operations and control."
prompt = "What is the optimal yaw angle for a wind turbine at 10 m/s wind speed?"

response = query_local_llm(
    api_key=config["api_key"],
    api_base=config["api_base"],
    model=config["model"],
    prompt=prompt,
    system_message=system_message,
    temperature=config["temperature"],
    max_tokens=config["max_tokens"]
)
print(response)
```

---

## **Benefits of This Approach**
- **Modularity**: Configuration is decoupled from the code, making it easier to update.
- **Reusability**: The `query_local_llm` function can be reused across different projects.
- **Scalability**: Easily switch between different LLM models or servers by updating the `config.yaml` file.
- **Error Handling**: Centralized error handling ensures robustness.

---

## **Troubleshooting**
- **Error: `query_local_llm() got an unexpected keyword argument 'temperature'`**
  - Ensure the `query_local_llm` function signature matches the parameters passed.
  - Verify the `config.yaml` file contains the correct keys and values.

- **LLM Server Connection Issues**
  - Ensure the local LLM server is running and accessible at the `api_base` URL.
  - Check for firewall or network issues.

- **Error: `You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0`**
  - Update your code to use the new OpenAI API interface as shown above.

---

## 🔄 How to Reuse This LLM Workflow in Other Projects

### **Files to Copy/Use**
To reuse this LLM integration workflow in another project, copy or adapt the following files:

- **`wind_turbine_gui.py`**  
  (or your main application file; contains the LLM query logic and integration)
- **`config.yaml`**  
  (stores LLM API configuration, model, and parameters)
- **`llm_utils.py`** *(if you modularize the LLM functions)*  
  (contains `load_llm_config`, `query_local_llm`, and related helpers)
- **`base.py` and `ntnu_llm.py`** *(if you want to use or extend the LLM agent classes)*
- **`readme_llm_use.md`**  
  (this documentation for reference)

---

### **Challenges Encountered and Solutions**

#### 1. **API Parameter Compatibility**
- **Challenge:**  
  The OpenAI Python API changed in version 1.0.0, deprecating the old `openai.ChatCompletion.create` interface.
- **Solution:**  
  Updated the code to use the new `openai.OpenAI` client and `client.chat.completions.create` method, ensuring compatibility with openai>=1.0.0.

#### 2. **Parameter Passing (temperature, timeout, etc.)**
- **Challenge:**  
  Errors like `query_local_llm() got an unexpected keyword argument 'temperature'` or `'timeout'` occurred due to mismatches between function signatures and how parameters were passed.
- **Solution:**  
  Explicitly added `temperature`, `max_tokens`, and `timeout` as parameters to the `query_local_llm` function, with defaults and proper forwarding to the API call.

#### 3. **Configuration Management**
- **Challenge:**  
  Hardcoding API keys and parameters is insecure and inflexible.
- **Solution:**  
  Used a `config.yaml` file and a loader function to centralize and manage LLM configuration.

#### 4. **Error Handling**
- **Challenge:**  
  LLM server connection issues, API errors, or misconfigurations can cause runtime failures.
- **Solution:**  
  Wrapped LLM calls in `try-except` blocks, returned informative error messages, and displayed user-friendly warnings in the GUI.

#### 5. **Modularity and Reusability**
- **Challenge:**  
  LLM logic was mixed with application logic, making reuse difficult.
- **Solution:**  
  Modularized LLM configuration loading and querying into standalone functions (`load_llm_config`, `query_local_llm`) for easy reuse in other projects.

---

**To reuse:**  
Copy the above files, update `config.yaml` for your LLM server, and call the LLM functions as shown in this project.

---