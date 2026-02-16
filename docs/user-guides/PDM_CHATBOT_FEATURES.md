# Predictive Maintenance Chatbot Features

## Overview

Enhanced the PdM analysis mode with two major new features:
1. **Downloadable Report Generator** - Comprehensive Markdown reports with recommendations
2. **Interactive Chatbot** - LLM-powered Q&A for analysis interpretation

---

## 1. Downloadable Report Generator

### Location
After the 5-tab analysis display (Health Indicator, State Distribution, Fault Probabilities, RUL, Feature Importance)

### Features
- **One-click generation** - "📥 Generate Report" button
- **Comprehensive analysis** including:
  - Executive summary with overall health assessment
  - Detailed breakdown of health indicators, fault probabilities, and RUL predictions
  - Risk-based recommendations (Critical/High/Medium priority)
  - Preventive measures and long-term strategy suggestions
  - Technical details about ML models and methodology
  - Appendix with data collection process

### Report Content Sections

#### Executive Summary
- Overall health assessment
- Average Health Indicator vs threshold
- Health state distribution (Healthy/Degrading/Critical)
- Average fault probability

#### Detailed Analysis
1. **Health Indicator Analysis**: Statistical metrics (mean, std, min, max, samples above threshold)
2. **Fault Probability Assessment**: Binary and multi-class predictions breakdown
3. **Remaining Useful Life (RUL) Predictions**: Distribution by urgency (Critical <24h, Warning 1-7 days, Safe >7 days)

#### Recommendations
- **Immediate Actions**: Based on risk level (urgent, high priority, medium, normal)
- **Preventive Measures**: Checklist for lubrication, temperature, vibration, oil analysis, generator
- **Long-term Strategy**: Predictive maintenance scheduling, component tracking, failure pattern analysis

#### Technical Details
- ML models used (Autoencoder, GMM, Gradient Boosting, Random Forest, LSTM)
- 27 input features with descriptions
- Model performance metrics (accuracy, F1 score, training/test data)
- Methodology appendix

### Output Format
- **File Type**: Markdown (.md)
- **File Name**: `pdm_analysis_report_YYYYMMDD_HHMMSS.md`
- **Compatible with**: Any Markdown viewer, can be converted to PDF/HTML

---

## 2. Interactive Chatbot

### Location
At the bottom of the PdM analysis page, after the download report section

### Features

#### LLM Provider Selection
Located in expandable "⚙️ LLM Settings" panel with:
- **Provider dropdown**: NTNU, OpenAI, Ollama, Google, Anthropic
- **Model dropdown**: Context-aware list based on selected provider

**Supported Models by Provider:**
- **NTNU**: Kimi-K2.5, QwQ-32B-Preview, DeepSeek-R1, Llama-3.3-70B-Instruct-Turbo
- **OpenAI**: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo
- **Ollama**: Llama2, Mistral, CodeLlama (local models)
- **Google**: Gemini-Pro, Gemini-Ultra
- **Anthropic**: Claude-3-Opus, Claude-3-Sonnet, Claude-2

#### Suggested Questions
8 pre-defined questions displayed as clickable buttons:
1. "What do the health indicator values mean?"
2. "Why is my turbine classified as 'Degrading'?"
3. "How accurate is the RUL prediction?"
4. "What maintenance actions should I take first?"
5. "Explain the difference between binary and multi-class predictions"
6. "What causes high health indicator values?"
7. "How often should I retrain the ML models?"
8. "What are the most important features for fault prediction?"

**Behavior**: Clicking a button populates the question in the chat input

#### Chat Interface
- **Text input**: "Your question:" with placeholder "Ask anything about the analysis..."
- **Submit button**: "🚀 Ask" to send question
- **Spinner**: Shows "Consulting {provider} {model}..." during query

#### Context-Aware Responses
The chatbot has access to:
- Current analysis results (Health Indicator, fault probabilities, RUL)
- Health state distribution (Healthy/Degrading/Critical counts)
- Turbine specifications (Fuhrlander FL2500, 2.5MW)
- Number of samples analyzed
- ML model details (5 models with architecture descriptions)

This context is automatically included in the LLM prompt for informed answers.

#### Expert System Message
The LLM is primed with expert knowledge in:
- Machine learning for anomaly detection and fault prediction
- Wind turbine SCADA data analysis
- Fuhrlander FL2500 turbine specifications and failure modes
- Maintenance strategies and cost-benefit analysis

#### Chat History
- **Display**: Last 5 conversations shown in expandable panels
- **Format**: Question + Answer with provider/model attribution
- **Most recent first**: Latest conversation expanded by default
- **Clear button**: "🗑️ Clear Chat History" to reset

---

## Usage Workflow

### For Report Generation:
1. Run PdM analysis on SCADA data
2. Review 5-tab visualization results
3. Click "📥 Generate Report" button
4. Download the generated Markdown file
5. Share with maintenance team or save for records

### For Chatbot:
1. Run PdM analysis on SCADA data
2. Scroll to bottom "📬 Analysis Q&A Chatbot" section
3. (Optional) Adjust LLM settings via "⚙️ LLM Settings" expander
4. Either:
   - Click a suggested question button
   - Type custom question in text input
5. Click "🚀 Ask" button
6. Wait for LLM response (~5-15 seconds)
7. Review answer in chat history
8. Ask follow-up questions as needed

---

## Technical Implementation

### Functions Added

#### `generate_pdm_report(pdm_results, models_dict)`
- **Purpose**: Generate comprehensive Markdown report
- **Input**: Analysis results dictionary, models dictionary
- **Output**: Formatted Markdown string (3000-5000 lines)
- **Logic**: 
  - Extracts metrics from pdm_results
  - Generates risk-based recommendations
  - Formats technical specifications
  - Creates actionable maintenance checklist

#### `pdm_chatbot_section(pdm_results, models_dict)`
- **Purpose**: Render interactive chatbot interface
- **Input**: Analysis results, models
- **Components**:
  - LLM provider/model selection UI
  - Suggested questions as clickable buttons
  - Text input for custom questions
  - Chat history display with expandable panels
- **Integration**: Uses existing `query_local_llm()` function from Wind Farm mode

### Session State Variables
- `st.session_state.pdm_llm_provider` - Selected LLM provider
- `st.session_state.pdm_selected_model` - Selected model
- `st.session_state.pdm_chat_input` - Current question text
- `st.session_state.pdm_chat_history` - List of Q&A pairs

### LLM Integration
Reuses existing `query_local_llm()` function with:
- **API Key**: Provider-specific (NTNU hardcoded, others from env vars)
- **API Base URL**: Provider-specific endpoints
- **Temperature**: 0.7 for balanced creativity/accuracy
- **Max Tokens**: 1000 for comprehensive answers
- **Timeout**: 30 seconds
- **Context**: Current analysis metrics automatically included
- **System Message**: Expert wind turbine maintenance specialist persona

---

## Benefits

### For Operators
1. **Quick Decision Making**: Download report with clear recommendations
2. **Knowledge Access**: Ask questions without reading documentation
3. **Risk Assessment**: Understand severity of detected issues
4. **Maintenance Planning**: Get specific action items with priorities

### For Maintenance Teams
1. **Comprehensive Reports**: All analysis in one shareable document
2. **Expert Guidance**: LLM provides maintenance best practices
3. **Cost Analysis**: Report includes failure cost vs planned maintenance cost
4. **Preventive Checklists**: Actionable items for routine checks

### For Management
1. **Audit Trail**: Downloaded reports serve as records
2. **Risk Communication**: Clear health status indicators
3. **Resource Planning**: Long-term strategy recommendations
4. **Training Tool**: Suggested questions educate new staff

---

## Example Use Cases

### Use Case 1: Critical Turbine Health
**Scenario**: Health Indicator is 0.45 (threshold: 0.3916), multiple critical samples

**Report Output**:
- 🔴 URGENT: Average Health Indicator exceeds failure threshold
- 🔴 CRITICAL: Minimum RUL is 18.2 hours. Component failure imminent
- Recommendation: Schedule immediate inspection within 24 hours
- Estimated cost of unplanned failure: €500,000 - €2,000,000
- Estimated cost of planned maintenance: €50,000 - €150,000

**Chatbot Interaction**:
```
User: "Why is my turbine classified as 'Degrading'?"
LLM: "Your turbine has been classified as 'Degrading' by the Gaussian Mixture Model 
based on 15% of samples falling into the intermediate health state. This indicates 
the autoencoder is detecting abnormal patterns in your SCADA data, particularly in 
gearbox temperature (28% above normal) and bearing oil pressure (12% below baseline). 
The GMM clustering suggests early-stage component wear that will progress to 'Critical' 
if not addressed. I recommend reviewing the Feature Importance tab to identify which 
specific components are contributing most to this classification."
```

### Use Case 2: Normal Operations
**Scenario**: Health Indicator is 0.28, low fault probability

**Report Output**:
- ✅ NORMAL OPERATION: Turbine appears healthy
- Recommendation: Continue routine monitoring schedule
- Average fault probability: 12%
- Safe RUL: 95% of samples show >7 days

**Chatbot Interaction**:
```
User: "What are the most important features for fault prediction?"
LLM: "Based on the trained Random Forest Classifier, the top 5 features are:
1. Gearbox Temperature (importance: 0.24) - Primary indicator of transmission health
2. Bearing Temperature Variability (0.19) - Early warning of bearing degradation
3. Thermal Stress Index (0.15) - Engineered feature combining multiple temps
4. Generator Speed Variability (0.11) - Indicates mechanical imbalances
5. Oil Pressure Ratio (0.09) - Lubrication system effectiveness

For your current analysis with healthy status, all these features are within normal 
ranges. The model's 75.2% accuracy makes these predictions reliable for planning 
maintenance windows."
```

---

## Configuration

### API Keys Required
Set environment variables for non-NTNU providers:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-key-here"
$env:GOOGLE_API_KEY = "your-google-key"
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"
```

### NTNU Configuration
Set via `.env` file or Streamlit secrets (see `.env.example`):
- API Key: Set `NTNU_API_KEY` in `.env` or Streamlit Cloud secrets
- API Base: `https://llm.hpc.ntnu.no/v1`

### Ollama Configuration
Requires local Ollama installation:
```bash
# Install Ollama
# Download models: ollama pull llama2
# API automatically uses: http://localhost:11434/v1
```

---

## Future Enhancements

### Potential Additions
1. **Multi-format reports**: Add PDF, HTML, Word export options
2. **Report customization**: Let users select which sections to include
3. **Email integration**: Automatically send reports to maintenance team
4. **Scheduled monitoring**: Generate reports automatically at intervals
5. **Chatbot memory**: Maintain conversation context across multiple questions
6. **Voice input**: Allow voice questions via speech-to-text
7. **Multi-language**: Support for Norwegian, German, Spanish
8. **Report templates**: Pre-configured formats for different audiences
9. **Chart embedding**: Include matplotlib figures in reports (requires PDF)
10. **Comparison mode**: Compare current analysis with historical reports

---

## Files Modified

### wind_turbine_gui.py
**Lines 810-835**: Added report download section after analysis results
- "📥 Generate Report" button
- `st.download_button` for Markdown file
- Timestamp-based filename generation

**Lines 837-1070**: New `generate_pdm_report()` function
- 230+ lines of report generation logic
- Risk-based recommendation engine
- Technical documentation formatting

**Lines 1072-1265**: New `pdm_chatbot_section()` function
- 190+ lines of chatbot UI and logic
- LLM provider/model selection
- Suggested questions as buttons
- Chat history management
- Integration with `query_local_llm()`

**Total Addition**: ~440 lines of new code

---

## Testing Checklist

- [ ] Report generation works with normal health status
- [ ] Report generation works with critical health status
- [ ] Download button creates valid Markdown file
- [ ] Chatbot displays suggested questions correctly
- [ ] Clicking suggested question populates input field
- [ ] Custom questions can be typed and submitted
- [ ] NTNU LLM integration works (default provider)
- [ ] OpenAI integration works (if API key provided)
- [ ] Chat history displays last 5 conversations
- [ ] Clear chat history button resets conversation
- [ ] LLM receives correct context about current analysis
- [ ] Responses are relevant and technically accurate
- [ ] Error handling works for API failures
- [ ] Multiple providers can be switched without issues
- [ ] Model dropdown updates correctly per provider

---

## Troubleshooting

### Report not downloading
- **Issue**: Button click doesn't trigger download
- **Fix**: Check browser popup blocker settings
- **Workaround**: Try different browser

### Chatbot not responding
- **Issue**: Spinner runs but no response appears
- **Fix**: Check API key configuration for selected provider
- **Check**: Network connectivity to LLM API endpoint
- **NTNU**: Verify VPN connection if required

### Slow LLM responses
- **Issue**: Takes >30 seconds to respond
- **Fix**: Reduce max_tokens parameter (currently 1000)
- **Alternative**: Switch to faster model (e.g., GPT-3.5-Turbo vs GPT-4)

### Inaccurate chatbot answers
- **Issue**: LLM provides generic or incorrect responses
- **Fix**: Context might not be passing correctly
- **Check**: Verify `pdm_results` dictionary has all required keys
- **Debug**: Print `context` string to verify metrics included

### Suggested questions not working
- **Issue**: Clicking button doesn't populate input
- **Fix**: Check `st.session_state.pdm_chat_input` is being set
- **Workaround**: Type question manually

---

## Maintenance Notes

### Updating Suggested Questions
Edit the `suggested_questions` list in `pdm_chatbot_section()` (line ~1125):
```python
suggested_questions = [
    'Your new question here',
    # ... existing questions
]
```

### Adding New LLM Providers
1. Add provider name to selectbox options (line ~1093)
2. Add model list for that provider (line ~1099)
3. Add API configuration in LLM query section (line ~1203)
4. Test with actual API credentials

### Customizing Report Format
Modify the `report` string in `generate_pdm_report()`:
- Add new sections after line ~920 (after recommendations)
- Adjust executive summary format (line ~865)
- Change risk threshold logic (line ~935)

### Changing System Message
Edit the `system_message` in chatbot section (line ~1176):
```python
system_message = """You are an expert..."""
```

---

*Document created: 2026-02-12*  
*Last updated: 2026-02-12*  
*Version: 1.0*
