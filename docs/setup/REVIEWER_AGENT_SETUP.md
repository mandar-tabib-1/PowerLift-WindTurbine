# Expert Reviewer Agent - Setup & Documentation

## Overview

The **Expert Reviewer Agent** is an AI-powered validation system that reviews outputs from all agents at critical checkpoints to catch errors, validate physical feasibility, and provide expert recommendations. It acts as a domain expert with deep knowledge of wind turbine operations, wake effects, and power optimization.

## Purpose

Many wind farm analysis workflows can make costly mistakes, such as:
- ❌ Running optimization when wind speed is below cut-in (turbine not operating)
- ❌ Accepting power predictions that exceed rated capacity
- ❌ Using turbine pairs that aren't aligned with wind direction
- ❌ Extrapolating ML models far beyond their training range

The Expert Reviewer Agent prevents these issues by:
- ✅ Validating outputs against physical constraints
- ✅ Checking consistency across agents
- ✅ Providing actionable recommendations
- ✅ Optionally blocking workflow on critical issues

## Architecture

### Three-Checkpoint Review System

**Checkpoint 1: After Agent 2 (Weather & Yaw Recommendations)**
- Reviews: Weather data, yaw angle suggestions, operating region classification
- Validates: Wind speed within operational range (3-25 m/s), yaw angles valid, turbine pair alignment
- Example Issues Caught:
  - "Wind speed 2.5 m/s is BELOW cut-in speed (3.0 m/s). Turbine should be parked."
  - "Turbine pair T3→T7 has 28° deviation from wind direction - not in wake"

**Checkpoint 2: After Agent 3 (Power & Optimization)**
- Reviews: Power predictions, optimization results
- Validates: Power within 0-5 MW range, optimization only runs when operating, gains realistic
- Example Issues Caught:
  - "❌ CRITICAL: Optimization ran with wind speed 2.1 m/s BELOW cut-in. Meaningless results."
  - "Power gain of 35% exceeds typical wake steering gains (5-15%). Check model validity."

**Checkpoint 3: After Agent 4 (Wake Flow)**
- Reviews: Wake flow predictions, velocity fields
- Validates: Wake deflection consistent with yaw angle, velocity statistics reasonable
- Example Issues Caught:
  - "Wake steering active with 12° yaw misalignment. Expected deflection: ~1.5 rotor diameters."

**Final Review: Before Report Generation**
- Synthesizes all checkpoint findings
- Provides overall assessment: APPROVED / WARNING / FAILED
- Lists key recommendations

### Reviewer Modes

**Advisory Mode (Default)**
- Provides feedback but **never halts workflow**
- All issues logged in final report with severity levels
- Best for: Exploration, testing, iterative development
- User sees: Info/warning messages, continues analysis

**Blocking Mode**
- **Halts workflow** on critical issues
- Requires user acknowledgment to continue
- Best for: Production, safety-critical operations
- User sees: Error message with findings, analysis stopped

## Configuration

### Basic Setup (config.yaml)

```yaml
reviewer:
  # Enable/disable globally
  enabled: true
  
  # Mode: "advisory" or "blocking"
  mode: "advisory"
  
  # Use same LLM as Agent 2B/2C
  use_same_provider_as_agent2b: true
  
  # Reviewer-specific parameters
  temperature: 0.3  # Balance determinism & insight
  max_tokens: 1500  # Detailed feedback
```

### Advanced: Dedicated Reviewer LLM

```yaml
reviewer:
  enabled: true
  mode: "blocking"
  use_same_provider_as_agent2b: false
  
  # Dedicated configuration
  provider: "openai"
  model: "gpt-4o"
  api_key: "your-api-key"
  api_base: "https://api.openai.com/v1"
  
  temperature: 0.3
  max_tokens: 1500
```

### GUI Controls

In [wind_turbine_gui.py](wind_turbine_gui.py), sidebar section **"Expert Reviewer Agent"**:

1. **Checkbox**: "🔍 Enable LLM Expert Reviewer" (default: ON)
2. **Radio Buttons**: 
   - "Advisory Only" (default)
   - "Blocking on Critical Issues"

## Usage

### Running with Orchestrator (Programmatic)

```python
from wind_turbine_orchestrator import WindTurbineOrchestrator
import yaml
import asyncio

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize orchestrator with reviewer
orchestrator = WindTurbineOrchestrator(
    config=config,
    reviewer_mode="advisory",  # or "blocking"
    reviewer_enabled=True
)

# Run analysis (async)
results = await orchestrator.run_analysis(
    location="Bessaker Wind Farm",
    agent2b_result=agent2b_output,
    turbine_pairs=turbine_pairs,
    optimization_result=opt_result
)

# Check review status
final_review = results["reviews"]["final_review"]
if final_review["overall_status"] == "FAILED":
    print(f"❌ Analysis failed: {final_review['status_message']}")
    for finding in final_review["key_findings"]:
        print(f"  - {finding['message']}")
```

### Running with GUI

1. Launch GUI: `streamlit run wind_turbine_gui.py`
2. Sidebar → **Expert Reviewer Agent**:
   - ✅ Enable checkbox
   - Select mode (Advisory / Blocking)
3. Configure agents and run analysis
4. View review in results:
   - **"Expert Review Assessment"** section
   - Overall status badge
   - Key findings with severity icons
   - Actionable recommendations

## Review Output Structure

### Checkpoint Review Format

```python
{
    "checkpoint": "checkpoint1_agent2",
    "timestamp": "2026-02-13T10:30:00",
    "severity": "warning",  # "info", "warning", or "critical"
    "findings": [
        {
            "type": "warning",
            "message": "Wind speed 3.8 m/s is marginal...",
            "rule": "marginal_wind_speed"
        }
    ],
    "llm_assessment": "Weather conditions are acceptable but...",
    "allow_continue": True,  # False if blocking mode + critical
    "reviewer_mode": "advisory",
    "summary": "Found 1 warning(s). LLM expert provided additional assessment."
}
```

### Final Review Format

```python
{
    "overall_status": "WARNING",  # "APPROVED", "WARNING", or "FAILED"
    "status_message": "Analysis completed with 2 warning(s).",
    "critical_count": 0,
    "warning_count": 2,
    "info_count": 3,
    "key_findings": [
        {
            "type": "warning",
            "message": "High uncertainty detected...",
            "rule": "high_uncertainty"
        }
    ],
    "recommendations": [
        "⚠️ Wait for wind speed to stabilize above 4 m/s",
        "✅ Turbine pair alignment verified for current wind direction"
    ],
    "checkpoint_count": 3,
    "timestamp": "2026-02-13T10:35:00"
}
```

## Severity Levels

### Critical (🔴)
- **Definition**: Issues that invalidate analysis or create safety concerns
- **Examples**:
  - Wind speed outside operational range + optimization attempted
  - Power predictions exceed rated capacity
  - Negative power values
- **Blocking Mode**: Halts workflow
- **Advisory Mode**: Logged, workflow continues

### Warning (⚠️)
- **Definition**: Issues requiring attention but not fatal
- **Examples**:
  - Marginal wind conditions (3-4 m/s)
  - Model extrapolation warnings
  - High prediction uncertainty
  - Turbine pair slight misalignment (15-20°)
- **Both Modes**: Logged, workflow continues

### Info (ℹ️)
- **Definition**: No issues detected, informational only
- **Examples**:
  - "All validation checks passed"
  - "Wake steering active with expected deflection"
- **Both Modes**: Logged for completeness

## Validation Rules

### Agent 2 Rules
| Rule ID | Check | Severity | Threshold |
|---------|-------|----------|-----------|
| `cut_in_violation` | Wind speed ≥ cut-in | Critical | 3.0 m/s |
| `cut_out_violation` | Wind speed ≤ cut-out | Critical | 25.0 m/s |
| `marginal_wind_speed` | Wind speed margin | Warning | 3.0-4.0 m/s |
| `yaw_range_check` | Yaw angle valid | Warning | 272-285° |
| `turbine_pair_alignment` | Wind direction match | Warning | ±20° tolerance |

### Agent 3 Rules
| Rule ID | Check | Severity | Threshold |
|---------|-------|----------|-----------|
| `negative_power` | Power ≥ 0 | Critical | 0 MW |
| `power_exceeds_rated` | Power ≤ rated * 1.1 | Critical | 5.5 MW |
| `optimization_below_cutoff` | Wind speed check | Critical | IF optimizing AND ws < 3.0 |
| `high_uncertainty` | Uncertainty ratio | Warning | σ > 0.5 * μ |
| `excessive_power_gain` | Gain plausibility | Warning | >20% |
| `yaw_extrapolation` | Misalignment range | Warning | >15° |

### Agent 4 Rules
| Rule ID | Check | Severity | Notes |
|---------|-------|----------|-------|
| `wake_generated` | Prediction exists | Info | Basic validation |
| `wake_steering_active` | Deflection expected | Info | If yaw ≠ 0 |

## Example Scenarios

### Scenario 1: Critical Error (Blocking Mode)

**Workflow**:
```
1. Agent 1 fetches weather: Wind speed = 2.3 m/s
2. Agent 2 suggests yaw angle (generic, not operating)
3. 🎓 CHECKPOINT 1: ❌ CRITICAL: Wind below cut-in
4. Workflow continues to Agent 3...
5. Agent 3 optimization runs (shouldn't happen!)
6. 🎓 CHECKPOINT 2: 🔴 CRITICAL: Optimization ran below cut-in!
7. 🛑 WORKFLOW HALTED (blocking mode)
```

**User sees**:
```
🔴 Review Failed: Optimization ran with wind speed 2.3 m/s BELOW cut-in. 
    Optimization is meaningless when turbine is not operating.

⚠️ Workflow halted by Expert Reviewer in blocking mode.
    Fix issues and try again.
```

### Scenario 2: Warnings (Advisory Mode)

**Workflow**:
```
1. Weather: Wind = 3.6 m/s (marginal)
2. Agent 2: Yaw angle suggested
3. 🎓 CHECKPOINT 1: ⚠️ WARNING: Marginal wind speed
4. Agent 3: Power prediction with high uncertainty
5. 🎓 CHECKPOINT 2: ⚠️ WARNING: High uncertainty detected
6. Agent 4: Wake flow prediction
7. 🎓 CHECKPOINT 3: ✅ INFO: Wake consistent
8. 📋 FINAL REVIEW: WARNING status, 2 warnings
```

**User sees**:
```
⚠️ Expert Review: Analysis completed with 2 warning(s).

Key Findings:
⚠️ Wind speed 3.6 m/s is marginal (just above cut-in). Power minimal.
⚠️ High uncertainty (±0.8 MW) relative to mean power (1.2 MW).

Recommendations:
⚠️ Wait for wind speed to stabilize above 4 m/s before optimization
✅ Predictions are physically plausible but use with caution
```

### Scenario 3: All Clear

**Workflow**:
```
1. Weather: Wind = 8.5 m/s (ideal)
2. Agent 2: Optimal yaw angle
3. 🎓 CHECKPOINT 1: ✅ All checks passed
4. Agent 3: Power = 2.8 MW, gain = 0.15 MW (5.3%)
5. 🎓 CHECKPOINT 2: ✅ Predictions plausible
6. Agent 4: Wake flow generated
7. 🎓 CHECKPOINT 3: ✅ Wake consistent
8. 📋 FINAL REVIEW: APPROVED
```

**User sees**:
```
✅ Expert Review: All validation checks passed.

Recommendations:
✅ Analysis looks good. No critical issues detected. Proceed with confidence.
```

## Technical Details

### LLM Prompts

Each checkpoint has a tailored prompt incorporating:
- **Turbine specifications** (NREL 5MW: 3-25 m/s, 5 MW rated, 126m rotor)
- **Current conditions** (weather, yaw angles, operating region)
- **Agent outputs** (predictions, recommendations)
- **Rule-based findings** (from pre-LLM validation)
- **Domain expertise** (wake effects, power curves, optimization theory)

**Temperature**: 0.3 (balance between deterministic and insightful)
**Max Tokens**: 1500 per checkpoint (detailed analysis)

### Fallback Strategy

If LLM fails (API error, timeout, etc.):
1. Log error message
2. Return rule-based validation only
3. Add finding: "LLM expert review unavailable"
4. **Never block** due to LLM failure (safety first)
5. Continue workflow

### Performance

- **Checkpoint duration**: ~2-5 seconds per checkpoint (LLM dependent)
- **Total overhead**: ~10-15 seconds for full 3-checkpoint review
- **Caching**: Not used (each analysis is unique)
- **Async**: Fully async implementation for non-blocking I/O

## Integration with Other Agents

The reviewer agent is **non-intrusive**:
- ✅ No modifications to existing agent logic
- ✅ Agents don't need to know reviewer exists
- ✅ Can be enabled/disabled without code changes
- ✅ Works with all agent combinations (2B/2C/2D, 3, 4, 5)

**Data flow**:
```
Agent 1 → Agent 2 → 🎓 Review → Agent 3 → 🎓 Review → Agent 4 → 🎓 Review → 📋 Final
                     ↓                      ↓                      ↓          ↓
                  results["reviews"]["checkpoint1_agent2"]         final_review
```

## Troubleshooting

### Issue: Reviewer not appearing in GUI

**Check**:
1. `reviewer_agent.py` file exists in project root
2. No import errors in terminal
3. Checkbox "Enable LLM Expert Reviewer" is checked
4. `config.yaml` has `reviewer:` section

### Issue: LLM review fails

**Common causes**:
- API key invalid/expired
- Network connectivity issues
- LLM service rate limiting
- Model not available

**Solution**: Reviewer falls back to rule-based validation only. Reviews still run but without LLM insights.

### Issue: Blocking mode too aggressive

**Solution**: Switch to advisory mode in sidebar or `config.yaml`:
```yaml
reviewer:
  mode: "advisory"
```

### Issue: Want to skip reviewer for testing

**Quick disable**:
- GUI: Uncheck "Enable LLM Expert Reviewer"
- Code: `reviewer_enabled=False`
- Config: `enabled: false`

## Best Practices

### Development Phase
- ✅ Use **advisory mode**
- ✅ Keep reviewer enabled to learn what errors are possible
- ✅ Review findings even if analysis succeeds
- ✅ Use findings to improve workflows

### Production Phase
- ✅ Use **blocking mode** for safety-critical operations
- ✅ Review configuration in `config.yaml` before deployment
- ✅ Monitor critical finding frequency
- ✅ Update validation rules based on experience

### Custom Deployments
- ✅ Adjust severity thresholds in `reviewer_agent.py`
- ✅ Add domain-specific rules
- ✅ Customize LLM prompts for your use case
- ✅ Test both modes thoroughly

## Future Enhancements

Potential improvements for future versions:
- [ ] Configurable validation rules via YAML
- [ ] Historical analysis of review patterns
- [ ] Machine learning on review outcomes
- [ ] Integration with monitoring/alerting systems
- [ ] Multi-language support for reviews
- [ ] Detailed VTK field validation for Agent 4
- [ ] Custom user-defined validation rules

## Support

For issues or questions:
1. Check terminal output for detailed error messages
2. Review `config.yaml` configuration
3. Test with advisory mode first
4. Examine `reviewer_agent.py` validation logic
5. Check LLM connectivity separately

## References

- **NREL 5MW Turbine**: [NREL Technical Report](https://www.nrel.gov/docs/fy09osti/38060.pdf)
- **Wake Steering**: Bastankhah, M., & Porté-Agel, F. (2016). Energies, 9(7), 506.
- **Agent 2B Documentation**: [AGENT_2B_SETUP.md](AGENT_2B_SETUP.md)
- **Optimization Methods**: [README_Optimization.md](README_Optimization.md)

---

**Version**: 1.0  
**Date**: February 13, 2026  
**Author**: Wind Turbine Multi-Agent System Team
