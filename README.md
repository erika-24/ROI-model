# BMW ROI Economic Model — Integration Guide

## Overview
This tool evaluates the **financial impact of robotics, AI, and automation technologies**.  
It accepts a **single structured row (JSON)**, uses an LLM to generate economic assumptions, and returns:

- ROI
- NPV
- Payback period
- Yearly cashflows
- Generated (and editable) model inputs

It is designed to integrate directly with the **technology matchmaking tool**.

## Running the Economic Model

To run the Streamlit app locally:

```bash
streamlit run bmw_economic_model.py
```

---

## Integration Pattern

### Recommended Architecture

```
Match-Making Tool → POST /roi/evaluate → ROI Model → JSON response → Match-Making UI
```

---

## Input Format (POST Request)

Send a **single row JSON object**:

```json
{
  "application": "MANUFACTURING",
  "country": "GERMANY",
  "row": {
    "technologyName": "OpenOOD v1.5",
    "technologyDescription": "Provides standardized evaluation for OOD detection under near/far shifts and nuisance factors.",
    "trlLevel": "4",
    "jobTitle": "Validation Engineers",
    "task": "Analyze validation test data to determine whether systems or processes have met validation criteria or identify root causes.",
    "capability": "detect input data outside training distributions",
    "idea": "Validation Engineers can integrate OpenOOD v1.5 into the post-deployment analysis pipeline...",
    "fitLevel": "HIGH",
    "fitRationale": "This directly addresses root cause analysis by flagging OOD production data."
  }
}
```
The first two rows are required context. The values shown are set as default.

## Example Output from Economic Model tool

```json
{
  "generated_inputs": {
    "scenario_overrides": {
      "application_type": "manufacturing",
      "solution_type": "humanoid",
      "region": "Germany",
      "deployment_units": 8,
      "current_labor_cost_per_year": 1800000,
      "robot_capex_per_unit": 95000,
      "integration_cost_initial": 650000,
      "annual_maintenance_pct": 0.12,
      "labor_reduction_pct": 0.35,
      "throughput_improvement_pct": 0.08,
      "discount_rate": 0.1
    }
  },
  "current_horizon_summary": {
    "Horizon_Years": 7,
    "NPV": 2450000,
    "ROI": 60.5,
    "Payback_Year": 2
  },
  "horizon_outputs": [
    {
      "Horizon_Years": 3,
      "NPV": 420000,
      "ROI": 14.2,
      "Payback_Year": 2
    },
    {
      "Horizon_Years": 5,
      "NPV": 1380000,
      "ROI": 38.7,
      "Payback_Year": 2
    },
    {
      "Horizon_Years": 7,
      "NPV": 2450000,
      "ROI": 60.5,
      "Payback_Year": 2
    }
  ]
}
```

---

## LLM Behavior
The LLM performs:
1. Idea breakdown  
2. Economic assumption generation  

Outputs include:
- labor assumptions
- cost structure
- performance improvements
- deployment ramp

---

## Design Notes

### Solution Selection
- The provided row is treated as the selected solution

### Editable Inputs
- All generated assumptions are editable in the UI

### Deterministic Recalculation
- After generation, all updates are deterministic

---

## Suggested Endpoint

```
POST /roi/evaluate
```

Request:
```json
{
  "row": { },
  "application": "MANUFACTURING",
  "country": "GERMANY"
}
```