# BMW ROI Model Integration Guide

This guide explains:
1. How to run the demo locally using the simulator
2. How to replace the simulator with your matchmaking tool
3. How the integration works under the hood

---

## 🧠 High-Level Architecture

Matchmaking Tool → get_roi_for_selected_row(...) → shared_roi_state.json → Streamlit ROI Model

---

## 🚀 Running the Demo Locally

### Step 1 — Start the ROI Model (Streamlit)

```bash
streamlit run model_bmw_matchmaking_final.py --server.port 8501
```

### Step 2 — Start the Matchmaking Simulator

```bash
streamlit run matchmaking_simulator.py --server.port 8502
```

### Step 3 — Open Simulator

http://localhost:8502

---

## 🔌 Integration

Call this function from your tool:

```python
get_roi_for_selected_row(row_dict, row_id, application, country)
```

---

## 📤 Output Format

{
  "status": "success",
  "technologyName": "...",
  "jobTitle": "...",
  "row_id": "...",
  "streamlit_url": "...",
  "horizon_outputs": [...]
}

---

## 🎯 Demo Scenario

Full Factory Digital Twin Coverage (NVIDIA Omniverse)

---

## Notes

- LLM is called once per row
- Editing assumptions does NOT call LLM
- Shared state stored in shared_roi_state.json
