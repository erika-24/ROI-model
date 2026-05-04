# roi_shared_state.py

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


STATE_PATH = Path("shared_roi_state.json")

def load_roi_state(path: Path = STATE_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": {}}

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return {"rows": {}}

    try:
        state = json.loads(text)
    except json.JSONDecodeError:
        return {"rows": {}}

    if not isinstance(state, dict):
        return {"rows": {}}

    if "rows" not in state:
        state["rows"] = {}

    return state


def save_roi_state(state: Dict[str, Any], path: Path = STATE_PATH) -> None:
    state["updated_at"] = datetime.now().isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def save_row_state(
    row_id: str,
    row: Dict[str, Any],
    context: Dict[str, Any],
    scenario_overrides: Optional[Dict[str, Any]] = None,
    horizon_outputs: Optional[list] = None,
    status: str = "initialized",
    streamlit_url: Optional[str] = None,
) -> Dict[str, Any]:
    state = load_roi_state()

    state["rows"][row_id] = {
        "row_id": row_id,
        "status": status,
        "row": row,
        "context": context,
        "scenario_overrides": scenario_overrides or {},
        "horizon_outputs": horizon_outputs or [],
        "streamlit_url": streamlit_url,
        "updated_at": datetime.now().isoformat(),
    }

    save_roi_state(state)
    return state["rows"][row_id]


def get_row_state(row_id: str) -> Optional[Dict[str, Any]]:
    state = load_roi_state()
    return state.get("rows", {}).get(row_id)


def update_row_state(row_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    state = load_roi_state()

    if row_id not in state["rows"]:
        raise KeyError(f"No row state found for row_id={row_id}")

    state["rows"][row_id].update(updates)
    state["rows"][row_id]["updated_at"] = datetime.now().isoformat()

    save_roi_state(state)
    return state["rows"][row_id]


def get_assumption_notes(row_id: str) -> list[str]:
    row_state = get_row_state(row_id)

    if not row_state:
        return []

    generated_inputs = row_state.get("generated_inputs", {})
    notes = generated_inputs.get("assumption_notes", [])

    if isinstance(notes, list):
        return [str(note) for note in notes if note]

    return []