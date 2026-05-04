# matchmaking_simulator.py

import hashlib
import pandas as pd
import streamlit as st
import json
from pathlib import Path

from bmw_economic_model import get_roi_for_selected_row, get_row_state


def load_jobs(path: str = "sample_jobs.json") -> list[dict]:
    jobs_path = Path(path)

    if not jobs_path.exists():
        raise FileNotFoundError(f"Could not find {jobs_path.resolve()}")

    text = jobs_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"{jobs_path.resolve()} is empty")

    data = json.loads(text)

    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]

    if not isinstance(data, list):
        raise ValueError("Expected sample_jobs.json to contain a list of row objects")

    return data


def make_row_id(row: dict, index: int) -> str:
    raw = "|".join(
        [
            str(index),
            row.get("technologyName", ""),
            row.get("jobTitle", ""),
            row.get("capability", ""),
            row.get("idea", ""),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def format_roi(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:,.1f}%"


def format_npv(value):
    if value is None or pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def main():
    st.set_page_config(page_title="Matchmaking Tool Simulator", layout="wide")
    st.title("Matchmaking Tool Simulator")
    st.caption("This simulates a row-level Get ROI button from the technology matchmaking table.")

    rows = load_jobs("jobs.json")

    if "match_rows" not in st.session_state:
        st.session_state["match_rows"] = rows

    if "roi_results_by_row_id" not in st.session_state:
        st.session_state["roi_results_by_row_id"] = {}

    app = st.sidebar.selectbox("Application", ["MANUFACTURING", "LOGISTICS", "PRODUCT", "OFFICE", "SALES"])
    country = st.sidebar.selectbox("Country", ["GERMANY", "USA", "CHINA", "MEXICO"])

    st.markdown("### Matchmaking Results")

    header = st.columns([1.2, 1.4, 1.6, 2.2, 0.7, 1.0, 1.0, 1.0, 1.0])
    header[0].markdown("**Technology**")
    header[1].markdown("**Job**")
    header[2].markdown("**Capability**")
    header[3].markdown("**Idea**")
    header[4].markdown("**Fit**")
    header[5].markdown("**3Y ROI**")
    header[6].markdown("**3Y NPV**")
    header[7].markdown("**7Y ROI**")
    header[8].markdown("**Action**")

    for idx, row in enumerate(st.session_state["match_rows"]):
        row_id = make_row_id(row, idx)
        row_state = get_row_state(row_id)
        result = st.session_state["roi_results_by_row_id"].get(row_id)

        horizon_outputs = result.get("horizon_outputs", []) if result else []
        h3 = next((h for h in horizon_outputs if h["Horizon_Years"] == 3), {})
        h7 = next((h for h in horizon_outputs if h["Horizon_Years"] == 7), {})

        cols = st.columns([1.2, 1.4, 1.6, 2.2, 0.7, 1.0, 1.0, 1.0, 1.0])

        cols[0].write(row["technologyName"])
        cols[1].write(row["jobTitle"])
        cols[2].write(row["capability"])
        cols[3].write(row["idea"][:140] + ("..." if len(row["idea"]) > 140 else ""))
        cols[4].write(row["fitLevel"])
        cols[5].write(format_roi(h3.get("ROI")) if result else "—")
        cols[6].write(format_npv(h3.get("NPV")) if result else "—")
        cols[7].write(format_roi(h7.get("ROI")) if result else "—")

        if cols[8].button("Get ROI", key=f"get_roi_{row_id}"):
            with st.spinner("Running ROI model..."):
                result = get_roi_for_selected_row(
                    row_dict=row,
                    row_id=row_id,
                    application=app,
                    country=country,
                    streamlit_base_url="http://localhost:8501",
                )

                st.session_state["roi_results_by_row_id"][row_id] = result
                st.rerun()

        if result:
            cols[8].link_button("Open Model", result["streamlit_url"])

    if st.button("Refresh ROI results"):
        for idx, row in enumerate(st.session_state["match_rows"]):
            row_id = make_row_id(row, idx)
            row_state = get_row_state(row_id)

            if row_state and row_state.get("horizon_outputs"):
                st.session_state["roi_results_by_row_id"][row_id] = {
                    "status": row_state.get("status", "success"),
                    "technologyName": row.get("technologyName"),
                    "jobTitle": row.get("jobTitle"),
                    "row_id": row_id,
                    "streamlit_url": row_state.get("streamlit_url", f"http://localhost:8501/?row_id={row_id}"),
                    "horizon_outputs": row_state.get("horizon_outputs", []),
                }

        st.rerun()


if __name__ == "__main__":
    main()