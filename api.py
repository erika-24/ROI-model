from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

from model_bmw_matchmaking_apr20 import (
    Model,
    ApplicationType,
    Country,
    evaluate_matchmaking_row,
)

app = FastAPI(title="BMW ROI Economic Model API")


class ROIEvaluateRequest(BaseModel):
    row: Dict[str, Any]
    application: str = "MANUFACTURING"
    country: str = "GERMANY"
    regional_inputs: Optional[Dict[str, float]] = None


@app.post("/api/roi/evaluate")
def evaluate_roi(req: ROIEvaluateRequest):
    application = ApplicationType[req.application.upper()]
    country = Country[req.country.upper()]

    base_model = Model.default(application)

    regional_inputs = req.regional_inputs or {
        "wage_multiplier_vs_baseline": 1.25,
        "energy_cost_per_kwh": 0.25,
        "maintenance_multiplier": 1.10,
    }

    return evaluate_matchmaking_row(
        row_dict=req.row,
        base_model=base_model,
        country=country,
        regional_inputs=regional_inputs,
    )