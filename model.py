"""
BMW Robotics ROI & Cost Degression Model
--------------------------------------
A generalizable Python program to evaluate ROI for robotics adoption across
five application areas: Logistics, Product, Manufacturing, Office, Sales.

Key features
- Parameterized TCO vs. Status Quo
- Cost degression via Wright's Law (learning curves)
- Scenario engine (sliders/overrides)
- Sensitivity analysis (tornado & grid)
- Payback, NPV, IRR, ROI metrics
- Per-year cash flow breakdown
- Domain templates with realistic defaults for each application area

Dependencies: numpy, pandas, matplotlib (optional: streamlit, altair)
Tested with Python 3.10+

Usage
-----
As a script (CLI examples):
  python model.py --app manufacturing --years 7 --plot
  python model.py --app logistics --override labor.wage_per_hour 38 --plot
  python model.py --app sales --revenue-uplift --plot

Streamlit dashboard:
  streamlit run model.py
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -----------------------------
# Domain & Scenario Structures
# -----------------------------


class ApplicationType(Enum):
    LOGISTICS = auto()
    PRODUCT = auto()
    MANUFACTURING = auto()
    OFFICE = auto()
    SALES = auto()


@dataclass
class EconomicAssumptions:
    discount_rate: float = 0.10  # WACC / hurdle rate (10%)
    horizon_years: int = 7  # planning horizon
    inflation_labor: float = 0.03  # annual labor inflation
    inflation_maintenance: float = 0.02  # annual maintenance inflation
    inflation_energy: float = 0.03  # annual energy inflation
    carbon_price_per_ton: float = 0.0  # optional monetization of CO2


@dataclass
class DemandParams:
    base_units_per_year: int = 1_000_000  # tasks/units demanded in Y1
    annual_growth: float = 0.00  # demand growth (exogenous)
    revenue_per_unit: float = 0.0  # optional revenue monetization (gross margin per unit)


@dataclass
class LaborParams:
    wage_per_hour: float = 35.0  # fully-loaded wage
    manual_hours_per_unit: float = 0.05  # baseline manual time (hrs/unit)
    automated_supervision_hours_per_unit: float = 0.005  # human-in-loop
    safety_incident_cost_delta_per_year: float = -5_000  # negative = savings


@dataclass
class CostParams:
    capex_per_unit: float = 250_000  # robot cell cost
    install_commission_per_unit: float = 25_000
    integration_per_unit: float = 50_000
    maintenance_per_unit_per_year: float = 8_000
    energy_kwh_per_unit: float = 0.1
    energy_cost_per_kwh: float = 0.12
    consumables_per_unit: float = 0.00


@dataclass
class PerformanceParams:
    cycle_time_seconds: float = 5.0  # automated cycle
    uptime_pct: float = 0.90
    manual_cycle_time_seconds: float = 8.0
    defect_rate_change_pct: float = -0.20  # -20% defects vs. manual
    scrap_cost_per_unit: float = 0.50
    rework_cost_per_unit: float = 0.25
    throughput_change_pct: float = 0.10  # +10% capacity (optional)


@dataclass
class DeploymentPlan:
    units_initial: int = 2
    ramp: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 16, 20, 24])
    utilization_pct: float = 0.85
    shifts_per_day: float = 2.0


@dataclass
class LearningCurve:
    learning_rate: float = 0.15  # 15% cost reduction per doubling
    apply_to: Tuple[str, ...] = ("capex_per_unit",)


@dataclass
class Scenario:
    application: ApplicationType
    economics: EconomicAssumptions
    demand: DemandParams
    labor: LaborParams
    costs: CostParams
    perf: PerformanceParams
    deploy: DeploymentPlan
    learning: LearningCurve
    include_revenue_uplift: bool = False  # toggles monetization of throughput


# -----------------------------
# Utility & Financial Math
# -----------------------------


def wrights_law(base_cost: float, cumulative_units: int, learning_rate: float) -> float:
    """Apply Wright's Law: cost = base * (cum_units)^(log2(1 - lr))
    learning_rate is e.g., 0.15 (15% per doubling). For cum_units<=1, return base_cost.
    """
    if cumulative_units <= 1 or learning_rate <= 0:
        return base_cost
    b = math.log2(1 - learning_rate)  # negative number
    return base_cost * (cumulative_units ** b)


def npv(cashflows: List[float], rate: float) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def irr(cashflows: List[float], guess: float = 0.1) -> Optional[float]:
    # Using numpy's IRR with fallback
    try:
        val = np.irr(cashflows)
        if val is None or np.isnan(val):
            return None
        return float(val)
    except Exception:
        # Simple Newton fallback
        r = guess
        for _ in range(100):
            # f(r) = NPV
            f = sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))
            df = sum(-t * cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cashflows))
            if abs(df) < 1e-9:
                break
            r_new = r - f / df
            if abs(r_new - r) < 1e-7:
                return r_new
            r = r_new
        return None


# -----------------------------
# Core Model
# -----------------------------


@dataclass
class Results:
    yearly: pd.DataFrame
    summary_metrics: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return self.summary_metrics


class Model:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    @staticmethod
    def default(app: ApplicationType) -> "Model":
        """
        Build a Scenario with realistic example assumptions per application area.
        The structure (inputs & equations) is identical across all areas; only
        default values change to reflect operational reality.
        """
        economics = EconomicAssumptions()

        # We can vary learning curve slightly by domain, but keep structure identical.
        if app == ApplicationType.LOGISTICS:
            # Example: automated palletizing / case handling in a distribution center
            demand = DemandParams(
                base_units_per_year=4_000_000,  # cases / picks
                annual_growth=0.03,
                revenue_per_unit=0.0,
            )
            labor = LaborParams(
                wage_per_hour=30.0,  # warehouse associates
                manual_hours_per_unit=0.02,  # ~1.2 min per case
                automated_supervision_hours_per_unit=0.003,  # ~11 sec supervision
                safety_incident_cost_delta_per_year=-15_000,  # safety improvement
            )
            costs = CostParams(
                capex_per_unit=180_000,  # standardized palletizing cell
                install_commission_per_unit=20_000,
                integration_per_unit=40_000,
                maintenance_per_unit_per_year=7_000,
                energy_kwh_per_unit=0.03,
                energy_cost_per_kwh=0.11,
                consumables_per_unit=0.01,
            )
            perf = PerformanceParams(
                cycle_time_seconds=3.0,
                uptime_pct=0.93,
                manual_cycle_time_seconds=7.0,
                defect_rate_change_pct=-0.05,  # fewer mispicks / damage
                scrap_cost_per_unit=0.05,
                rework_cost_per_unit=0.02,
                throughput_change_pct=0.15,  # 15% more throughput
            )
            deploy = DeploymentPlan(
                units_initial=4,
                ramp=[4, 8, 12, 16, 20, 24, 28],
                utilization_pct=0.90,
                shifts_per_day=2.5,
            )
            learning = LearningCurve(learning_rate=0.12)

        elif app == ApplicationType.PRODUCT:
            # Example: final assembly / trim cell on a vehicle line
            demand = DemandParams(
                base_units_per_year=300_000,  # vehicles or assemblies
                annual_growth=0.02,
                revenue_per_unit=0.0,
            )
            labor = LaborParams(
                wage_per_hour=38.0,  # skilled production operators
                manual_hours_per_unit=0.08,  # ~4.8 min per unit
                automated_supervision_hours_per_unit=0.010,  # 0.6 min supervision
                safety_incident_cost_delta_per_year=-20_000,
            )
            costs = CostParams(
                capex_per_unit=260_000,
                install_commission_per_unit=30_000,
                integration_per_unit=80_000,  # higher integration for product-specific tooling
                maintenance_per_unit_per_year=9_000,
                energy_kwh_per_unit=0.18,
                energy_cost_per_kwh=0.12,
                consumables_per_unit=0.05,
            )
            perf = PerformanceParams(
                cycle_time_seconds=6.0,
                uptime_pct=0.91,
                manual_cycle_time_seconds=11.0,
                defect_rate_change_pct=-0.30,  # big quality gain
                scrap_cost_per_unit=1.50,
                rework_cost_per_unit=0.75,
                throughput_change_pct=0.10,
            )
            deploy = DeploymentPlan(
                units_initial=3,
                ramp=[3, 6, 9, 12, 16, 20, 24],
                utilization_pct=0.88,
                shifts_per_day=2.0,
            )
            learning = LearningCurve(learning_rate=0.15)

        elif app == ApplicationType.MANUFACTURING:
            # Example: machine tending / welding cells in a body or powertrain plant
            demand = DemandParams(
                base_units_per_year=1_200_000,  # parts or weld operations
                annual_growth=0.015,
                revenue_per_unit=0.0,
            )
            labor = LaborParams(
                wage_per_hour=42.0,  # highly skilled trades / operators
                manual_hours_per_unit=0.07,  # ~4.2 min per unit
                automated_supervision_hours_per_unit=0.006,
                safety_incident_cost_delta_per_year=-25_000,
            )
            costs = CostParams(
                capex_per_unit=320_000,
                install_commission_per_unit=35_000,
                integration_per_unit=100_000,  # custom fixturing, safety, PLC integration
                maintenance_per_unit_per_year=12_000,
                energy_kwh_per_unit=0.22,
                energy_cost_per_kwh=0.12,
                consumables_per_unit=0.10,
            )
            perf = PerformanceParams(
                cycle_time_seconds=5.0,
                uptime_pct=0.92,
                manual_cycle_time_seconds=10.0,
                defect_rate_change_pct=-0.22,
                scrap_cost_per_unit=1.00,
                rework_cost_per_unit=0.50,
                throughput_change_pct=0.12,
            )
            deploy = DeploymentPlan(
                units_initial=4,
                ramp=[4, 8, 12, 16, 20, 24, 28],
                utilization_pct=0.90,
                shifts_per_day=2.5,
            )
            learning = LearningCurve(learning_rate=0.14)

        elif app == ApplicationType.OFFICE:
            # Example: back-office document / transaction processing automation
            demand = DemandParams(
                base_units_per_year=10_000_000,  # documents / transactions
                annual_growth=0.02,
                revenue_per_unit=0.0,
            )
            labor = LaborParams(
                wage_per_hour=27.0,
                manual_hours_per_unit=0.004,  # ~14.4 seconds per item
                automated_supervision_hours_per_unit=0.0007,  # ~2.5 sec
                safety_incident_cost_delta_per_year=-5_000,
            )
            costs = CostParams(
                capex_per_unit=90_000,  # servers + licenses + small robots
                install_commission_per_unit=10_000,
                integration_per_unit=40_000,
                maintenance_per_unit_per_year=4_000,
                energy_kwh_per_unit=0.001,
                energy_cost_per_kwh=0.12,
                consumables_per_unit=0.0,
            )
            perf = PerformanceParams(
                cycle_time_seconds=0.3,
                uptime_pct=0.97,
                manual_cycle_time_seconds=2.0,
                defect_rate_change_pct=-0.20,  # fewer errors
                scrap_cost_per_unit=0.01,
                rework_cost_per_unit=0.02,
                throughput_change_pct=0.08,
            )
            deploy = DeploymentPlan(
                units_initial=2,
                ramp=[2, 4, 6, 8, 10, 12, 14],
                utilization_pct=0.95,
                shifts_per_day=1.5,
            )
            learning = LearningCurve(learning_rate=0.20)

        else:  # SALES
            # Example: showroom / digital sales support, kiosks, lead-handling robots
            demand = DemandParams(
                base_units_per_year=150_000,  # customer interactions or leads
                annual_growth=0.03,
                revenue_per_unit=1_500.0,  # average gross margin per converted sale (if uplift monetized)
            )
            labor = LaborParams(
                wage_per_hour=32.0,
                manual_hours_per_unit=0.10,  # ~6 minutes per customer interaction
                automated_supervision_hours_per_unit=0.015,  # ~0.9 min supervision
                safety_incident_cost_delta_per_year=-2_500,
            )
            costs = CostParams(
                capex_per_unit=140_000,
                install_commission_per_unit=15_000,
                integration_per_unit=30_000,
                maintenance_per_unit_per_year=6_000,
                energy_kwh_per_unit=0.01,
                energy_cost_per_kwh=0.12,
                consumables_per_unit=0.02,
            )
            perf = PerformanceParams(
                cycle_time_seconds=2.5,
                uptime_pct=0.94,
                manual_cycle_time_seconds=5.0,
                defect_rate_change_pct=-0.05,  # fewer lost leads / misquotes
                scrap_cost_per_unit=0.00,
                rework_cost_per_unit=0.30,  # re-contact attempts etc.
                throughput_change_pct=0.12,  # can handle more customers/leads
            )
            deploy = DeploymentPlan(
                units_initial=2,
                ramp=[2, 4, 6, 8, 10, 12, 14],
                utilization_pct=0.85,
                shifts_per_day=1.5,
            )
            learning = LearningCurve(learning_rate=0.18)

        scenario = Scenario(
            application=app,
            economics=economics,
            demand=demand,
            labor=labor,
            costs=costs,
            perf=perf,
            deploy=deploy,
            learning=learning,
            include_revenue_uplift=False,  # can be toggled in UI/CLI
        )
        return Model(scenario)

    # -------------------------
    # Main execution
    # -------------------------

    def run(self) -> Results:
        sc = self.scenario
        years = sc.economics.horizon_years
        # Track cumulative deployments for learning
        cum_units = 0

        rows = []
        cashflows_automation: List[float] = []
        cashflows_manual: List[float] = []
        cashflows_net: List[float] = []

        # Precompute demand per year
        demand_by_year = [
            sc.demand.base_units_per_year * ((1 + sc.demand.annual_growth) ** t)
            for t in range(years)
        ]

        # Labor inflation schedule
        wage_by_year = [
            sc.labor.wage_per_hour * ((1 + sc.economics.inflation_labor) ** t)
            for t in range(years)
        ]

        # Maintenance & energy inflation
        maint_base = sc.costs.maintenance_per_unit_per_year
        energy_cost_base = sc.costs.energy_cost_per_kwh
        maint_by_year = [
            maint_base * ((1 + sc.economics.inflation_maintenance) ** t)
            for t in range(years)
        ]
        energy_cost_by_year = [
            energy_cost_base * ((1 + sc.economics.inflation_energy) ** t)
            for t in range(years)
        ]

        # Deployment schedule
        ramp = sc.deploy.ramp
        if len(ramp) < years:
            # extend flat if shorter
            ramp = ramp + [ramp[-1]] * (years - len(ramp))

        for t in range(years):
            year = t + 1
            demand_units = demand_by_year[t]

            # Status Quo (Manual) costs
            manual_hours = sc.labor.manual_hours_per_unit * demand_units
            manual_labor_cost = manual_hours * wage_by_year[t]
            manual_defect_cost = (
                sc.perf.scrap_cost_per_unit + sc.perf.rework_cost_per_unit
            ) * demand_units
            # Energy/consumables baseline assumed negligible for manual by default (override if desired)
            manual_energy_cost = 0.0
            manual_consumables = 0.0
            manual_total = (
                manual_labor_cost
                + manual_defect_cost
                + manual_energy_cost
                + manual_consumables
            )

            # Robot deployment this year
            deployed_units = ramp[t]
            add_units = max(deployed_units - cum_units, 0)
            # Learning-adjusted CAPEX for the *new* units only
            capex_unit = wrights_law(
                sc.costs.capex_per_unit, max(cum_units, 1), sc.learning.learning_rate
            )
            capex_new = add_units * (
                capex_unit
                + sc.costs.install_commission_per_unit
                + sc.costs.integration_per_unit
            )
            cum_units = deployed_units

            # Automated operational costs
            # Supervision labor
            auto_hours = sc.labor.automated_supervision_hours_per_unit * demand_units
            auto_labor_cost = auto_hours * wage_by_year[t]
            # Maintenance is per deployed cell per year
            auto_maintenance = cum_units * maint_by_year[t]
            # Energy & consumables per unit of demand processed
            auto_energy = (
                sc.costs.energy_kwh_per_unit
                * energy_cost_by_year[t]
                * demand_units
            )
            auto_consumables = sc.costs.consumables_per_unit * demand_units

            # Quality deltas
            defect_delta_factor = 1 + sc.perf.defect_rate_change_pct
            auto_defect_cost = (
                sc.perf.scrap_cost_per_unit + sc.perf.rework_cost_per_unit
            ) * demand_units * max(0.0, defect_delta_factor)

            # Safety deltas
            safety_delta = sc.labor.safety_incident_cost_delta_per_year  # negative -> savings

            # Revenue uplift if enabled (throughput/capacity increase monetized)
            revenue_uplift = 0.0
            if (
                sc.include_revenue_uplift
                and sc.demand.revenue_per_unit > 0
                and sc.perf.throughput_change_pct > 0
            ):
                extra_units = demand_units * sc.perf.throughput_change_pct
                revenue_uplift = extra_units * sc.demand.revenue_per_unit

            auto_total_opex = (
                auto_labor_cost
                + auto_maintenance
                + auto_energy
                + auto_consumables
                + auto_defect_cost
                + safety_delta
            )
            auto_total_cash = capex_new + auto_total_opex

            # Year-over-year savings (manual minus automated opex), minus capex where applicable
            yearly_net_savings = (
                manual_total - auto_total_opex
            ) + revenue_uplift - capex_new

            rows.append(
                {
                    "Year": year,
                    "DemandUnits": demand_units,
                    "DeployedCells": cum_units,
                    "NewCells": add_units,
                    "CAPEX_new": capex_new,
                    "Manual_Labor": manual_labor_cost,
                    "Manual_Defects": manual_defect_cost,
                    "Manual_Total": manual_total,
                    "Auto_Labor": auto_labor_cost,
                    "Auto_Maint": auto_maintenance,
                    "Auto_Energy": auto_energy,
                    "Auto_Consum": auto_consumables,
                    "Auto_Defects": auto_defect_cost,
                    "Safety_Delta": safety_delta,
                    "Revenue_Uplift": revenue_uplift,
                    "Auto_OPEX": auto_total_opex,
                    "Auto_TotalCash": auto_total_cash,
                    "Net_Savings": yearly_net_savings,
                }
            )

            cashflows_automation.append(-auto_total_cash)
            cashflows_manual.append(-manual_total)
            cashflows_net.append(yearly_net_savings)

        df = pd.DataFrame(rows)

        # Financial metrics
        rate = sc.economics.discount_rate
        npv_net = npv(cashflows_net, rate)
        irr_net = irr(cashflows_net)
        cumulative = df["Net_Savings"].cumsum()
        payback_year = next(
            (int(y) for y, val in zip(df["Year"], cumulative) if val >= 0), None
        )

        total_capex = df["CAPEX_new"].sum()
        total_manual = df["Manual_Total"].sum()
        total_auto_opex = df["Auto_OPEX"].sum()
        total_savings = total_manual - total_auto_opex
        simple_roi = (
            (total_savings - total_capex) / total_capex if total_capex > 0 else np.nan
        )

        summary = {
            "NPV": npv_net,
            "IRR": irr_net,
            "Payback_Year": payback_year,
            "Total_CAPEX": total_capex,
            "Total_Manual_OPEX": total_manual,
            "Total_Auto_OPEX": total_auto_opex,
            "Total_Savings_vs_Manual": total_savings,
            "Simple_ROI": simple_roi,
        }

        return Results(yearly=df, summary_metrics=summary)

    # -------------------------
    # Sensitivity & Utilities
    # -------------------------

    def sensitivity_grid(self, param_path: str, values: List[float]) -> pd.DataFrame:
        """Vary a single scalar parameter and compute NPV/IRR/Payback."""
        out = []
        baseline_val = self._get_param(param_path)
        for v in values:
            self._set_param(param_path, v)
            res = self.run().summary()
            out.append({"value": v, **res})
        # restore baseline
        self._set_param(param_path, baseline_val)
        return pd.DataFrame(out)

    def tornado(self, deltas: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Return a tornado-style table for NPV by parameter low/high overrides.
        deltas: {"param.path": (low, high)}
        """
        baseline = self.run().summary()["NPV"]
        rows = []
        for p, (low, high) in deltas.items():
            orig_val = self._get_param(p)

            self._set_param(p, low)
            npv_low = self.run().summary()["NPV"]

            self._set_param(p, high)
            npv_high = self.run().summary()["NPV"]

            self._set_param(p, orig_val)  # restore
            rows.append(
                {
                    "param": p,
                    "NPV_baseline": baseline,
                    "NPV_low": npv_low,
                    "NPV_high": npv_high,
                    "Delta": npv_high - npv_low,
                }
            )
        df = pd.DataFrame(rows).sort_values("Delta", ascending=False)
        return df

    def _set_param(self, path: str, value: Any):
        node, attr = self._resolve(path)
        setattr(node, attr, value)

    def _get_param(self, path: str) -> Any:
        node, attr = self._resolve(path)
        return getattr(node, attr)

    def _resolve(self, path: str):
        # path form: "labor.wage_per_hour" or "costs.capex_per_unit"
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Param path must be 'section.field', got: {path}")
        section, field_name = parts
        node = getattr(self.scenario, section)
        if not hasattr(node, field_name):
            raise AttributeError(f"No field '{field_name}' in section '{section}'")
        return node, field_name


# -----------------------------
# Plotting helpers (optional)
# -----------------------------


def plot_cashflows(results: Results, title: str = "Cashflows & Savings"):
    if not _HAS_PLT:
        return
    df = results.yearly
    plt.figure()
    plt.plot(df["Year"], df["Manual_Total"], label="Manual OPEX")
    plt.plot(df["Year"], df["Auto_OPEX"], label="Automated OPEX")
    plt.bar(df["Year"], df["CAPEX_new"], alpha=0.3, label="New CAPEX")
    plt.plot(
        df["Year"],
        df["Net_Savings"].cumsum(),
        label="Cumulative Net Savings",
    )
    plt.xlabel("Year")
    plt.ylabel("Cost / Savings ($)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI & (optional) Streamlit UI
# -----------------------------


def build_model_from_args(args: argparse.Namespace) -> Model:
    app_map = {
        "logistics": ApplicationType.LOGISTICS,
        "product": ApplicationType.PRODUCT,
        "manufacturing": ApplicationType.MANUFACTURING,
        "office": ApplicationType.OFFICE,
        "sales": ApplicationType.SALES,
    }
    model = Model.default(app_map[args.app])

    # Overrides (simple dotted path -> value)
    for ov in args.override:
        path, sval = ov
        # try int/float casting
        try:
            if "." in sval:
                val: Any = float(sval)
            else:
                val = int(sval)
        except Exception:
            # bool or leave as string
            if isinstance(sval, str) and sval.lower() in ("true", "false"):
                val = sval.lower() == "true"
            else:
                val = sval
        model._set_param(path, val)

    # Horizon override
    if args.years:
        model.scenario.economics.horizon_years = int(args.years)

    # Revenue uplift toggle
    if args.revenue_uplift:
        model.scenario.include_revenue_uplift = True

    return model


def run_cli(args: argparse.Namespace):
    model = build_model_from_args(args)
    res = model.run()

    # Print summary
    print("\n=== SUMMARY ===")
    for k, v in res.summary().items():
        if isinstance(v, float):
            print(f"{k:>24}: {v:,.2f}")
        else:
            print(f"{k:>24}: {v}")

    # CSV export
    if args.out:
        out_path = args.out
        res.yearly.to_csv(out_path, index=False)
        print(f"\nSaved yearly breakdown to: {out_path}")

    # Plot
    if args.plot:
        plot_cashflows(res, title=f"{model.scenario.application.name.title()} — Cashflows")


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="BMW Robotics ROI", layout="wide")
    st.title("BMW Robotics ROI & Cost Degression Model")

    # Sidebar selectors
    app_choice = st.sidebar.selectbox(
        "Application", [e.name.title() for e in ApplicationType]
    )
    model = Model.default(ApplicationType[app_choice.upper()])

    st.sidebar.header("Horizon & Finance")
    econ = model.scenario.economics
    econ.horizon_years = st.sidebar.slider(
        "Years", 3, 15, econ.horizon_years, help="Planning horizon for ROI evaluation."
    )
    econ.discount_rate = st.sidebar.slider(
        "Discount Rate",
        0.02,
        0.2,
        float(econ.discount_rate),
        0.01,
        help="BMW hurdle rate / cost of capital.",
    )

    st.sidebar.header("Learning Curve")
    model.scenario.learning.learning_rate = st.sidebar.slider(
        "Learning Rate (per doubling)",
        0.0,
        0.3,
        float(model.scenario.learning.learning_rate),
        0.01,
        help="Cost reduction percentage each time cumulative deployments double.",
    )

    st.sidebar.header("Labor")
    lab = model.scenario.labor
    lab.wage_per_hour = st.sidebar.number_input(
        "Wage $/hr",
        value=float(lab.wage_per_hour),
        step=1.0,
        help="Fully-loaded labor cost per hour.",
    )

    st.sidebar.header("Costs")
    c = model.scenario.costs
    c.capex_per_unit = st.sidebar.number_input(
        "CAPEX per Cell",
        value=float(c.capex_per_unit),
        step=5_000.0,
        help="Hardware cost per robotic cell.",
    )
    c.integration_per_unit = st.sidebar.number_input(
        "Integration per Cell",
        value=float(c.integration_per_unit),
        step=5_000.0,
        help="Engineering, controls, safety integration cost per cell.",
    )
    c.maintenance_per_unit_per_year = st.sidebar.number_input(
        "Maintenance /yr/cell",
        value=float(c.maintenance_per_unit_per_year),
        step=500.0,
        help="Annual maintenance cost per deployed cell.",
    )

    st.sidebar.header("Performance & Revenue")
    p = model.scenario.perf
    p.uptime_pct = st.sidebar.slider(
        "Uptime %",
        0.5,
        0.99,
        float(p.uptime_pct),
        0.01,
        help="Expected operational uptime of the robotic system.",
    )
    p.defect_rate_change_pct = st.sidebar.slider(
        "Defect Rate Change %",
        -0.5,
        0.5,
        float(p.defect_rate_change_pct),
        0.01,
        help="Negative values = defect reduction vs. manual.",
    )

    st.sidebar.checkbox(
        "Include Revenue Uplift (throughput)",
        value=model.scenario.include_revenue_uplift,
        key="uplift",
        help="Monetize additional units sold when throughput increases.",
    )
    model.scenario.include_revenue_uplift = st.session_state["uplift"]

    # Run model
    res = model.run()

    # Summary metrics
    st.subheader("Summary")
    sm = res.summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NPV ($)", f"{sm['NPV']:,.0f}")
    irr_display = "—" if sm["IRR"] is None else f"{sm['IRR']*100:,.1f}%"
    col2.metric("IRR", irr_display)
    col3.metric(
        "Payback (Year)", sm["Payback_Year"] if sm["Payback_Year"] else "No payback"
    )
    col4.metric(
        "Simple ROI",
        f"{sm['Simple_ROI']*100:,.1f}%"
        if not math.isnan(sm["Simple_ROI"])
        else "—",
    )

    st.subheader("Yearly Breakdown")
    st.dataframe(
        res.yearly.style.format(
            {
                "DemandUnits": "{:.0f}",
                "CAPEX_new": "{:.0f}",
                "Manual_Total": "{:.0f}",
                "Auto_OPEX": "{:.0f}",
                "Net_Savings": "{:.0f}",
            }
        )
    )

    # Charts
    try:
        import altair as alt

        df = res.yearly.copy()
        base = alt.Chart(df).encode(x="Year:O")
        c1 = base.mark_line().encode(y="Manual_Total:Q", color=alt.value("steelblue"))
        c2 = base.mark_line().encode(y="Auto_OPEX:Q", color=alt.value("orange"))
        c3 = base.mark_bar(opacity=0.3).encode(
            y="CAPEX_new:Q", color=alt.value("grey")
        )
        st.altair_chart(
            (c1 + c2 + c3).properties(height=300), use_container_width=True
        )
    except Exception:
        st.info("Altair not available — charts skipped.")


# -----------------------------
# Entry point
# -----------------------------


if __name__ == "__main__":
    import sys

    # If running via `streamlit run`, Streamlit will already be imported.
    if "streamlit" in sys.modules:
        run_streamlit()
    else:
        parser = argparse.ArgumentParser(
            description="BMW Robotics ROI & Cost Degression Model"
        )
        parser.add_argument(
            "--app",
            choices=["logistics", "product", "manufacturing", "office", "sales"],
            default="manufacturing",
        )
        parser.add_argument("--years", type=int, default=None)
        parser.add_argument("--plot", action="store_true")
        parser.add_argument(
            "--out", type=str, default=None, help="CSV path for yearly table"
        )
        parser.add_argument(
            "--revenue-uplift",
            action="store_true",
            help="Monetize throughput increase",
        )
        parser.add_argument(
            "--override",
            nargs=2,
            action="append",
            default=[],
            metavar=("param.path", "value"),
            help="Override any parameter, e.g., labor.wage_per_hour 45",
        )
        parser.add_argument(
            "--streamlit", action="store_true", help="Launch Streamlit UI"
        )
        args = parser.parse_args()

        if args.streamlit:
            run_streamlit()
        else:
            run_cli(args)
