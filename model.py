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

class Country(Enum):
    GERMANY = auto()
    CHINA = auto()
    USA = auto()
    MEXICO = auto()

class SolutionType(Enum):
    HUMANOID = auto()
    DIGITAL_TWIN = auto()

@dataclass
class EconomicAssumptions:
    discount_rate: float = 0.10  # WACC / hurdle rate (10%)
    horizon_years: int = 7  # planning horizon
    inflation_labor: float = 0.03  # annual labor inflation
    inflation_maintenance: float = 0.02  # annual maintenance inflation
    inflation_energy: float = 0.03  # annual energy inflation
    carbon_price_per_ton: float = 0.0  # optional monetization of CO2

@dataclass(frozen=True)
class CountryAssumptions:
    wage_multiplier_vs_baseline: float = 1.0
    energy_cost_per_kwh: Optional[float] = None
    maintenance_multiplier: float = 1.0
    # optional: add other opex drivers later (rent, overhead adder, etc.)

COUNTRY_DEFAULTS: Dict[Country, CountryAssumptions] = {
    # Baseline in your defaults is roughly “US-like”; treat USA as 1.0
    Country.USA: CountryAssumptions(
        wage_multiplier_vs_baseline=1.0,
        energy_cost_per_kwh=0.12,
        maintenance_multiplier=1.0,
    ),

    # BMW note: Germany labor ~8x China labor (use multiplier on the same baseline wage)
    Country.GERMANY: CountryAssumptions(
        wage_multiplier_vs_baseline=1.25,   # example vs your current baseline; tune later
        energy_cost_per_kwh=0.25,           # example; tune later
        maintenance_multiplier=1.10,
    ),
    Country.CHINA: CountryAssumptions(
        wage_multiplier_vs_baseline=0.16,   # ~1/6–1/8 of high-income baseline (tune later)
        energy_cost_per_kwh=0.08,           # example; tune later
        maintenance_multiplier=0.95,
    ),
    Country.MEXICO: CountryAssumptions(
        wage_multiplier_vs_baseline=0.35,
        energy_cost_per_kwh=0.10,
        maintenance_multiplier=1.00,
    ),
}


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
class DigitalTwinParams:
    # One-time program costs (Yr 1)
    capex_initial: float = 1_200_000     # platform + initial build
    integration_initial: float = 600_000 # data connectors, MES/SCADA, models

    # Recurring costs
    software_per_year: float = 450_000
    data_ops_per_year: float = 200_000   # MLOps/Model maintenance, engineers

    # Benefits (modeled as % improvements vs manual baseline)
    defect_reduction_pct: float = 0.15   # reduce scrap+rework costs by 15%
    labor_efficiency_pct: float = 0.02   # reduce manual labor time by 2% via fewer disruptions
    throughput_uplift_pct: float = 0.02  # optional revenue uplift if monetized



# @dataclass
# class LearningCurve:
#     learning_rate: float = 0.15  # 15% cost reduction per doubling
#     apply_to: Tuple[str, ...] = ("capex_per_unit",)


@dataclass
class Scenario:
    application: ApplicationType
    country: Country
    solution: SolutionType
    economics: EconomicAssumptions
    demand: DemandParams
    labor: LaborParams
    costs: CostParams
    perf: PerformanceParams
    deploy: DeploymentPlan
    # learning: LearningCurve
    digital_twin: DigitalTwinParams = field(default_factory=DigitalTwinParams)
    include_revenue_uplift: bool = False  # toggles monetization of throughput


# -----------------------------
# Utility & Financial Math
# -----------------------------


# def wrights_law(base_cost: float, cumulative_units: int, learning_rate: float) -> float:
#     """Apply Wright's Law: cost = base * (cum_units)^(log2(1 - lr))
#     learning_rate is e.g., 0.15 (15% per doubling). For cum_units<=1, return base_cost.
#     """
#     if cumulative_units <= 1 or learning_rate <= 0:
#         return base_cost
#     b = math.log2(1 - learning_rate)  # negative number
#     return base_cost * (cumulative_units ** b)


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

    def apply_country(self):
        """Apply country-specific multipliers/overrides for labor & operating costs."""
        ca = COUNTRY_DEFAULTS.get(self.scenario.country, CountryAssumptions())

        # Labor: multiplier on the scenario’s base wage
        self.scenario.labor.wage_per_hour *= ca.wage_multiplier_vs_baseline

        # Operating costs: override energy price if provided
        if ca.energy_cost_per_kwh is not None:
            self.scenario.costs.energy_cost_per_kwh = ca.energy_cost_per_kwh

        # Maintenance: multiplier (useful if service/parts differ by region)
        self.scenario.costs.maintenance_per_unit_per_year *= ca.maintenance_multiplier
    
    def apply_solution_preset(self):
        """Adjust parameters to represent HUMANOID vs DIGITAL_TWIN scenarios."""
        sc = self.scenario

        if sc.solution == SolutionType.HUMANOID:
            # Humanoid: higher capex+maintenance, but can replace more direct labor time
            # (tune later; this is a demo-ready starting point)
            sc.costs.capex_per_unit *= 1.6
            sc.costs.integration_per_unit *= 1.3
            sc.costs.maintenance_per_unit_per_year *= 1.4

            # Typically more human oversight early than fixed automation, but less than manual
            sc.labor.automated_supervision_hours_per_unit *= 1.5

            # Humanoids are pitched on flexibility → assume bigger effective labor displacement
            # (reduce manual baseline time required per unit by improving automation coverage)
            sc.labor.manual_hours_per_unit *= 1.00  # keep baseline for manual scenario
            # Instead, represent benefits through lower supervision burden vs manual:
            # leave the manual baseline unchanged and rely on automated supervision hours

            # Deployment ramp: usually slower ramp than mature cobots
            sc.deploy.ramp = [sc.deploy.ramp[0]] + [max(1, int(x * 0.8)) for x in sc.deploy.ramp[1:]]

            # Optional: if execs want revenue uplift, humanoids often pitched as capacity unlock
            # sc.include_revenue_uplift = True

        elif sc.solution == SolutionType.DIGITAL_TWIN:
            # Digital twin: not "cells". We’ll model 0 robotic cells and shift spend into software program costs.
            sc.deploy.ramp = [0] * sc.economics.horizon_years

            # Set robot-cell costs to zero so we don’t double-count
            sc.costs.capex_per_unit = 0.0
            sc.costs.install_commission_per_unit = 0.0
            sc.costs.integration_per_unit = 0.0
            sc.costs.maintenance_per_unit_per_year = 0.0
            sc.costs.energy_kwh_per_unit = 0.0
            sc.costs.consumables_per_unit = 0.0


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
            # learning = LearningCurve(learning_rate=0.12)

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
            # learning = LearningCurve(learning_rate=0.15)

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
            # learning = LearningCurve(learning_rate=0.14)

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
            # learning = LearningCurve(learning_rate=0.20)

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
            # learning = LearningCurve(learning_rate=0.18)

        scenario = Scenario(
            application=app,
            country=Country.USA,
            solution=SolutionType.HUMANOID,
            economics=economics,
            demand=demand,
            labor=labor,
            costs=costs,
            perf=perf,
            deploy=deploy,
            # learning=learning,
            include_revenue_uplift=False,  # can be toggled in UI/CLI
        )
        model = Model(scenario)
        model.apply_country()
        model.apply_solution_preset()
        return model

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
            # deployed_units = ramp[t]
            # add_units = max(deployed_units - cum_units, 0)
            # # Learning-adjusted CAPEX for the *new* units only
            # # capex_unit = wrights_law(
            # #     sc.costs.capex_per_unit, max(cum_units, 1), sc.learning.learning_rate
            # # )
            # capex_unit = sc.costs.capex_per_unit
            # capex_new = add_units * (
            #     capex_unit
            #     + sc.costs.install_commission_per_unit
            #     + sc.costs.integration_per_unit
            # )
            # cum_units = deployed_units

            if sc.solution == SolutionType.DIGITAL_TWIN:
                # No cells; program costs instead
                deployed_units = 0
                add_units = 0
                capex_new = 0.0

                # Yr1 one-time program costs
                if t == 0:
                    capex_new = sc.digital_twin.capex_initial + sc.digital_twin.integration_initial

                cum_units = 0
            else:
                # HUMANOID (or other embodied robotics)
                deployed_units = ramp[t]
                add_units = max(deployed_units - cum_units, 0)
                capex_unit = sc.costs.capex_per_unit
                capex_new = add_units * (
                    capex_unit + sc.costs.install_commission_per_unit + sc.costs.integration_per_unit
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

            if sc.solution == SolutionType.DIGITAL_TWIN:
                # Digital twin opex
                auto_labor_cost = sc.digital_twin.data_ops_per_year  # flat annual team cost (simplified)
                auto_maintenance = 0.0
                auto_energy = 0.0
                auto_consumables = 0.0

                # Benefits: reduce defect costs and a small labor efficiency improvement
                auto_defect_cost = manual_defect_cost * (1 - sc.digital_twin.defect_reduction_pct)

                # Apply small labor efficiency improvement vs manual labor
                manual_labor_cost_adj = manual_labor_cost * (1 - sc.digital_twin.labor_efficiency_pct)

                # Safety delta not really relevant → set to 0 for DT (or keep if you want)
                safety_delta = 0.0

                # Subscription cost
                subscription = sc.digital_twin.software_per_year

                auto_total_opex = (
                    auto_labor_cost
                    + subscription
                    + auto_defect_cost
                    + safety_delta
                )

                # For DT scenario, the baseline manual_total should reflect adjusted labor + original defects
                manual_total = manual_labor_cost_adj + manual_defect_cost

                # Optional revenue uplift (capacity unlock) if monetized
                revenue_uplift = 0.0
                if sc.include_revenue_uplift and sc.demand.revenue_per_unit > 0 and sc.digital_twin.throughput_uplift_pct > 0:
                    extra_units = demand_units * sc.digital_twin.throughput_uplift_pct
                    revenue_uplift = extra_units * sc.demand.revenue_per_unit

            else:
                # Existing embodied-robotics logic (your current block)
                auto_hours = sc.labor.automated_supervision_hours_per_unit * demand_units
                auto_labor_cost = auto_hours * wage_by_year[t]
                auto_maintenance = cum_units * maint_by_year[t]
                auto_energy = sc.costs.energy_kwh_per_unit * energy_cost_by_year[t] * demand_units
                auto_consumables = sc.costs.consumables_per_unit * demand_units

                defect_delta_factor = 1 + sc.perf.defect_rate_change_pct
                auto_defect_cost = (sc.perf.scrap_cost_per_unit + sc.perf.rework_cost_per_unit) * demand_units * max(0.0, defect_delta_factor)

                safety_delta = sc.labor.safety_incident_cost_delta_per_year

                revenue_uplift = 0.0
                if sc.include_revenue_uplift and sc.demand.revenue_per_unit > 0 and sc.perf.throughput_change_pct > 0:
                    extra_units = demand_units * sc.perf.throughput_change_pct
                    revenue_uplift = extra_units * sc.demand.revenue_per_unit

                auto_total_opex = (
                    auto_labor_cost + auto_maintenance + auto_energy + auto_consumables + auto_defect_cost + safety_delta
                )


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

    country_map = {
        "usa": Country.USA,
        "germany": Country.GERMANY,
        "china": Country.CHINA,
        "mexico": Country.MEXICO,
    }
    model.scenario.country = country_map[args.country]
    model.apply_country()

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

def assumptions_table(solution: str) -> pd.DataFrame:
    rows = []

    if solution.lower() == "humanoid":
        rows += [
            {
                "Category": "Robot hardware",
                "Parameter": "CAPEX per humanoid cell",
                "Model Value": "$512,000",
                "Why this value is reasonable": (
                    "Humanoid robots capable of industrial duty are currently quoted between "
                    "$150k–$300k for hardware alone; additional safety hardware, compute, tooling, "
                    "and installation roughly double this cost for plant-ready deployment."
                ),
                "Industry evidence": "Morgan Stanley Research 2024; Reuters 2025"
            },
            {
                "Category": "Integration",
                "Parameter": "Integration per cell",
                "Model Value": "$130,000",
                "Why this value is reasonable": (
                    "FOAK integration in automotive plants typically adds 30–50% of hardware cost "
                    "for controls, safety, validation, and process tuning."
                ),
                "Industry evidence": "McKinsey Industrial Automation Benchmarks"
            },
            {
                "Category": "Annual maintenance",
                "Parameter": "Maintenance / year",
                "Model Value": "$16,800",
                "Why this value is reasonable": (
                    "Higher mechanical complexity and lower maturity increases service burden vs. "
                    "standard robot arms."
                ),
                "Industry evidence": "ABB & FANUC automotive service contracts (proxy)"
            },
            {
                "Category": "Oversight labor",
                "Parameter": "Human supervision time",
                "Model Value": "0.0075 hr / unit",
                "Why this value is reasonable": (
                    "Humanoids require more exception handling and training than fixed automation, "
                    "especially in early deployments."
                ),
                "Industry evidence": "Early humanoid pilot deployments"
            },
            {
                "Category": "Deployment ramp",
                "Parameter": "Cells after Yr 7",
                "Model Value": "22 cells",
                "Why this value is reasonable": (
                    "Early humanoid rollouts expand more slowly than standardized cobot cells due "
                    "to safety approvals and workflow tuning."
                ),
                "Industry evidence": "FOAK→NOAK industrial deployment pattern"
            },
        ]

    elif solution.lower() == "digital twin":
        rows += [
            {
                "Category": "Platform build",
                "Parameter": "Initial platform CAPEX (Yr1)",
                "Model Value": "$1,200,000",
                "Why this value is reasonable": (
                    "Enterprise digital twin programs typically require a seven-figure initial "
                    "platform investment to model assets and connect plant systems."
                ),
                "Industry evidence": "Autodesk Manufacturing Digital Twin, Oxmaint"
            },
            {
                "Category": "System integration",
                "Parameter": "Initial integration (Yr1)",
                "Model Value": "$600,000",
                "Why this value is reasonable": (
                    "MES/SCADA integration and data modeling represent heavy first-year effort."
                ),
                "Industry evidence": "Oxmaint Maintenance Twin Implementation Guides"
            },
            {
                "Category": "Platform license",
                "Parameter": "Annual subscription",
                "Model Value": "$450,000 / yr",
                "Why this value is reasonable": (
                    "Enterprise SaaS digital twin platforms are licensed in the mid-six-figure range."
                ),
                "Industry evidence": "Autodesk / Siemens Xcelerator pricing (enterprise ranges)"
            },
            {
                "Category": "Data & MLOps",
                "Parameter": "Annual data ops",
                "Model Value": "$200,000 / yr",
                "Why this value is reasonable": (
                    "Represents 2–3 FTE engineers for data pipelines, models, and monitoring."
                ),
                "Industry evidence": "Typical data engineering salary benchmarks"
            },
            {
                "Category": "Quality improvement",
                "Parameter": "Defect reduction",
                "Model Value": "15%",
                "Why this value is reasonable": (
                    "Digital twins reduce scrap/rework by identifying deviations before failure."
                ),
                "Industry evidence": "Autodesk Manufacturing Twin case studies"
            },
        ]

    return pd.DataFrame(rows)

def run_streamlit():
    import streamlit as st
    import math

    st.set_page_config(page_title="BMW Robotics ROI", layout="wide")
    st.title("BMW Robotics ROI & Cost Degression Model")

    @st.dialog("Assumptions & Sources")
    def show_assumptions_dialog():
        st.write("Model assumptions used in this dashboard. Values are starting points and are meant to be tuned with BMW data.")
        tab1, tab2 = st.tabs(["Humanoid", "Digital Twin"])

        with tab1:
            df = assumptions_table("humanoid")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab2:
            df = assumptions_table("digital twin")
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            "Tip: Replace 'general industry practice' rows with BMW internal benchmarks as they become available."
        )

    # -------------------------
    # Sidebar: core selectors
    # -------------------------
    app_choice = st.sidebar.selectbox(
        "Application", [e.name.title() for e in ApplicationType]
    )
    country_choice = st.sidebar.selectbox(
        "Country", [c.name.title() for c in Country]
    )

    st.sidebar.header("Robotic Solutions")
    scenario_view = st.sidebar.radio(
        "View",
        ["Compare Solutions", "Single Scenario"],
        index=0,
    )

    # If single scenario, choose which one
    single_solution_choice = None
    if scenario_view == "Single Scenario":
        single_solution_choice = st.sidebar.selectbox(
            "Solution", [s.name.replace("_", " ").title() for s in SolutionType]
        )

    # Helper to build a model consistently
    def make_model(app: ApplicationType, country: Country, solution: SolutionType) -> Model:
        m = Model.default(app)
        m.scenario.country = country
        m.scenario.solution = solution

        # IMPORTANT: re-apply transforms after setting country/solution
        m.apply_country()
        m.apply_solution_preset()
        return m

    app = ApplicationType[app_choice.upper()]
    country = Country[country_choice.upper()]

    if st.sidebar.button("Show assumptions & sources"):
        show_assumptions_dialog()

    # -------------------------
    # Sidebar: horizon & finance (apply to both scenarios)
    # -------------------------
    st.sidebar.header("Horizon & Finance")
    years = st.sidebar.slider("Years", 3, 15, 7)
    discount_rate = st.sidebar.slider(
        "Discount Rate", 0.02, 0.2, 0.10, 0.01
    )

    include_uplift = st.sidebar.checkbox(
        "Include Revenue Uplift (throughput)",
        value=False,
        help="Monetize additional units sold when throughput increases.",
    )

    # -------------------------
    # Build models (either 2 for compare or 1 for single)
    # -------------------------
    if scenario_view == "Compare Solutions":
        humanoid_model = make_model(app, country, SolutionType.HUMANOID)
        twin_model = make_model(app, country, SolutionType.DIGITAL_TWIN)

        # Apply shared finance settings
        for m in (humanoid_model, twin_model):
            m.scenario.economics.horizon_years = int(years)
            m.scenario.economics.discount_rate = float(discount_rate)
            m.scenario.include_revenue_uplift = bool(include_uplift)

        # Country caption (same for both)
        st.sidebar.caption(
            f"Applied wage multiplier: {COUNTRY_DEFAULTS[country].wage_multiplier_vs_baseline:.2f}x"
        )

        # -------------------------
        # Optional: scenario-specific tuning controls
        # Keep this light for execs (only the biggest levers)
        # -------------------------
        st.sidebar.header("Key Levers (Optional)")

        # Humanoid levers
        st.sidebar.subheader("Humanoid")
        h_lab = humanoid_model.scenario.labor
        h_cost = humanoid_model.scenario.costs
        h_lab.wage_per_hour = st.sidebar.number_input(
            "Humanoid: Wage $/hr",
            value=float(h_lab.wage_per_hour),
            step=1.0,
        )
        h_cost.capex_per_unit = st.sidebar.number_input(
            "Humanoid: CAPEX per Unit",
            value=float(h_cost.capex_per_unit),
            step=10_000.0,
        )
        h_cost.integration_per_unit = st.sidebar.number_input(
            "Humanoid: Integration per Unit",
            value=float(h_cost.integration_per_unit),
            step=10_000.0,
        )

        # Digital twin levers
        st.sidebar.subheader("Digital Twin")
        dt = twin_model.scenario.digital_twin
        dt.capex_initial = st.sidebar.number_input(
            "DT: Initial Platform Cost (Yr1)",
            value=float(dt.capex_initial),
            step=50_000.0,
        )
        dt.integration_initial = st.sidebar.number_input(
            "DT: Initial Integration (Yr1)",
            value=float(dt.integration_initial),
            step=50_000.0,
        )
        dt.software_per_year = st.sidebar.number_input(
            "DT: Software / Year",
            value=float(dt.software_per_year),
            step=25_000.0,
        )
        dt.defect_reduction_pct = st.sidebar.slider(
            "DT: Defect Reduction",
            0.0, 0.5, float(dt.defect_reduction_pct), 0.01
        )

        # Run both
        humanoid_res = humanoid_model.run()
        twin_res = twin_model.run()

        # -------------------------
        # Summary metrics: side-by-side
        # -------------------------
        st.subheader("Overview Dashboard")
        colA, colB = st.columns(2)

        def show_summary(col, label: str, res: Results):
            sm = res.summary()
            col.markdown(f"### {label}")
            col.metric("NPV ($)", f"{sm['NPV']:,.0f}")
            irr_display = "—" if sm["IRR"] is None else f"{sm['IRR']*100:,.1f}%"
            col.metric("IRR", irr_display)
            col.metric("Payback (Year)", sm["Payback_Year"] if sm["Payback_Year"] else "No payback")
            col.metric(
                "Simple ROI",
                f"{sm['Simple_ROI']*100:,.1f}%" if not math.isnan(sm["Simple_ROI"]) else "—",
            )

        show_summary(colA, "Humanoid Robots", humanoid_res)
        show_summary(colB, "Digital Twin", twin_res)

        # -------------------------
        # Cumulative net savings comparison
        # -------------------------
        st.subheader("Cumulative Net Savings")
        try:
            import altair as alt
            d1 = humanoid_res.yearly[["Year", "Net_Savings"]].copy()
            d1["Scenario"] = "Humanoid"
            d1["Cumulative_Net_Savings"] = d1["Net_Savings"].cumsum()

            d2 = twin_res.yearly[["Year", "Net_Savings"]].copy()
            d2["Scenario"] = "Digital Twin"
            d2["Cumulative_Net_Savings"] = d2["Net_Savings"].cumsum()

            dfc = pd.concat([d1, d2], ignore_index=True)

            chart = (
                alt.Chart(dfc)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Year:O"),
                    y=alt.Y("Cumulative_Net_Savings:Q", title="Cumulative Net Savings ($)"),
                    color="Scenario:N",
                    tooltip=["Scenario", "Year", "Cumulative_Net_Savings"],
                )
                .properties(height=350)
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
            st.altair_chart(chart + zero, use_container_width=True)
        except Exception:
            st.info("Altair not available — comparison chart skipped.")

        # -------------------------
        # Deep dive tables (optional)
        # -------------------------
        with st.expander("Yearly Breakdown (Humanoid)"):
            st.dataframe(
                humanoid_res.yearly.style.format(
                    {
                        "DemandUnits": "{:.0f}",
                        "CAPEX_new": "{:.0f}",
                        "Manual_Total": "{:.0f}",
                        "Auto_OPEX": "{:.0f}",
                        "Net_Savings": "{:.0f}",
                    }
                )
            )
        with st.expander("Yearly Breakdown (Digital Twin)"):
            st.dataframe(
                twin_res.yearly.style.format(
                    {
                        "DemandUnits": "{:.0f}",
                        "CAPEX_new": "{:.0f}",
                        "Manual_Total": "{:.0f}",
                        "Auto_OPEX": "{:.0f}",
                        "Net_Savings": "{:.0f}",
                    }
                )
            )

    else:
        # -------------------------
        # Single scenario mode
        # -------------------------
        solution = SolutionType[single_solution_choice.upper().replace(" ", "_")]
        model = make_model(app, country, solution)

        # Apply shared finance settings
        model.scenario.economics.horizon_years = int(years)
        model.scenario.economics.discount_rate = float(discount_rate)
        model.scenario.include_revenue_uplift = bool(include_uplift)

        st.sidebar.caption(
            f"Applied wage multiplier: {COUNTRY_DEFAULTS[country].wage_multiplier_vs_baseline:.2f}x"
        )

        # Minimal tuning controls
        st.sidebar.header("Key Levers")
        if solution == SolutionType.HUMANOID:
            st.sidebar.subheader("Humanoid")
            lab = model.scenario.labor
            c = model.scenario.costs
            lab.wage_per_hour = st.sidebar.number_input(
                "Wage $/hr", value=float(lab.wage_per_hour), step=1.0
            )
            c.capex_per_unit = st.sidebar.number_input(
                "CAPEX per Unit", value=float(c.capex_per_unit), step=10_000.0
            )
            c.integration_per_unit = st.sidebar.number_input(
                "Integration per Unit", value=float(c.integration_per_unit), step=10_000.0
            )
        else:
            st.sidebar.subheader("Digital Twin")
            dt = model.scenario.digital_twin
            dt.capex_initial = st.sidebar.number_input(
                "Initial Platform Cost (Yr1)", value=float(dt.capex_initial), step=50_000.0
            )
            dt.integration_initial = st.sidebar.number_input(
                "Initial Integration (Yr1)", value=float(dt.integration_initial), step=50_000.0
            )
            dt.software_per_year = st.sidebar.number_input(
                "Software / Year", value=float(dt.software_per_year), step=25_000.0
            )
            dt.defect_reduction_pct = st.sidebar.slider(
                "Defect Reduction", 0.0, 0.5, float(dt.defect_reduction_pct), 0.01
            )

        res = model.run()

        # Summary
        st.subheader("Summary")
        sm = res.summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NPV ($)", f"{sm['NPV']:,.0f}")
        irr_display = "—" if sm["IRR"] is None else f"{sm['IRR']*100:,.1f}%"
        col2.metric("IRR", irr_display)
        col3.metric("Payback (Year)", sm["Payback_Year"] if sm["Payback_Year"] else "No payback")
        col4.metric(
            "Simple ROI",
            f"{sm['Simple_ROI']*100:,.1f}%" if not math.isnan(sm["Simple_ROI"]) else "—",
        )

        # Charts (single scenario): show manual vs auto OPEX + capex + cumulative savings
        st.subheader("Cashflows & Savings")
        try:
            import altair as alt

            df = res.yearly.copy()
            df["Cumulative_Net_Savings"] = df["Net_Savings"].cumsum()

            base = alt.Chart(df).encode(x="Year:O")
            manual_line = base.mark_line().encode(y=alt.Y("Manual_Total:Q", title="$/Year"), color=alt.value("steelblue"))
            auto_line = base.mark_line().encode(y="Auto_OPEX:Q", color=alt.value("orange"))
            capex_bar = base.mark_bar(opacity=0.25).encode(y="CAPEX_new:Q", color=alt.value("grey"))
            st.altair_chart((manual_line + auto_line + capex_bar).properties(height=300), use_container_width=True)

            cum = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x="Year:O",
                    y=alt.Y("Cumulative_Net_Savings:Q", title="Cumulative Net Savings ($)"),
                    tooltip=["Year", "Cumulative_Net_Savings"],
                )
                .properties(height=250)
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
            st.altair_chart(cum + zero, use_container_width=True)
        except Exception:
            st.info("Altair not available — charts skipped.")

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


# def run_streamlit():
#     import streamlit as st

#     st.set_page_config(page_title="BMW Robotics ROI", layout="wide")
#     st.title("BMW Robotics ROI & Cost Degression Model")

#     # Sidebar selectors
#     app_choice = st.sidebar.selectbox(
#         "Application", [e.name.title() for e in ApplicationType]
#     )
#     model = Model.default(ApplicationType[app_choice.upper()])

#     country_choice = st.sidebar.selectbox(
#         "Country", [c.name.title() for c in Country]
#     )
#     model.scenario.country = Country[country_choice.upper()]
#     model.apply_country()
#     st.sidebar.caption(
#         f"Applied wage multiplier: {COUNTRY_DEFAULTS[model.scenario.country].wage_multiplier_vs_baseline:.2f}x"
#     )


#     st.sidebar.header("Horizon & Finance")
#     econ = model.scenario.economics
#     econ.horizon_years = st.sidebar.slider(
#         "Years", 3, 15, econ.horizon_years, help="Planning horizon for ROI evaluation."
#     )
#     econ.discount_rate = st.sidebar.slider(
#         "Discount Rate",
#         0.02,
#         0.2,
#         float(econ.discount_rate),
#         0.01,
#         help="BMW hurdle rate / cost of capital.",
#     )

#     # st.sidebar.header("Learning Curve")
#     # model.scenario.learning.learning_rate = st.sidebar.slider(
#     #     "Learning Rate (per doubling)",
#     #     0.0,
#     #     0.3,
#     #     float(model.scenario.learning.learning_rate),
#     #     0.01,
#     #     help="Cost reduction percentage each time cumulative deployments double.",
#     # )

#     st.sidebar.header("Labor")
#     lab = model.scenario.labor
#     lab.wage_per_hour = st.sidebar.number_input(
#         "Wage $/hr",
#         value=float(lab.wage_per_hour),
#         step=1.0,
#         help="Fully-loaded labor cost per hour.",
#     )

#     st.sidebar.header("Costs")
#     c = model.scenario.costs
#     c.capex_per_unit = st.sidebar.number_input(
#         "CAPEX per Cell",
#         value=float(c.capex_per_unit),
#         step=5_000.0,
#         help="Hardware cost per robotic cell.",
#     )
#     c.integration_per_unit = st.sidebar.number_input(
#         "Integration per Cell",
#         value=float(c.integration_per_unit),
#         step=5_000.0,
#         help="Engineering, controls, safety integration cost per cell.",
#     )
#     c.maintenance_per_unit_per_year = st.sidebar.number_input(
#         "Maintenance /yr/cell",
#         value=float(c.maintenance_per_unit_per_year),
#         step=500.0,
#         help="Annual maintenance cost per deployed cell.",
#     )

#     st.sidebar.header("Performance & Revenue")
#     p = model.scenario.perf
#     p.uptime_pct = st.sidebar.slider(
#         "Uptime %",
#         0.5,
#         0.99,
#         float(p.uptime_pct),
#         0.01,
#         help="Expected operational uptime of the robotic system.",
#     )
#     p.defect_rate_change_pct = st.sidebar.slider(
#         "Defect Rate Change %",
#         -0.5,
#         0.5,
#         float(p.defect_rate_change_pct),
#         0.01,
#         help="Negative values = defect reduction vs. manual.",
#     )

#     st.sidebar.checkbox(
#         "Include Revenue Uplift (throughput)",
#         value=model.scenario.include_revenue_uplift,
#         key="uplift",
#         help="Monetize additional units sold when throughput increases.",
#     )
#     model.scenario.include_revenue_uplift = st.session_state["uplift"]

#     # Run model
#     res = model.run()

#     # Summary metrics
#     st.subheader("Summary")
#     sm = res.summary()
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("NPV ($)", f"{sm['NPV']:,.0f}")
#     irr_display = "—" if sm["IRR"] is None else f"{sm['IRR']*100:,.1f}%"
#     col2.metric("IRR", irr_display)
#     col3.metric(
#         "Payback (Year)", sm["Payback_Year"] if sm["Payback_Year"] else "No payback"
#     )
#     col4.metric(
#         "Simple ROI",
#         f"{sm['Simple_ROI']*100:,.1f}%"
#         if not math.isnan(sm["Simple_ROI"])
#         else "—",
#     )

#     st.subheader("Yearly Breakdown")
#     st.dataframe(
#         res.yearly.style.format(
#             {
#                 "DemandUnits": "{:.0f}",
#                 "CAPEX_new": "{:.0f}",
#                 "Manual_Total": "{:.0f}",
#                 "Auto_OPEX": "{:.0f}",
#                 "Net_Savings": "{:.0f}",
#             }
#         )
#     )

#     # Charts
#     try:
#         import altair as alt

#         df = res.yearly.copy()
#         base = alt.Chart(df).encode(x="Year:O")
#         c1 = base.mark_line().encode(y="Manual_Total:Q", color=alt.value("steelblue"))
#         c2 = base.mark_line().encode(y="Auto_OPEX:Q", color=alt.value("orange"))
#         c3 = base.mark_bar(opacity=0.3).encode(
#             y="CAPEX_new:Q", color=alt.value("grey")
#         )
#         st.altair_chart(
#             (c1 + c2 + c3).properties(height=300), use_container_width=True
#         )
#     except Exception:
#         st.info("Altair not available — charts skipped.")


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
        parser.add_argument(
            "--country",
            choices=["usa", "germany", "china", "mexico"],
            default="usa",
            help="Country used to set labor and operating cost assumptions.",
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
