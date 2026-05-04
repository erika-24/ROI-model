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
- Payback, NPV, and ROI metrics
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
import json
import math
import os
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import time

import streamlit as st
from google import genai
import matplotlib.pyplot as plt

from roi_shared_state import load_roi_state, save_roi_state, save_row_state, get_row_state, update_row_state, get_assumption_notes


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
    HARDWARE = auto()
    SOFTWARE = auto()

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
    automation_capture_pct: float = 1.0  # share of manual labor time displaced by the technology
    automated_wage_per_hour: Optional[float] = None  # optional lower wage for residual/supervisory labor
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
class SoftwareParams:
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
    software: SoftwareParams = field(default_factory=SoftwareParams)
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
        """Adjust parameters to represent HARDWARE vs SOFTWARE scenarios."""
        sc = self.scenario

        if sc.solution == SolutionType.HARDWARE:
            # HARDWARE: higher capex+maintenance, but can replace more direct labor time
            # (tune later; this is a demo-ready starting point)
            sc.costs.capex_per_unit *= 1.6
            sc.costs.integration_per_unit *= 1.3
            sc.costs.maintenance_per_unit_per_year *= 1.4

            # Typically more human oversight early than fixed automation, but less than manual
            sc.labor.automated_supervision_hours_per_unit *= 1.5

            # HARDWARE solutions are pitched on flexibility → assume bigger effective labor displacement
            # (reduce manual baseline time required per unit by improving automation coverage)
            sc.labor.manual_hours_per_unit *= 1.00  # keep baseline for manual scenario
            # Instead, represent benefits through lower supervision burden vs manual:
            # leave the manual baseline unchanged and rely on automated supervision hours

            # Deployment ramp: usually slower ramp than mature cobots
            sc.deploy.ramp = [sc.deploy.ramp[0]] + [max(1, int(x * 0.8)) for x in sc.deploy.ramp[1:]]

            # Optional: if execs want revenue uplift, HARDWARE solutions often pitched as capacity unlock
            # sc.include_revenue_uplift = True

        elif sc.solution == SolutionType.SOFTWARE:
            # Software: not "cells". We’ll model 0 robotic cells and shift spend into software CAPEX costs.
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
            solution=SolutionType.HARDWARE,
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
        automated_wage_base = (
            sc.labor.automated_wage_per_hour
            if sc.labor.automated_wage_per_hour is not None
            else sc.labor.wage_per_hour
        )
        automated_wage_by_year = [
            automated_wage_base * ((1 + sc.economics.inflation_labor) ** t)
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

            if sc.solution == SolutionType.SOFTWARE:
                # No cells; program costs instead
                deployed_units = 0
                add_units = 0
                capex_new = 0.0

                # Yr1 one-time program costs
                if t == 0:
                    capex_new = sc.software.capex_initial + sc.software.integration_initial

                cum_units = 0
            else:
                # HARDWARE (embodied robotics)
                deployed_units = ramp[t]
                add_units = max(deployed_units - cum_units, 0)
                capex_unit = sc.costs.capex_per_unit
                capex_new = add_units * (
                    capex_unit + sc.costs.install_commission_per_unit + sc.costs.integration_per_unit
                )
                cum_units = deployed_units

            # Automated operational costs
            # Residual labor model:
            # - automation_capture_pct is the share of manual labor removed by the technology
            # - residual manual labor remains when automation_capture_pct < 1
            # - supervision labor is added on top
            manual_hours_total = sc.labor.manual_hours_per_unit * demand_units
            automation_capture_pct = max(0.0, min(1.0, sc.labor.automation_capture_pct))

            residual_manual_hours = manual_hours_total * (1 - automation_capture_pct)
            supervision_hours = sc.labor.automated_supervision_hours_per_unit * demand_units
            auto_hours = residual_manual_hours + supervision_hours

            auto_labor_cost = auto_hours * automated_wage_by_year[t]

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

            if sc.solution == SolutionType.SOFTWARE:
                # Software does not deploy physical cells.
                # It changes economics through:
                # - one-time software CAPEX in Year 1
                # - recurring software/data-ops costs
                # - partial labor efficiency
                # - defect/rework reduction
                # - optional throughput revenue uplift

                auto_maintenance = 0.0
                auto_energy = 0.0
                auto_consumables = 0.0
                safety_delta = 0.0

                subscription = sc.software.software_per_year
                data_ops = sc.software.data_ops_per_year

                labor_efficiency_pct = max(0.0, min(1.0, sc.software.labor_efficiency_pct))
                software_defect_reduction_pct = max(0.0, min(1.0, sc.software.defect_reduction_pct))

                # Software typically augments rather than fully replaces workers.
                # Automated-state labor cost is manual labor after software efficiency improvement,
                # plus data/MLOps support.
                auto_labor_cost = manual_labor_cost * (1 - labor_efficiency_pct) + data_ops

                auto_defect_cost = manual_defect_cost * (1 - software_defect_reduction_pct)

                revenue_uplift = 0.0
                if (
                    sc.include_revenue_uplift
                    and sc.demand.revenue_per_unit > 0
                    and sc.software.throughput_uplift_pct > 0
                ):
                    extra_units = demand_units * sc.software.throughput_uplift_pct
                    revenue_uplift = extra_units * sc.demand.revenue_per_unit

                auto_total_opex = (
                    auto_labor_cost
                    + subscription
                    + auto_defect_cost
                    + safety_delta
                )
            else:
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
        cumulative = df["Net_Savings"].cumsum()
        payback_year = next(
            (int(y) for y, val in zip(df["Year"], cumulative) if val >= 0), None
        )

        total_capex = df["CAPEX_new"].sum()
        total_manual = df["Manual_Total"].sum()
        total_auto_opex = df["Auto_OPEX"].sum()
        total_revenue_uplift = df["Revenue_Uplift"].sum()

        # Operating savings before upfront investment. Using CAPEX-only
        # ROI can behave backwards when operating savings are negative because
        # (negative savings - CAPEX) / CAPEX becomes less negative as CAPEX grows.
        total_savings = total_manual - total_auto_opex

        # Total-cost ROI / benefit-cost style ROI:
        #   benefits = avoided manual operating cost + monetized revenue uplift
        #   costs    = automated operating cost + upfront CAPEX
        # This keeps CAPEX as a true cost: holding all else constant, increasing
        # CAPEX always lowers ROI.
        total_project_cost = total_auto_opex + total_capex
        net_benefit_after_all_costs = (
            total_manual + total_revenue_uplift - total_project_cost
        )

        roi = (
            net_benefit_after_all_costs / total_project_cost
            if total_project_cost > 0
            else np.nan
        )
        

        summary = {
            "NPV": npv_net,
            "Payback_Year": payback_year,
            "Total_CAPEX": total_capex,
            "Total_Manual_OPEX": total_manual,
            "Total_Auto_OPEX": total_auto_opex,
            "Total_Revenue_Uplift": total_revenue_uplift,
            "Total_Project_Cost": total_project_cost,
            "Total_Savings_vs_Manual": total_savings,
            "Net_Benefit_After_All_Costs": net_benefit_after_all_costs,
            "ROI": roi,
        }

        return Results(yearly=df, summary_metrics=summary)

    # -------------------------
    # Sensitivity & Utilities
    # -------------------------

    def sensitivity_grid(self, param_path: str, values: List[float]) -> pd.DataFrame:
        """Vary a single scalar parameter and compute NPV/Payback."""
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
        "USA": Country.USA,
        "germany": Country.GERMANY,
        "china": Country.CHINA,
        "mexico": Country.MEXICO,
    }
    model.scenario.country = country_map[args.country]

    # Horizon override before presets so any horizon-based transforms stay aligned
    if args.years:
        model.scenario.economics.horizon_years = int(args.years)

    model.apply_country()
    model.apply_solution_preset()

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

    # Revenue uplift toggle
    if args.revenue_uplift:
        model.scenario.include_revenue_uplift = True

    return model


PARAMETER_BUCKET_ORDER = [
    "Finance",
    "Demand & Revenue",
    "Labor Baseline",
    "Technology Costs",
    "Performance & Quality",
    "Deployment & Rollout",
    "Software Costs",
]

PARAMETER_BUCKET_GUIDES = {
    "Finance": "Financial assumptions that drive discounting and annual cost escalation.",
    "Demand & Revenue": "Business volume assumptions. Revenue per unit is only used when revenue uplift is turned on.",
    "Labor Baseline": "Manual-state labor assumptions and safety impact. These set the baseline that automation is compared against.",
    "Technology Costs": "Direct technology spend for embodied robotics. These are not used in software mode once the software preset is applied.",
    "Performance & Quality": "Operational performance and quality assumptions. Defect and throughput changes are important drivers of ROI.",
    "Deployment & Rollout": "How quickly the technology is rolled out over time. Ramp should be a comma-separated list by year.",
    "Software Costs": "One-time and recurring costs plus benefits for the software option.",
}

PARAMETER_METADATA = [
    {
        "path": "economics.discount_rate",
        "bucket": "Finance",
        "label": "Discount rate",
        "help": "BMW hurdle rate or weighted average cost of capital used for NPV.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 0.5,
        "step": 0.005,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "economics.horizon_years",
        "bucket": "Finance",
        "label": "Planning horizon (years)",
        "help": "Number of years included in the ROI calculation.",
        "kind": "int",
        "min_value": 1,
        "max_value": 20,
        "step": 1,
        "solutions": ["hardware", "software"],
    },
    {
        "path": "economics.inflation_labor",
        "bucket": "Finance",
        "label": "Labor inflation",
        "help": "Annual escalation applied to labor cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 0.2,
        "step": 0.005,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "economics.inflation_maintenance",
        "bucket": "Finance",
        "label": "Maintenance inflation",
        "help": "Annual escalation applied to maintenance cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 0.2,
        "step": 0.005,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "economics.inflation_energy",
        "bucket": "Finance",
        "label": "Energy inflation",
        "help": "Annual escalation applied to energy price.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 0.2,
        "step": 0.005,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "include_revenue_uplift",
        "bucket": "Demand & Revenue",
        "label": "Include revenue uplift",
        "help": "Turn on monetization of extra throughput or capacity created by the technology.",
        "kind": "bool",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "demand.base_units_per_year",
        "bucket": "Demand & Revenue",
        "label": "Base units per year",
        "help": "Year-1 demand volume used as the base for the forecast.",
        "kind": "int",
        "min_value": 0,
        "max_value": 100000000,
        "step": 1000,
        "solutions": ["hardware", "software"],
    },
    {
        "path": "demand.annual_growth",
        "bucket": "Demand & Revenue",
        "label": "Annual demand growth",
        "help": "Expected annual demand growth rate.",
        "kind": "float",
        "min_value": -0.2,
        "max_value": 0.5,
        "step": 0.005,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "demand.revenue_per_unit",
        "bucket": "Demand & Revenue",
        "label": "Revenue per unit",
        "help": "Contribution margin or monetized value for each additional unit processed. Used only when revenue uplift is enabled.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 100000.0,
        "step": 10.0,
        "format": "%.3f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.wage_per_hour",
        "bucket": "Labor Baseline",
        "label": "Wage per hour",
        "help": "Fully loaded hourly labor cost before country multiplier is applied.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 500.0,
        "step": 1.0,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.manual_hours_per_unit",
        "bucket": "Labor Baseline",
        "label": "Manual hours per unit",
        "help": "Manual-state labor time needed to process one unit.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10.0,
        "step": 0.001,
        "format": "%.1f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.automated_supervision_hours_per_unit",
        "bucket": "Labor Baseline",
        "label": "Automated supervision hours per unit",
        "help": "Residual human oversight required for each unit in the automation case. Keep low for near-full replacement and higher for assistive technologies.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10.0,
        "step": 0.001,
        "format": "%.1f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.automation_capture_pct",
        "bucket": "Labor Baseline",
        "label": "Automation capture",
        "help": "Share of baseline manual labor time displaced by the technology. Use 1.0 for full replacement, lower values for assistive or augmenting technologies.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.automated_wage_per_hour",
        "bucket": "Labor Baseline",
        "label": "Automation-side wage per hour",
        "help": "Optional wage for residual supervisory labor in the automated state. Set this below the baseline wage to model higher-skill to lower-skill handoff.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 500.0,
        "step": 1.0,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "labor.safety_incident_cost_delta_per_year",
        "bucket": "Labor Baseline",
        "label": "Safety incident cost delta per year",
        "help": "Negative values mean annual savings versus the manual baseline. Positive values mean added cost.",
        "kind": "float",
        "min_value": -10000000.0,
        "max_value": 10000000.0,
        "step": 1000.0,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "costs.capex_per_unit",
        "bucket": "Technology Costs",
        "label": "CAPEX per unit",
        "help": "Hardware acquisition cost per deployed robotic cell before solution preset multipliers are applied.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10000000.0,
        "step": 10000.0,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.install_commission_per_unit",
        "bucket": "Technology Costs",
        "label": "Install and commission per unit",
        "help": "On-site installation and commissioning cost per new deployed cell.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 5000000.0,
        "step": 5000.0,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.integration_per_unit",
        "bucket": "Technology Costs",
        "label": "Integration per unit",
        "help": "Controls, safety, tooling, and process-integration cost per new deployed cell.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 5000000.0,
        "step": 5000.0,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.maintenance_per_unit_per_year",
        "bucket": "Technology Costs",
        "label": "Maintenance per unit per year",
        "help": "Annual service and upkeep cost per deployed cell before country and solution multipliers are applied.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 1000000.0,
        "step": 1000.0,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.energy_kwh_per_unit",
        "bucket": "Technology Costs",
        "label": "Energy kWh per unit",
        "help": "Electricity consumption per processed unit.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 1000.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.energy_cost_per_kwh",
        "bucket": "Technology Costs",
        "label": "Energy cost per kWh",
        "help": "Baseline energy price before regional override is applied.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 5.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "costs.consumables_per_unit",
        "bucket": "Technology Costs",
        "label": "Consumables per unit",
        "help": "Variable consumables cost per processed unit.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10000.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "perf.defect_rate_change_pct",
        "bucket": "Performance & Quality",
        "label": "Defect rate change",
        "help": "Negative values mean defect reduction versus the manual baseline.",
        "kind": "float",
        "min_value": -1.0,
        "max_value": 1.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "perf.scrap_cost_per_unit",
        "bucket": "Performance & Quality",
        "label": "Scrap cost per unit",
        "help": "Manual-state scrap cost per unit. Automation changes this through defect-rate assumptions.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 100000.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "perf.rework_cost_per_unit",
        "bucket": "Performance & Quality",
        "label": "Rework cost per unit",
        "help": "Manual-state rework cost per unit. Automation changes this through defect-rate assumptions.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 100000.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware", "software"],
    },
    {
        "path": "perf.throughput_change_pct",
        "bucket": "Performance & Quality",
        "label": "Throughput change",
        "help": "Capacity increase versus manual. Used only when revenue uplift is enabled.",
        "kind": "float",
        "min_value": -1.0,
        "max_value": 5.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["hardware"],
    },
    {
        "path": "deploy.ramp",
        "bucket": "Deployment & Rollout",
        "label": "Deployment ramp by year",
        "help": "Comma-separated cumulative deployed cells for each year, such as 4,8,12,16,20,24,28.",
        "kind": "int_list",
        "solutions": ["hardware"],
    },
    {
        "path": "software.capex_initial",
        "bucket": "Software Costs",
        "label": "Initial platform CAPEX",
        "help": "One-time year-1 software platform build cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 50000000.0,
        "step": 50000.0,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.integration_initial",
        "bucket": "Software Costs",
        "label": "Initial integration",
        "help": "One-time year-1 systems integration and data modeling cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 50000000.0,
        "step": 50000.0,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.software_per_year",
        "bucket": "Software Costs",
        "label": "Software per year",
        "help": "Annual license or subscription cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 50000000.0,
        "step": 10000.0,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.data_ops_per_year",
        "bucket": "Software Costs",
        "label": "Data ops per year",
        "help": "Annual data, MLOps, and engineering support cost.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 50000000.0,
        "step": 10000.0,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.defect_reduction_pct",
        "bucket": "Software Costs",
        "label": "Defect reduction",
        "help": "Reduction in scrap and rework cost versus the manual baseline.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.labor_efficiency_pct",
        "bucket": "Software Costs",
        "label": "Labor efficiency improvement",
        "help": "Currently stored for future use. The present cash flow logic does not directly monetize this parameter.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["software"],
    },
    {
        "path": "software.throughput_uplift_pct",
        "bucket": "Software Costs",
        "label": "Throughput uplift",
        "help": "Capacity increase used for revenue uplift when revenue uplift is enabled.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 5.0,
        "step": 0.01,
        "format": "%.2f",
        "solutions": ["software"],
    },
]

REGIONAL_PARAMETER_METADATA = [
    {
        "key": "wage_multiplier_vs_baseline",
        "label": "Regional wage multiplier",
        "help": "Multiplier applied to the selected country's baseline labor rate.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10.0,
        "step": 0.01,
        "format": "%.2f",
    },
    {
        "key": "energy_cost_per_kwh",
        "label": "Regional energy cost per kWh",
        "help": "Override for the selected country's energy price.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 5.0,
        "step": 0.01,
        "format": "%.2f",
    },
    {
        "key": "maintenance_multiplier",
        "label": "Regional maintenance multiplier",
        "help": "Multiplier applied to annual maintenance cost for the selected country.",
        "kind": "float",
        "min_value": 0.0,
        "max_value": 10.0,
        "step": 0.01,
        "format": "%.2f",
    },
]

def get_param_value(model: Model, path: str):
    if path == "include_revenue_uplift":
        return model.scenario.include_revenue_uplift
    return model._get_param(path)

def set_param_value(model: Model, path: str, value: Any):
    if path == "include_revenue_uplift":
        model.scenario.include_revenue_uplift = bool(value)
    else:
        model._set_param(path, value)

def format_ramp(values: List[int]) -> str:
    return ",".join(str(int(v)) for v in values)

def parse_ramp(text: str) -> List[int]:
    if not str(text).strip():
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]

def get_default_regional_inputs(country: Country) -> Dict[str, float]:
    defaults = COUNTRY_DEFAULTS[country]
    return {
        "wage_multiplier_vs_baseline": float(defaults.wage_multiplier_vs_baseline),
        "energy_cost_per_kwh": float(defaults.energy_cost_per_kwh if defaults.energy_cost_per_kwh is not None else 0.0),
        "maintenance_multiplier": float(defaults.maintenance_multiplier),
    }

def apply_country_overrides(model: Model, regional_inputs: Dict[str, float]):
    model.scenario.labor.wage_per_hour *= regional_inputs["wage_multiplier_vs_baseline"]
    model.scenario.costs.energy_cost_per_kwh = regional_inputs["energy_cost_per_kwh"]
    model.scenario.costs.maintenance_per_unit_per_year *= regional_inputs["maintenance_multiplier"]

def build_configured_model(base_model: Model, country: Country, solution: SolutionType, regional_inputs: Dict[str, float]) -> Model:
    model = deepcopy(base_model)
    model.scenario.country = country
    model.scenario.solution = solution
    apply_country_overrides(model, regional_inputs)
    model.apply_solution_preset()
    return model

def safe_for_display(v: Any) -> str:
    """Return a Streamlit/Arrow-safe display string for mixed metadata tables."""
    if v is None:
        return "Not set"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def build_parameter_table(model: Model, solution: SolutionType, regional_inputs: Dict[str, float]) -> pd.DataFrame:
    rows = []
    sol_key = solution.name.lower()
    for meta in PARAMETER_METADATA:
        if sol_key not in meta["solutions"]:
            continue
        rows.append(
            {
                "Bucket": meta["bucket"],
                "Parameter": meta["label"],
                "Path": meta["path"],
                "Value": safe_for_display(get_param_value(model, meta["path"])),
                "Description": meta["help"],
            }
        )
    rows += [
        {
            "Bucket": "Regional Assumptions",
            "Parameter": item["label"],
            "Path": f"_country.{item['key']}",
            "Value": safe_for_display(regional_inputs[item["key"]]),
            "Description": item["help"],
        }
        for item in REGIONAL_PARAMETER_METADATA
    ]
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df
    

def build_assumption_rationale_table(model: Model, solution: SolutionType) -> pd.DataFrame:
    sol_key = solution.name.lower()

    # BMW-scale software deployment assumptions
    num_plants = 12
    units_per_plant = model.scenario.demand.base_units_per_year / num_plants

    gpus_per_plant = 8
    central_gpus = 48
    total_gpus = (num_plants * gpus_per_plant) + central_gpus

    omniverse_cost_per_gpu_year = 4500
    estimated_omniverse_license_cost = total_gpus * omniverse_cost_per_gpu_year

    rationale_map = {
        "economics.discount_rate": (
            "Uses an 8% enterprise hurdle/WACC-style rate, appropriate for a large OEM capital allocation case."
        ),
        "economics.horizon_years": (
            "Seven years captures enterprise rollout, integration maturity, and recurring operational benefits."
        ),
        "economics.inflation_labor": (
            "3.5% labor inflation reflects rising skilled manufacturing labor costs."
        ),
        "economics.inflation_maintenance": (
            "2.5% maintenance inflation reflects moderate annual escalation in technical support costs."
        ),
        "economics.inflation_energy": (
            "4.0% energy inflation is conservative for industrial compute and manufacturing energy exposure."
        ),

        "include_revenue_uplift": (
            "Kept off by default so ROI is driven by cost savings, not speculative incremental revenue."
        ),
        "demand.base_units_per_year": (
            f"Models {model.scenario.demand.base_units_per_year:,.0f} units/year across {num_plants} plants, "
            f"or about {units_per_plant:,.0f} units per plant per year."
        ),
        "demand.annual_growth": (
            "2.0% annual growth is modest and avoids overstating volume-driven benefits."
        ),
        "demand.revenue_per_unit": (
            "Set to $0 because revenue uplift is disabled; throughput benefits are not monetized unless explicitly enabled."
        ),

        "labor.wage_per_hour": (
            "Uses fully loaded skilled automotive labor cost, including wage, benefits, and overhead burden."
        ),
        "labor.manual_hours_per_unit": (
            "Represents baseline manual planning, inspection, coordination, or disruption-response effort per unit."
        ),
        "labor.automated_supervision_hours_per_unit": (
            "Captures remaining human oversight after software-enabled automation; not assumed to be lights-out."
        ),
        "labor.automation_capture_pct": (
            "Models partial automation capture rather than full labor replacement, which is more defensible for software-led factory optimization."
        ),
        "labor.automated_wage_per_hour": (
            "Represents automation-side supervisory or technical operations labor."
        ),
        "labor.safety_incident_cost_delta_per_year": (
            "Annual safety benefit from reducing manual intervention, line disruptions, and high-risk operational tasks."
        ),

        "perf.scrap_cost_per_unit": (
            "Estimated unit-level cost of scrap or wasted production output affected by better simulation and process control."
        ),
        "perf.rework_cost_per_unit": (
            "Estimated unit-level cost of rework, quality correction, and production recovery."
        ),

        "software.capex_initial": (
            "One-time platform and infrastructure CAPEX for a 12-plant digital twin / simulation deployment, including compute, storage, networking, and enterprise setup."
        ),
        "software.integration_initial": (
            "Initial integration cost for connecting plant systems such as MES, SCADA, PLC data, CAD/BOM sources, and production data pipelines."
        ),
        "software.software_per_year": (
            f"Assumes {gpus_per_plant} GPUs per plant across {num_plants} plants plus {central_gpus} central GPUs "
            f"= {total_gpus} total GPUs. At ${omniverse_cost_per_gpu_year:,.0f}/GPU/year, estimated Omniverse-style "
            f"license cost is about ${estimated_omniverse_license_cost:,.0f}/year before enterprise discounts or support adders."
        ),
        "software.data_ops_per_year": (
            "Annual cost for data engineering, simulation model upkeep, MLOps, model validation, and factory-data pipeline maintenance."
        ),
        "software.defect_reduction_pct": (
            "Conservative quality improvement from better virtual commissioning, simulation, anomaly detection, and process optimization."
        ),
        "software.labor_efficiency_pct": (
            "Expected productivity gain from faster planning, fewer disruptions, better root-cause analysis, and reduced manual coordination."
        ),
        "software.throughput_uplift_pct": (
            "Conservative capacity uplift from improved bottleneck analysis, scheduling, and factory-flow optimization."
        ),
    }

    rows = []
    for meta in PARAMETER_METADATA:
        if sol_key not in meta["solutions"]:
            continue

        path = meta["path"]
        value = get_param_value(model, path)

        rows.append({
            "Parameter": meta["label"],
            "Value": safe_for_display(value),
            "Rationale": rationale_map.get(
                path,
                "Selected as a conservative starting assumption for the modeled BMW scenario."
            ),
        })

    return pd.DataFrame(rows)


def render_meta_widget(st, model: Model, meta: Dict[str, Any], key_prefix: str):
    current_value = get_param_value(model, meta["path"])
    widget_key = f"{key_prefix}__{meta['path']}"
    label = meta["label"]
    help_text = meta["help"]
    kind = meta["kind"]

    if kind == "bool":
        value = st.checkbox(label, value=bool(current_value), help=help_text, key=widget_key)
    elif kind == "int":
        safe_value = current_value
        if safe_value is None:
            safe_value = meta.get("default", meta.get("min_value", 0))
        value = st.number_input(
            label,
            # min_value=int(meta["min_value"]) if meta.get("min_value") is not None else None,
            # max_value=int(meta["max_value"]) if meta.get("max_value") is not None else None,
            value=int(safe_value),
            step=int(meta.get("step", 1)),
            help=help_text,
            key=widget_key,
        )
    elif kind == "float":
        safe_value = current_value
        if safe_value is None:
            safe_value = meta.get("default", meta.get("min_value", 0.0))
        value = st.number_input(
            label,
            # min_value=float(meta["min_value"]) if meta.get("min_value") is not None else None,
            # max_value=float(meta["max_value"]) if meta.get("max_value") is not None else None,
            value=float(safe_value),
            step=float(meta.get("step", 0.01)),
            format=meta.get("format", "%.4f"),
            help=help_text,
            key=widget_key,
        )
    elif kind == "int_list":
        value = st.text_input(
            label,
            value=format_ramp(current_value),
            help=help_text,
            key=widget_key,
        )
        try:
            value = parse_ramp(value)
        except Exception:
            st.warning(f"Could not parse '{label}'. Use comma-separated integers such as 4,8,12,16.")
            value = current_value
    else:
        value = current_value

    set_param_value(model, meta["path"], value)

def render_regional_widget(st, regional_inputs: Dict[str, float], meta: Dict[str, Any], key_prefix: str):
    widget_key = f"{key_prefix}__country__{meta['key']}"
    value = st.number_input(
        meta["label"],
        min_value=float(meta["min_value"]),
        max_value=float(meta["max_value"]),
        value=float(regional_inputs[meta["key"]]),
        step=float(meta.get("step", 0.01)),
        format=meta.get("format", "%.4f"),
        help=meta["help"],
        key=widget_key,
    )
    regional_inputs[meta["key"]] = value

def render_parameter_editor(
    st,
    model: Model,
    solution: SolutionType,
    regional_inputs: Dict[str, float],
    key_prefix: str,
    title: str,
):
    st.markdown(f"### {title}")
    sol_key = solution.name.lower()

    for bucket in PARAMETER_BUCKET_ORDER:
        bucket_items = [m for m in PARAMETER_METADATA if m["bucket"] == bucket and sol_key in m["solutions"]]
        if not bucket_items:
            continue
        with st.expander(bucket, expanded=False):
            st.caption(PARAMETER_BUCKET_GUIDES[bucket])
            for meta in bucket_items:
                render_meta_widget(st, model, meta, key_prefix)

    # with st.expander("Regional Assumptions", expanded=False):
    #     st.caption("These are editable run-level overrides for the currently selected country. They are applied after baseline assumptions and before solution presets.")
    #     for meta in REGIONAL_PARAMETER_METADATA:
    #         render_regional_widget(st, regional_inputs, meta, key_prefix)

def show_model_guide(st):
    with st.expander("How to use this model", expanded=False):
        st.markdown(
            """
            1. Choose the application area and country context in the sidebar.  
            2. The LLM converts that row into ROI model assumptions.  
            43. Review and edit the generated assumptions by category.  
            4. Click **Re-run economic model with edited inputs** to recalculate ROI without calling the LLM again.

            Notes:
            - The deployment ramp is cumulative deployed units by year.
            """
        )


def summarize_horizon_metrics(base_model: Model, country: Country, solution: SolutionType, regional_inputs: Dict[str, float], horizons: Tuple[int, ...] = (3, 5, 7)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        horizon_model = build_configured_model(base_model, country, solution, regional_inputs)
        horizon_model.scenario.economics.horizon_years = horizon
        res = horizon_model.run()
        sm = res.summary()
        rows.append(
            {
                "Horizon_Years": horizon,
                "NPV": sm["NPV"],
                "Payback_Year": sm["Payback_Year"],
                "ROI": sm["ROI"],
            }
        )
    return pd.DataFrame(rows)


def summarize_horizon_metrics_from_model(model: Model, horizons: Tuple[int, ...] = (3, 5, 7)) -> pd.DataFrame:
    """Compute 3/5/7-year ROI outputs from the currently edited model without rebuilding assumptions or calling the LLM."""
    rows = []
    for horizon in horizons:
        horizon_model = deepcopy(model)
        horizon_model.scenario.economics.horizon_years = horizon
        res = horizon_model.run()
        sm = res.summary()
        rows.append(
            {
                "Horizon_Years": horizon,
                "NPV": sm["NPV"],
                "Payback_Year": sm["Payback_Year"],
                "ROI": sm["ROI"],
            }
        )
    return pd.DataFrame(rows)


def collect_editable_model_inputs(model: Model, solution: SolutionType) -> Dict[str, Any]:
    """Collect the currently displayed/generated ROI inputs as dotted-path overrides."""
    sol_key = solution.name.lower()
    overrides: Dict[str, Any] = {}
    for meta in PARAMETER_METADATA:
        if sol_key in meta["solutions"]:
            overrides[meta["path"]] = get_param_value(model, meta["path"])
    return overrides


def idea_template_payload(idea_text: str, app: ApplicationType, country: Country, solution: Optional[SolutionType] = None) -> Dict[str, Any]:
    return {
        "idea_name": idea_text[:80] if idea_text else "New technology idea",
        "idea_description": idea_text or "Describe the technology concept, target workflow, and expected impact.",
        "application": app.name.lower(),
        "country": country.name.lower(),
        "preferred_solution": None if solution is None else solution.name.lower(),
        "labor_mode": {
            "mode": "augment",
            "automation_capture_pct": 0.50,
            "baseline_wage_per_hour": 42.0,
            "automated_wage_per_hour": 28.0,
        },
        "business_context": {
            "base_units_per_year": 1_200_000,
            "annual_growth": 0.015,
            "revenue_per_unit": 0.0,
        },
        "notes": [
            "mode can be replacement, augment, or skill_shift",
            "use automation_capture_pct below 1.0 for assistive technologies",
            "use a lower automated_wage_per_hour to model higher-skill to lower-skill handoff",
        ],
    }


def build_breakdown_prompt(idea_text: str, app: ApplicationType, country: Country) -> str:
    payload = json.dumps(idea_template_payload(idea_text, app, country), indent=2)
    return textwrap.dedent(
        f"""
        You are helping BMW evaluate a technology idea for an ROI model.
        Break the idea into the economic building blocks needed for modeling.
        The 'x' placeholders indicate a place where you need to substitute a reasonable numerical value.

        Return valid JSON only with this schema:
        {{
          "idea_summary": "...",
          "application": "{app.name.lower()}",
          "country": "{country.name.lower()}",
          "labor_mode": {{
            "mode": "replacement | augment | skill_shift",
            "reasoning": "...",
            "automation_capture_pct": x,
            "baseline_wage_per_hour": x,
            "automated_wage_per_hour": x
          }},
          "components": [
            {{"name": "hardware", "included": true, "description": "...", "cost_driver_type": "capex_per_unit | capex_initial | none"}},
            {{"name": "integration", "included": true, "description": "...", "cost_driver_type": "integration_per_unit | integration_initial | none"}},
            {{"name": "software", "included": true, "description": "...", "cost_driver_type": "software_per_year | none"}},
            {{"name": "data_ops", "included": false, "description": "...", "cost_driver_type": "data_ops_per_year | none"}},
            {{"name": "maintenance", "included": true, "description": "...", "cost_driver_type": "maintenance_per_unit_per_year | none"}},
            {{"name": "energy", "included": true, "description": "...", "cost_driver_type": "energy_kwh_per_unit | none"}}
          ],
          "expected_benefits": [
            {{"name": "labor_savings", "description": "..."}},
            {{"name": "quality_improvement", "description": "..."}},
            {{"name": "throughput_uplift", "description": "..."}},
            {{"name": "safety_delta", "description": "..."}}
          ],
          "confidence_notes": ["..."]
        }}

        User idea payload:
        {payload}
        """
    ).strip()


def build_input_generation_prompt(idea_text: str, app: ApplicationType, country: Country, solution: SolutionType) -> str:
    payload = json.dumps(idea_template_payload(idea_text, app, country, solution), indent=2)
    return textwrap.dedent(
        f"""
        You are generating starting-point ROI inputs for BMW's technology economics model.
        Use realistic but conservative assumptions, and specifically overestimate on CAPEX. Prefer transparent ranges over aggressive values. 
        Use a relatively large-scale for units; consider the size of a given BMW operation. Costs / CAPEX can never total $0.
        Fill in reasonable numerical values for the 'x' placeholders. Base units per year must be greater than 0.

        Return valid JSON only with this schema:
        {{
          "reasoning": "...",
          "scenario_overrides": {{
            "include_revenue_uplift": false,
            "demand.base_units_per_year": x,
            "demand.annual_growth": x,
            "demand.revenue_per_unit": x,
            "labor.wage_per_hour": x,
            "labor.manual_hours_per_unit": x,
            "labor.automated_supervision_hours_per_unit": x,
            "labor.automation_capture_pct": x,
            "labor.automated_wage_per_hour": x,
            "labor.safety_incident_cost_delta_per_year": x,
            "costs.capex_per_unit": x,
            "costs.install_commission_per_unit": x,
            "costs.integration_per_unit": x,
            "costs.maintenance_per_unit_per_year": x,
            "costs.energy_kwh_per_unit": x,
            "costs.consumables_per_unit": x,
            "perf.defect_rate_change_pct": x,
            "perf.throughput_change_pct": x,
            "perf.scrap_cost_per_unit": x,
            "perf.rework_cost_per_unit": x,
            "deploy.ramp": [1, 2, 4, 6, 8, 10, 12],
            "software.capex_initial": x,
            "software.integration_initial": x,
            "software.software_per_year": x,
            "software.data_ops_per_year": x,
            "software.defect_reduction_pct": x,
            "software.labor_efficiency_pct": x,
            "software.throughput_uplift_pct": x
          }},
          "assumption_notes": ["..."],
          "open_questions": ["..."]
        }}

        User idea payload:
        {payload}
        """
    ).strip()


def technology_row_payload(row: Dict[str, Any], app: ApplicationType, country: Country) -> Dict[str, Any]:
    return {
        "technology_name": row.get("technologyName"),
        "technology_description": row.get("technologyDescription"),
        "trl_level": row.get("trlLevel"),
        "job_title": row.get("jobTitle"),
        "task": row.get("task"),
        "capability": row.get("capability"),
        "idea": row.get("idea"),
        "fit_level": row.get("fitLevel"),
        "fit_rationale": row.get("fitRationale"),
        "custom_ratings": row.get("customRatings", []),
        "application": app.name,
        "country": country.name,
    }


def build_breakdown_prompt_from_row(row: Dict[str, Any], app: ApplicationType, country: Country) -> str:
    payload = json.dumps(technology_row_payload(row, app, country), indent=2)
    return textwrap.dedent(
        f"""
        You are an industrial automation strategist helping BMW convert a technology-matchmaking row into economic-model assumptions.

        Interpret the job, task, technology, capability, and idea. Classify whether the idea primarily replaces labor, augments a worker, or shifts work from a high-skill worker to a lower-skill/operator role. Be conservative and realistic for an automotive enterprise environment.

        Return ONLY valid JSON with this exact schema:
        {{
          "idea_summary": "string",
          "application": "LOGISTICS | PRODUCT | MANUFACTURING | OFFICE | SALES",
          "country": "{country.name}",
          "labor_mode": {{
            "mode": "replacement | augmentation | skill_shift",
            "reasoning": "string",
            "automation_capture_pct": x,
            "baseline_wage_per_hour": x,
            "automated_wage_per_hour": x
          }},
          "deployment_archetype": {{
            "type": "software_first | embodied_robotics | hybrid",
            "reasoning": "string"
          }},
          "benefit_map": {{
            "labor_savings": "low | medium | high",
            "quality_improvement": "low | medium | high",
            "throughput_uplift": "low | medium | high",
            "safety_improvement": "low | medium | high"
          }},
          "assumption_notes": ["string"],
          "open_questions": ["string"]
        }}

        Matchmaking row:
        {payload}
        """
    ).strip()


def build_input_generation_prompt_from_row(row: Dict[str, Any], breakdown: Dict[str, Any], app: ApplicationType, country: Country) -> str:
    payload = json.dumps(technology_row_payload(row, app, country), indent=2)
    breakdown_json = json.dumps(breakdown, indent=2)
    return textwrap.dedent(
        f"""
        You are generating starting-point inputs for BMW's ROI economics model from a technology-matchmaking row.

        Use the row and the prior economic breakdown to generate concrete numerical inputs. Be conservative and internally consistent. Use automotive/manufacturing defaults when uncertain. Use partial automation unless the idea clearly supports full replacement.

        Return ONLY valid JSON with this exact schema. Fill in reasonable values for the 'x' placeholders. Base units per year must be greater than 0.
        {{
          "reasoning": "string",
          "scenario_overrides": {{
            "include_revenue_uplift": false,
            "demand.base_units_per_year": x,
            "demand.annual_growth": x,
            "demand.revenue_per_unit": x,
            "labor.wage_per_hour": x,
            "labor.manual_hours_per_unit": x,
            "labor.automated_supervision_hours_per_unit": x,
            "labor.automation_capture_pct": x,
            "labor.automated_wage_per_hour": x,
            "labor.safety_incident_cost_delta_per_year": x,
            "costs.capex_per_unit": x,
            "costs.install_commission_per_unit": x,
            "costs.integration_per_unit": x,
            "costs.maintenance_per_unit_per_year": x,
            "costs.energy_kwh_per_unit": x,
            "costs.consumables_per_unit": x,
            "perf.defect_rate_change_pct": x,
            "perf.throughput_change_pct": x,
            "perf.scrap_cost_per_unit": x,
            "perf.rework_cost_per_unit": x,
            "deploy.ramp": [1, 2, 4, 6, 8, 10, 12],
            "software.capex_initial": x,
            "software.integration_initial": x,
            "software.software_per_year": x,
            "software.data_ops_per_year": x,
            "software.defect_reduction_pct": x,
            "software.labor_efficiency_pct": x,
            "software.throughput_uplift_pct": x
          }},
          "bucket_explanations": {{
            "demand": "string",
            "labor": "string",
            "costs": "string",
            "performance": "string",
            "deployment": "string"
          }},
          "assumption_notes": ["string"],
          "open_questions": ["string"]
        }}

        Matchmaking row:
        {payload}

        Economic breakdown:
        {breakdown_json}
        """
    ).strip()


def safe_json_loads(text_value: str) -> Dict[str, Any]:
    text_value = text_value.strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text_value = "\n".join(lines).strip()
    start_idx = text_value.find("{")
    end_idx = text_value.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text_value = text_value[start_idx:end_idx + 1]
    return json.loads(text_value)


def call_gemini_json(system_prompt: str, user_prompt: str, max_retries: int = 4) -> Dict[str, Any]:
    """Call Gemini using env vars only. Expected env: GEMINI_API_KEY or GOOGLE_API_KEY; optional GEMINI_MODEL."""
    import time
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

    primary_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    models_to_try = []
    for model_name in [primary_model, "gemini-2.5-flash-lite", "gemini-2.5-flash"]:
        if model_name and model_name not in models_to_try:
            models_to_try.append(model_name)

    prompt = f"""SYSTEM INSTRUCTION:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}\n\nReturn ONLY valid JSON."""
    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return safe_json_loads(response.text)
            except Exception as exc:
                last_error = exc
                err = str(exc)
                retryable = any(token in err for token in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])
                if not retryable or attempt == max_retries - 1:
                    break
                time.sleep(min(2 ** attempt, 15))
    raise last_error


def coerce_json_value(raw: Any) -> Any:
    if isinstance(raw, str):
        raw_strip = raw.strip()
        if raw_strip.startswith("[") and raw_strip.endswith("]"):
            try:
                return json.loads(raw_strip)
            except Exception:
                return raw
        if raw_strip.lower() in {"true", "false"}:
            return raw_strip.lower() == "true"
        try:
            if "." in raw_strip:
                return float(raw_strip)
            return int(raw_strip)
        except Exception:
            return raw
    return raw


def apply_override_dict(model: Model, overrides: Dict[str, Any]):
    for path, raw_value in overrides.items():
        value = coerce_json_value(raw_value)
        if path == "include_revenue_uplift":
            model.scenario.include_revenue_uplift = bool(value)
            continue
        try:
            model._set_param(path, value)
        except Exception:
            # Ignore unknown fields from imperfect LLM output; prompt docs should minimize these.
            pass


def build_model_for_solution(
    application: ApplicationType,
    country: Country,
    solution: SolutionType,
    regional_inputs: Dict[str, float],
) -> Model:
    model = Model.default(application)
    model.scenario.country = country
    model.scenario.solution = solution
    apply_country_overrides(model, regional_inputs)
    model.apply_solution_preset()
    return model



def infer_economic_model_type_from_row(row: Dict[str, Any], breakdown: Dict[str, Any]) -> SolutionType:
    """Map the selected matchmaking row to the current ROI model's internal cost archetype.

    The row is already the chosen technology/use case. This helper does not recommend a
    solution; it only chooses which existing cost structure should run under the hood.
    """
    deployment_type = str((breakdown.get("deployment_archetype") or {}).get("type", "")).lower()
    row_text = " ".join(
        str(row.get(key, ""))
        for key in ["technologyName", "technologyDescription", "capability", "idea", "task"]
    ).lower()

    embodied_terms = [
        "robot arm", "robotic arm", "humanoid", "mobile robot", "manipulator", "gripper",
        "pick", "place", "insert", "fasten", "assembly", "fixture", "motion", "trajectory",
        "fleet", "physical", "hardware", "sensor robot", "test fixture"
    ]

    if deployment_type == "embodied_robotics":
        return SolutionType.HARDWARE
    if deployment_type == "software_first":
        return SolutionType.SOFTWARE
    if any(term in row_text for term in embodied_terms):
        return SolutionType.HARDWARE
    return SolutionType.SOFTWARE
def build_model_with_generated_inputs(
    base_model: Model,
    country: Country,
    regional_inputs: Dict[str, float],
    solution: SolutionType,
    overrides: Dict[str, Any],
) -> Model:
    generated_model = build_model_for_solution(
        base_model.scenario.application,
        country,
        solution,
        regional_inputs,
    )
    generated_model.scenario.economics.discount_rate = base_model.scenario.economics.discount_rate
    generated_model.scenario.economics.horizon_years = base_model.scenario.economics.horizon_years
    apply_override_dict(generated_model, overrides)
    return generated_model


def evaluate_matchmaking_row(
    row_dict: Dict[str, Any],
    base_model: Model,
    country: Country,
    regional_inputs: Dict[str, float],
) -> Dict[str, Any]:
    """Shared backend pipeline for Streamlit, CLI testing, and future POST integration."""
    system_prompt = "You are a careful industrial economics analyst. Return valid JSON only. Use conservative BMW-relevant assumptions."
    breakdown_json = call_gemini_json(
        system_prompt=system_prompt,
        user_prompt=build_breakdown_prompt_from_row(row_dict, base_model.scenario.application, country),
    )
    generated_json = call_gemini_json(
        system_prompt=system_prompt,
        user_prompt=build_input_generation_prompt_from_row(row_dict, breakdown_json, base_model.scenario.application, country),
    )
    # The row is already the selected technology/use case. This only maps it to
    # the ROI model's existing internal cost archetype so the equations can run.
    economic_model_type = infer_economic_model_type_from_row(row_dict, breakdown_json)
    overrides = generated_json.get("scenario_overrides", {}) or {}
    generated_model = build_model_with_generated_inputs(base_model, country, regional_inputs, economic_model_type, overrides)
    current_results = generated_model.run()
    horizon_df = summarize_horizon_metrics_from_model(generated_model)
    return {
        "input_row": row_dict,
        "breakdown": breakdown_json,
        "generated_inputs": generated_json,
        "editor_key": str(abs(hash(json.dumps({"row": row_dict, "generated": generated_json}, sort_keys=True, default=str)))),
        "economic_model_type": economic_model_type.name,
        "regional_inputs": dict(regional_inputs),
        "current_horizon_summary": current_results.summary(),
        "horizon_outputs": horizon_df.to_dict(orient="records"),
        "yearly": current_results.yearly.to_dict(orient="records"),
    }


def render_generated_json_summary(st, generated: Dict[str, Any]):
    overrides = generated.get("scenario_overrides", {}) or {}
    rows = [{"Parameter": k, "Value": safe_for_display(v)} for k, v in overrides.items()]
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn(width="medium"),
                "Value": st.column_config.TextColumn(width="large"),
            },
        )
    explanations = generated.get("bucket_explanations", {}) or {}
    if explanations:
        st.markdown("**Why these assumptions were selected**")
        for bucket, explanation in explanations.items():
            st.markdown(f"- **{bucket.replace('_', ' ').title()}**: {explanation}")


def render_matchmaking_handoff(
    st,
    base_model: Model,
    country: Country,
    regional_inputs: Dict[str, float],
):
    st.header("Technology Matchmaking Input")

    row_id = st.query_params.get("row_id")

    if not row_id:
        st.warning("No row_id provided. Open this page from Job-tech match-making tool.")
        st.stop()

    row_state = get_row_state(row_id)

    if not row_state:
        st.warning(
            f"No saved ROI state found for row_id={row_id}. "
            "Run Get ROI from match-making tool first."
        )
        st.stop()

    row_dict = row_state.get("row", {})
    context = row_state.get("context", {})

    app_name = context.get("application", "MANUFACTURING")
    country_name = context.get("country", country.name)

    try:
        app = ApplicationType[app_name.upper()]
    except Exception:
        app = base_model.scenario.application

    try:
        country_enum = Country[country_name.upper()]
    except Exception:
        country_enum = country

    # Build base model from row context so the opened Streamlit page matches match-making tool's selected row.
    working_base_model = Model.default(app)
    working_base_model.scenario.country = country_enum
    working_base_model.scenario.economics.discount_rate = base_model.scenario.economics.discount_rate
    working_base_model.scenario.economics.horizon_years = base_model.scenario.economics.horizon_years
    working_base_model.scenario.economics.inflation_labor = base_model.scenario.economics.inflation_labor
    working_base_model.scenario.economics.inflation_maintenance = base_model.scenario.economics.inflation_maintenance
    working_base_model.scenario.economics.inflation_energy = base_model.scenario.economics.inflation_energy

    st.subheader("Selected Technology")
    st.write(f"Technology: {row_dict.get('technologyName')}")
    st.write(f"Job: {row_dict.get('jobTitle')}")
    st.write(f"Idea: {row_dict.get('idea')}")

    st.markdown("### Generated ROI Inputs")
    st.caption("Review or edit the generated assumptions below. Editing these values recalculates the model without calling the LLM again.")

    generated_inputs = row_state.get("generated_inputs", {}) or {}
    generated_overrides = (
        row_state.get("scenario_overrides")
        or generated_inputs.get("scenario_overrides", {})
        or {}
    )

    editable_regional_inputs = dict(
        row_state.get("regional_inputs")
        or get_default_regional_inputs(country_enum)
        or regional_inputs
    )

    solution_name = row_state.get(
        "economic_model_type",
        working_base_model.scenario.solution.name,
    )

    try:
        solution = SolutionType[solution_name]
    except Exception:
        solution = working_base_model.scenario.solution

    editable_model = build_model_with_generated_inputs(
        base_model=working_base_model,
        country=country_enum,
        regional_inputs=editable_regional_inputs,
        solution=solution,
        overrides=generated_overrides,
    )

    render_parameter_editor(
        st,
        editable_model,
        solution,
        editable_regional_inputs,
        key_prefix=f"generated_roi_inputs_editor_{row_id}",
        title="Editable generated assumptions",
    )

    explanations = generated_inputs.get("bucket_explanations", {}) or {}
    if explanations:
        with st.expander("Why these assumptions were selected", expanded=False):
            for bucket, explanation in explanations.items():
                st.markdown(f"- **{bucket.replace('_', ' ').title()}**: {explanation}")

            assumption_notes = get_assumption_notes(row_id)

            if assumption_notes:
                st.markdown("### Additional Assumptions & Context")

                for note in assumption_notes:
                    st.markdown(f"- {note}")

            st.caption(
                "Concise rationale for each current model input, including BMW-scale deployment logic and key cost drivers."
            )

            rationale_df = build_assumption_rationale_table(
                editable_model,
                solution
            )

            st.dataframe(
                rationale_df,
                use_container_width=True,
                hide_index=True,
            )

    # Recalculate current horizon directly from the edited model.
    current_results_for_explain = editable_model.run()
    render_metric_cards_with_explainers(st, editable_model, current_results_for_explain)

    st.markdown("### 3-, 5-, and 7-Year Outputs")

    # Recalculate horizon table from edited values, not from stale saved output.
    horizon_df = summarize_horizon_metrics_from_model(editable_model)

    horizon_outputs = horizon_df.to_dict(orient="records")
    scenario_overrides = collect_editable_model_inputs(editable_model, solution)

    # if st.button("Update ROI results for Nik's tool", type="primary"):
    #     update_row_state(
    #         row_id,
    #         {
    #             "status": "edited",
    #             "scenario_overrides": scenario_overrides,
    #             "regional_inputs": dict(editable_regional_inputs),
    #             "horizon_outputs": horizon_outputs,
    #             "economic_model_type": solution.name,
    #             "generated_inputs": {
    #                 **generated_inputs,
    #                 "scenario_overrides": scenario_overrides,
    #             },
    #         },
    #     )
    #     st.success("Updated ROI results for Match-making tool.")

    if not horizon_df.empty:
        st.dataframe(
            horizon_df.style.format(
                {
                    "NPV": "{:,.0f}",
                    "ROI": lambda x: "—" if pd.isna(x) else f"{x * 100:,.1f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No horizon outputs available yet.")

    horizon_outputs = horizon_df.to_dict(orient="records")
    scenario_overrides = collect_editable_model_inputs(editable_model, solution)

    updated_row_state = update_row_state(
        row_id,
        {
            "status": "edited7",
            "scenario_overrides": scenario_overrides,
            "regional_inputs": dict(editable_regional_inputs),
            "horizon_outputs": horizon_outputs,
            "economic_model_type": solution.name,
            "generated_inputs": {
                **generated_inputs,
                "scenario_overrides": scenario_overrides,
            },
        },
    )


def _fmt_money(value: Any) -> str:
    """Human-readable currency formatting for UI explanations."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    sign = "-" if float(value) < 0 else ""
    return f"{sign}${abs(float(value)):,.0f}"


def _fmt_pct(value: Any) -> str:
    """Format a decimal ROI/rate as a percent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value) * 100:,.1f}%"


def _fmt_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):,.0f}"


def _discounted_terms_table(results: Results, discount_rate: float) -> pd.DataFrame:
    """Build an explainability table for NPV terms."""
    df = results.yearly.copy()
    rows = []
    for idx, row in df.iterrows():
        # Model npv() enumerates from t=0, so Year 1 is discounted by (1+r)^0.
        t = int(idx)
        year = int(row["Year"])
        cashflow = float(row["Net_Savings"])
        discount_factor = (1 + discount_rate) ** t
        discounted = cashflow / discount_factor
        rows.append(
            {
                "Year": year,
                "t used in NPV": t,
                "Net Savings": cashflow,
                "Discount Factor": discount_factor,
                "Discounted Term": discounted,
                "Calculation": f"{cashflow:,.0f} / (1 + {discount_rate:.2%})^{t} = {discounted:,.0f}",
            }
        )
    return pd.DataFrame(rows)


def _payback_trace_table(results: Results) -> pd.DataFrame:
    df = results.yearly.copy()
    df["Cumulative Net Savings"] = df["Net_Savings"].cumsum()
    return df[["Year", "Net_Savings", "Cumulative Net Savings"]]


def render_metric_cards_with_explainers(st, model: Model, results: Results):
    """
    Show headline ROI metrics with a directly attached calculation breakdown under each number.
    This keeps the dashboard clean while making extreme values like 6000% ROI auditable.
    """
    sm = results.summary()
    df = results.yearly.copy()
    rate = float(model.scenario.economics.discount_rate)
    horizon = int(model.scenario.economics.horizon_years)

    total_capex = float(sm.get("Total_CAPEX", df["CAPEX_new"].sum()))
    total_manual = float(sm.get("Total_Manual_OPEX", df["Manual_Total"].sum()))
    total_auto = float(sm.get("Total_Auto_OPEX", df["Auto_OPEX"].sum()))
    total_savings = float(sm.get("Total_Savings_vs_Manual", total_manual - total_auto))
    total_revenue_uplift = float(sm.get("Total_Revenue_Uplift", df.get("Revenue_Uplift", pd.Series([0.0])).sum()))
    total_project_cost = float(sm.get("Total_Project_Cost", total_auto + total_capex))
    net_benefit_after_all_costs = float(
        sm.get(
            "Net_Benefit_After_All_Costs",
            total_manual + total_revenue_uplift - total_project_cost,
        )
    )
    roi = sm.get("ROI", np.nan)
    npv_value = float(sm.get("NPV", 0.0))
    payback = sm.get("Payback_Year")

    st.markdown("### Current Horizon Output")
    st.caption("Expand any metric to see the formula and the exact numbers used in the calculation.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("NPV", _fmt_money(npv_value))
        with st.expander("Explain NPV"):
            st.markdown(
                f"""
                **Formula**

                `NPV = Σ Net_Savings_t / (1 + discount_rate)^t`

                **Variables**

                - `discount_rate = {rate:.2%}`
                - `Net_Savings_t = Manual_Total_t - Auto_OPEX_t + Revenue_Uplift_t - CAPEX_new_t`
                - Current horizon = `{horizon}` years

                **Substituted calculation**
                """
            )
            terms = _discounted_terms_table(results, rate)
            st.dataframe(
                terms.style.format(
                    {
                        "Net Savings": "{:,.0f}",
                        "Discount Factor": "{:,.3f}",
                        "Discounted Term": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown(f"**NPV = sum of discounted terms = {_fmt_money(npv_value)}**")

    with col2:
        st.metric("ROI", _fmt_pct(roi))
        with st.expander("Explain ROI"):
            st.markdown(
                f"""
            **Formula**

            `ROI = Net_Benefit_After_All_Costs / Total_Project_Cost`

            **Variables**

            - `Total_Project_Cost = Total_Auto_OPEX + Total_CAPEX`
            - `Net_Benefit_After_All_Costs = Total_Manual_OPEX + Total_Revenue_Uplift - Total_Project_Cost`

            **Substituted calculation**

            `Total_Project_Cost = {_fmt_money(total_auto)} + {_fmt_money(total_capex)} = {_fmt_money(total_project_cost)}`

            `Net_Benefit_After_All_Costs = {_fmt_money(total_manual)} + {_fmt_money(total_revenue_uplift)} - {_fmt_money(total_project_cost)} = {_fmt_money(net_benefit_after_all_costs)}`

            `ROI = {_fmt_money(net_benefit_after_all_costs)} / {_fmt_money(total_project_cost)} = {_fmt_pct(roi)}`
            """
            )
            if total_project_cost > 0 and not pd.isna(roi) and roi > 5:
                st.info(
                    "Large ROI percentages usually occur when avoided manual cost is very large relative to the total automation project cost. "
                    "Compare Total_Project_Cost against Total_Manual_OPEX to sanity-check the result."
                )
            elif total_project_cost <= 0:
                st.warning("ROI is undefined or not meaningful when Total_Project_Cost is zero.")

    with col3:
        st.metric("Payback", payback if payback else "No payback")
        with st.expander("Explain payback"):
            st.markdown(
                """
                **Formula / logic**

                `Payback_Year = first year where cumulative Net_Savings >= 0`

                Because yearly `Net_Savings` already subtracts that year's new CAPEX, payback is based on the cumulative net cash-flow trace below.
                """
            )
            trace = _payback_trace_table(results)
            st.dataframe(
                trace.style.format(
                    {
                        "Net_Savings": "{:,.0f}",
                        "Cumulative Net Savings": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            if payback:
                st.markdown(f"**Payback occurs in Year {payback}, when cumulative net savings first become non-negative.**")
            else:
                st.markdown("**No payback occurs within the modeled horizon.**")

    with col4:
        st.metric("Savings vs. Manual", _fmt_money(total_savings))
        with st.expander("Explain savings"):
            st.markdown(
                f"""
                **Formula**

                `Total_Savings_vs_Manual = Total_Manual_OPEX - Total_Auto_OPEX`

                **Substituted calculation**

                `Total_Savings_vs_Manual = {_fmt_money(total_manual)} - {_fmt_money(total_auto)} = {_fmt_money(total_savings)}`

                This number measures operating-cost reduction before subtracting total CAPEX. ROI then subtracts CAPEX and divides by CAPEX.
                """
            )
            yearly = df[["Year", "Manual_Total", "Auto_OPEX", "CAPEX_new", "Revenue_Uplift", "Net_Savings"]].copy()
            yearly["Manual - Auto OPEX"] = yearly["Manual_Total"] - yearly["Auto_OPEX"]
            st.dataframe(
                yearly[["Year", "Manual_Total", "Auto_OPEX", "Manual - Auto OPEX", "CAPEX_new", "Revenue_Uplift", "Net_Savings"]].style.format(
                    {
                        "Manual_Total": "{:,.0f}",
                        "Auto_OPEX": "{:,.0f}",
                        "Manual - Auto OPEX": "{:,.0f}",
                        "CAPEX_new": "{:,.0f}",
                        "Revenue_Uplift": "{:,.0f}",
                        "Net_Savings": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

def compact_horizon_outputs_from_model(model: Model) -> list:
    horizon_df = summarize_horizon_metrics_from_model(model)

    outputs = []
    for _, row in horizon_df.iterrows():
        outputs.append(
            {
                "Horizon_Years": int(row["Horizon_Years"]),
                "NPV": float(row["NPV"]),
                "ROI": None if pd.isna(row["ROI"]) else float(row["ROI"]),
                "Payback_Year": None if pd.isna(row["Payback_Year"]) else int(row["Payback_Year"]),
            }
        )

    return outputs


def get_roi_for_selected_row(
    row_dict: dict,
    row_id: str,
    application: str = "MANUFACTURING",
    country: str = "GERMANY",
    streamlit_base_url: str = "http://localhost:8501",
    force_regenerate: bool = False,
) -> dict:
    """
    Entry point for Matchmaking tool.

    Behavior:
    - If row_id already has saved assumptions, reuse them.
    - If not, call the LLM once to generate assumptions.
    - Calculate 3-, 5-, and 7-year outputs.
    - Save all row-specific state needed by the Streamlit model page.
    - Return compact JSON for Matchmaking table.
    """

    existing = get_row_state(row_id)
    streamlit_url = f"{streamlit_base_url}/?row_id={row_id}"

    app = ApplicationType[application.upper()]
    country_enum = Country[country.upper()]

    base_model = Model.default(app)
    base_model.scenario.country = country_enum

    regional_inputs = (
        existing.get("regional_inputs")
        if existing and existing.get("regional_inputs")
        else get_default_regional_inputs(country_enum)
    )

    generated_inputs = {}
    scenario_overrides = {}
    solution = base_model.scenario.solution

    should_reuse_existing = (
        existing is not None
        and existing.get("scenario_overrides")
        and not force_regenerate
    )

    if should_reuse_existing:
        scenario_overrides = existing.get("scenario_overrides", {}) or {}
        generated_inputs = existing.get("generated_inputs", {}) or {
            "scenario_overrides": scenario_overrides
        }

        solution_name = existing.get("economic_model_type", base_model.scenario.solution.name)
        try:
            solution = SolutionType[solution_name]
        except Exception:
            solution = base_model.scenario.solution

    else:
        output = evaluate_matchmaking_row(
            row_dict=row_dict,
            base_model=base_model,
            country=country_enum,
            regional_inputs=regional_inputs,
        )

        generated_inputs = output.get("generated_inputs", {}) or {}
        scenario_overrides = generated_inputs.get("scenario_overrides", {}) or {}

        solution_name = output.get("economic_model_type", base_model.scenario.solution.name)
        try:
            solution = SolutionType[solution_name]
        except Exception:
            solution = base_model.scenario.solution

    generated_model = build_model_with_generated_inputs(
        base_model=base_model,
        country=country_enum,
        regional_inputs=regional_inputs,
        solution=solution,
        overrides=scenario_overrides,
    )

    horizon_outputs = compact_horizon_outputs_from_model(generated_model)

    row_state_payload = {
        "row": row_dict,
        "context": {
            "application": application,
            "country": country,
        },
        "generated_inputs": generated_inputs,
        "scenario_overrides": scenario_overrides,
        "regional_inputs": dict(regional_inputs),
        "economic_model_type": solution.name,
        "horizon_outputs": horizon_outputs,
        "streamlit_url": streamlit_url,
        "status": "generated" if not should_reuse_existing else existing.get("status", "generated"),
    }

    if existing:
        update_row_state(row_id, row_state_payload)
    else:
        save_row_state(
            row_id=row_id,
            row=row_dict,
            context=row_state_payload["context"],
            scenario_overrides=scenario_overrides,
            horizon_outputs=horizon_outputs,
            status=row_state_payload["status"],
            streamlit_url=streamlit_url,
        )
        update_row_state(
            row_id,
            {
                "generated_inputs": generated_inputs,
                "regional_inputs": dict(regional_inputs),
                "economic_model_type": solution.name,
            },
        )

    return {
        "status": "success",
        "technologyName": row_dict.get("technologyName"),
        "jobTitle": row_dict.get("jobTitle"),
        "row_id": row_id,
        "streamlit_url": streamlit_url,
        "horizon_outputs": horizon_outputs,
    }


def run_streamlit():

    st.set_page_config(page_title="BMW Robotics ROI", layout="wide")
    st.title("BMW Robotics ROI & Cost Degression Model")
    st.caption("ROI model for a selected technology-matchmaking solution.")

    st.sidebar.header("Scenario Context")
    app_choice = st.sidebar.selectbox("Application", [e.name.title() for e in ApplicationType])
    country_choice = st.sidebar.selectbox("Country", [c.name.title() for c in Country])

    st.sidebar.header("Finance")
    horizon_years = st.sidebar.slider("Current horizon years", 3, 15, 7)
    discount_rate = st.sidebar.slider("Discount rate", 0.02, 0.20, 0.10, 0.01)
    include_uplift = st.sidebar.checkbox(
        "Include revenue uplift if generated",
        value=False,
        help="Only enable when extra throughput can realistically be monetized.",
    )

    app = ApplicationType[app_choice.upper()]
    country = Country[country_choice.upper()]

    state_key = f"bmw_roi_single_row_base_{app.name}_{country.name}"
    region_key = f"bmw_roi_single_row_region_{country.name}"

    if state_key not in st.session_state:
        st.session_state[state_key] = Model.default(app)
    if region_key not in st.session_state:
        st.session_state[region_key] = get_default_regional_inputs(country)

    base_model = deepcopy(st.session_state[state_key])
    base_model.scenario.application = app
    base_model.scenario.country = country
    base_model.scenario.economics.horizon_years = int(horizon_years)
    base_model.scenario.economics.discount_rate = float(discount_rate)
    base_model.scenario.include_revenue_uplift = bool(include_uplift)

    regional_inputs = dict(st.session_state[region_key])

    show_model_guide(st)

    st.session_state[state_key] = deepcopy(base_model)
    st.session_state[region_key] = dict(regional_inputs)

    render_matchmaking_handoff(st, base_model, country, regional_inputs)
    output = st.session_state.get("latest_matchmaking_output")
    if output:
        yearly_df = pd.DataFrame(output.get("yearly", []))
        if not yearly_df.empty:
            st.subheader("Cashflows & Savings")
            try:
                import altair as alt

                chart_df = yearly_df.copy()
                chart_df["Cumulative_Net_Savings"] = chart_df["Net_Savings"].cumsum()
                base = alt.Chart(chart_df).encode(x="Year:O")
                manual_line = base.mark_line().encode(
                    y=alt.Y("Manual_Total:Q", title="$/Year"),
                    tooltip=["Year", "Manual_Total"],
                )
                auto_line = base.mark_line().encode(
                    y="Auto_OPEX:Q",
                    tooltip=["Year", "Auto_OPEX"],
                )
                capex_bar = base.mark_bar(opacity=0.25).encode(
                    y="CAPEX_new:Q",
                    tooltip=["Year", "CAPEX_new"],
                )
                st.altair_chart((manual_line + auto_line + capex_bar).properties(height=300), width="stretch")

                cum = (
                    alt.Chart(chart_df)
                    .mark_line(point=True)
                    .encode(
                        x="Year:O",
                        y=alt.Y("Cumulative_Net_Savings:Q", title="Cumulative Net Savings ($)"),
                        tooltip=["Year", "Cumulative_Net_Savings"],
                    )
                    .properties(height=250)
                )
                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
                st.altair_chart(cum + zero, width="stretch")
            except Exception:
                st.info("Altair not available — charts skipped.")

            with st.expander("Yearly Breakdown", expanded=False):
                st.dataframe(
                    yearly_df.style.format(
                        {
                            "DemandUnits": "{:.0f}",
                            "CAPEX_new": "{:,.0f}",
                            "Manual_Total": "{:,.0f}",
                            "Auto_OPEX": "{:,.0f}",
                            "Net_Savings": "{:,.0f}",
                        }
                    ),
                    width="stretch",
                )


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    import sys

    run_streamlit()
 