"""
Spain Government System Simulation Engine
==========================================
Core simulation loop and state management.
All model coefficients are loaded from an external weights JSON file.
This keeps the "physics" separate from the government system logic.
"""

import copy
import json
import random
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# WEIGHTS
# ---------------------------------------------------------------------------

class Weights:
    """Container for all model coefficients loaded from a JSON file."""

    def __init__(self, path: str = None):
        if path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, "weights", "default.json")
        with open(path, encoding="utf-8") as f:
            self._data = json.load(f)
        self.path = path

    def get(self, dotpath: str, default=None):
        parts = dotpath.split(".")
        node = self._data
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                return default
        return node

    def __getitem__(self, dotpath: str):
        v = self.get(dotpath)
        if v is None:
            raise KeyError(dotpath)
        return v

    def set(self, dotpath: str, value):
        parts = dotpath.split(".")
        node = self._data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value

    def save(self, path: str = None):
        path = path or self.path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def clone(self, new_path: str) -> "Weights":
        import copy as _copy
        w = Weights.__new__(Weights)
        w._data = _copy.deepcopy(self._data)
        w.path = new_path
        w.save(new_path)
        return w

    def flat(self) -> Dict[str, float]:
        out = {}
        def _walk(prefix, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k.startswith("_"):
                        continue
                    _walk(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(node, (int, float)):
                out[prefix] = node
        _walk("", self._data)
        return out

    @property
    def data(self):
        return self._data


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

@dataclass
class EconomicState:
    gdp_billion_eur: float = 1_462.0
    gdp_growth_rate: float = 0.035
    gdp_per_capita_eur: float = 30_800.0
    inflation_rate: float = 0.028
    unemployment_rate: float = 0.113
    youth_unemployment_rate: float = 0.265
    public_debt_pct_gdp: float = 1.04
    budget_deficit_pct_gdp: float = -0.032
    tax_revenue_pct_gdp: float = 0.392
    gov_spending_pct_gdp: float = 0.455
    healthcare_spending_pct_gdp: float = 0.067
    education_spending_pct_gdp: float = 0.046
    defense_spending_pct_gdp: float = 0.014
    rd_spending_pct_gdp: float = 0.014
    social_protection_pct_gdp: float = 0.17
    infrastructure_spending_pct_gdp: float = 0.02
    exports_pct_gdp: float = 0.38
    imports_pct_gdp: float = 0.35
    tourism_revenue_pct_gdp: float = 0.12
    fdi_inflow_billion_eur: float = 35.0
    labor_force_participation: float = 0.746
    avg_annual_wage_eur: float = 28_400.0
    minimum_wage_monthly_eur: float = 1_134.0
    informal_economy_pct: float = 0.20
    corporate_tax_rate: float = 0.25
    top_income_tax_rate: float = 0.47
    vat_rate: float = 0.21
    interest_rate: float = 0.0315
    housing_price_index: float = 100.0
    home_ownership_rate: float = 0.75

@dataclass
class DemographicState:
    population_million: float = 48.0
    population_growth_rate: float = 0.01
    fertility_rate: float = 1.19
    life_expectancy: float = 84.0
    median_age: float = 46.8
    pct_under_15: float = 0.14
    pct_15_64: float = 0.65
    pct_over_65: float = 0.21
    dependency_ratio: float = 0.54
    net_migration_per_1000: float = 5.0
    urbanization_rate: float = 0.81

@dataclass
class GovernanceState:
    rule_of_law: float = 70.0
    corruption_control: float = 65.0
    government_effectiveness: float = 72.0
    regulatory_quality: float = 70.0
    voice_accountability: float = 78.0
    political_stability: float = 55.0
    judicial_independence: float = 62.0
    bureaucratic_efficiency: float = 60.0
    decentralization_level: float = 75.0

@dataclass
class SocialState:
    life_satisfaction: float = 6.3
    trust_in_government: float = 0.30
    trust_in_judiciary: float = 0.35
    interpersonal_trust: float = 0.38
    gini_coefficient: float = 0.33
    poverty_rate: float = 0.205
    social_mobility_index: float = 55.0
    voter_turnout: float = 0.70
    crime_rate_per_100k: float = 45.0
    housing_affordability: float = 40.0
    work_life_balance: float = 6.0
    education_quality: float = 65.0
    healthcare_quality: float = 78.0
    environmental_quality: float = 68.0
    mental_health_index: float = 62.0
    regional_separatism_tension: float = 45.0

@dataclass
class SimulationState:
    year: int = 2024
    economy: EconomicState = field(default_factory=EconomicState)
    demographics: DemographicState = field(default_factory=DemographicState)
    governance: GovernanceState = field(default_factory=GovernanceState)
    social: SocialState = field(default_factory=SocialState)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "economy": asdict(self.economy),
            "demographics": asdict(self.demographics),
            "governance": asdict(self.governance),
            "social": asdict(self.social),
        }


# ---------------------------------------------------------------------------
# PARTIES
# ---------------------------------------------------------------------------

@dataclass
class Party:
    name: str
    abbreviation: str
    ideology_score: float
    base_vote_share: float
    tax_pressure_mult: float = 1.0
    gov_spending_mult: float = 1.0
    healthcare_spending_mult: float = 1.0
    education_spending_mult: float = 1.0
    defense_spending_mult: float = 1.0
    rd_spending_mult: float = 1.0
    social_protection_mult: float = 1.0
    infrastructure_mult: float = 1.0
    labor_regulation_mult: float = 1.0
    business_deregulation_mult: float = 1.0
    environmental_regulation_mult: float = 1.0
    immigration_openness_mult: float = 1.0
    decentralization_mult: float = 1.0
    corruption_effort_mult: float = 1.0
    gdp_growth_tendency: float = 0.0
    inequality_tendency: float = 0.0
    def __repr__(self):
        return f"{self.abbreviation} ({self.name})"

def create_spanish_parties() -> List[Party]:
    return [
        Party("Partido Socialista Obrero Español","PSOE",-2.5,0.317,1.02,1.02,1.03,1.03,1.01,1.02,1.03,1.01,1.03,0.98,1.03,1.02,1.02,1.01,0.001,-0.003),
        Party("Partido Popular","PP",3.0,0.331,0.97,0.98,1.0,0.99,1.02,1.01,0.98,1.02,0.97,1.04,0.98,0.99,0.98,1.01,0.002,0.002),
        Party("Vox","VOX",7.0,0.122,0.95,0.95,0.98,0.97,1.06,0.98,0.96,1.01,0.95,1.06,0.93,0.90,0.90,1.0,0.001,0.004),
        Party("Sumar","SUMAR",-5.5,0.123,1.04,1.04,1.04,1.05,0.98,1.04,1.05,1.02,1.05,0.94,1.06,1.04,1.03,1.03,-0.001,-0.005),
        Party("Esquerra Republicana de Catalunya","ERC",-3.0,0.027,1.02,1.02,1.02,1.03,0.98,1.02,1.02,1.01,1.02,0.99,1.02,1.01,1.08,1.02,0.0,-0.002),
        Party("Junts per Catalunya","JUNTS",1.0,0.023,0.99,1.0,1.01,1.01,0.99,1.02,1.0,1.02,1.0,1.02,1.01,1.0,1.10,1.01,0.001,0.0),
        Party("Partido Nacionalista Vasco","PNV",1.5,0.017,1.0,1.01,1.02,1.02,1.0,1.02,1.01,1.02,1.01,1.01,1.01,1.0,1.06,1.02,0.001,-0.001),
        Party("EH Bildu","BILDU",-6.0,0.030,1.04,1.03,1.04,1.04,0.95,1.03,1.04,1.02,1.05,0.95,1.05,1.03,1.08,1.03,-0.001,-0.004),
    ]


# ---------------------------------------------------------------------------
# GOVERNMENT SYSTEM INTERFACE
# ---------------------------------------------------------------------------

@dataclass
class GovernmentAction:
    tax_pressure_mult: float = 1.0
    gov_spending_mult: float = 1.0
    healthcare_spending_mult: float = 1.0
    education_spending_mult: float = 1.0
    defense_spending_mult: float = 1.0
    rd_spending_mult: float = 1.0
    social_protection_mult: float = 1.0
    infrastructure_mult: float = 1.0
    labor_regulation_mult: float = 1.0
    business_deregulation_mult: float = 1.0
    environmental_regulation_mult: float = 1.0
    immigration_openness_mult: float = 1.0
    decentralization_mult: float = 1.0
    corruption_effort_mult: float = 1.0
    gdp_growth_tendency: float = 0.0
    inequality_tendency: float = 0.0
    governing_parties: List[str] = field(default_factory=list)
    coalition_stability: float = 1.0
    election_held: bool = False
    description: str = ""

class GovernmentSystem(ABC):
    @abstractmethod
    def initialize(self, state: SimulationState, parties: List[Party]): pass
    @abstractmethod
    def step(self, state: SimulationState) -> GovernmentAction: pass
    @abstractmethod
    def get_status(self) -> dict: pass


# ---------------------------------------------------------------------------
# ECONOMIC MODEL
# ---------------------------------------------------------------------------

def apply_economic_model(state, action, rng, w):
    eco, demo, gov, soc = state.economy, state.demographics, state.governance, state.social
    effectiveness = action.coalition_stability * (gov.government_effectiveness / 100.0)
    g = "economy.gdp."
    base_growth = w[g+"base_growth"]
    momentum = eco.gdp_growth_rate * w[g+"momentum_factor"]
    policy_growth = action.gdp_growth_tendency * effectiveness
    business_boost = (action.business_deregulation_mult - 1.0) * w[g+"business_dereg_boost"] * effectiveness
    infra_boost = (action.infrastructure_mult - 1.0) * w[g+"infra_boost"] * effectiveness
    rd_boost = (eco.rd_spending_pct_gdp - w[g+"rd_baseline"]) * w[g+"rd_boost"]
    demo_effect = (demo.pct_15_64 - w[g+"demo_working_age_baseline"]) * w[g+"demo_working_age_effect"]
    debt_drag = max(0, (eco.public_debt_pct_gdp - w[g+"debt_drag_threshold"])) * w[g+"debt_drag_coeff"]
    tourism_effect = (eco.tourism_revenue_pct_gdp - w[g+"tourism_baseline"]) * w[g+"tourism_effect"]
    shock = rng.gauss(0, w[g+"shock_std"])
    regulation_drag = (action.labor_regulation_mult - 1.0) * w[g+"regulation_drag"]
    mean_reversion = (base_growth - eco.gdp_growth_rate) * w[g+"mean_reversion_speed"]
    new_growth = (base_growth * w[g+"base_growth_weight"] + momentum + mean_reversion +
                  policy_growth + business_boost + infra_boost + rd_boost +
                  demo_effect + debt_drag + tourism_effect + shock + regulation_drag)
    eco.gdp_growth_rate = max(w[g+"growth_min"], min(w[g+"growth_max"], new_growth))
    # GDP level grows nominally: real growth + inflation
    # (The target data — spain_2024.json etc — uses nominal GDP in current euros,
    #  so the simulation must compound both real growth and inflation.)
    nominal_growth = eco.gdp_growth_rate + eco.inflation_rate
    eco.gdp_billion_eur *= (1 + nominal_growth)
    eco.gdp_per_capita_eur = (eco.gdp_billion_eur * 1e9) / (demo.population_million * 1e6)

    i = "economy.inflation."
    spending_pressure = (action.gov_spending_mult - 1.0) * w[i+"spending_pressure_coeff"]
    demand_pull = max(0, eco.gdp_growth_rate - w[i+"demand_pull_threshold"]) * w[i+"demand_pull_coeff"]
    monetary = (eco.interest_rate - w[i+"monetary_baseline"]) * w[i+"monetary_coeff"]
    # Inflation has inertia — it doesn't reset to base_rate each year.
    # Real inflation is sticky: expectations, wage contracts, indexation.
    # The model blends the base rate with the previous year's inflation.
    inertia = w.get(i+"inertia", 0.5)
    target_inflation = w[i+"base_rate"] + spending_pressure + demand_pull + monetary
    new_inflation = inertia * eco.inflation_rate + (1 - inertia) * target_inflation + rng.gauss(0, w[i+"shock_std"])
    eco.inflation_rate = max(w[i+"min"], min(w[i+"max"], new_inflation))

    u = "economy.unemployment."
    su = "economy.structural_unemployment."
    # --- UNEMPLOYMENT: Structural floor + cyclical dynamics ---
    # Spain has a high structural unemployment that slowly declines with reform
    struct_floor_initial = w.get(su+"initial_floor", 0.18)
    floor_decay = w.get(su+"floor_decay_per_year", 0.004)
    floor_min = w.get(su+"floor_min", 0.06)
    # Labor deregulation actively reduces the structural floor
    reform_push = max(0, action.business_deregulation_mult - 1.0) * w.get(su+"labor_reform_effect", 0.3)
    # Years since start: use year to approximate structural improvement
    # The floor decays over time as institutions modernize
    years_elapsed = max(0, state.year - 1994)
    structural_floor = max(floor_min,
        struct_floor_initial - floor_decay * years_elapsed - reform_push * 0.01)

    # Cyclical component: Okun's law around the structural floor
    okun_effect = (eco.gdp_growth_rate - w[u+"okun_trend"]) * w[u+"okun_coeff"]
    reg_effect = (action.labor_regulation_mult - 1.0) * w[u+"labor_reg_effect"]
    dereg_effect = (action.business_deregulation_mult - 1.0) * w[u+"business_dereg_effect"]
    imm_effect = (action.immigration_openness_mult - 1.0) * w[u+"immigration_effect"]

    # Boom absorption: strong growth can push unemployment below structural floor temporarily
    boom_absorption = w.get(su+"boom_absorption_rate", 0.5)
    cyclical_change = okun_effect + reg_effect + dereg_effect + imm_effect + rng.gauss(0, w[u+"shock_std"])
    new_unemployment = eco.unemployment_rate + cyclical_change

    # Hysteresis: if unemployment was high, it tends to stay high (sticky)
    hysteresis = w.get(su+"hysteresis_coeff", 0.1)
    # Pull toward structural floor if below it (market forces), but allow booms to push below
    if new_unemployment < structural_floor:
        new_unemployment += (structural_floor - new_unemployment) * (1 - boom_absorption) * 0.1
    else:
        # Above floor: hysteresis makes it sticky, slow to come down
        new_unemployment += (structural_floor - new_unemployment) * hysteresis

    eco.unemployment_rate = max(w[u+"min"], min(w[u+"max"], new_unemployment))
    eco.youth_unemployment_rate = min(w[u+"youth_max"], max(w[u+"youth_min"],
        eco.unemployment_rate * w[u+"youth_multiplier"] + rng.gauss(0, w[u+"youth_shock_std"])))

    gf = "economy.government_finances."
    eco.tax_revenue_pct_gdp *= action.tax_pressure_mult
    eco.tax_revenue_pct_gdp += eco.gdp_growth_rate * w[gf+"tax_auto_stabilizer"]
    eco.tax_revenue_pct_gdp = max(w[gf+"tax_min"], min(w[gf+"tax_max"], eco.tax_revenue_pct_gdp))
    eco.healthcare_spending_pct_gdp *= action.healthcare_spending_mult
    eco.education_spending_pct_gdp *= action.education_spending_mult
    eco.defense_spending_pct_gdp *= action.defense_spending_mult
    eco.rd_spending_pct_gdp *= action.rd_spending_mult
    eco.social_protection_pct_gdp *= action.social_protection_mult
    eco.infrastructure_spending_pct_gdp *= action.infrastructure_mult
    pension_pressure = max(0, demo.pct_over_65 - w[gf+"pension_pressure_threshold"]) * w[gf+"pension_pressure_coeff"]
    eco.social_protection_pct_gdp += pension_pressure
    auto_stab = max(0, eco.unemployment_rate - w[gf+"unemployment_auto_stabilizer_threshold"]) * w[gf+"unemployment_auto_stabilizer_coeff"]
    interest_spending = eco.public_debt_pct_gdp * eco.interest_rate
    component = (eco.healthcare_spending_pct_gdp + eco.education_spending_pct_gdp +
                 eco.defense_spending_pct_gdp + eco.rd_spending_pct_gdp +
                 eco.social_protection_pct_gdp + eco.infrastructure_spending_pct_gdp)
    eco.gov_spending_pct_gdp = component + w[gf+"other_spending"] + interest_spending + auto_stab
    if eco.public_debt_pct_gdp > w[gf+"austerity_threshold"]:
        austerity = (eco.public_debt_pct_gdp - w[gf+"austerity_threshold"]) * w[gf+"austerity_spending_coeff"]
        eco.gov_spending_pct_gdp -= austerity
        eco.tax_revenue_pct_gdp += austerity * w[gf+"austerity_tax_coeff"]
    eco.gov_spending_pct_gdp = max(w[gf+"spending_min"], min(w[gf+"spending_max"], eco.gov_spending_pct_gdp))
    eco.budget_deficit_pct_gdp = eco.tax_revenue_pct_gdp - eco.gov_spending_pct_gdp
    nominal_growth = eco.gdp_growth_rate + eco.inflation_rate
    primary_deficit = max(0, -eco.budget_deficit_pct_gdp)
    eco.public_debt_pct_gdp = (eco.public_debt_pct_gdp + primary_deficit) / max(0.9, 1 + nominal_growth)
    if eco.budget_deficit_pct_gdp > 0:
        eco.public_debt_pct_gdp -= eco.budget_deficit_pct_gdp * w[gf+"surplus_debt_reduction"]
    eco.public_debt_pct_gdp = max(w[gf+"debt_min"], min(w[gf+"debt_max"], eco.public_debt_pct_gdp))

    wg = "economy.wages."
    wage_growth = eco.gdp_growth_rate * w[wg+"gdp_passthrough"] + eco.inflation_rate * w[wg+"inflation_passthrough"]
    eco.avg_annual_wage_eur *= (1 + wage_growth)
    eco.minimum_wage_monthly_eur *= (1 + max(wage_growth, eco.inflation_rate * w[wg+"min_wage_inflation_floor"]))

    h = "economy.housing."
    dp = max(0, eco.gdp_growth_rate) * w[h+"demand_pressure_coeff"]
    ip = max(0, demo.net_migration_per_1000) * w[h+"immigration_pressure_coeff"]
    re = (eco.interest_rate - w[h+"interest_rate_baseline"]) * w[h+"interest_rate_effect"]
    eco.housing_price_index += dp + ip + re + rng.gauss(0, w[h+"shock_std"])
    eco.housing_price_index = max(w[h+"min"], eco.housing_price_index)

    t = "economy.tourism."
    env_eff = (action.environmental_regulation_mult - 1.0) * w[t+"env_regulation_effect"]
    stab_eff = (gov.political_stability - w[t+"stability_baseline"]) * w[t+"stability_effect_coeff"]
    eco.tourism_revenue_pct_gdp += env_eff + stab_eff + rng.gauss(0, w[t+"shock_std"])
    eco.tourism_revenue_pct_gdp = max(w[t+"min"], min(w[t+"max"], eco.tourism_revenue_pct_gdp))

    ir = "economy.interest_rate."
    # --- INTEREST RATE: Period-aware regime model ---
    # The ECB (or Bank of Spain pre-1999) targets different rates in different eras.
    # The rate converges toward the regime target, with inflation deviations on top.
    regimes = w.get("economy.interest_rate.regimes", [])
    if not regimes:
        # Historical fallback: actual ECB/BoS rate regimes for Spain
        # These capture the major monetary policy eras that shaped Spain's economy.
        regimes = [
            {"start_year": 1994, "end_year": 1998, "target_rate": 0.06},   # Pre-Euro: BoS convergence
            {"start_year": 1999, "end_year": 2001, "target_rate": 0.035},  # Early Euro
            {"start_year": 2002, "end_year": 2005, "target_rate": 0.02},   # Low rates era
            {"start_year": 2006, "end_year": 2008, "target_rate": 0.04},   # Pre-crisis tightening
            {"start_year": 2009, "end_year": 2015, "target_rate": 0.005},  # Crisis: near-zero rates
            {"start_year": 2016, "end_year": 2021, "target_rate": 0.0},    # Negative/zero rate era
            {"start_year": 2022, "end_year": 2024, "target_rate": 0.04},   # Post-COVID tightening
            {"start_year": 2025, "end_year": 2100, "target_rate": 0.03},   # Projected neutral rate
        ]
    regime_target = w.get(ir+"inflation_target", 0.02)  # fallback
    for regime in regimes:
        if regime["start_year"] <= state.year <= regime["end_year"]:
            regime_target = regime["target_rate"]
            break
    convergence = w.get(ir+"regime_convergence_speed", 0.3)
    eco.interest_rate += (regime_target - eco.interest_rate) * convergence
    eco.interest_rate += (eco.inflation_rate - w[ir+"inflation_target"]) * w[ir+"inflation_response"] * 0.3
    eco.interest_rate = max(w[ir+"min"], min(w[ir+"max"], eco.interest_rate))


def apply_demographic_model(state, action, rng, w):
    demo, eco = state.demographics, state.economy
    f = "demographics.fertility."
    # Immigration-driven fertility boost: immigrants tend to have higher fertility
    imm_fertility = max(0, demo.net_migration_per_1000) * w.get(f+"immigration_fertility_boost", 0.008)
    demo.fertility_rate += ((state.social.housing_affordability - w[f+"affordability_baseline"]) * w[f+"affordability_coeff"] +
                            (eco.social_protection_pct_gdp - w[f+"social_protection_baseline"]) * w[f+"social_protection_coeff"] +
                            (eco.unemployment_rate - w[f+"unemployment_baseline"]) * w[f+"unemployment_coeff"] +
                            imm_fertility +
                            rng.gauss(0, w[f+"shock_std"]))
    demo.fertility_rate = max(w[f+"min"], min(w[f+"max"], demo.fertility_rate))

    le = "demographics.life_expectancy."
    demo.life_expectancy += ((eco.healthcare_spending_pct_gdp - w[le+"healthcare_baseline"]) * w[le+"healthcare_coeff"] +
                             eco.gdp_growth_rate * w[le+"wealth_coeff"] + rng.gauss(0, w[le+"shock_std"]))
    demo.life_expectancy = max(w[le+"min"], min(w[le+"max"], demo.life_expectancy))

    m = "demographics.migration."
    # Migration is driven by economic attractiveness, policy openness, and era-specific patterns.
    # Spain had near-zero migration pre-2000, massive inflows 2000-2008, negative 2009-2013,
    # and rising again 2014+. The GDP-per-capita effect captures the main economic pull.
    pull = (1 - eco.unemployment_rate) * w[m+"pull_factor_coeff"]
    gdp_pull = (eco.gdp_per_capita_eur / max(1.0, w.get(m+"gdp_pull_baseline", 20000.0))) * w.get(m+"gdp_pull_coeff", 2.0)
    openness = (action.immigration_openness_mult - w[m+"openness_baseline"]) * w[m+"openness_coeff"]
    demo.net_migration_per_1000 = max(w[m+"min"], min(w[m+"max"], pull + gdp_pull + openness + rng.gauss(0, w[m+"shock_std"])))

    p = "demographics.population."
    natural = (demo.fertility_rate / w[p+"natural_growth_replacement"] - 1) * w[p+"natural_growth_coeff"]
    mig = demo.net_migration_per_1000 / 1000
    demo.population_growth_rate = natural + mig
    demo.population_million *= (1 + demo.population_growth_rate)

    a = "demographics.age_structure."
    ar = w[a+"aging_rate"]
    fo = (demo.fertility_rate - w[a+"fertility_offset_baseline"]) * w[a+"fertility_offset_coeff"]
    my = mig * w[a+"migration_youth_factor"]
    demo.pct_over_65 += ar - fo + rng.gauss(0, w[a+"over65_shock_std"])
    demo.pct_under_15 += fo + my * w[a+"migration_youth_under15"] + rng.gauss(0, w[a+"under15_shock_std"])
    demo.pct_over_65 = max(w[a+"over65_min"], min(w[a+"over65_max"], demo.pct_over_65))
    demo.pct_under_15 = max(w[a+"under15_min"], min(w[a+"under15_max"], demo.pct_under_15))
    demo.pct_15_64 = 1.0 - demo.pct_over_65 - demo.pct_under_15
    demo.dependency_ratio = (demo.pct_under_15 + demo.pct_over_65) / max(0.01, demo.pct_15_64)
    demo.median_age += ar * w[a+"median_age_aging_scale"] - my * w[a+"median_age_migration_scale"]


def apply_social_model(state, action, rng, w):
    soc, eco, gov, demo = state.social, state.economy, state.governance, state.demographics

    iq = "social.inequality."
    soc.gini_coefficient += action.inequality_tendency
    soc.gini_coefficient += ((eco.unemployment_rate - w[iq+"unemployment_baseline"]) * w[iq+"unemployment_coeff"] +
                             (eco.tax_revenue_pct_gdp - w[iq+"tax_baseline"]) * w[iq+"tax_coeff"] + rng.gauss(0, w[iq+"shock_std"]))
    soc.gini_coefficient = max(w[iq+"min"], min(w[iq+"max"], soc.gini_coefficient))

    pv = "social.poverty."
    soc.poverty_rate = w[pv+"base"] + soc.gini_coefficient * w[pv+"gini_coeff"] + eco.unemployment_rate * w[pv+"unemployment_coeff"] + rng.gauss(0, w[pv+"shock_std"])
    soc.poverty_rate = max(w[pv+"min"], min(w[pv+"max"], soc.poverty_rate))

    ls = "social.life_satisfaction."
    inc = min(8, max(2, 5 + (eco.gdp_per_capita_eur - w[ls+"income_baseline"]) / w[ls+"income_scale"]))
    emp = min(8, max(2, 8 - eco.unemployment_rate * w[ls+"employment_scale"]))
    hth = min(9, max(3, eco.healthcare_spending_pct_gdp * w[ls+"health_scale"]))
    hou = min(8, max(2, soc.housing_affordability / w[ls+"housing_scale"]))
    saf = min(9, max(3, 9 - soc.crime_rate_per_100k / w[ls+"safety_scale"]))
    equ = min(8, max(2, 8 - soc.gini_coefficient * w[ls+"equality_scale"]))
    gvn = min(8, max(2, gov.government_effectiveness / w[ls+"governance_scale"]))
    target = (inc*w[ls+"income_weight"] + emp*w[ls+"employment_weight"] + hth*w[ls+"health_weight"] +
              hou*w[ls+"housing_weight"] + saf*w[ls+"safety_weight"] + equ*w[ls+"equality_weight"] + gvn*w[ls+"governance_weight"])
    soc.life_satisfaction += (target - soc.life_satisfaction) * w[ls+"adaptation_speed"] + rng.gauss(0, w[ls+"shock_std"])
    soc.life_satisfaction = max(w[ls+"min"], min(w[ls+"max"], soc.life_satisfaction))

    tr = "social.trust."
    soc.trust_in_government += ((gov.government_effectiveness - w[tr+"effectiveness_baseline"]) * w[tr+"effectiveness_coeff"] +
        eco.gdp_growth_rate * w[tr+"economy_growth_coeff"] - max(0, eco.unemployment_rate - w[tr+"economy_unemployment_baseline"]) * abs(w[tr+"economy_unemployment_coeff"]) +
        (gov.corruption_control - w[tr+"corruption_baseline"]) * w[tr+"corruption_coeff"] +
        (action.coalition_stability - w[tr+"stability_baseline"]) * w[tr+"stability_coeff"] +
        w[tr+"polarization_drag"] + rng.gauss(0, w[tr+"shock_std"]))
    soc.trust_in_government = max(w[tr+"min"], min(w[tr+"max"], soc.trust_in_government))

    ha = "social.housing_affordability."
    soc.housing_affordability = max(w[ha+"min"], min(w[ha+"max"],
        w[ha+"base"] + (eco.housing_price_index - w[ha+"price_baseline"]) * w[ha+"price_coeff"] +
        (eco.avg_annual_wage_eur - w[ha+"wage_baseline"]) / w[ha+"wage_coeff_divisor"] + rng.gauss(0, w[ha+"shock_std"])))

    ed = "social.education_quality."
    soc.education_quality += (eco.education_spending_pct_gdp - w[ed+"spending_baseline"]) * w[ed+"spending_coeff"] * w[ed+"change_rate"] + rng.gauss(0, w[ed+"shock_std"])
    soc.education_quality = max(w[ed+"min"], min(w[ed+"max"], soc.education_quality))

    hc = "social.healthcare_quality."
    soc.healthcare_quality += (((eco.healthcare_spending_pct_gdp - w[hc+"spending_baseline"]) * w[hc+"spending_coeff"] +
        (demo.pct_over_65 - w[hc+"aging_baseline"]) * w[hc+"aging_coeff"]) * w[hc+"change_rate"] + rng.gauss(0, w[hc+"shock_std"]))
    soc.healthcare_quality = max(w[hc+"min"], min(w[hc+"max"], soc.healthcare_quality))

    rt = "social.regional_tensions."
    soc.regional_separatism_tension += (action.decentralization_mult - 1.0) * w[rt+"decentralization_coeff"] + rng.gauss(0, w[rt+"shock_std"])
    soc.regional_separatism_tension = max(w[rt+"min"], min(w[rt+"max"], soc.regional_separatism_tension))

    vt = "social.voter_turnout."
    soc.voter_turnout += ((soc.trust_in_government - w[vt+"trust_baseline"]) * w[vt+"trust_coeff"] +
        (soc.life_satisfaction - w[vt+"satisfaction_baseline"]) * w[vt+"satisfaction_coeff"] +
        max(0, eco.unemployment_rate - w[vt+"crisis_threshold"]) * w[vt+"crisis_coeff"] + rng.gauss(0, w[vt+"shock_std"]))
    soc.voter_turnout = max(w[vt+"min"], min(w[vt+"max"], soc.voter_turnout))

    gi = "social.governance_indices."
    gov.government_effectiveness += (action.corruption_effort_mult - 1.0) * w[gi+"effectiveness_policy_coeff"] + rng.gauss(0, w[gi+"effectiveness_shock_std"])
    gov.corruption_control += (action.corruption_effort_mult - 1.0) * w[gi+"corruption_policy_coeff"] + rng.gauss(0, w[gi+"corruption_shock_std"])
    gov.political_stability += (action.coalition_stability - w[gi+"stability_coalition_baseline"]) * w[gi+"stability_coalition_coeff"] + rng.gauss(0, w[gi+"stability_shock_std"])
    gov.regulatory_quality += (action.business_deregulation_mult - 1.0) * w[gi+"regulatory_dereg_coeff"] + rng.gauss(0, w[gi+"regulatory_shock_std"])

    ev = "social.environment."
    soc.environmental_quality += ((action.environmental_regulation_mult - 1.0) * w[ev+"regulation_coeff"] +
        eco.gdp_growth_rate * w[ev+"growth_pollution_coeff"] + rng.gauss(0, w[ev+"shock_std"]))
    soc.environmental_quality = max(w[ev+"min"], min(w[ev+"max"], soc.environmental_quality))

    mh = "social.mental_health."
    soc.mental_health_index = max(w[mh+"min"], min(w[mh+"max"],
        (1 - eco.unemployment_rate) * w[mh+"employment_coeff"] + (1 - soc.gini_coefficient) * w[mh+"inequality_coeff"] +
        soc.housing_affordability * w[mh+"housing_coeff"] + rng.gauss(0, w[mh+"shock_std"])))

    cr = "social.crime."
    soc.crime_rate_per_100k = max(w[cr+"min"], min(w[cr+"max"],
        eco.unemployment_rate * w[cr+"unemployment_coeff"] + soc.gini_coefficient * w[cr+"inequality_coeff"] -
        gov.rule_of_law * w[cr+"policing_coeff"] + rng.gauss(0, w[cr+"shock_std"])))

    for attr in ['rule_of_law','corruption_control','government_effectiveness',
                 'regulatory_quality','voice_accountability','political_stability',
                 'judicial_independence','bureaucratic_efficiency']:
        setattr(gov, attr, max(w[gi+"index_min"], min(w[gi+"index_max"], getattr(gov, attr))))


# ---------------------------------------------------------------------------
# JSON LOADING + ENGINE
# ---------------------------------------------------------------------------

def load_state_from_json(path: str) -> SimulationState:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SimulationState(
        year=data.get("start_year", 2024),
        economy=EconomicState(**{k: v for k, v in data.get("economy", {}).items()}),
        demographics=DemographicState(**{k: v for k, v in data.get("demographics", {}).items()}),
        governance=GovernanceState(**{k: v for k, v in data.get("governance", {}).items()}),
        social=SocialState(**{k: v for k, v in data.get("social", {}).items()}),
    )

class SimulationEngine:
    def __init__(self, government_system, seed=42, initial_state=None, weights=None):
        self.rng = random.Random(seed)
        self.state = initial_state if initial_state else SimulationState()
        self.parties = create_spanish_parties()
        self.government_system = government_system
        self.government_system.initialize(self.state, self.parties)
        self.weights = weights if weights else Weights()
        self.history: List[dict] = []
        self._record_state()

    def _record_state(self):
        record = self.state.to_dict()
        record["government"] = self.government_system.get_status()
        self.history.append(record)

    def step(self):
        self.state.year += 1
        action = self.government_system.step(self.state)
        apply_economic_model(self.state, action, self.rng, self.weights)
        apply_demographic_model(self.state, action, self.rng, self.weights)
        apply_social_model(self.state, action, self.rng, self.weights)
        self._record_state()
        return self.state

    def run(self, years):
        for _ in range(years):
            self.step()
        return self.history

    def get_summary(self):
        s = self.state
        return {"year": s.year, "gdp_growth": f"{s.economy.gdp_growth_rate:.1%}",
                "unemployment": f"{s.economy.unemployment_rate:.1%}",
                "debt_to_gdp": f"{s.economy.public_debt_pct_gdp:.0%}",
                "inflation": f"{s.economy.inflation_rate:.1%}",
                "life_satisfaction": f"{s.social.life_satisfaction:.1f}/10",
                "gini": f"{s.social.gini_coefficient:.3f}",
                "trust_in_gov": f"{s.social.trust_in_government:.0%}",
                "population_m": f"{s.demographics.population_million:.1f}M",
                "fertility": f"{s.demographics.fertility_rate:.2f}",
                "pct_over_65": f"{s.demographics.pct_over_65:.1%}"}