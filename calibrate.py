"""
Spain Simulation — Multi-Strategy Calibrator (CMA-ES + CoordDescent + Perturbation)
====================================================================================
Combines three complementary optimization strategies to break through
plateaus and find globally good weight configurations:

  1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
     Population-based, derivative-free optimizer that adapts its search
     distribution. Gold standard for noisy black-box optimization with
     50-200 parameters. Handles the non-differentiable simulation perfectly.

  2. Per-variable gradient descent
     Uses the known causal structure (VAR_TO_WEIGHTS mapping) to make
     targeted adjustments based on signed errors. Fast local improvement
     where causal links are known.

  3. Stochastic perturbation (finite differences)
     Estimates gradients by evaluating random perturbations. Catches
     improvements that CMA-ES's population hasn't explored and that
     coord descent misses due to cross-variable interactions.

Each epoch runs all three strategies and keeps the best result.

Features:
  - Full checkpoint/resume: saves after every epoch
  - Graceful termination via SIGTERM/SIGINT (safe for SSH)
  - Endless mode: --epochs 0
  - Adaptive step sizes that DON'T decay to zero
  - Population diversity prevents premature convergence
  - No PyTorch dependency (pure numpy) — runs anywhere

Usage:
    python calibrate.py
    python calibrate.py --epochs 0                  # endless
    python calibrate.py --max-seeds 200 --pop-size 20
    python calibrate.py --reset                     # fresh start

On your server:
    nohup python calibrate.py --epochs 0 > calibrate.log 2>&1 &
    tail -f calibrate.log
    kill $(cat calibrate.pid)                       # graceful stop
"""

import argparse, copy, json, math, os, signal, sys, time
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import SimulationEngine, Weights, load_state_from_json
from spanish_parliamentary import SpanishParliamentarySystem


# ---------------------------------------------------------------------------
# GRACEFUL SHUTDOWN HANDLER
# ---------------------------------------------------------------------------

class GracefulShutdown:
    """
    Handles SIGTERM/SIGINT for clean shutdown. The training loop checks
    should_stop after each epoch, saves everything, and exits.

    Writes a PID file so you can `kill $(cat calibrate.pid)` from SSH.
    """
    def __init__(self, pid_path: str = None):
        self.should_stop = False
        self._signal_received = None
        self.pid_path = pid_path

        if self.pid_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.pid_path)), exist_ok=True)
            with open(self.pid_path, "w") as f:
                f.write(str(os.getpid()))

        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        if not self.should_stop:
            print(f"\n  *** Received {sig_name} — will save and exit after current epoch ***")
            print(f"  *** Send again to force-quit (NOT recommended) ***\n", flush=True)
            self.should_stop = True
            self._signal_received = sig_name
        else:
            print(f"\n  *** Second {sig_name} — force quitting ***\n")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        if self.pid_path and os.path.isfile(self.pid_path):
            os.remove(self.pid_path)


# ---------------------------------------------------------------------------
# CMA-ES OPTIMIZER
# ---------------------------------------------------------------------------

class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy.

    Maintains a multivariate normal search distribution and adapts both
    the mean (step-size) and covariance (search direction) based on
    which candidates in the population performed best.

    This is the right optimizer for noisy, non-differentiable, moderate-
    dimensional (10-500 params) problems like simulation calibration.
    """
    def __init__(self, x0: np.ndarray, sigma0: float = 0.3, pop_size: int = None):
        self.n = len(x0)
        self.mean = x0.copy().astype(np.float64)
        self.sigma = sigma0

        # Population size: default heuristic from Hansen (2006)
        self.lam = pop_size or (4 + int(3 * np.log(self.n)))
        self.mu = self.lam // 2

        # Recombination weights (log-weighted)
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))

        # Covariance adaptation
        self.cc = (4 + self.mu_eff/self.n) / (self.n + 4 + 2*self.mu_eff/self.n)
        self.c1 = 2 / ((self.n + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2*(self.mu_eff - 2 + 1/self.mu_eff) / ((self.n+2)**2 + self.mu_eff))

        # State
        self.ps = np.zeros(self.n)
        self.pc = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.invsqrtC = np.eye(self.n)
        self.eigenvalues = np.ones(self.n)
        self.B = np.eye(self.n)
        self._eigen_countdown = 0

        self.gen = 0

    def ask(self) -> List[np.ndarray]:
        """Generate population of candidate solutions."""
        self._update_eigen()
        candidates = []
        for _ in range(self.lam):
            z = np.random.randn(self.n)
            x = self.mean + self.sigma * (self.B @ (self.eigenvalues * z))
            candidates.append(x)
        return candidates

    def tell(self, candidates: List[np.ndarray], fitnesses: List[float]):
        """Update distribution based on evaluated candidates."""
        idx = np.argsort(fitnesses)
        candidates = [candidates[i] for i in idx]

        old_mean = self.mean.copy()
        self.mean = np.zeros(self.n)
        for i in range(self.mu):
            self.mean += self.weights[i] * candidates[i]

        step = (self.mean - old_mean) / self.sigma
        self.ps = ((1 - self.cs) * self.ps +
                   np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.invsqrtC @ step))

        hs = (np.linalg.norm(self.ps) /
              np.sqrt(1 - (1-self.cs)**(2*(self.gen+1))) / self.chi_n
              < 1.4 + 2/(self.n+1))

        self.pc = ((1 - self.cc) * self.pc +
                   hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * step)

        artmp = np.zeros((self.mu, self.n))
        for i in range(self.mu):
            artmp[i] = (candidates[i] - old_mean) / self.sigma

        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.n, self.n))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(artmp[i], artmp[i])

        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (rank_one + (1-hs) * self.cc * (2-self.cc) * self.C) +
                  self.cmu * rank_mu)

        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = max(1e-8, min(2.0, self.sigma))

        self.gen += 1
        self._eigen_countdown -= 1

    def _update_eigen(self):
        if self._eigen_countdown <= 0:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D, self.B = np.linalg.eigh(self.C)
            D = np.maximum(D, 1e-20)
            self.eigenvalues = np.sqrt(D)
            self.invsqrtC = self.B @ np.diag(1.0 / self.eigenvalues) @ self.B.T
            self._eigen_countdown = max(1, int(self.lam / (self.c1 + self.cmu) / self.n / 10))

    def get_state(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "ps": self.ps.tolist(),
            "pc": self.pc.tolist(),
            "C": self.C.tolist(),
            "gen": self.gen,
            "lam": self.lam,
            "mu": self.mu,
        }

    def set_state(self, state: dict):
        self.mean = np.array(state["mean"])
        self.sigma = state["sigma"]
        self.ps = np.array(state["ps"])
        self.pc = np.array(state["pc"])
        self.C = np.array(state["C"])
        self.gen = state["gen"]
        self._eigen_countdown = 0


# ---------------------------------------------------------------------------
# CALIBRATION VARIABLES
# ---------------------------------------------------------------------------

CALIB_VARS = {
    "economy.gdp_billion_eur":         {"w": 3.0},
    "economy.gdp_growth_rate":         {"w": 2.0, "pct": True},
    "economy.gdp_per_capita_eur":      {"w": 2.0},
    "economy.unemployment_rate":       {"w": 3.0, "pct": True},
    "economy.youth_unemployment_rate": {"w": 1.5, "pct": True},
    "economy.inflation_rate":          {"w": 2.0, "pct": True},
    "economy.public_debt_pct_gdp":     {"w": 2.5, "pct": True},
    "economy.tax_revenue_pct_gdp":     {"w": 1.5, "pct": True},
    "economy.gov_spending_pct_gdp":    {"w": 1.5, "pct": True},
    "economy.avg_annual_wage_eur":     {"w": 2.0},
    "economy.interest_rate":           {"w": 1.0, "pct": True},
    "economy.housing_price_index":     {"w": 1.5},
    "demographics.population_million": {"w": 2.5},
    "demographics.fertility_rate":     {"w": 1.5},
    "demographics.life_expectancy":    {"w": 1.5},
    "demographics.pct_over_65":        {"w": 1.5, "pct": True},
    "demographics.net_migration_per_1000": {"w": 1.0},
    "social.life_satisfaction":        {"w": 1.0},
    "social.gini_coefficient":         {"w": 1.5},
    "social.poverty_rate":             {"w": 1.0, "pct": True},
    "social.housing_affordability":    {"w": 1.0},
    "social.education_quality":        {"w": 1.0},
    "social.healthcare_quality":       {"w": 1.0},
    "governance.government_effectiveness": {"w": 1.0},
    "governance.corruption_control":   {"w": 0.5},
    "governance.political_stability":  {"w": 0.5},
}

CALIB_VAR_LIST = list(CALIB_VARS.keys())
CALIB_WEIGHTS = np.array([CALIB_VARS[k]["w"] for k in CALIB_VAR_LIST])

VAR_TO_WEIGHTS = {
    "economy.gdp_billion_eur": [("economy.gdp.base_growth",+1),("economy.gdp.base_growth_weight",+1)],
    "economy.gdp_growth_rate": [("economy.gdp.base_growth",+1),("economy.gdp.mean_reversion_speed",+1)],
    "economy.gdp_per_capita_eur": [("economy.gdp.base_growth",+1)],
    "economy.unemployment_rate": [("economy.structural_unemployment.initial_floor",+1),("economy.structural_unemployment.floor_decay_per_year",-1),("economy.structural_unemployment.hysteresis_coeff",+1),("economy.structural_unemployment.boom_absorption_rate",-1),("economy.unemployment.okun_coeff",-1)],
    "economy.youth_unemployment_rate": [("economy.unemployment.youth_multiplier",+1)],
    "economy.inflation_rate": [("economy.inflation.base_rate",+1),("economy.inflation.demand_pull_coeff",+1),("economy.inflation.inertia",+1)],
    "economy.public_debt_pct_gdp": [("economy.government_finances.surplus_debt_reduction",-1),("economy.government_finances.other_spending",+1),("economy.government_finances.austerity_spending_coeff",-1)],
    "economy.tax_revenue_pct_gdp": [("economy.government_finances.tax_auto_stabilizer",+1)],
    "economy.gov_spending_pct_gdp": [("economy.government_finances.other_spending",+1),("economy.government_finances.unemployment_auto_stabilizer_coeff",+1)],
    "economy.avg_annual_wage_eur": [("economy.wages.gdp_passthrough",+1),("economy.wages.inflation_passthrough",+1)],
    "economy.interest_rate": [("economy.interest_rate.regime_convergence_speed",+1),("economy.interest_rate.inflation_response",+1)],
    "economy.housing_price_index": [("economy.housing.demand_pressure_coeff",+1),("economy.housing.immigration_pressure_coeff",+1)],
    "demographics.population_million": [("demographics.migration.pull_factor_coeff",+1),("demographics.population.natural_growth_coeff",+1)],
    "demographics.fertility_rate": [("demographics.fertility.affordability_coeff",+1),("demographics.fertility.immigration_fertility_boost",+1)],
    "demographics.life_expectancy": [("demographics.life_expectancy.healthcare_coeff",+1)],
    "demographics.pct_over_65": [("demographics.age_structure.aging_rate",+1)],
    "demographics.net_migration_per_1000": [("demographics.migration.pull_factor_coeff",+1),("demographics.migration.openness_coeff",+1),("demographics.migration.gdp_pull_coeff",+1),("demographics.migration.gdp_pull_baseline",-1)],
    "social.life_satisfaction": [("social.life_satisfaction.adaptation_speed",+1),("social.life_satisfaction.income_baseline",-1)],
    "social.gini_coefficient": [("social.inequality.unemployment_coeff",+1),("social.inequality.tax_coeff",+1)],
    "social.poverty_rate": [("social.poverty.base",+1),("social.poverty.unemployment_coeff",+1)],
    "social.housing_affordability": [("social.housing_affordability.base",+1),("social.housing_affordability.price_coeff",+1)],
    "social.education_quality": [("social.education_quality.spending_coeff",+1),("social.education_quality.change_rate",+1)],
    "social.healthcare_quality": [("social.healthcare_quality.spending_coeff",+1),("social.healthcare_quality.change_rate",+1)],
    "governance.government_effectiveness": [("social.governance_indices.effectiveness_policy_coeff",+1)],
    "governance.corruption_control": [("social.governance_indices.corruption_policy_coeff",+1)],
    "governance.political_stability": [("social.governance_indices.stability_coalition_coeff",+1)],
}


def _getval(data, dotkey):
    parts = dotkey.split(".")
    v = data
    for p in parts:
        if isinstance(v, dict) and p in v:
            v = v[p]
        else:
            return None
    return v


# ---------------------------------------------------------------------------
# WEIGHT VECTOR HELPERS
# ---------------------------------------------------------------------------

def get_tunable_keys(weights: Weights) -> List[str]:
    flat = weights.flat()
    skip_suffixes = ('_min', '_max', 'min', 'max', '_std', 'shock_std',
                     '_version', 'start_year', 'end_year', 'label')
    skip_prefixes = ('economy.interest_rate.regimes',)
    keys = []
    for k, v in sorted(flat.items()):
        if any(k.endswith(s) for s in skip_suffixes):
            continue
        if any(k.startswith(s) for s in skip_prefixes):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keys.append(k)
    return keys


def weights_to_vector(weights: Weights, keys: List[str]) -> np.ndarray:
    return np.array([weights.get(k, 0.0) for k in keys], dtype=np.float64)


def vector_to_weights(vec: np.ndarray, keys: List[str], weights: Weights):
    for i, k in enumerate(keys):
        weights.set(k, float(vec[i]))


def normalize_weights(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    safe_ref = np.where(np.abs(ref) > 1e-8, ref, 1.0)
    return vec / safe_ref


def denormalize_weights(norm_vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    safe_ref = np.where(np.abs(ref) > 1e-8, ref, 1.0)
    return norm_vec * safe_ref


def clamp_vector(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Clamp each weight to [0.01x, 10x] of reference value."""
    result = vec.copy()
    for i in range(len(vec)):
        if ref[i] > 0:
            result[i] = max(ref[i] * 0.01, min(ref[i] * 10, result[i]))
        elif ref[i] < 0:
            result[i] = min(ref[i] * 0.01, max(ref[i] * 10, result[i]))
        elif abs(ref[i]) < 1e-8:
            result[i] = max(-1.0, min(1.0, result[i]))
    return result


# ---------------------------------------------------------------------------
# SIMULATION EVALUATION
# ---------------------------------------------------------------------------

def evaluate_weights_batch(start_path, checkpoints, weights):
    """
    Run simulation with seed 0 against all checkpoints and return:
    - wmape: scalar loss
    - signed_errors: np.array of shape (n_calib_vars,) — signed relative error per var
    """
    all_signed = [[] for _ in CALIB_VAR_LIST]
    all_weighted_rel = []
    total_w = CALIB_WEIGHTS.sum()

    for cp in checkpoints:
        initial = load_state_from_json(start_path)
        years = cp["year"] - initial.year
        if years <= 0:
            continue
        system = SpanishParliamentarySystem()
        engine = SimulationEngine(system, seed=0, initial_state=initial, weights=weights)
        engine.run(years)

        sim_rec = None
        for r in engine.history:
            if r["year"] == cp["year"]:
                sim_rec = r
                break
        if sim_rec is None:
            continue

        for i, var in enumerate(CALIB_VAR_LIST):
            sv = _getval(sim_rec, var)
            av = _getval(cp["data"], var)
            if sv is None or av is None:
                continue
            if av != 0:
                signed = (sv - av) / abs(av)
            else:
                signed = sv if sv != 0 else 0.0
            all_signed[i].append(signed)
            all_weighted_rel.append(abs(signed) * CALIB_WEIGHTS[i])

    mean_signed = np.zeros(len(CALIB_VAR_LIST))
    for i in range(len(CALIB_VAR_LIST)):
        if all_signed[i]:
            mean_signed[i] = np.mean(all_signed[i])

    n_cp = len(checkpoints)
    wmape = sum(all_weighted_rel) / (n_cp * total_w) if all_weighted_rel else 999.0
    return wmape, mean_signed


def evaluate_vector(vec, ref_vec, tunable_keys, start_path, checkpoints,
                    base_weights_data):
    """Evaluate a candidate weight vector. Returns wmape."""
    vec_clamped = clamp_vector(vec, ref_vec)
    weights = Weights.__new__(Weights)
    weights._data = copy.deepcopy(base_weights_data)
    weights.path = ""
    vector_to_weights(vec_clamped, tunable_keys, weights)
    wmape, _ = evaluate_weights_batch(start_path, checkpoints, weights)
    return wmape


# ---------------------------------------------------------------------------
# COORDINATE DESCENT
# ---------------------------------------------------------------------------

def coord_descent_step(weights, signed_errors, lr_cd=0.12):
    """One step of coordinate descent using known var->weight mappings."""
    adjustments = 0
    for i, var in enumerate(CALIB_VAR_LIST):
        err = signed_errors[i]
        if abs(err) < 0.02:
            continue
        mappings = VAR_TO_WEIGHTS.get(var, [])
        for wpath, sign in mappings:
            cur = weights.get(wpath)
            if cur is None:
                continue
            direction = -1 if err > 0 else +1
            mag = min(abs(err), 2.0)
            step = lr_cd * mag * direction * sign
            if abs(cur) > 1e-6:
                weights.set(wpath, cur * (1 + step))
            else:
                weights.set(wpath, cur + step * 0.01)
            adjustments += 1
    return adjustments


def individual_weight_probe(weights, tunable_keys, start_path, checkpoints,
                            best_wmape, probe_scale=0.05, max_probes=20):
    """
    Try nudging individual weights one at a time (not just the ones in
    VAR_TO_WEIGHTS). This catches improvements in the ~40 weights that
    coord descent never touches because they have no known causal mapping.

    Picks random weights to probe each call to avoid always trying the same ones.
    """
    indices = list(range(len(tunable_keys)))
    np.random.shuffle(indices)
    improved = False

    for idx in indices[:max_probes]:
        key = tunable_keys[idx]
        cur = weights.get(key)
        if cur is None:
            continue

        for direction in [+1, -1]:
            if abs(cur) > 1e-6:
                new_val = cur * (1 + direction * probe_scale)
            else:
                new_val = cur + direction * probe_scale * 0.01
            weights.set(key, new_val)
            wmape, _ = evaluate_weights_batch(start_path, checkpoints, weights)
            if wmape < best_wmape:
                best_wmape = wmape
                improved = True
                break  # keep this change, move to next weight
            else:
                weights.set(key, cur)  # revert

    return best_wmape, improved


# Coupled weight groups: each defines a DIRECTION to search along.
# The optimizer tries multiple scales (1% to 200%) in each direction.
# "direction" is a multiplier relative to current value: >1 = increase, <1 = decrease.
COUPLED_WEIGHT_GROUPS = [
    {
        "name": "migration_boost",
        "weights": {
            "demographics.migration.pull_factor_coeff": "flip_positive",  # special: flip sign if negative
            "demographics.migration.gdp_pull_coeff": 1.5,     # increase
            "demographics.migration.gdp_pull_baseline": 0.5,  # decrease (lower baseline = stronger effect)
            "demographics.migration.openness_coeff": 1.3,     # increase
        }
    },
    {
        "name": "unemployment_fix",
        "weights": {
            "economy.structural_unemployment.floor_decay_per_year": 1.5,  # faster decay = lower floor
            "economy.structural_unemployment.hysteresis_coeff": 1.5,      # stronger pull toward floor
            "economy.structural_unemployment.boom_absorption_rate": 1.3,  # booms reduce unemp more
            "economy.structural_unemployment.initial_floor": 0.8,        # lower starting floor
        }
    },
    {
        "name": "inflation_raise",
        "weights": {
            "economy.inflation.inertia": 0.85,           # lower inertia frees inflation to adjust
            "economy.inflation.demand_pull_coeff": 1.4,   # growth -> more inflation
            "economy.inflation.base_rate": 1.5,           # higher base inflation rate
            "economy.inflation.spending_pressure_coeff": 1.3,  # spending -> more inflation
        }
    },
    {
        "name": "inflation_lower",
        "weights": {
            "economy.inflation.inertia": 0.8,
            "economy.inflation.demand_pull_coeff": 0.7,
            "economy.inflation.base_rate": 0.5,
        }
    },
    {
        "name": "growth_moderate",
        "weights": {
            "economy.gdp.base_growth_weight": 0.8,
            "economy.gdp.momentum_factor": 0.7,
            "economy.gdp.mean_reversion_speed": 1.5,
        }
    },
    {
        "name": "growth_boost",
        "weights": {
            "economy.gdp.base_growth_weight": 1.3,
            "economy.gdp.base_growth": 1.3,
            "economy.gdp.infra_boost": 1.3,
        }
    },
    {
        "name": "housing_boost",
        "weights": {
            "economy.housing.demand_pressure_coeff": 1.5,
            "economy.housing.immigration_pressure_coeff": 1.5,
            "economy.housing.interest_rate_effect": 1.3,
        }
    },
    {
        "name": "wage_inflation_link",
        "weights": {
            "economy.wages.inflation_passthrough": 1.3,
            "economy.wages.gdp_passthrough": 0.9,
            "economy.inflation.demand_pull_coeff": 1.2,
        }
    },
]


# ---------------------------------------------------------------------------
# CORE DYNAMICS GRID SEARCH
# ---------------------------------------------------------------------------
# The optimizer often gets trapped in degenerate basins where key weights
# have wrong signs, near-zero values with compensating huge multipliers,
# or absurd thresholds. This grid search explores diverse configurations
# of the most critical "core dynamics" weights to find better basins.
#
# Key degenerate patterns to escape:
#   - inflation.monetary_coeff near zero with monetary_baseline near zero
#   - inflation.inertia too low (inflation doesn't persist)
#   - unemployment.okun_coeff near zero (Okun's law disabled)
#   - migration.pull_factor_coeff negative (economic pull repels migrants)
#   - gdp.base_growth near zero with huge base_growth_weight

CORE_DYNAMICS_GRID = {
    # INFLATION — most commonly degenerate
    "economy.inflation.inertia": [0.1, 0.35, 0.55, 0.7, 0.85],
    "economy.inflation.monetary_baseline": [0.0001, 0.01, 0.02, 0.04, 0.06],
    "economy.inflation.demand_pull_coeff": [0.2, 0.5, 1.0, 1.9],
    "economy.inflation.base_rate": [-0.003, 0.005, 0.015],
}


def _safe_coord_descent(w, start_path, checkpoints, n_rounds=20, lr=0.10):
    """Coord descent with revert-if-worse after each round."""
    local_wmape, errors = evaluate_weights_batch(start_path, checkpoints, w)
    local_best_data = copy.deepcopy(w._data)
    for _ in range(n_rounds):
        coord_descent_step(w, errors, lr_cd=lr)
        wmape, errors = evaluate_weights_batch(start_path, checkpoints, w)
        if wmape < local_wmape:
            local_wmape = wmape
            local_best_data = copy.deepcopy(w._data)
        else:
            w._data = copy.deepcopy(local_best_data)
            _, errors = evaluate_weights_batch(start_path, checkpoints, w)
    return local_wmape, local_best_data


def grid_search_initialization(weights, start_path, checkpoints, verbose=True):
    """
    Escape degenerate weight basins via adaptive subsystem search.

    Searches subsystems one at a time (inflation, GDP, unemployment,
    migration) with ranges that include the CURRENT value plus several
    alternatives. Each subsystem search uses safe coord descent
    (revert-if-worse) so it can only improve or stay the same.

    The search is sequential: each phase builds on the best result
    from the previous phase. Because the current value is always
    in the grid, this can never make things worse.

    Returns:
        (best_wmape, best_weights_data, improved)
    """
    import itertools

    base_data = copy.deepcopy(weights._data)

    best_wmape, _ = evaluate_weights_batch(start_path, checkpoints, weights)
    best_data = copy.deepcopy(base_data)
    initial_wmape = best_wmape

    if verbose:
        print(f"\n  --- Adaptive Grid Search ---")
        print(f"  Current WMAPE: {best_wmape:.4f}")

    def _build_range(current, alternatives):
        """Current value + alternatives, deduplicated."""
        values = [current] + alternatives
        unique = []
        for v in values:
            if not any(abs(v - u) < abs(u) * 0.01 + 1e-10 for u in unique):
                unique.append(v)
        return unique

    def _search_subsystem(name, grid, search_base, best_wmape, best_data,
                          cd_rounds=15):
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        all_combos = list(itertools.product(*value_lists))
        if verbose:
            dims = " × ".join(str(len(v)) for v in value_lists)
            print(f"  {name}: {len(all_combos)} configs ({dims})")
        for combo in all_combos:
            config = dict(zip(keys, combo))
            w = Weights.__new__(Weights)
            w._data = copy.deepcopy(search_base)
            w.path = ""
            for k, v in config.items():
                w.set(k, v)
            local_wmape, local_data = _safe_coord_descent(
                w, start_path, checkpoints, n_rounds=cd_rounds, lr=0.10)
            if local_wmape < best_wmape:
                best_wmape = local_wmape
                best_data = copy.deepcopy(local_data)
                if verbose:
                    vals = " ".join(f"{k.split('.')[-1]}={v:.4g}"
                                   for k, v in config.items())
                    print(f"    WMAPE={local_wmape:.4f}  ({vals})")
        return best_wmape, best_data

    # Access nested dict values
    def _g(d, path):
        for p in path.split("."):
            d = d[p]
        return d

    # ------------------------------------------------------------------
    # Search each subsystem sequentially, building on prior improvements
    # ------------------------------------------------------------------
    search_base = copy.deepcopy(base_data)

    # 1. Inflation
    cur = _g(search_base, "economy.inflation")
    infl_grid = {
        "economy.inflation.inertia":
            _build_range(cur["inertia"], [0.15, 0.3, 0.45, 0.6, 0.75]),
        "economy.inflation.monetary_baseline":
            _build_range(cur["monetary_baseline"], [0.005, 0.015, 0.03, 0.05]),
        "economy.inflation.demand_pull_coeff":
            _build_range(cur["demand_pull_coeff"], [0.2, 0.5, 0.8, 1.2]),
    }
    best_wmape, best_data = _search_subsystem(
        "Inflation", infl_grid, search_base, best_wmape, best_data)
    if best_wmape < initial_wmape:
        search_base = copy.deepcopy(best_data)

    # 2. GDP dynamics
    cur = _g(search_base, "economy.gdp")
    gdp_grid = {
        "economy.gdp.base_growth":
            _build_range(cur["base_growth"], [0.003, 0.008, 0.015, 0.025]),
        "economy.gdp.base_growth_weight":
            _build_range(cur["base_growth_weight"], [0.5, 1.0, 3.0, 6.0]),
        "economy.gdp.mean_reversion_speed":
            _build_range(cur["mean_reversion_speed"], [0.02, 0.06, 0.12, 0.2]),
    }
    best_wmape, best_data = _search_subsystem(
        "GDP", gdp_grid, search_base, best_wmape, best_data)
    if best_wmape < initial_wmape:
        search_base = copy.deepcopy(best_data)

    # 3. Unemployment
    cur_u = _g(search_base, "economy.unemployment")
    unemp_grid = {
        "economy.unemployment.okun_coeff":
            _build_range(cur_u["okun_coeff"], [-0.4, -0.2, -0.08, -0.01]),
        "economy.unemployment.okun_trend":
            _build_range(cur_u["okun_trend"], [0.02, 0.05, 0.15, 0.5]),
    }
    best_wmape, best_data = _search_subsystem(
        "Unemployment", unemp_grid, search_base, best_wmape, best_data,
        cd_rounds=10)
    if best_wmape < initial_wmape:
        search_base = copy.deepcopy(best_data)

    # 4. Migration
    cur_m = _g(search_base, "demographics.migration")
    mig_grid = {
        "demographics.migration.pull_factor_coeff":
            _build_range(cur_m["pull_factor_coeff"],
                         [-0.5, 0.5, 1.5, 3.0, 5.0]),
        "demographics.migration.gdp_pull_baseline":
            _build_range(cur_m["gdp_pull_baseline"],
                         [12000, 20000, 35000, 50000]),
    }
    best_wmape, best_data = _search_subsystem(
        "Migration", mig_grid, search_base, best_wmape, best_data,
        cd_rounds=10)

    # ------------------------------------------------------------------
    # Phase 2: Final polish with individual probing
    # ------------------------------------------------------------------
    if best_wmape < initial_wmape:
        w = Weights.__new__(Weights)
        w._data = copy.deepcopy(best_data)
        w.path = ""
        tunable = get_tunable_keys(w)
        for _ in range(5):
            pw, pi2 = individual_weight_probe(
                w, tunable, start_path, checkpoints,
                best_wmape, probe_scale=0.04, max_probes=30)
            if pi2:
                best_wmape = pw
                best_data = copy.deepcopy(w._data)
            else:
                break
        if verbose:
            print(f"  After probing: {best_wmape:.4f}")

    improved = best_wmape < initial_wmape - 1e-6
    if verbose:
        if improved:
            print(f"  Grid search: {initial_wmape:.4f} -> {best_wmape:.4f} "
                  f"({(1-best_wmape/initial_wmape)*100:.1f}% improvement)")
        else:
            print(f"  Grid search: no improvement (best remains {initial_wmape:.4f})")

    return best_wmape, best_data, improved


def coupled_descent_step(weights, start_path, checkpoints, best_wmape, verbose=False):
    """
    For each coupled group, try moving ALL weights in the group simultaneously
    at multiple scales. This breaks correlated local minima by making coordinated
    changes that no single-weight optimizer can find.
    
    Uses a line search: try the direction at scales 0.05, 0.1, 0.2, 0.5, 1.0, 2.0
    (where 1.0 = full step as defined in the group). Keeps the best scale.
    Also tries the inverse direction at the same scales.
    """
    improved = False
    scales = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    for group in COUPLED_WEIGHT_GROUPS:
        # Save current values
        saved = {}
        for wpath in group["weights"]:
            cur = weights.get(wpath)
            if cur is not None:
                saved[wpath] = cur

        if not saved:
            continue

        # Try both the forward and inverse direction
        for direction_sign in [+1, -1]:
            best_scale_wmape = best_wmape
            best_scale = None

            for scale in scales:
                # Apply scaled adjustments
                for wpath, target_mult in group["weights"].items():
                    cur = saved.get(wpath)
                    if cur is None:
                        continue

                    if target_mult == "flip_positive":
                        # Special: interpolate toward |cur| (flip sign if negative)
                        target = abs(cur) * 3.0  # target is positive and larger
                        new_val = cur + direction_sign * scale * (target - cur)
                    else:
                        # Interpolate: cur + scale * (cur * mult - cur) = cur * (1 + scale * (mult - 1))
                        if direction_sign > 0:
                            mult = target_mult
                        else:
                            mult = 1.0 / target_mult  # inverse direction
                        new_val = cur * (1 + scale * (mult - 1))

                    weights.set(wpath, new_val)

                wmape, _ = evaluate_weights_batch(start_path, checkpoints, weights)

                if wmape < best_scale_wmape:
                    best_scale_wmape = wmape
                    best_scale = (scale, direction_sign)

                # Revert for next scale
                for wpath, cur in saved.items():
                    weights.set(wpath, cur)

            # If we found an improvement, apply the best scale permanently
            if best_scale is not None and best_scale_wmape < best_wmape:
                scale, dsign = best_scale
                for wpath, target_mult in group["weights"].items():
                    cur = saved.get(wpath)
                    if cur is None:
                        continue
                    if target_mult == "flip_positive":
                        target = abs(cur) * 3.0
                        new_val = cur + dsign * scale * (target - cur)
                    else:
                        mult = target_mult if dsign > 0 else 1.0 / target_mult
                        new_val = cur * (1 + scale * (mult - 1))
                    weights.set(wpath, new_val)

                best_wmape = best_scale_wmape
                improved = True
                if verbose:
                    print(f"    >>> Coupled '{group['name']}' scale={scale:.2f} "
                          f"dir={'fwd' if dsign>0 else 'inv'} -> WMAPE={best_wmape:.4f}",
                          flush=True)
                # Update saved values for subsequent groups
                for wpath in group["weights"]:
                    cur = weights.get(wpath)
                    if cur is not None:
                        saved[wpath] = cur
                break  # don't try inverse if forward worked

    return best_wmape, improved


# ---------------------------------------------------------------------------
# STOCHASTIC PERTURBATION
# ---------------------------------------------------------------------------

def perturbation_step(current_vec, ref_vec, tunable_keys, start_path, checkpoints,
                      base_weights_data, current_wmape,
                      n_perturbations=8, perturbation_scale=0.05):
    """
    Estimate gradient via random perturbations with antithetic sampling.
    """
    n = len(current_vec)
    best_vec = current_vec.copy()
    best_wmape = current_wmape

    for _ in range(n_perturbations):
        direction = np.random.randn(n)
        direction /= (np.linalg.norm(direction) + 1e-10)
        scale = perturbation_scale * np.maximum(np.abs(ref_vec), 1e-6)
        delta = direction * scale

        plus_vec = current_vec + delta
        plus_wmape = evaluate_vector(plus_vec, ref_vec, tunable_keys, start_path,
                                     checkpoints, base_weights_data)
        if plus_wmape < best_wmape:
            best_wmape = plus_wmape
            best_vec = plus_vec.copy()

        minus_vec = current_vec - delta
        minus_wmape = evaluate_vector(minus_vec, ref_vec, tunable_keys, start_path,
                                      checkpoints, base_weights_data)
        if minus_wmape < best_wmape:
            best_wmape = minus_wmape
            best_vec = minus_vec.copy()

    return best_vec, best_wmape


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------------------------------

def save_checkpoint(path, cma_state, epoch, best_wmape, best_weights_data,
                    total_epochs, history, init_wmape, best_vec):
    checkpoint = {
        "cma_state": cma_state,
        "epoch": epoch,
        "best_wmape": best_wmape,
        "best_weights_data": best_weights_data,
        "total_epochs": total_epochs,
        "history": history,
        "init_wmape": init_wmape,
        "best_vec": best_vec.tolist(),
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MAIN CALIBRATION LOOP
# ---------------------------------------------------------------------------

def run_calibration(start_path, checkpoint_paths,
                    epochs=50, lr=0.10, pop_size=None, verbose=True,
                    grid_threshold=0.20, force_grid=False):

    endless = (epochs == 0)

    checkpoints = []
    for cp_path in checkpoint_paths:
        with open(cp_path) as f:
            cp_data = json.load(f)
        checkpoints.append({"path": cp_path, "data": cp_data, "year": cp_data["start_year"]})

    # --- Paths ---
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "weights", "default.json")
    tuned_path   = os.path.join(base_dir, "weights", "tuned.json")
    ckpt_path    = os.path.join(base_dir, "weights", "calibrator_checkpoint.json")
    state_path   = os.path.join(base_dir, "weights", "calibrator_state.json")
    pid_path     = os.path.join(base_dir, "calibrate.pid")

    shutdown = GracefulShutdown(pid_path=pid_path)

    # --- Weights ---
    resuming = os.path.isfile(tuned_path)
    if resuming:
        weights = Weights(tuned_path)
    else:
        weights = Weights(default_path)
        weights = weights.clone(tuned_path)

    default_weights = Weights(default_path)

    # Merge any NEW keys from default.json into tuned.json so they become tunable.
    # This handles the case where engine.py uses w.get(key, fallback) for weights
    # that were never in the JSON — adding them to default.json makes them visible
    # to the calibrator, but only if they also get into tuned.json.
    def _deep_merge(src, dst):
        """Recursively copy keys from src into dst where dst is missing them."""
        count = 0
        for k, v in src.items():
            if k.startswith("_"):
                continue
            if k not in dst:
                dst[k] = copy.deepcopy(v)
                count += 1
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                count += _deep_merge(v, dst[k])
        return count

    merged_count = _deep_merge(default_weights.data, weights._data)
    if merged_count > 0:
        weights.save()
        if verbose:
            print(f"  Merged {merged_count} new weight(s) from default.json into tuned.json")

    tunable_keys = get_tunable_keys(weights)
    n_weights = len(tunable_keys)
    ref_vec = weights_to_vector(default_weights, tunable_keys)
    current_vec = weights_to_vector(weights, tunable_keys)

    # --- CMA-ES ---
    norm_vec = normalize_weights(current_vec, ref_vec)
    actual_pop_size = pop_size or max(12, 4 + int(3 * np.log(n_weights)))
    cma = CMAES(norm_vec, sigma0=0.2, pop_size=actual_pop_size)

    # --- Resume ---
    start_epoch = 0
    best_wmape = 999.0
    best_weights_data = copy.deepcopy(weights.data)
    history = []
    init_wmape_saved = None
    stagnation_counter = 0

    saved_state = {}
    if os.path.isfile(state_path):
        with open(state_path) as f:
            saved_state = json.load(f)
        best_wmape = float(saved_state.get("best_wmape", 999.0))

    if os.path.isfile(ckpt_path):
        try:
            ckpt = load_checkpoint(ckpt_path)
            start_epoch = ckpt["epoch"]
            best_wmape = ckpt["best_wmape"]
            best_weights_data = ckpt["best_weights_data"]
            history = ckpt.get("history", [])
            init_wmape_saved = ckpt.get("init_wmape")

            # Merge new keys into best_weights_data from default
            # (same as we did for tuned.json above)
            def _deep_merge(src, dst):
                """Copy keys from src into dst where dst is missing them."""
                count = 0
                for k, v in src.items():
                    if k.startswith("_"):
                        continue
                    if k not in dst:
                        dst[k] = copy.deepcopy(v)
                        count += 1
                    elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                        count += _deep_merge(v, dst[k])
                return count
            merge_count = _deep_merge(default_weights.data, best_weights_data)
            if merge_count > 0 and verbose:
                print(f"  Merged {merge_count} new key(s) into checkpoint best_weights")

            # Check if CMA dimensions match (new tunable keys = different dimension)
            ckpt_cma_dim = len(ckpt["cma_state"]["mean"])
            if ckpt_cma_dim == n_weights:
                cma.set_state(ckpt["cma_state"])
                if "best_vec" in ckpt:
                    current_vec = np.array(ckpt["best_vec"])
                    norm_vec = normalize_weights(current_vec, ref_vec)
                    cma.mean = norm_vec.copy()
            else:
                # Dimensions changed — new weights were added. Reset CMA but keep
                # the best weights and epoch counter so we don't lose calibration progress.
                current_vec = weights_to_vector(weights, tunable_keys)
                norm_vec = normalize_weights(current_vec, ref_vec)
                cma = CMAES(norm_vec, sigma0=0.3, pop_size=actual_pop_size)
                if verbose:
                    print(f"  CMA dimensions changed ({ckpt_cma_dim} -> {n_weights}) — "
                          f"fresh CMA, keeping best weights & epoch")

            if verbose:
                print(f"  Resumed from checkpoint at epoch {start_epoch} "
                      f"(best WMAPE: {best_wmape:.4f}, CMA gen: {cma.gen})")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Could not load checkpoint ({e}), starting fresh")
            start_epoch = 0

    if verbose:
        mode_str = "ENDLESS" if endless else f"{epochs} epochs"
        resume_str = (f"RESUMING (epoch {start_epoch})" if (resuming or start_epoch > 0)
                      else "FRESH START")
        print(f"\n{'='*80}")
        print(f"  MULTI-STRATEGY CALIBRATOR (CMA-ES + CoordDescent + Perturbation)")
        print(f"{'='*80}")
        print(f"  Mode:            {resume_str}")
        print(f"  Epochs:          {mode_str}")
        print(f"  Start:           {start_path}")
        for cp in checkpoints:
            print(f"  Checkpoint:      {cp['path']} (year {cp['year']})")
        print(f"  Tunable weights: {n_weights}")
        print(f"  Calib variables: {len(CALIB_VAR_LIST)}")
        print(f"  Seed:            0 (deterministic)")
        print(f"  CMA-ES pop:      {actual_pop_size} (mu={cma.mu})")
        print(f"  CMA-ES sigma:    {cma.sigma:.4f}")
        print(f"  Coord descent:   lr={lr}")
        print(f"  Perturbation:    8 directions, scale=0.05")
        print(f"  Grid search:     threshold={grid_threshold}, force={'yes' if force_grid else 'no'}")
        print(f"  Output:          {tuned_path}")
        print(f"  PID:             {pid_path} ({os.getpid()})")
        if best_wmape < 999:
            print(f"  Prior best WMAPE: {best_wmape:.4f}")
        print()
        if endless:
            print(f"  ENDLESS MODE — stop gracefully with: kill $(cat {pid_path})")
            print()

    t0 = time.time()

    # Initial evaluation
    init_wmape, init_errors = evaluate_weights_batch(start_path, checkpoints, weights)
    if init_wmape_saved is None:
        init_wmape_saved = init_wmape
    if init_wmape < best_wmape:
        best_wmape = init_wmape
        best_weights_data = copy.deepcopy(weights.data)
    if verbose:
        print(f"  Current WMAPE:   {init_wmape:.4f}  (best: {best_wmape:.4f})")
        top_errs = sorted(zip(CALIB_VAR_LIST, init_errors), key=lambda x: -abs(x[1]))[:6]
        for k, v in top_errs:
            print(f"    {k.split('.')[-1]:<35} {v:+.4f}")
        print()

    # --- GRID SEARCH INITIALIZATION ---
    # On fresh starts OR when resuming with high WMAPE, run a grid search
    # over core dynamics weights to escape degenerate basins.
    # This is the key innovation: the optimizer gets trapped when weights
    # form coupled pathological configurations that no single-weight change
    # can escape. The grid search tests diverse core dynamics configurations.
    grid_threshold_val = 999.0 if force_grid else grid_threshold
    if force_grid or (grid_threshold > 0 and best_wmape > grid_threshold):
        if verbose:
            if force_grid:
                print(f"  FORCED grid search (--force-grid) — current WMAPE: {best_wmape:.4f}",
                      flush=True)
            else:
                print(f"  WMAPE ({best_wmape:.4f}) > {grid_threshold} — running grid search...",
                      flush=True)
        weights._data = copy.deepcopy(best_weights_data)
        grid_wmape, grid_data, grid_improved = grid_search_initialization(
            weights, start_path, checkpoints, verbose=verbose)
        if grid_improved:
            best_wmape = grid_wmape
            best_weights_data = copy.deepcopy(grid_data)
            weights._data = copy.deepcopy(grid_data)
            current_vec = weights_to_vector(weights, tunable_keys)
            norm_vec = normalize_weights(current_vec, ref_vec)
            cma = CMAES(norm_vec, sigma0=0.2, pop_size=actual_pop_size)
            if verbose:
                print(f"  Grid search improved WMAPE to {best_wmape:.4f} — "
                      f"CMA-ES restarted from new basin\n", flush=True)
        else:
            weights._data = copy.deepcopy(best_weights_data)

    epoch = start_epoch
    epochs_this_run = 0

    while True:
        if shutdown.should_stop:
            if verbose:
                print(f"\n  Graceful shutdown requested — saving state...", flush=True)
            break

        if not endless and epoch >= (start_epoch + epochs):
            break

        epoch_t0 = time.time()
        epoch_best_wmape = best_wmape
        strategy_used = ""

        # =================================================================
        # STRATEGY 1: CMA-ES
        # =================================================================
        candidates_norm = cma.ask()
        candidate_fitnesses = []

        for cand_norm in candidates_norm:
            cand_vec = denormalize_weights(cand_norm, ref_vec)
            cand_vec = clamp_vector(cand_vec, ref_vec)
            fitness = evaluate_vector(cand_vec, ref_vec, tunable_keys, start_path,
                                      checkpoints, best_weights_data)
            candidate_fitnesses.append(fitness)

            if fitness < best_wmape:
                best_wmape = fitness
                vector_to_weights(cand_vec, tunable_keys, weights)
                best_weights_data = copy.deepcopy(weights.data)
                current_vec = cand_vec.copy()
                strategy_used = "CMA"

        cma.tell(candidates_norm, candidate_fitnesses)

        if strategy_used == "CMA":
            cma.mean = normalize_weights(current_vec, ref_vec)

        # =================================================================
        # STRATEGY 2: Coordinate descent
        # =================================================================
        weights._data = copy.deepcopy(best_weights_data)
        _, cd_errors = evaluate_weights_batch(start_path, checkpoints, weights)

        cd_lr = lr * max(0.3, min(2.0, best_wmape / 0.20))
        coord_descent_step(weights, cd_errors, lr_cd=cd_lr)

        cd_wmape, _ = evaluate_weights_batch(start_path, checkpoints, weights)

        if cd_wmape < best_wmape:
            best_wmape = cd_wmape
            best_weights_data = copy.deepcopy(weights.data)
            current_vec = weights_to_vector(weights, tunable_keys)
            cma.mean = normalize_weights(current_vec, ref_vec)
            strategy_used = "CD"
        else:
            weights._data = copy.deepcopy(best_weights_data)

        # =================================================================
        # STRATEGY 3: Stochastic perturbation
        # =================================================================
        # Scale up aggressively with stagnation: 0.03 → 0.05 → 0.13 → 0.33 → ...
        perturb_scale = 0.03 * (1.3 ** min(stagnation_counter, 30))
        perturb_scale = min(perturb_scale, 0.5)  # cap at 50% of weight magnitude
        n_perturb = 8 + min(stagnation_counter // 10, 16)  # more tries when stuck
        current_vec = weights_to_vector(weights, tunable_keys)

        perturb_vec, perturb_wmape = perturbation_step(
            current_vec, ref_vec, tunable_keys, start_path, checkpoints,
            best_weights_data, best_wmape,
            n_perturbations=n_perturb, perturbation_scale=perturb_scale
        )

        if perturb_wmape < best_wmape:
            best_wmape = perturb_wmape
            perturb_vec = clamp_vector(perturb_vec, ref_vec)
            vector_to_weights(perturb_vec, tunable_keys, weights)
            best_weights_data = copy.deepcopy(weights.data)
            current_vec = perturb_vec.copy()
            cma.mean = normalize_weights(current_vec, ref_vec)
            strategy_used = "PERTURB"

        # =================================================================
        # STRATEGY 4: Individual weight probing (unmapped weights)
        # =================================================================
        # Run every 3 epochs or when stagnating, to keep it from slowing
        # down the loop too much (it does max_probes sequential evals)
        if epochs_this_run % 3 == 0 or stagnation_counter > 10:
            weights._data = copy.deepcopy(best_weights_data)
            probe_scale = 0.03 * (1.2 ** min(stagnation_counter, 20))
            probe_scale = min(probe_scale, 0.3)
            probe_wmape, probe_improved = individual_weight_probe(
                weights, tunable_keys, start_path, checkpoints,
                best_wmape, probe_scale=probe_scale, max_probes=20
            )
            if probe_improved:
                best_wmape = probe_wmape
                best_weights_data = copy.deepcopy(weights.data)
                current_vec = weights_to_vector(weights, tunable_keys)
                cma.mean = normalize_weights(current_vec, ref_vec)
                strategy_used = "PROBE"
            else:
                weights._data = copy.deepcopy(best_weights_data)

        # =================================================================
        # STRATEGY 5: Coupled multi-weight descent (breaks correlated traps)
        # =================================================================
        # Run every 5 epochs or when stagnating
        if epochs_this_run % 5 == 0 or stagnation_counter > 5:
            weights._data = copy.deepcopy(best_weights_data)
            coupled_wmape, coupled_improved = coupled_descent_step(
                weights, start_path, checkpoints, best_wmape, verbose=verbose
            )
            if coupled_improved:
                best_wmape = coupled_wmape
                best_weights_data = copy.deepcopy(weights.data)
                current_vec = weights_to_vector(weights, tunable_keys)
                cma.mean = normalize_weights(current_vec, ref_vec)
                strategy_used = "COUPLED"
            else:
                weights._data = copy.deepcopy(best_weights_data)

        # Stagnation tracking
        if best_wmape >= epoch_best_wmape - 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        # --- ESCALATING STAGNATION RESPONSE ---
        # Level 1 (25 epochs): Boost CMA sigma
        # Level 2 (50 epochs): Full CMA restart (reset covariance matrix)
        # Level 3 (100 epochs): Random restart — jump to a random point near best,
        #                       evaluate, keep if better
        if stagnation_counter == 25:
            cma.sigma = 0.4
            if verbose:
                print(f"  >>> Stagnation L1 ({stagnation_counter}) — "
                      f"boosting CMA sigma to {cma.sigma:.4f}", flush=True)

        elif stagnation_counter > 0 and stagnation_counter % 50 == 0:
            # Full CMA restart: keep the mean at best known, but reset
            # the covariance matrix so it explores fresh directions
            old_sigma = cma.sigma
            best_norm = normalize_weights(current_vec, ref_vec)
            cma = CMAES(best_norm, sigma0=0.3, pop_size=actual_pop_size)
            if verbose:
                print(f"  >>> Stagnation L2 ({stagnation_counter}) — "
                      f"CMA-ES full restart (sigma 0.3, fresh covariance)", flush=True)

        elif stagnation_counter > 0 and stagnation_counter % 100 == 0:
            # Random restart: try a big random jump
            jump_scale = 0.15
            for attempt in range(5):
                noise = np.random.randn(n_weights) * jump_scale
                jump_vec = current_vec * (1 + noise)
                jump_vec = clamp_vector(jump_vec, ref_vec)
                jump_wmape = evaluate_vector(jump_vec, ref_vec, tunable_keys,
                                             start_path, checkpoints, best_weights_data)
                if jump_wmape < best_wmape:
                    best_wmape = jump_wmape
                    vector_to_weights(jump_vec, tunable_keys, weights)
                    best_weights_data = copy.deepcopy(weights.data)
                    current_vec = jump_vec.copy()
                    cma.mean = normalize_weights(current_vec, ref_vec)
                    stagnation_counter = 0
                    strategy_used = "JUMP"
                    if verbose:
                        print(f"  >>> Random jump found improvement! "
                              f"WMAPE={best_wmape:.4f}", flush=True)
                    break
                jump_scale *= 1.5  # try bigger jumps
            else:
                if verbose:
                    print(f"  >>> Stagnation L3 ({stagnation_counter}) — "
                          f"random jumps didn't help, continuing...", flush=True)

        # Level 4 (200 epochs): Grid search — systematically explore core
        # dynamics configurations to find a completely new basin
        if stagnation_counter > 0 and stagnation_counter % 200 == 0:
            if verbose:
                print(f"  >>> Stagnation L4 ({stagnation_counter}) — "
                      f"running grid search over core dynamics...", flush=True)
            weights._data = copy.deepcopy(best_weights_data)
            grid_wmape, grid_data, grid_improved = grid_search_initialization(
                weights, start_path, checkpoints, verbose=verbose)
            if grid_improved:
                best_wmape = grid_wmape
                best_weights_data = copy.deepcopy(grid_data)
                weights._data = copy.deepcopy(grid_data)
                current_vec = weights_to_vector(weights, tunable_keys)
                cma.mean = normalize_weights(current_vec, ref_vec)
                # Full CMA restart from the new basin
                cma = CMAES(cma.mean.copy(), sigma0=0.2, pop_size=actual_pop_size)
                stagnation_counter = 0
                strategy_used = "GRID"
                if verbose:
                    print(f"  >>> Grid search found new basin! "
                          f"WMAPE={best_wmape:.4f}", flush=True)
            else:
                weights._data = copy.deepcopy(best_weights_data)
                if verbose:
                    print(f"  >>> Grid search didn't find improvement", flush=True)

        epochs_this_run += 1

        # Log
        if verbose:
            elapsed = time.time() - epoch_t0
            total_elapsed = time.time() - t0
            cma_best = min(candidate_fitnesses) if candidate_fitnesses else 999
            epoch_display = f"{epoch+1}" if endless else f"{epoch+1}/{start_epoch + epochs}"
            improved = "+" if strategy_used else " "
            print(f"  [{epoch_display:>8}] WMAPE={best_wmape:.4f} "
                  f"cma_pop_best={cma_best:.4f} sigma={cma.sigma:.3f} "
                  f"| {elapsed:.0f}s ({total_elapsed:.0f}s) "
                  f"| {strategy_used or 'none':>7}{improved}",
                  flush=True)

            # Periodic diagnostic: every 50 epochs or on first stagnation hit,
            # print per-variable error breakdown so we can see what's blocking progress
            if epochs_this_run % 50 == 0 or stagnation_counter == 25:
                weights._data = copy.deepcopy(best_weights_data)
                _, diag_errors = evaluate_weights_batch(start_path, checkpoints, weights)
                total_cw = CALIB_WEIGHTS.sum()
                print(f"\n  --- Per-variable error breakdown (WMAPE contribution) ---")
                var_contributions = []
                for vi, var in enumerate(CALIB_VAR_LIST):
                    contrib = abs(diag_errors[vi]) * CALIB_WEIGHTS[vi] / total_cw
                    var_contributions.append((var, diag_errors[vi], contrib))
                var_contributions.sort(key=lambda x: -x[2])
                cumulative = 0.0
                for var, err, contrib in var_contributions:
                    cumulative += contrib
                    bar = "#" * int(contrib / best_wmape * 40)
                    print(f"    {var.split('.')[-1]:<35} err={err:+.4f}  "
                          f"contrib={contrib:.4f} cum={cumulative:.4f} {bar}")
                print(f"  --- Total WMAPE: {sum(c for _,_,c in var_contributions):.4f} ---\n",
                      flush=True)

        history.append({
            "epoch": epoch + 1,
            "best_wmape": float(best_wmape),
            "cma_pop_best": float(min(candidate_fitnesses)) if candidate_fitnesses else None,
            "cma_sigma": float(cma.sigma),
            "strategy": strategy_used or "none",
            "stagnation": stagnation_counter,
        })

        # --- SAVE EVERY EPOCH ---
        weights._data = best_weights_data
        weights.save()

        total_epochs_now = start_epoch + epochs_this_run
        save_checkpoint(
            ckpt_path, cma.get_state(),
            epoch=epoch + 1, best_wmape=best_wmape,
            best_weights_data=best_weights_data,
            total_epochs=total_epochs_now, history=history,
            init_wmape=init_wmape_saved, best_vec=current_vec,
        )

        cal_state = {
            "best_wmape": float(best_wmape),
            "total_epochs_run": total_epochs_now,
            "cma_sigma": float(cma.sigma),
            "cma_gen": cma.gen,
            "stagnation_counter": stagnation_counter,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid(),
        }
        with open(state_path, "w") as f:
            json.dump(cal_state, f, indent=2)

        if best_wmape < 0.02:
            if verbose:
                print(f"  Converged — WMAPE below 2%")
            break

        epoch += 1

    # --- FINAL ---
    weights._data = best_weights_data
    weights.save()

    if verbose:
        print(f"\n  Running final evaluation...")

    final_wmape, final_errors = evaluate_weights_batch(start_path, checkpoints, weights)
    total_epochs_now = start_epoch + epochs_this_run

    if verbose:
        print(f"\n{'='*80}")
        print(f"  CALIBRATION {'PAUSED' if shutdown.should_stop else 'COMPLETE'}")
        print(f"{'='*80}")
        print(f"  WMAPE at session start: {init_wmape_saved:.4f}")
        print(f"  Final WMAPE:            {final_wmape:.4f}")
        print(f"  All-time best WMAPE:    {min(final_wmape, best_wmape):.4f}")
        if init_wmape_saved > 0:
            print(f"  Improvement this run:   {(1 - final_wmape/init_wmape_saved)*100:.1f}%")
        print(f"  Total epochs:           {total_epochs_now}")
        print(f"  Epochs this run:        {epochs_this_run}")
        print(f"  CMA-ES generations:     {cma.gen}")
        print(f"  Time:                   {time.time()-t0:.1f}s")
        print(f"  Weights:                {tuned_path}")
        print(f"\n  Remaining biases (|error| > 5%):")
        for var, err in sorted(zip(CALIB_VAR_LIST, final_errors), key=lambda x: -abs(x[1])):
            if abs(err) > 0.05:
                d = "HIGH" if err > 0 else "LOW"
                print(f"    {var.split('.')[-1]:<35} {d:<5} {abs(err)*100:>6.1f}%")

    # Final save
    save_checkpoint(
        ckpt_path, cma.get_state(),
        epoch=epoch, best_wmape=min(final_wmape, best_wmape),
        best_weights_data=best_weights_data,
        total_epochs=total_epochs_now, history=history,
        init_wmape=init_wmape_saved, best_vec=current_vec,
    )

    final_state = {
        "best_wmape": float(min(final_wmape, best_wmape)),
        "total_epochs_run": total_epochs_now,
        "cma_sigma": float(cma.sigma),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paused" if shutdown.should_stop else "complete",
    }
    with open(state_path, "w") as f:
        json.dump(final_state, f, indent=2)

    if verbose:
        print(f"  Checkpoint:      {ckpt_path}")
        print(f"  State:           {state_path}")

    report = {
        "start_file": start_path,
        "checkpoints": [cp["path"] for cp in checkpoints],
        "seed": 0,
        "epochs_this_run": epochs_this_run,
        "total_epochs_all_runs": total_epochs_now,
        "initial_wmape_this_run": float(init_wmape_saved),
        "final_wmape": float(final_wmape),
        "improvement_pct_this_run": float((1 - final_wmape / max(init_wmape_saved, 1e-8)) * 100),
        "optimizer": "CMA-ES + CoordDescent + Perturbation",
        "cma_pop_size": actual_pop_size,
        "cma_generations": cma.gen,
        "status": "paused" if shutdown.should_stop else "complete",
        "history": history,
        "final_biases": {var: round(float(err), 4)
                         for var, err in zip(CALIB_VAR_LIST, final_errors)},
        "tuned_weights_path": tuned_path,
        "checkpoint_path": ckpt_path,
        "state_path": state_path,
    }
    report_path = os.path.join(base_dir, "output", "calibration_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"  Report:          {report_path}")
        if shutdown.should_stop:
            print(f"\n  To resume: python calibrate.py --epochs 0")
        print()

    shutdown.cleanup()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Calibrator (CMA-ES + Coord Descent + Perturbation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python calibrate.py                          # 50 epochs (default)
  python calibrate.py --epochs 0               # endless (stop with kill/Ctrl+C)
  python calibrate.py --pop-size 20
  python calibrate.py --reset                  # discard all progress
  python calibrate.py --force-grid             # force grid search over core dynamics
  python calibrate.py --grid-threshold 0.25    # grid search if WMAPE > 0.25
  python calibrate.py --grid-threshold 0       # disable grid search

Server usage:
  nohup python -u calibrate.py --epochs 0 > calibrate.log 2>&1 &
  tail -f calibrate.log
  kill $(cat calibrate.pid)                    # graceful stop
""")
    parser.add_argument("--start", default="input/spain_1994.json")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (0 = endless until terminated)")
    parser.add_argument("--lr", type=float, default=0.10,
                        help="Learning rate for coordinate descent phase")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="CMA-ES population size (default: auto)")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--reset", action="store_true",
                        help="Discard all previous calibration and start fresh")
    parser.add_argument("--grid-threshold", type=float, default=0.20,
                        help="Run grid search if WMAPE > this (default: 0.20, 0=disable)")
    parser.add_argument("--force-grid", action="store_true",
                        help="Force grid search even if WMAPE is below threshold")
    args = parser.parse_args()

    if args.reset:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        removed = []
        for fname in ("weights/tuned.json", "weights/calibrator_checkpoint.json",
                      "weights/calibrator_state.json",
                      "weights/calibrator_checkpoint.pt",
                      "weights/calibrator_nn.npz"):
            p = os.path.join(base_dir, fname)
            if os.path.isfile(p):
                os.remove(p)
                removed.append(p)
        if removed:
            print("  RESET: removed previous calibration files:")
            for p in removed:
                print(f"    {p}")
        else:
            print("  RESET: no previous calibration files found, starting fresh.")
        print()

    if args.checkpoints is None:
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
        start_data = json.load(open(args.start))
        start_year = start_data.get("start_year", 1994)
        args.checkpoints = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.startswith("spain_") and f.endswith(".json")
            and json.load(open(os.path.join(input_dir, f))).get("start_year", 0) > start_year
        ])

    run_calibration(args.start, args.checkpoints,
                    args.epochs, args.lr, args.pop_size, not args.quiet,
                    grid_threshold=args.grid_threshold,
                    force_grid=args.force_grid)


if __name__ == "__main__":
    main()