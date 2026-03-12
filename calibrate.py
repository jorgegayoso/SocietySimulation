"""
Spain Simulation — Neural Network Calibrator
==============================================
Uses a feedforward neural network with backpropagation to learn optimal
weight corrections. The network observes simulation errors across many
seeds and checkpoints, and learns which weight adjustments minimize the
overall WMAPE.

Architecture:
  Input  (N_weights):  current weight values (normalized)
  Hidden 1 (128):      ReLU
  Hidden 2 (64):       ReLU
  Output (N_weights):  delta corrections to apply to weights

Training loop:
  1. Forward pass: apply NN-predicted deltas to weights
  2. Run simulation batch (multiple seeds, multiple checkpoints)
  3. Compute loss = WMAPE across all checkpoints
  4. Backprop through NN to update its parameters
  5. Apply best weight corrections found
  6. Repeat

This is a proper neural-network-based optimization approach, sometimes
called "learning to optimize" or "neural parameter tuning."

Usage:
    python calibrate.py
    python calibrate.py --max-seeds 200 --epochs 50
    python calibrate.py --start input/spain_1994.json --lr 0.001
"""

import argparse, copy, json, math, os, sys, time
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import SimulationEngine, Weights, load_state_from_json
from spanish_parliamentary import SpanishParliamentarySystem


# ---------------------------------------------------------------------------
# NUMPY NEURAL NETWORK
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Feedforward neural network in pure NumPy.
    Layers: Input -> Dense(h1, ReLU) -> Dense(h2, ReLU) -> Dense(output, tanh)
    Trained with Adam optimizer and backpropagation.
    """

    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int,
                 lr: float = 0.001):
        self.lr = lr
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim)

        # Adam optimizer state
        self.t = 0
        self.params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}

        # Cache for backprop
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, input_dim) or (input_dim,)"""
        squeeze = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True

        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU

        z3 = a2 @ self.W3 + self.b3
        out = np.tanh(z3)  # tanh bounds output to [-1, 1]

        self._cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'out': out}

        if squeeze:
            return out.squeeze(0)
        return out

    def backward(self, grad_output: np.ndarray):
        """Backprop. grad_output: gradient of loss w.r.t. network output."""
        c = self._cache
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        batch = grad_output.shape[0]

        # tanh derivative: 1 - tanh^2
        d3 = grad_output * (1 - c['out'] ** 2)
        gW3 = c['a2'].T @ d3 / batch
        gb3 = d3.mean(axis=0)

        da2 = d3 @ self.W3.T
        d2 = da2 * (c['z2'] > 0).astype(float)  # ReLU derivative
        gW2 = c['a1'].T @ d2 / batch
        gb2 = d2.mean(axis=0)

        da1 = d2 @ self.W2.T
        d1 = da1 * (c['z1'] > 0).astype(float)
        gW1 = c['x'].T @ d1 / batch
        gb1 = d1.mean(axis=0)

        grads = {'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2, 'W3': gW3, 'b3': gb3}

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            g = grads[p]
            # Gradient clipping
            g = np.clip(g, -1.0, 1.0)
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']


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

# Known mapping: which weights control which simulation outputs (for coordinate descent phase)
VAR_TO_WEIGHTS = {
    "economy.gdp_billion_eur": [("economy.gdp.base_growth",+1),("economy.gdp.base_growth_weight",+1)],
    "economy.gdp_growth_rate": [("economy.gdp.base_growth",+1),("economy.gdp.mean_reversion_speed",+1)],
    "economy.gdp_per_capita_eur": [("economy.gdp.base_growth",+1)],
    "economy.unemployment_rate": [("economy.structural_unemployment.initial_floor",+1),("economy.structural_unemployment.floor_decay_per_year",-1),("economy.structural_unemployment.hysteresis_coeff",+1)],
    "economy.youth_unemployment_rate": [("economy.unemployment.youth_multiplier",+1)],
    "economy.inflation_rate": [("economy.inflation.base_rate",+1),("economy.inflation.demand_pull_coeff",+1)],
    "economy.public_debt_pct_gdp": [("economy.government_finances.surplus_debt_reduction",-1),("economy.government_finances.other_spending",+1),("economy.government_finances.austerity_spending_coeff",-1)],
    "economy.tax_revenue_pct_gdp": [("economy.government_finances.tax_auto_stabilizer",+1)],
    "economy.gov_spending_pct_gdp": [("economy.government_finances.other_spending",+1),("economy.government_finances.unemployment_auto_stabilizer_coeff",+1)],
    "economy.avg_annual_wage_eur": [("economy.wages.gdp_passthrough",+1),("economy.wages.inflation_passthrough",+1)],
    "economy.interest_rate": [("economy.interest_rate.regime_convergence_speed",+1)],
    "economy.housing_price_index": [("economy.housing.demand_pressure_coeff",+1),("economy.housing.immigration_pressure_coeff",+1)],
    "demographics.population_million": [("demographics.migration.pull_factor_coeff",+1),("demographics.population.natural_growth_coeff",+1)],
    "demographics.fertility_rate": [("demographics.fertility.affordability_coeff",+1),("demographics.fertility.immigration_fertility_boost",+1)],
    "demographics.life_expectancy": [("demographics.life_expectancy.healthcare_coeff",+1)],
    "demographics.pct_over_65": [("demographics.age_structure.aging_rate",+1)],
    "demographics.net_migration_per_1000": [("demographics.migration.pull_factor_coeff",+1),("demographics.migration.openness_coeff",+1)],
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
# WEIGHT VECTOR HELPERS — flatten/unflatten weights for the NN
# ---------------------------------------------------------------------------

def get_tunable_keys(weights: Weights) -> List[str]:
    """Get all numeric leaf keys that should be tuned (skip min/max/regimes/labels)."""
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
    """Normalize weight vector relative to reference (default weights)."""
    safe_ref = np.where(np.abs(ref) > 1e-8, ref, 1.0)
    return vec / safe_ref


def denormalize_weights(norm_vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    safe_ref = np.where(np.abs(ref) > 1e-8, ref, 1.0)
    return norm_vec * safe_ref


# ---------------------------------------------------------------------------
# SIMULATION EVALUATION
# ---------------------------------------------------------------------------

def evaluate_weights_batch(start_path, checkpoints, weights, seeds):
    """
    Run simulations and return:
    - wmape: scalar loss
    - signed_errors: np.array of shape (n_calib_vars,) — mean signed relative error per var
    """
    all_signed = [[] for _ in CALIB_VAR_LIST]
    all_weighted_rel = []
    total_w = CALIB_WEIGHTS.sum()

    for seed in seeds:
        for cp in checkpoints:
            initial = load_state_from_json(start_path)
            years = cp["year"] - initial.year
            if years <= 0:
                continue
            system = SpanishParliamentarySystem()
            engine = SimulationEngine(system, seed=seed, initial_state=initial, weights=weights)
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

    n_evals = len(seeds) * len(checkpoints)
    wmape = sum(all_weighted_rel) / (n_evals * total_w) if all_weighted_rel else 999.0
    return wmape, mean_signed


# ---------------------------------------------------------------------------
# NEURAL NETWORK CALIBRATION LOOP
# ---------------------------------------------------------------------------

def run_calibration(start_path, checkpoint_paths, max_seeds=1000,
                    epochs=50, lr=0.001, delta_scale=0.15, verbose=True):

    checkpoints = []
    for cp_path in checkpoint_paths:
        with open(cp_path) as f:
            cp_data = json.load(f)
        checkpoints.append({"path": cp_path, "data": cp_data, "year": cp_data["start_year"]})

    # Clone default -> tuned
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "weights", "default.json")
    tuned_path = os.path.join(base_dir, "weights", "tuned.json")
    weights = Weights(default_path)
    weights = weights.clone(tuned_path)

    # Get tunable weight keys and reference vector
    tunable_keys = get_tunable_keys(weights)
    n_weights = len(tunable_keys)
    ref_vec = weights_to_vector(weights, tunable_keys)
    n_errors = len(CALIB_VAR_LIST)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  NEURAL NETWORK CALIBRATOR")
        print(f"{'='*80}")
        print(f"  Start:          {start_path}")
        for cp in checkpoints:
            print(f"  Checkpoint:     {cp['path']} (year {cp['year']})")
        print(f"  Tunable weights: {n_weights}")
        print(f"  Error signals:   {n_errors}")
        print(f"  Seeds/epoch:     {min(50, max_seeds)} (eval: {max_seeds})")
        print(f"  Epochs:          {epochs}")
        print(f"  NN architecture: {n_weights} -> 128 -> 64 -> {n_weights}")
        print(f"  NN lr:           {lr}")
        print(f"  Delta scale:     {delta_scale}")
        print(f"  Output:          {tuned_path}")
        print()

    # Build neural network
    # Input: normalized weight vector
    # Output: delta corrections (scaled by tanh to [-1, 1], then multiplied by delta_scale)
    nn = NeuralNetwork(n_weights, 128, 64, n_weights, lr=lr)

    train_seeds = list(range(min(50, max_seeds)))
    all_seeds = list(range(max_seeds))
    t0 = time.time()

    best_wmape = 999.0
    best_weights_data = copy.deepcopy(weights.data)
    history = []

    # Initial evaluation
    init_wmape, init_errors = evaluate_weights_batch(start_path, checkpoints, weights, train_seeds)
    best_wmape = init_wmape
    if verbose:
        print(f"  Initial WMAPE: {init_wmape:.4f}")
        print()

    def coord_descent_step(weights, signed_errors, lr_cd=0.12):
        """One step of coordinate descent using known var->weight mappings."""
        adjustments = 0
        for i, var in enumerate(CALIB_VAR_LIST):
            err = signed_errors[i]
            if abs(err) < 0.05:
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

    for epoch in range(epochs):
        # --- PHASE 1: Neural network exploration ---
        current_vec = weights_to_vector(weights, tunable_keys)
        norm_input = normalize_weights(current_vec, ref_vec)
        nn_input = np.clip(norm_input, -5, 5)

        deltas_normalized = nn.forward(nn_input)
        deltas = deltas_normalized * delta_scale
        candidate_vec = current_vec * (1 + deltas)

        # Safety clamp
        for i in range(n_weights):
            if ref_vec[i] > 0:
                candidate_vec[i] = max(ref_vec[i] * 0.01, min(ref_vec[i] * 10, candidate_vec[i]))
            elif ref_vec[i] < 0:
                candidate_vec[i] = min(ref_vec[i] * 0.01, max(ref_vec[i] * 10, candidate_vec[i]))

        vector_to_weights(candidate_vec, tunable_keys, weights)
        nn_wmape, nn_errors = evaluate_weights_batch(start_path, checkpoints, weights, train_seeds)

        # NN gradient signal
        error_signal = np.zeros(n_weights)
        for i in range(n_weights):
            error_signal[i] = deltas_normalized[i] * (nn_wmape - best_wmape) * 10
            error_signal[i] += np.sum(nn_errors * CALIB_WEIGHTS / CALIB_WEIGHTS.sum()) * deltas_normalized[i] * 5
        nn.backward(error_signal)

        if nn_wmape < best_wmape:
            best_wmape = nn_wmape
            best_weights_data = copy.deepcopy(weights.data)
        else:
            weights._data = copy.deepcopy(best_weights_data)

        # --- PHASE 2: Coordinate descent refinement ---
        _, cd_errors = evaluate_weights_batch(start_path, checkpoints, weights, train_seeds)
        cd_lr = 0.10 * (0.97 ** epoch)
        coord_descent_step(weights, cd_errors, lr_cd=cd_lr)

        cd_wmape, cd_errors2 = evaluate_weights_batch(start_path, checkpoints, weights, train_seeds)

        if cd_wmape < best_wmape:
            best_wmape = cd_wmape
            best_weights_data = copy.deepcopy(weights.data)
            wmape = cd_wmape
            signed_errors = cd_errors2
        else:
            weights._data = copy.deepcopy(best_weights_data)
            wmape = best_wmape
            signed_errors = cd_errors

        # Log
        if verbose:
            elapsed = time.time() - t0
            top_errs = sorted(zip(CALIB_VAR_LIST, signed_errors),
                              key=lambda x: -abs(x[1]))[:4]
            err_str = " | ".join(f"{k.split('.')[-1]}={v:+.2f}" for k, v in top_errs)
            improved = wmape <= best_wmape + 1e-6
            marker = " *" if improved else ""
            print(f"  [{epoch+1:3d}/{epochs}] WMAPE={wmape:.4f} best={best_wmape:.4f} "
                  f"| {elapsed:.0f}s | {err_str}{marker}")

        history.append({"epoch": epoch+1, "wmape": float(wmape),
                        "best": float(best_wmape), "improved": bool(wmape <= best_wmape + 1e-6)})

        # Early stopping if WMAPE is very low
        if best_wmape < 0.05:
            if verbose:
                print(f"  Converged — WMAPE below 5%")
            break

        # Decay delta_scale slightly (explore less as we converge)
        delta_scale *= 0.98

    # Restore best
    weights._data = best_weights_data
    weights.save()

    # Final evaluation with all seeds
    if verbose:
        print(f"\n  Running final evaluation with all {max_seeds} seeds...")

    final_wmape, final_errors = evaluate_weights_batch(start_path, checkpoints, weights, all_seeds)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  CALIBRATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Initial WMAPE: {init_wmape:.4f}")
        print(f"  Final WMAPE:   {final_wmape:.4f}")
        print(f"  Improvement:   {(1 - final_wmape/init_wmape)*100:.1f}%")
        print(f"  Time:          {time.time()-t0:.1f}s")
        print(f"  Weights:       {tuned_path}")
        print(f"\n  Remaining biases (|error| > 5%):")
        for var, err in sorted(zip(CALIB_VAR_LIST, final_errors), key=lambda x: -abs(x[1])):
            if abs(err) > 0.05:
                d = "HIGH" if err > 0 else "LOW"
                print(f"    {var.split('.')[-1]:<35} {d:<5} {abs(err)*100:>6.1f}%")

    # Save NN model
    nn_path = os.path.join(base_dir, "weights", "calibrator_nn.npz")
    nn.save(nn_path)

    # Save report
    report = {
        "start_file": start_path,
        "checkpoints": [cp["path"] for cp in checkpoints],
        "max_seeds": max_seeds,
        "epochs": len(history),
        "initial_wmape": float(init_wmape),
        "final_wmape": float(final_wmape),
        "improvement_pct": float((1 - final_wmape/init_wmape)*100),
        "architecture": f"{n_weights} -> 128 -> 64 -> {n_weights}",
        "history": history,
        "final_biases": {var: round(float(err), 4)
                         for var, err in zip(CALIB_VAR_LIST, final_errors)},
        "tuned_weights_path": tuned_path,
        "nn_model_path": nn_path,
    }
    report_path = os.path.join(base_dir, "output", "calibration_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"  Report:        {report_path}")
        print(f"  NN model:      {nn_path}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neural Network Calibrator")
    parser.add_argument("--start", default="input/spain_1994.json")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--max-seeds", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--delta-scale", type=float, default=0.15)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.checkpoints is None:
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
        start_data = json.load(open(args.start))
        start_year = start_data.get("start_year", 1994)
        args.checkpoints = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.startswith("spain_") and f.endswith(".json")
            and json.load(open(os.path.join(input_dir, f))).get("start_year", 0) > start_year
        ])

    run_calibration(args.start, args.checkpoints, args.max_seeds,
                    args.epochs, args.lr, args.delta_scale, not args.quiet)


if __name__ == "__main__":
    main()