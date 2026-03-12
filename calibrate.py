"""
Spain Simulation — AI Calibrator
==================================
Clones weights/default.json -> weights/tuned.json, then iteratively:
  1. Runs simulation from start (e.g. 1994) across many seeds
  2. Compares against ALL checkpoint years (e.g. 2000, 2004, 2024)
  3. Identifies systematic biases (what's consistently too high/low)
  4. Adjusts the relevant weights to correct the bias
  5. Repeats until error converges

Usage:
    python calibrate.py                       # auto-discover checkpoints, 1000 seeds
    python calibrate.py --max-seeds 200 --iterations 20
    python calibrate.py --start input/spain_1994.json --max-seeds 500
"""

import argparse, copy, json, math, os, sys, statistics, time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import SimulationEngine, Weights, load_state_from_json
from spanish_parliamentary import SpanishParliamentarySystem


# ---------------------------------------------------------------------------
# WHICH VARIABLES WE MEASURE AND HOW THEY MAP TO WEIGHTS
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

# Maps a simulation variable to the weight paths that most directly control it,
# along with the direction: if the sim var is TOO HIGH, do we INCREASE or DECREASE
# the weight to fix it?
# sign = +1 means "increase weight -> sim var goes UP"
# sign = -1 means "increase weight -> sim var goes DOWN"
VAR_TO_WEIGHTS = {
    "economy.gdp_billion_eur": [
        ("economy.gdp.base_growth", +1),
        ("economy.gdp.base_growth_weight", +1),
        ("economy.gdp.momentum_factor", +1),
    ],
    "economy.gdp_growth_rate": [
        ("economy.gdp.base_growth", +1),
        ("economy.gdp.base_growth_weight", +1),
        ("economy.gdp.mean_reversion_speed", +1),
    ],
    "economy.gdp_per_capita_eur": [
        ("economy.gdp.base_growth", +1),
    ],
    "economy.unemployment_rate": [
        ("economy.unemployment.okun_coeff", +1),   # more negative -> unemployment drops faster with growth
        ("economy.unemployment.okun_trend", -1),    # lower trend -> more growth-above-trend -> less unemployment
    ],
    "economy.youth_unemployment_rate": [
        ("economy.unemployment.youth_multiplier", +1),
        ("economy.unemployment.okun_coeff", +1),
    ],
    "economy.inflation_rate": [
        ("economy.inflation.base_rate", +1),
        ("economy.inflation.demand_pull_coeff", +1),
        ("economy.inflation.spending_pressure_coeff", +1),
    ],
    "economy.public_debt_pct_gdp": [
        ("economy.government_finances.surplus_debt_reduction", -1),
        ("economy.government_finances.other_spending", +1),
        ("economy.government_finances.austerity_spending_coeff", -1),
    ],
    "economy.tax_revenue_pct_gdp": [
        ("economy.government_finances.tax_auto_stabilizer", +1),
    ],
    "economy.gov_spending_pct_gdp": [
        ("economy.government_finances.other_spending", +1),
        ("economy.government_finances.unemployment_auto_stabilizer_coeff", +1),
    ],
    "economy.avg_annual_wage_eur": [
        ("economy.wages.gdp_passthrough", +1),
        ("economy.wages.inflation_passthrough", +1),
    ],
    "economy.interest_rate": [
        ("economy.interest_rate.inflation_response", +1),
    ],
    "economy.housing_price_index": [
        ("economy.housing.demand_pressure_coeff", +1),
        ("economy.housing.immigration_pressure_coeff", +1),
    ],
    "demographics.population_million": [
        ("demographics.migration.pull_factor_coeff", +1),
        ("demographics.population.natural_growth_coeff", +1),
    ],
    "demographics.fertility_rate": [
        ("demographics.fertility.affordability_coeff", +1),
        ("demographics.fertility.social_protection_coeff", +1),
    ],
    "demographics.life_expectancy": [
        ("demographics.life_expectancy.healthcare_coeff", +1),
        ("demographics.life_expectancy.wealth_coeff", +1),
    ],
    "demographics.pct_over_65": [
        ("demographics.age_structure.aging_rate", +1),
    ],
    "demographics.net_migration_per_1000": [
        ("demographics.migration.pull_factor_coeff", +1),
        ("demographics.migration.openness_coeff", +1),
    ],
    "social.life_satisfaction": [
        ("social.life_satisfaction.adaptation_speed", +1),
        ("social.life_satisfaction.income_baseline", -1),
    ],
    "social.gini_coefficient": [
        ("social.inequality.unemployment_coeff", +1),
        ("social.inequality.tax_coeff", +1),
    ],
    "social.poverty_rate": [
        ("social.poverty.base", +1),
        ("social.poverty.gini_coeff", +1),
        ("social.poverty.unemployment_coeff", +1),
    ],
    "social.housing_affordability": [
        ("social.housing_affordability.base", +1),
        ("social.housing_affordability.price_coeff", +1),
    ],
    "social.education_quality": [
        ("social.education_quality.spending_coeff", +1),
        ("social.education_quality.change_rate", +1),
    ],
    "social.healthcare_quality": [
        ("social.healthcare_quality.spending_coeff", +1),
        ("social.healthcare_quality.change_rate", +1),
    ],
    "governance.government_effectiveness": [
        ("social.governance_indices.effectiveness_policy_coeff", +1),
    ],
    "governance.corruption_control": [
        ("social.governance_indices.corruption_policy_coeff", +1),
    ],
    "governance.political_stability": [
        ("social.governance_indices.stability_coalition_coeff", +1),
    ],
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
# EVALUATE: run N seeds, return mean signed error per variable
# ---------------------------------------------------------------------------

def evaluate_weights(start_path, checkpoints, weights, seeds, quiet=False):
    """
    Returns: overall_wmape, {var_dotkey: mean_signed_error} averaged over
    all checkpoints and all seeds.
    """
    all_signed = {k: [] for k in CALIB_VARS}
    all_rel = []

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

            for var, meta in CALIB_VARS.items():
                sv = _getval(sim_rec, var)
                av = _getval(cp["data"], var)
                if sv is None or av is None:
                    continue
                if av != 0:
                    signed = (sv - av) / abs(av)
                    rel = abs(signed)
                else:
                    signed = sv
                    rel = abs(sv)
                all_signed[var].append(signed)
                all_rel.append(rel * meta["w"])

    mean_signed = {}
    for var, errs in all_signed.items():
        if errs:
            mean_signed[var] = statistics.mean(errs)

    total_w = sum(m["w"] for m in CALIB_VARS.values())
    wmape = sum(all_rel) / (len(seeds) * len(checkpoints) * total_w) if all_rel else 999
    return wmape, mean_signed


# ---------------------------------------------------------------------------
# TUNE: adjust weights based on bias
# ---------------------------------------------------------------------------

def tune_weights(weights: Weights, biases: Dict[str, float],
                 learning_rate: float = 0.15, min_bias: float = 0.05) -> int:
    """
    For each variable with a systematic bias, nudge the corresponding weights.
    Returns number of adjustments made.
    """
    adjustments = 0

    for var, signed_error in biases.items():
        if abs(signed_error) < min_bias:
            continue  # too small to bother

        mappings = VAR_TO_WEIGHTS.get(var)
        if not mappings:
            continue

        for weight_path, sign in mappings:
            current = weights.get(weight_path)
            if current is None:
                continue

            # If sim is too high (signed_error > 0), we want to DECREASE the sim value.
            # sign tells us: +1 = increasing this weight increases the sim value
            # So if sim is too high and sign is +1, we need to DECREASE the weight.
            # correction_direction = -1 if we need sim to go down, +1 if up
            correction_direction = -1 if signed_error > 0 else +1
            # Combined with sign: if sign=+1, correction_direction=-1 -> decrease weight
            adjust_direction = correction_direction * sign

            # Scale adjustment by the magnitude of the error (but cap it)
            magnitude = min(abs(signed_error), 2.0)  # cap at 200% error
            step = learning_rate * magnitude

            # Apply proportionally to current value (so we don't go crazy on small numbers)
            if abs(current) > 1e-6:
                new_val = current * (1 + adjust_direction * step)
            else:
                # For near-zero values, use absolute step
                new_val = current + adjust_direction * step * 0.01

            weights.set(weight_path, new_val)
            adjustments += 1

    return adjustments


# ---------------------------------------------------------------------------
# MAIN CALIBRATION LOOP
# ---------------------------------------------------------------------------

def run_calibration(start_path, checkpoint_paths, max_seeds=1000,
                    iterations=30, learning_rate=0.15, verbose=True):

    # Load checkpoints
    checkpoints = []
    for cp_path in checkpoint_paths:
        with open(cp_path, encoding="utf-8") as f:
            cp_data = json.load(f)
        checkpoints.append({"path": cp_path, "data": cp_data, "year": cp_data["start_year"]})

    # Clone default weights
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "weights", "default.json")
    tuned_path = os.path.join(base_dir, "weights", "tuned.json")
    weights = Weights(default_path)
    weights = weights.clone(tuned_path)

    seeds = list(range(max_seeds))
    # Use a subset of seeds for the tuning loop (speed), full set for final eval
    tune_seeds = seeds[:min(100, max_seeds)]

    if verbose:
        print(f"\n{'='*80}")
        print(f"  AI CALIBRATION — AUTOMATIC WEIGHT TUNING")
        print(f"{'='*80}")
        print(f"  Start:       {start_path}")
        for cp in checkpoints:
            print(f"  Checkpoint:  {cp['path']} (year {cp['year']})")
        print(f"  Seeds:       {max_seeds} total, {len(tune_seeds)} per tuning iteration")
        print(f"  Iterations:  {iterations}")
        print(f"  LR:          {learning_rate}")
        print(f"  Output:      {tuned_path}")
        print()

    # Initial eval
    t0 = time.time()
    best_wmape = 999
    best_weights_data = copy.deepcopy(weights.data)
    history = []

    for it in range(iterations):
        wmape, biases = evaluate_weights(start_path, checkpoints, weights, tune_seeds)

        # Track best
        if wmape < best_wmape:
            best_wmape = wmape
            best_weights_data = copy.deepcopy(weights.data)

        # Find top biases
        sorted_biases = sorted(biases.items(), key=lambda x: -abs(x[1]))
        top_biases = [(k, v) for k, v in sorted_biases if abs(v) > 0.05]

        if verbose:
            elapsed = time.time() - t0
            direction_str = " | ".join(
                f"{k.split('.')[-1]}={v:+.2f}" for k, v in top_biases[:5]
            )
            print(f"  [{it+1:3d}/{iterations}] WMAPE={wmape:.4f} (best={best_wmape:.4f}) "
                  f"| {elapsed:.0f}s | {direction_str}")

        history.append({"iteration": it+1, "wmape": wmape, "top_biases": dict(top_biases[:10])})

        # Tune
        n_adj = tune_weights(weights, biases, learning_rate=learning_rate)
        if n_adj == 0:
            if verbose:
                print(f"  No more adjustments to make — converged!")
            break

        # Decay learning rate slightly each iteration
        learning_rate *= 0.95

        weights.save()

    # Restore best weights
    weights._data = best_weights_data
    weights.save()

    # Final evaluation with ALL seeds
    if verbose:
        print(f"\n  Running final evaluation with all {max_seeds} seeds...")

    final_wmape, final_biases = evaluate_weights(start_path, checkpoints, weights, seeds)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  CALIBRATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Final WMAPE:  {final_wmape:.4f}")
        print(f"  Time:         {time.time()-t0:.1f}s")
        print(f"  Saved to:     {tuned_path}")
        print(f"\n  Remaining biases (|error| > 5%):")
        for k, v in sorted(final_biases.items(), key=lambda x: -abs(x[1])):
            if abs(v) > 0.05:
                direction = "HIGH" if v > 0 else "LOW"
                print(f"    {k.split('.')[-1]:<35} {direction:<5} {abs(v)*100:>6.1f}%")

    # Save report
    report = {
        "start_file": start_path,
        "checkpoints": [cp["path"] for cp in checkpoints],
        "max_seeds": max_seeds,
        "iterations": len(history),
        "final_wmape": final_wmape,
        "history": history,
        "final_biases": {k: round(v, 4) for k, v in final_biases.items()},
        "tuned_weights_path": tuned_path,
    }
    report_path = os.path.join(base_dir, "output", "calibration_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"  Report:       {report_path}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Calibration — automatic weight tuning")
    parser.add_argument("--start", default="input/spain_1994.json")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--max-seeds", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate")
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
                    args.iterations, args.lr, not args.quiet)


if __name__ == "__main__":
    main()