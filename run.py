"""
Spain Government Simulation — Runner
=====================================
Usage:
    python run.py                                     # defaults: input/spain_2024.json, 30y
    python run.py --input input/spain_1994.json --years 30
    python run.py -i input/spain_1994.json -y 50

Output goes to:  output/<input_name>_<years>y.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import SimulationEngine, load_state_from_json, Weights
from spanish_parliamentary import SpanishParliamentarySystem


def format_table_row(label, *values):
    cols = [f"{label:<28}"]
    for v in values:
        cols.append(f"{v:>14}")
    return " | ".join(cols)


def run_simulation(input_path: str, years: int, weights_path: str = None):
    # --- Load initial state ---
    if os.path.isfile(input_path):
        initial_state = load_state_from_json(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
    else:
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # --- Load weights ---
    weights = Weights(weights_path) if weights_path else Weights()

    seed = 0
    start_year = initial_state.year
    end_year = start_year + years

    print("=" * 90)
    print("  SPAIN GOVERNMENT SYSTEM SIMULATION")
    print("  System: Spanish Parliamentary Democracy (D'Hondt Proportional Representation)")
    print(f"  Input:  {input_path}")
    print(f"  Period: {start_year} -> {end_year}  ({years} years)")
    print("=" * 90)
    print()

    # --- Initialize & Run ---
    system = SpanishParliamentarySystem()
    engine = SimulationEngine(system, seed=seed, initial_state=initial_state, weights=weights)
    history = engine.run(years)

    # --- Yearly summary ---
    print("YEAR-BY-YEAR KEY INDICATORS")
    print("-" * 90)
    print(format_table_row("Year / Metric", "GDP Grow%", "Unemp%", "Debt/GDP%", "Infl%"))
    print(format_table_row("", "LifeSat", "Gini", "TrustGov%", "Pop(M)"))
    print("-" * 90)

    for record in history:
        yr = record["year"]
        eco = record["economy"]
        soc = record["social"]
        demo = record["demographics"]
        gov_s = record["government"]
        print(format_table_row(f"{yr}",
            f"{eco['gdp_growth_rate']*100:+.1f}%", f"{eco['unemployment_rate']*100:.1f}%",
            f"{eco['public_debt_pct_gdp']*100:.0f}%", f"{eco['inflation_rate']*100:.1f}%"))
        print(format_table_row(f"  Gov: {'+'.join(gov_s.get('coalition',['?'])[:3])}",
            f"{soc['life_satisfaction']:.1f}/10", f"{soc['gini_coefficient']:.3f}",
            f"{soc['trust_in_government']*100:.0f}%", f"{demo['population_million']:.1f}M"))
        if yr % 5 == 0 or yr == end_year:
            print("-" * 90)

    # --- Elections ---
    print("\nELECTION RESULTS\n" + "-" * 90)
    for el in system.election_history:
        print(f"\n  Election Year: {el['year']}")
        print(f"  {'Party':<10} {'Vote%':>8} {'Seats':>8}\n  {'-'*26}")
        for party in sorted(el["seats"], key=lambda x: -el["seats"].get(x, 0)):
            s = el["seats"].get(party, 0)
            if s > 0:
                print(f"  {party:<10} {el['vote_shares'].get(party,0)*100:>7.1f}% {s:>7}")
        print(f"  -> Government: {' + '.join(el['coalition'])} (stability: {el['stability']:.0%})")

    # --- Comparison table ---
    final, initial_rec = history[-1], history[0]
    print(f"\n{'='*90}\n  RESULTS: {start_year} vs {end_year}\n{'='*90}")
    sections = [
        ("ECONOMY", [
            ("GDP (EUR billion)", "economy", "gdp_billion_eur", ",.0f"),
            ("GDP Growth Rate", "economy", "gdp_growth_rate", ".1%"),
            ("GDP per Capita (EUR)", "economy", "gdp_per_capita_eur", ",.0f"),
            ("Unemployment Rate", "economy", "unemployment_rate", ".1%"),
            ("Youth Unemployment", "economy", "youth_unemployment_rate", ".1%"),
            ("Inflation Rate", "economy", "inflation_rate", ".1%"),
            ("Public Debt (% GDP)", "economy", "public_debt_pct_gdp", ".0%"),
            ("Budget Balance (% GDP)", "economy", "budget_deficit_pct_gdp", "+.1%"),
            ("Healthcare Spend (%GDP)", "economy", "healthcare_spending_pct_gdp", ".1%"),
            ("Education Spend (%GDP)", "economy", "education_spending_pct_gdp", ".1%"),
            ("R&D Spend (%GDP)", "economy", "rd_spending_pct_gdp", ".1%"),
            ("Avg Annual Wage (EUR)", "economy", "avg_annual_wage_eur", ",.0f"),
        ]),
        ("DEMOGRAPHICS", [
            ("Population (millions)", "demographics", "population_million", ".1f"),
            ("Fertility Rate", "demographics", "fertility_rate", ".2f"),
            ("Life Expectancy", "demographics", "life_expectancy", ".1f"),
            ("% Over 65", "demographics", "pct_over_65", ".1%"),
            ("Dependency Ratio", "demographics", "dependency_ratio", ".2f"),
        ]),
        ("WELLBEING", [
            ("Life Satisfaction (0-10)", "social", "life_satisfaction", ".1f"),
            ("Trust in Government", "social", "trust_in_government", ".0%"),
            ("Gini Coefficient", "social", "gini_coefficient", ".3f"),
            ("Poverty Rate", "social", "poverty_rate", ".1%"),
            ("Housing Affordability", "social", "housing_affordability", ".0f"),
            ("Education Quality", "social", "education_quality", ".0f"),
            ("Healthcare Quality", "social", "healthcare_quality", ".0f"),
            ("Mental Health Index", "social", "mental_health_index", ".0f"),
            ("Voter Turnout", "social", "voter_turnout", ".0%"),
        ]),
        ("GOVERNANCE", [
            ("Rule of Law", "governance", "rule_of_law", ".0f"),
            ("Corruption Control", "governance", "corruption_control", ".0f"),
            ("Gov Effectiveness", "governance", "government_effectiveness", ".0f"),
            ("Political Stability", "governance", "political_stability", ".0f"),
        ]),
    ]
    for section_name, metrics in sections:
        print(f"\n  {section_name}")
        print(f"  {'Metric':<30} {str(start_year):>12} {str(end_year):>12} {'Change':>12}")
        print(f"  {'-'*66}")
        for label, cat, key, fmt in metrics:
            vi, vf = initial_rec[cat][key], final[cat][key]
            if "%" in fmt:
                ch = f"{(vf-vi)*100:+.1f}pp"
            elif vi != 0:
                ch = f"{(vf-vi)/abs(vi)*100:+.1f}%"
            else:
                ch = "n/a"
            vi_s = f"{vi:{fmt}}"
            vf_s = f"{vf:{fmt}}"
            print(f"  {label:<30} {vi_s:>12} {vf_s:>12} {ch:>12}")

    # --- Save output ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{input_name}_{years}y.json"
    output_path = os.path.join(output_dir, output_filename)

    output_data = {
        "simulation": {
            "system": "Spanish Parliamentary (D'Hondt PR)",
            "input_file": input_path,
            "years_simulated": years,
            "start_year": start_year,
            "end_year": end_year,
        },
        "election_history": system.election_history,
        "yearly_data": history,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n  Output saved to: {output_path}")
    print()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spain Government System Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run.py
  python run.py --input input/spain_1994.json --years 30
  python run.py -i input/spain_2024.json -y 50""")
    parser.add_argument("-i", "--input", default="input/spain_2024.json",
                        help="Path to initial state JSON (default: input/spain_2024.json)")
    parser.add_argument("-y", "--years", type=int, default=30,
                        help="Number of years to simulate (default: 30)")
    parser.add_argument("-w", "--weights", default=None,
                        help="Path to weights JSON (default: weights/default.json)")
    args = parser.parse_args()
    run_simulation(input_path=args.input, years=args.years,
                   weights_path=args.weights)