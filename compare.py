"""
Spain Simulation — Reality Check / Comparison Tool
====================================================
Compares a simulation output against a real-world input file to measure
how close the simulation got to reality.

Usage:
    python compare.py output/spain_1994_30y_seed42.json input/spain_2024.json
    python compare.py <simulated_output> <real_world_input>

The tool:
1. Reads the simulation output (yearly time-series)
2. Reads the real-world input (a single snapshot at a known year)
3. Finds the simulated data for that same year
4. Compares every variable: simulated vs actual
5. Computes error metrics (absolute error, % error, MAPE, etc.)
6. Generates a standalone HTML comparison dashboard
"""

import json
import os
import sys
import math
import argparse
from typing import Dict, Tuple, List


# ---------------------------------------------------------------------------
# VARIABLE METADATA — labels, formatting, and what "good accuracy" looks like
# ---------------------------------------------------------------------------

VARIABLE_META = {
    # Economy
    "economy.gdp_billion_eur":           {"label": "GDP (€ billion)",            "fmt": ",.0f", "unit": "€B",  "tolerance": 0.15},
    "economy.gdp_growth_rate":           {"label": "GDP Growth Rate",            "fmt": ".1%",  "unit": "%",   "tolerance": 0.5, "is_pct": True},
    "economy.gdp_per_capita_eur":        {"label": "GDP per Capita (€)",         "fmt": ",.0f", "unit": "€",   "tolerance": 0.15},
    "economy.inflation_rate":            {"label": "Inflation Rate",             "fmt": ".1%",  "unit": "%",   "tolerance": 0.5, "is_pct": True},
    "economy.unemployment_rate":         {"label": "Unemployment Rate",          "fmt": ".1%",  "unit": "%",   "tolerance": 0.25, "is_pct": True},
    "economy.youth_unemployment_rate":   {"label": "Youth Unemployment",         "fmt": ".1%",  "unit": "%",   "tolerance": 0.25, "is_pct": True},
    "economy.public_debt_pct_gdp":       {"label": "Public Debt (% GDP)",        "fmt": ".0%",  "unit": "%",   "tolerance": 0.20, "is_pct": True},
    "economy.budget_deficit_pct_gdp":    {"label": "Budget Balance (% GDP)",     "fmt": "+.1%", "unit": "%",   "tolerance": 0.5, "is_pct": True},
    "economy.tax_revenue_pct_gdp":       {"label": "Tax Revenue (% GDP)",        "fmt": ".1%",  "unit": "%",   "tolerance": 0.15, "is_pct": True},
    "economy.gov_spending_pct_gdp":      {"label": "Gov Spending (% GDP)",       "fmt": ".1%",  "unit": "%",   "tolerance": 0.15, "is_pct": True},
    "economy.healthcare_spending_pct_gdp": {"label": "Healthcare Spend (% GDP)", "fmt": ".1%",  "unit": "%",   "tolerance": 0.20, "is_pct": True},
    "economy.education_spending_pct_gdp":  {"label": "Education Spend (% GDP)",  "fmt": ".1%",  "unit": "%",   "tolerance": 0.20, "is_pct": True},
    "economy.rd_spending_pct_gdp":       {"label": "R&D Spend (% GDP)",          "fmt": ".1%",  "unit": "%",   "tolerance": 0.30, "is_pct": True},
    "economy.social_protection_pct_gdp": {"label": "Social Protection (% GDP)",  "fmt": ".1%",  "unit": "%",   "tolerance": 0.20, "is_pct": True},
    "economy.avg_annual_wage_eur":       {"label": "Avg Annual Wage (€)",        "fmt": ",.0f", "unit": "€",   "tolerance": 0.20},
    "economy.interest_rate":             {"label": "Interest Rate",              "fmt": ".1%",  "unit": "%",   "tolerance": 0.5, "is_pct": True},
    "economy.tourism_revenue_pct_gdp":   {"label": "Tourism Revenue (% GDP)",    "fmt": ".1%",  "unit": "%",   "tolerance": 0.25, "is_pct": True},
    "economy.housing_price_index":       {"label": "Housing Price Index",        "fmt": ".0f",  "unit": "",    "tolerance": 0.25},
    # Demographics
    "demographics.population_million":   {"label": "Population (M)",             "fmt": ".1f",  "unit": "M",   "tolerance": 0.05},
    "demographics.fertility_rate":       {"label": "Fertility Rate",             "fmt": ".2f",  "unit": "",    "tolerance": 0.15},
    "demographics.life_expectancy":      {"label": "Life Expectancy",            "fmt": ".1f",  "unit": "yr",  "tolerance": 0.03},
    "demographics.median_age":           {"label": "Median Age",                 "fmt": ".1f",  "unit": "yr",  "tolerance": 0.10},
    "demographics.pct_under_15":         {"label": "% Under 15",                "fmt": ".1%",  "unit": "%",   "tolerance": 0.15, "is_pct": True},
    "demographics.pct_15_64":            {"label": "% Working Age (15-64)",      "fmt": ".1%",  "unit": "%",   "tolerance": 0.10, "is_pct": True},
    "demographics.pct_over_65":          {"label": "% Over 65",                 "fmt": ".1%",  "unit": "%",   "tolerance": 0.15, "is_pct": True},
    "demographics.dependency_ratio":     {"label": "Dependency Ratio",           "fmt": ".2f",  "unit": "",    "tolerance": 0.15},
    "demographics.net_migration_per_1000": {"label": "Net Migration (per 1000)", "fmt": ".1f",  "unit": "‰",   "tolerance": 0.50},
    # Social
    "social.life_satisfaction":          {"label": "Life Satisfaction (0-10)",    "fmt": ".1f",  "unit": "/10", "tolerance": 0.10},
    "social.trust_in_government":        {"label": "Trust in Government",        "fmt": ".0%",  "unit": "%",   "tolerance": 0.30, "is_pct": True},
    "social.gini_coefficient":           {"label": "Gini Coefficient",           "fmt": ".3f",  "unit": "",    "tolerance": 0.10},
    "social.poverty_rate":               {"label": "Poverty Rate",               "fmt": ".1%",  "unit": "%",   "tolerance": 0.20, "is_pct": True},
    "social.voter_turnout":              {"label": "Voter Turnout",              "fmt": ".0%",  "unit": "%",   "tolerance": 0.15, "is_pct": True},
    "social.housing_affordability":      {"label": "Housing Affordability",      "fmt": ".0f",  "unit": "/100","tolerance": 0.25},
    "social.education_quality":          {"label": "Education Quality",          "fmt": ".0f",  "unit": "/100","tolerance": 0.15},
    "social.healthcare_quality":         {"label": "Healthcare Quality",         "fmt": ".0f",  "unit": "/100","tolerance": 0.15},
    "social.environmental_quality":      {"label": "Environmental Quality",      "fmt": ".0f",  "unit": "/100","tolerance": 0.20},
    "social.mental_health_index":        {"label": "Mental Health Index",        "fmt": ".0f",  "unit": "/100","tolerance": 0.25},
    "social.regional_separatism_tension": {"label": "Regional Tensions",         "fmt": ".0f",  "unit": "/100","tolerance": 0.30},
    # Governance
    "governance.rule_of_law":            {"label": "Rule of Law",                "fmt": ".0f",  "unit": "/100","tolerance": 0.10},
    "governance.corruption_control":     {"label": "Corruption Control",         "fmt": ".0f",  "unit": "/100","tolerance": 0.15},
    "governance.government_effectiveness": {"label": "Gov Effectiveness",        "fmt": ".0f",  "unit": "/100","tolerance": 0.10},
    "governance.political_stability":    {"label": "Political Stability",        "fmt": ".0f",  "unit": "/100","tolerance": 0.20},
    "governance.regulatory_quality":     {"label": "Regulatory Quality",         "fmt": ".0f",  "unit": "/100","tolerance": 0.10},
}

CATEGORIES = {
    "economy": "Economy",
    "demographics": "Demographics",
    "social": "Wellbeing & Social",
    "governance": "Governance",
}


# ---------------------------------------------------------------------------
# COMPARISON LOGIC
# ---------------------------------------------------------------------------

def get_val(data: dict, dotkey: str):
    parts = dotkey.split(".")
    v = data
    for p in parts:
        if p not in v:
            return None
        v = v[p]
    return v


def compute_error(simulated: float, actual: float, meta: dict) -> dict:
    """Compute error metrics for a single variable."""
    if actual is None or simulated is None:
        return None

    abs_err = simulated - actual
    # For percentage-point variables, error is in pp
    is_pct = meta.get("is_pct", False)
    if is_pct:
        abs_err_display = (simulated - actual) * 100  # in percentage points
    else:
        abs_err_display = abs_err

    # Relative error (% of actual)
    if actual != 0:
        rel_err = abs(simulated - actual) / abs(actual)
    else:
        rel_err = abs(simulated) if simulated != 0 else 0.0

    tolerance = meta.get("tolerance", 0.15)
    grade = "excellent" if rel_err < tolerance * 0.5 else \
            "good" if rel_err < tolerance else \
            "fair" if rel_err < tolerance * 2 else "poor"

    return {
        "simulated": simulated,
        "actual": actual,
        "abs_error": abs_err,
        "abs_error_display": abs_err_display,
        "rel_error": rel_err,
        "tolerance": tolerance,
        "grade": grade,
        "is_pct": is_pct,
    }


def run_comparison(sim_path: str, real_path: str) -> dict:
    """Main comparison logic."""

    with open(sim_path, encoding="utf-8") as f:
        sim_data = json.load(f)
    with open(real_path, encoding="utf-8") as f:
        real_data = json.load(f)

    real_year = real_data.get("start_year", None)
    if real_year is None:
        print("ERROR: Real-world file has no 'start_year' field.")
        sys.exit(1)

    sim_start = sim_data["simulation"]["start_year"]
    sim_end = sim_data["simulation"]["end_year"]

    if real_year < sim_start or real_year > sim_end:
        print(f"ERROR: Real-world year {real_year} is outside simulation range {sim_start}-{sim_end}")
        sys.exit(1)

    # Find the simulated record for the real year
    sim_record = None
    for record in sim_data["yearly_data"]:
        if record["year"] == real_year:
            sim_record = record
            break

    if sim_record is None:
        print(f"ERROR: No simulated data found for year {real_year}")
        sys.exit(1)

    # Compare every variable
    results = {}
    for dotkey, meta in VARIABLE_META.items():
        sim_val = get_val(sim_record, dotkey)
        real_val = get_val(real_data, dotkey)
        if sim_val is not None and real_val is not None:
            err = compute_error(sim_val, real_val, meta)
            if err:
                err["label"] = meta["label"]
                err["fmt"] = meta["fmt"]
                err["unit"] = meta["unit"]
                err["category"] = dotkey.split(".")[0]
                results[dotkey] = err

    # Aggregate scores
    grades = [r["grade"] for r in results.values()]
    grade_counts = {g: grades.count(g) for g in ["excellent", "good", "fair", "poor"]}
    rel_errors = [r["rel_error"] for r in results.values()]
    mape = sum(rel_errors) / len(rel_errors) * 100 if rel_errors else 0

    overall = "A" if mape < 10 else "B" if mape < 20 else "C" if mape < 35 else "D" if mape < 50 else "F"

    # Simulated trajectory for charts (full time-series)
    trajectory = {}
    for dotkey in VARIABLE_META:
        series = []
        for record in sim_data["yearly_data"]:
            v = get_val(record, dotkey)
            if v is not None:
                series.append({"year": record["year"], "value": v})
        if series:
            trajectory[dotkey] = series

    return {
        "sim_file": sim_path,
        "real_file": real_path,
        "sim_start": sim_start,
        "sim_end": sim_end,
        "comparison_year": real_year,
        "years_simulated": real_year - sim_start,
        "variables_compared": len(results),
        "mape": mape,
        "overall_grade": overall,
        "grade_counts": grade_counts,
        "results": results,
        "trajectory": trajectory,
        "sim_years": [r["year"] for r in sim_data["yearly_data"]],
    }


# ---------------------------------------------------------------------------
# HTML REPORT GENERATION
# ---------------------------------------------------------------------------

def generate_html(comparison: dict) -> str:
    """Generate a standalone HTML comparison report."""

    comp_json = json.dumps(comparison, default=str)
    categories_json = json.dumps(CATEGORIES)

    # Build HTML with safe placeholders, then replace at the end
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spain Sim &#8212; Reality Check</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg: #0b0d11; --card: #13161e; --card2: #181c26; --border: #242936;
  --text: #e4e7ee; --text2: #8890a4; --muted: #4e566a;
  --blue: #3b82f6; --cyan: #22d3ee; --green: #10b981; --amber: #f59e0b;
  --rose: #f43f5e; --violet: #8b5cf6;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

.header { background:rgba(11,13,17,0.88); backdrop-filter:blur(16px); border-bottom:1px solid var(--border);
  padding:0 2rem; position:sticky; top:0; z-index:50; }
.header-inner { max-width:1400px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; height:60px; }
.header h1 { font-size:1.05rem; font-weight:600; letter-spacing:-0.02em; }
.header-sub { font-size:0.78rem; color:var(--muted); font-family:'JetBrains Mono',monospace; }

.container { max-width:1400px; margin:0 auto; padding:1.5rem 2rem 3rem; }

.score-strip { display:grid; grid-template-columns:200px 1fr; gap:20px; margin-bottom:1.5rem; }
.score-ring { background:var(--card); border:1px solid var(--border); border-radius:16px;
  display:flex; flex-direction:column; align-items:center; justify-content:center; padding:28px 20px; }
.score-letter { font-size:4rem; font-weight:700; line-height:1; }
.score-letter.A { color:var(--green); } .score-letter.B { color:var(--cyan); }
.score-letter.C { color:var(--amber); } .score-letter.D { color:var(--rose); }
.score-letter.F { color:var(--rose); }
.score-sub { font-size:0.75rem; color:var(--text2); margin-top:6px; text-align:center; }

.score-details { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:20px 24px;
  display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:14px; align-content:center; }
.sd-item { display:flex; flex-direction:column; gap:2px; }
.sd-label { font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; }
.sd-value { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:600; }
.sd-value.green { color:var(--green); } .sd-value.amber { color:var(--amber); }
.sd-value.rose { color:var(--rose); } .sd-value.blue { color:var(--blue); }

.tabs { display:flex; gap:6px; margin-bottom:1.25rem; flex-wrap:wrap; }
.tab { padding:8px 18px; border-radius:100px; background:var(--card); border:1px solid var(--border);
  color:var(--text2); font-size:0.82rem; font-weight:500; cursor:pointer; transition:all 0.15s;
  font-family:'DM Sans',sans-serif; }
.tab:hover { background:var(--card2); color:var(--text); }
.tab.active { background:var(--blue); border-color:var(--blue); color:#fff; }

.var-grid { display:flex; flex-direction:column; gap:8px; }
.var-row { background:var(--card); border:1px solid var(--border); border-radius:12px;
  display:grid; grid-template-columns:240px 110px 110px 90px 80px 1fr; align-items:center;
  padding:12px 18px; gap:10px; transition:border-color 0.15s; cursor:pointer; }
.var-row:hover { border-color:#333d50; }
.var-row.expanded { border-color:var(--blue); border-bottom-left-radius:0; border-bottom-right-radius:0; }
.var-name { font-size:0.85rem; font-weight:500; }
.var-val { font-family:'JetBrains Mono',monospace; font-size:0.85rem; text-align:right; }
.var-err { font-family:'JetBrains Mono',monospace; font-size:0.8rem; text-align:right; }
.var-grade { font-size:0.72rem; font-weight:600; text-transform:uppercase; text-align:center;
  padding:3px 10px; border-radius:100px; }
.var-grade.excellent { background:rgba(16,185,129,0.15); color:var(--green); }
.var-grade.good { background:rgba(34,211,238,0.12); color:var(--cyan); }
.var-grade.fair { background:rgba(245,158,11,0.12); color:var(--amber); }
.var-grade.poor { background:rgba(244,63,94,0.12); color:var(--rose); }
.var-bar-cell { padding-right:8px; }
.var-bar-track { height:8px; background:var(--card2); border-radius:4px; overflow:hidden; position:relative; }
.var-bar-fill { height:100%; border-radius:4px; transition:width 0.3s; }

.var-chart-area { background:var(--card); border:1px solid var(--blue); border-top:none;
  border-radius:0 0 12px 12px; padding:16px 18px; margin-bottom:8px; margin-top:-8px; }
.var-chart-wrap { height:200px; position:relative; }

.row-header { display:grid; grid-template-columns:240px 110px 110px 90px 80px 1fr;
  padding:6px 18px; gap:10px; font-size:0.68rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.04em; font-weight:500; }
.row-header span:nth-child(2), .row-header span:nth-child(3), .row-header span:nth-child(4) { text-align:right; }
.row-header span:nth-child(5) { text-align:center; }

@media(max-width:900px) {
  .var-row { grid-template-columns:1fr 80px 80px 70px 70px; }
  .var-bar-cell { display:none; }
  .row-header { grid-template-columns:1fr 80px 80px 70px 70px; }
  .row-header span:last-child { display:none; }
  .score-strip { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <h1>Reality Check &#8212; Simulation vs Actual</h1>
    <div class="header-sub" id="headerMeta"></div>
  </div>
</div>

<div class="container">
  <div class="score-strip" id="scoreStrip"></div>
  <div class="tabs" id="tabs"></div>
  <div class="row-header">
    <span>Variable</span><span>Simulated</span><span>Actual</span><span>Error</span><span>Grade</span><span></span>
  </div>
  <div class="var-grid" id="varGrid"></div>
</div>

<script>
const C = __COMP_DATA__;
const CAT_MAP = __CATEGORIES__;

// header
document.getElementById('headerMeta').textContent =
  'Simulated ' + C.sim_start + ' -> ' + C.sim_end + ' | Comparing at year ' + C.comparison_year + ' (' + C.years_simulated + 'y forward)';

// Score strip
var gc = {A:'green',B:'blue',C:'amber',D:'rose',F:'rose'};
document.getElementById('scoreStrip').innerHTML =
  '<div class="score-ring">' +
    '<div class="score-letter ' + gc[C.overall_grade] + '">' + C.overall_grade + '</div>' +
    '<div class="score-sub">Overall accuracy<br>MAPE: ' + C.mape.toFixed(1) + '%</div>' +
  '</div>' +
  '<div class="score-details">' +
    '<div class="sd-item"><div class="sd-label">Variables</div><div class="sd-value blue">' + C.variables_compared + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Excellent</div><div class="sd-value green">' + (C.grade_counts.excellent||0) + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Good</div><div class="sd-value" style="color:var(--cyan)">' + (C.grade_counts.good||0) + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Fair</div><div class="sd-value amber">' + (C.grade_counts.fair||0) + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Poor</div><div class="sd-value rose">' + (C.grade_counts.poor||0) + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Years Forward</div><div class="sd-value blue">' + C.years_simulated + '</div></div>' +
    '<div class="sd-item"><div class="sd-label">Mean Abs % Error</div><div class="sd-value ' + (C.mape<20?'green':C.mape<35?'amber':'rose') + '">' + C.mape.toFixed(1) + '%</div></div>' +
  '</div>';

// Category tabs
var cats = ['all'].concat(Object.keys(CAT_MAP));
var catLabels = Object.assign({all:'All'}, CAT_MAP);
var activeCat = 'all';

function renderTabs() {
  document.getElementById('tabs').innerHTML = cats.map(function(c) {
    return '<button class="tab ' + (c===activeCat?'active':'') + '" onclick="setCat(\'' + c + '\')">' + catLabels[c] + '</button>';
  }).join('');
}

function setCat(c) { activeCat = c; renderTabs(); renderVars(); }

var chartInstances = {};

function fmtVal(v, meta) {
  if (meta.is_pct) {
    var d = (meta.fmt && meta.fmt.indexOf('0%') >= 0) ? 0 : 1;
    return (v*100).toFixed(d) + '%';
  }
  if (meta.fmt && meta.fmt.indexOf(',') >= 0) return v.toLocaleString('en', {maximumFractionDigits:0});
  var m = meta.fmt ? meta.fmt.match(/\.(\d)/) : null;
  var dec = m ? parseInt(m[1]) : 2;
  return Number(v).toFixed(dec);
}

function renderVars() {
  var grid = document.getElementById('varGrid');
  var entries = Object.entries(C.results)
    .filter(function(e) { return activeCat === 'all' || e[1].category === activeCat; })
    .sort(function(a,b) { return b[1].rel_error - a[1].rel_error; });

  grid.innerHTML = entries.map(function(e) {
    var key = e[0], r = e[1];
    var safeKey = key.replace(/\./g,'_');
    var barPct = Math.min(100, r.rel_error / 0.5 * 100);
    var barColor = r.grade==='excellent'?'var(--green)':r.grade==='good'?'var(--cyan)':r.grade==='fair'?'var(--amber)':'var(--rose)';
    var errStr = r.is_pct
      ? (r.abs_error_display>=0?'+':'') + r.abs_error_display.toFixed(1) + 'pp'
      : (r.rel_error*100 < 1000 ? (r.rel_error*100).toFixed(1)+'%' : '>999%');
    return '<div class="var-row" id="row_' + safeKey + '" onclick="toggleChart(\'' + key + '\')">' +
        '<div class="var-name">' + r.label + '</div>' +
        '<div class="var-val">' + fmtVal(r.simulated, r) + '</div>' +
        '<div class="var-val" style="color:var(--cyan)">' + fmtVal(r.actual, r) + '</div>' +
        '<div class="var-err">' + errStr + '</div>' +
        '<div><span class="var-grade ' + r.grade + '">' + r.grade + '</span></div>' +
        '<div class="var-bar-cell"><div class="var-bar-track"><div class="var-bar-fill" style="width:' + Math.min(100,barPct) + '%;background:' + barColor + '"></div></div></div>' +
      '</div>' +
      '<div class="var-chart-area" id="chart_area_' + safeKey + '" style="display:none">' +
        '<div class="var-chart-wrap"><canvas id="canvas_' + safeKey + '"></canvas></div>' +
      '</div>';
  }).join('');
}

function toggleChart(key) {
  var safeKey = key.replace(/\./g,'_');
  var area = document.getElementById('chart_area_' + safeKey);
  var row = document.getElementById('row_' + safeKey);
  var visible = area.style.display !== 'none';

  document.querySelectorAll('.var-chart-area').forEach(function(el) { el.style.display = 'none'; });
  document.querySelectorAll('.var-row').forEach(function(el) { el.classList.remove('expanded'); });

  if (visible) return;

  area.style.display = 'block';
  row.classList.add('expanded');

  if (chartInstances[key]) { chartInstances[key].destroy(); delete chartInstances[key]; }

  var canvas = document.getElementById('canvas_' + safeKey);
  var traj = C.trajectory[key] || [];
  var r = C.results[key];
  var isPct = r.is_pct;

  var simData = traj.map(function(t) { return { x: t.year, y: isPct ? t.value*100 : t.value }; });
  var actualPoint = [{ x: C.comparison_year, y: isPct ? r.actual*100 : r.actual }];

  chartInstances[key] = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Simulated', data: simData, borderColor: 'rgba(59,130,246,0.9)',
          backgroundColor: 'rgba(59,130,246,0.08)', borderWidth: 2, pointRadius: 0,
          tension: 0.3, fill: true },
        { label: 'Actual (' + C.comparison_year + ')', data: actualPoint,
          borderColor: 'rgba(34,211,238,1)', backgroundColor: 'rgba(34,211,238,1)',
          pointRadius: 8, pointStyle: 'crossRot', pointBorderWidth: 3, showLine: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type:'linear', grid:{display:false}, ticks:{callback:function(v){return v.toFixed(0)}} },
        y: { grid:{color:'rgba(37,42,54,0.5)'}, ticks:{callback:function(v){return isPct ? v.toFixed(1)+'%' : v.toLocaleString()}} }
      },
      plugins: {
        legend: { position:'top', labels:{ boxWidth:10, padding:8, font:{size:11} } },
        tooltip: { backgroundColor:'rgba(18,21,28,0.95)', borderColor:'#252a36', borderWidth:1, cornerRadius:8, padding:10 }
      }
    }
  });
}

renderTabs();
renderVars();
</script>
</body>
</html>"""

    # Inject data via simple string replacement
    html = html.replace("__COMP_DATA__", comp_json)
    html = html.replace("__CATEGORIES__", categories_json)
    return html


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation output against real-world data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python compare.py output/spain_1994_30y_seed42.json input/spain_2024.json
  python compare.py output/spain_2024_30y_seed42.json input/spain_2024.json  (self-check at year 0)
""")
    parser.add_argument("simulation", help="Path to simulation output JSON")
    parser.add_argument("reality", help="Path to real-world input JSON (the ground truth)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML path (default: auto-generated in output/)")
    args = parser.parse_args()

    print(f"\n  Simulation: {args.simulation}")
    print(f"  Reality:    {args.reality}")

    comparison = run_comparison(args.simulation, args.reality)

    print(f"\n  Comparison year: {comparison['comparison_year']}")
    print(f"  Variables compared: {comparison['variables_compared']}")
    print(f"  MAPE: {comparison['mape']:.1f}%")
    print(f"  Overall grade: {comparison['overall_grade']}")
    print(f"  Breakdown: {comparison['grade_counts']}")

    # Console table
    print(f"\n  {'Variable':<30} {'Simulated':>12} {'Actual':>12} {'Error':>10} {'Grade':>10}")
    print(f"  {'-'*74}")
    for key in sorted(comparison["results"], key=lambda k: -comparison["results"][k]["rel_error"]):
        r = comparison["results"][key]
        sim_s = fmtVal_console(r["simulated"], r)
        act_s = fmtVal_console(r["actual"], r)
        err_s = f"{r['rel_error']*100:.1f}%"
        print(f"  {r['label']:<30} {sim_s:>12} {act_s:>12} {err_s:>10} {r['grade']:>10}")

    # Generate HTML
    html = generate_html(comparison)

    if args.output:
        out_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "comparisons")
        os.makedirs(out_dir, exist_ok=True)
        sim_name = os.path.splitext(os.path.basename(args.simulation))[0]
        real_name = os.path.splitext(os.path.basename(args.reality))[0]
        out_path = os.path.join(out_dir, f"compare_{sim_name}_vs_{real_name}.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report saved to: {out_path}")
    print()


def fmtVal_console(v, r):
    if r.get("is_pct"):
        return f"{v*100:.1f}%"
    elif v > 1000:
        return f"{v:,.0f}"
    else:
        return f"{v:.2f}"


if __name__ == "__main__":
    main()