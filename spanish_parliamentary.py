"""
Spanish Parliamentary System (Current System)
==============================================
Implements Spain's actual government system:
- Bicameral parliament (Congress of Deputies 350 seats, Senate)
- D'Hondt proportional representation at province level
- 52 constituencies with minimum 2 seats each
- 3% electoral threshold per constituency
- Parliamentary investiture (PM needs majority)
- Coalition government formation
- 4-year election cycles (can be called early)

This is the BASELINE system against which alternatives will be compared.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from engine import (GovernmentSystem, GovernmentAction, SimulationState,
                    Party, SimulationEngine)

# ---------------------------------------------------------------------------
# SIMPLIFIED CONSTITUENCY MAP
# We aggregate Spain's 52 constituencies into size categories
# to capture the key D'Hondt distortion effect without modeling each province
# ---------------------------------------------------------------------------

# Spain's seats distribution creates systematic biases:
# - Small provinces (2-5 seats): heavily favor big parties (PP, PSOE)
# - Large provinces (10+ seats): more proportional, allow small parties
# This is a key feature of the system.

CONSTITUENCY_STRUCTURE = [
    # (num_constituencies, seats_each, label)
    (28, 3, "small"),  # 28 provinces with ~3 seats avg (84 seats)
    (12, 5, "medium_small"),  # 12 provinces with ~5 seats (60 seats)
    (6, 8, "medium"),  # 6 provinces with ~8 seats (48 seats)
    (3, 15, "large"),  # 3 provinces with ~15 seats (45 seats)
    (1, 37, "madrid"),  # Madrid: 37 seats
    (1, 33, "barcelona"),  # Barcelona: 33 seats
    (1, 2, "ceuta_melilla"),  # Ceuta+Melilla: 2 seats (FPTP-like)
]
# Total: 52 constituencies, ~350 seats (simplified approximation)

TOTAL_SEATS = 350
ELECTORAL_THRESHOLD = 0.03  # 3% per constituency


def dhondt_allocation(votes: Dict[str, float], num_seats: int,
                      threshold: float = 0.03) -> Dict[str, int]:
    """
    D'Hondt method of seat allocation.
    This is the actual method used in Spanish elections.

    It systematically favors larger parties, especially in small constituencies.
    This is not a bug — it's a documented feature of Spain's system.
    """
    # Apply threshold
    total_votes = sum(votes.values())
    eligible = {k: v for k, v in votes.items()
                if total_votes > 0 and v / total_votes >= threshold}

    if not eligible:
        # If no party passes threshold, give all seats to largest party
        if votes:
            winner = max(votes, key=votes.get)
            return {winner: num_seats}
        return {}

    seats = {party: 0 for party in eligible}

    for _ in range(num_seats):
        # Calculate quotients: votes / (seats_won + 1)
        quotients = {party: eligible[party] / (seats[party] + 1)
                     for party in eligible}
        winner = max(quotients, key=quotients.get)
        seats[winner] += 1

    return seats


# ---------------------------------------------------------------------------
# VOTE SHARE MODEL
# ---------------------------------------------------------------------------

def calculate_vote_shares(parties: List[Party], state: SimulationState,
                          governing_parties: List[str],
                          years_in_power: int,
                          rng: random.Random) -> Dict[str, float]:
    """
    Model how citizens vote based on:
    1. Party base (ideological loyalty)
    2. Economic performance (retrospective voting)
    3. Incumbent fatigue
    4. Issue salience (which issues matter most right now)
    5. Random variation (campaigns, scandals, etc.)

    This is based on political science models of vote choice,
    particularly the economic voting literature applied to Spain.
    """
    eco = state.economy
    soc = state.social

    raw_shares = {}

    for party in parties:
        share = party.base_vote_share

        # --- ECONOMIC VOTING (retrospective) ---
        # Incumbents are rewarded/punished for economic performance
        if party.abbreviation in governing_parties:
            # GDP growth effect (strong positive = reward incumbent)
            growth_reward = (eco.gdp_growth_rate - 0.02) * 0.5
            # Unemployment effect (high = punish incumbent)
            unemployment_penalty = (eco.unemployment_rate - 0.10) * -0.3
            # Inflation effect (high = punish incumbent)
            inflation_penalty = (eco.inflation_rate - 0.02) * -0.2

            economic_vote = growth_reward + unemployment_penalty + inflation_penalty
            share += economic_vote

            # Incumbent fatigue: each year in power costs ~0.5-1% vote share
            fatigue = years_in_power * -0.005
            share += fatigue
        else:
            # Opposition gets a slight boost when economy is bad
            bad_economy_bonus = max(0, eco.unemployment_rate - 0.12) * 0.15
            share += bad_economy_bonus * (1 if party.ideology_score * -1 > 0 else 0.5)

        # --- ISSUE SALIENCE ---
        # When specific issues are salient, parties with strong positions benefit

        # Housing crisis: benefits left parties (affordability focus)
        if soc.housing_affordability < 35:
            if party.ideology_score < -2:
                share += 0.01

        # High inequality: benefits redistributive parties
        if soc.gini_coefficient > 0.34:
            if party.ideology_score < -1:
                share += (soc.gini_coefficient - 0.34) * 0.1

        # Regional tensions: benefits regionalist parties and Vox (opposite sides)
        if soc.regional_separatism_tension > 55:
            if party.abbreviation in ["ERC", "JUNTS", "BILDU", "PNV"]:
                share += 0.005
            if party.abbreviation == "VOX":
                share += 0.005  # Anti-separatism mobilization

        # Immigration concerns: benefits Vox when immigration is high
        if state.demographics.net_migration_per_1000 > 7:
            if party.abbreviation == "VOX":
                share += (state.demographics.net_migration_per_1000 - 7) * 0.003

        # Trust in system: low trust benefits anti-establishment parties
        if soc.trust_in_government < 0.25:
            if abs(party.ideology_score) > 4:  # Extreme parties benefit
                share += (0.25 - soc.trust_in_government) * 0.1

        # --- RANDOM CAMPAIGN EFFECTS ---
        # Scandals, media coverage, candidate charisma, etc.
        campaign_noise = rng.gauss(0, 0.015)
        share += campaign_noise

        # Floor: no party goes below 0.5% or above 45%
        share = max(0.005, min(0.45, share))
        raw_shares[party.abbreviation] = share

    # Normalize to sum to 1.0
    total = sum(raw_shares.values())
    return {k: v / total for k, v in raw_shares.items()}


# ---------------------------------------------------------------------------
# COALITION FORMATION MODEL
# ---------------------------------------------------------------------------

def form_coalition(seat_distribution: Dict[str, int],
                   parties: List[Party],
                   total_seats: int,
                   rng: random.Random) -> Tuple[List[str], float]:
    """
    Model Spanish coalition formation.

    Spain requires a PM to win an investiture vote (absolute majority on first
    ballot, simple majority on second). This means the PM's bloc needs at least
    176 seats, or more seats than the opposition.

    Coalition preferences are based on ideological proximity and historical
    patterns:
    - PSOE allies: SUMAR, ERC, JUNTS, PNV, BILDU
    - PP allies: VOX (reluctantly), PNV (occasionally)
    - Regional parties are kingmakers

    Returns: (list of governing party abbreviations, coalition_stability 0-1)
    """
    majority_threshold = total_seats // 2 + 1  # 176

    party_dict = {p.abbreviation: p for p in parties}
    seated_parties = {k: v for k, v in seat_distribution.items() if v > 0}

    # Sort by seats (largest party gets first try)
    sorted_parties = sorted(seated_parties.items(), key=lambda x: -x[1])

    if not sorted_parties:
        return (["PSOE"], 0.5)  # Fallback

    # Define coalition blocs (based on real Spanish political dynamics)
    left_bloc = ["PSOE", "SUMAR", "ERC", "BILDU"]
    right_bloc = ["PP", "VOX"]
    regionalist_swing = ["JUNTS", "PNV"]  # Can go either way

    def bloc_seats(bloc_parties):
        return sum(seated_parties.get(p, 0) for p in bloc_parties)

    # Try natural coalitions first
    # Left bloc
    left_seats = bloc_seats(left_bloc)
    left_with_swing = left_seats + bloc_seats(regionalist_swing)

    # Right bloc
    right_seats = bloc_seats(right_bloc)
    right_with_pnv = right_seats + seated_parties.get("PNV", 0)

    # Determine who can form government
    if left_seats >= majority_threshold:
        # Left bloc has majority without regionalists
        coalition = [p for p in left_bloc if p in seated_parties]
        stability = 0.7 + rng.gauss(0, 0.05)
    elif left_with_swing >= majority_threshold and left_seats > right_seats:
        # Left bloc needs regionalists (current situation)
        coalition = [p for p in left_bloc + regionalist_swing if p in seated_parties]
        stability = 0.5 + rng.gauss(0, 0.05)  # Fragile — as in real Spain
    elif right_seats >= majority_threshold:
        # Right bloc majority
        coalition = [p for p in right_bloc if p in seated_parties]
        stability = 0.75 + rng.gauss(0, 0.05)
    elif right_with_pnv >= majority_threshold:
        # PP + PNV (has historical precedent)
        coalition = ["PP"]
        if "PNV" in seated_parties:
            coalition.append("PNV")
        if right_with_pnv < majority_threshold and "VOX" in seated_parties:
            coalition.append("VOX")
        stability = 0.6 + rng.gauss(0, 0.05)
    else:
        # No clear majority — minority government by largest party
        largest = sorted_parties[0][0]
        coalition = [largest]
        # Try to find confidence-and-supply partners
        for name, seats in sorted_parties[1:]:
            if name in party_dict:
                # Partner if ideologically close enough
                main_ideology = party_dict[largest].ideology_score
                partner_ideology = party_dict[name].ideology_score
                if abs(main_ideology - partner_ideology) < 6:
                    coalition.append(name)
                    if sum(seated_parties.get(p, 0) for p in coalition) >= majority_threshold:
                        break
        stability = 0.4 + rng.gauss(0, 0.05)

    stability = max(0.2, min(0.95, stability))
    return (coalition, stability)


def blend_party_policies(coalition: List[str], seat_distribution: Dict[str, int],
                         parties: List[Party]) -> GovernmentAction:
    """
    Create a blended policy action weighted by coalition partners' seat share.
    The largest party has the most influence, but coalition partners pull policy.
    """
    party_dict = {p.abbreviation: p for p in parties}

    # Calculate weights (seat-based)
    coalition_seats = {p: seat_distribution.get(p, 0) for p in coalition if p in party_dict}
    total_coalition_seats = sum(coalition_seats.values())

    if total_coalition_seats == 0:
        return GovernmentAction(governing_parties=coalition)

    weights = {p: s / total_coalition_seats for p, s in coalition_seats.items()}

    # Blend multipliers
    action = GovernmentAction(governing_parties=coalition)

    multiplier_fields = [
        'tax_pressure_mult', 'gov_spending_mult', 'healthcare_spending_mult',
        'education_spending_mult', 'defense_spending_mult', 'rd_spending_mult',
        'social_protection_mult', 'infrastructure_mult', 'labor_regulation_mult',
        'business_deregulation_mult', 'environmental_regulation_mult',
        'immigration_openness_mult', 'decentralization_mult', 'corruption_effort_mult',
    ]

    for field_name in multiplier_fields:
        weighted_sum = sum(
            getattr(party_dict[p], field_name) * w
            for p, w in weights.items() if p in party_dict
        )
        setattr(action, field_name, weighted_sum)

    # GDP and inequality tendencies
    action.gdp_growth_tendency = sum(
        party_dict[p].gdp_growth_tendency * w
        for p, w in weights.items() if p in party_dict
    )
    action.inequality_tendency = sum(
        party_dict[p].inequality_tendency * w
        for p, w in weights.items() if p in party_dict
    )

    return action


# ---------------------------------------------------------------------------
# MAIN SYSTEM CLASS
# ---------------------------------------------------------------------------

class SpanishParliamentarySystem(GovernmentSystem):
    """
    Spain's current parliamentary system with D'Hondt PR.

    Key characteristics this models:
    1. D'Hondt favors large parties in small constituencies
    2. Coalition governments are the norm since 2015
    3. Regional parties are kingmakers
    4. PM needs investiture vote (majority in Congress)
    5. 4-year terms with possible early elections
    """

    def __init__(self):
        self.parties: List[Party] = []
        self.seat_distribution: Dict[str, int] = {}
        self.vote_shares: Dict[str, float] = {}
        self.coalition: List[str] = []
        self.coalition_stability: float = 0.7
        self.years_since_election: int = 0
        self.years_in_power: int = 0
        self.term_length: int = 4
        self.election_history: List[dict] = []
        self.rng = random.Random(42)

    def initialize(self, state: SimulationState, parties: List[Party]):
        self.parties = parties
        self.rng = random.Random(42)

        # Initialize with 2023 election results (approximate seat distribution)
        self.seat_distribution = {
            "PP": 137, "PSOE": 121, "VOX": 33, "SUMAR": 31,
            "ERC": 7, "JUNTS": 7, "BILDU": 6, "PNV": 5,
        }
        # Remaining seats to smaller parties (simplified)
        allocated = sum(self.seat_distribution.values())
        # 3 seats unaccounted (other small parties) — ignore for simplicity

        self.vote_shares = {p.abbreviation: p.base_vote_share for p in parties}

        # Current government: PSOE + SUMAR (with support from ERC, JUNTS, BILDU, PNV)
        self.coalition = ["PSOE", "SUMAR"]
        self.coalition_stability = 0.55  # Fragile minority coalition
        self.years_since_election = 1  # Election was 2023
        self.years_in_power = 6  # PSOE governing since 2018

    def _run_election(self, state: SimulationState) -> dict:
        """Simulate a full general election."""
        # Calculate vote shares
        self.vote_shares = calculate_vote_shares(
            self.parties, state, self.coalition,
            self.years_in_power, self.rng
        )

        # Run D'Hondt allocation across all constituencies
        total_seats = {}
        for num_const, seats_each, label in CONSTITUENCY_STRUCTURE:
            for _ in range(num_const):
                # Vote shares vary by constituency type
                local_votes = {}
                for abbr, share in self.vote_shares.items():
                    party = next((p for p in self.parties if p.abbreviation == abbr), None)
                    if party is None:
                        continue

                    local_share = share

                    # Regional parties only compete in their regions
                    # Simplified: they get 0 in most constituencies
                    if abbr in ["ERC", "JUNTS"]:
                        # Catalan parties: only ~15% of constituencies
                        if self.rng.random() > 0.15:
                            local_share = 0
                        else:
                            local_share *= 5  # Concentrated vote
                    elif abbr in ["PNV", "BILDU"]:
                        # Basque parties: only ~6% of constituencies
                        if self.rng.random() > 0.06:
                            local_share = 0
                        else:
                            local_share *= 10

                    # Add local variation
                    local_share *= (1 + self.rng.gauss(0, 0.1))
                    local_share = max(0, local_share)
                    local_votes[abbr] = local_share

                # Allocate seats in this constituency
                allocation = dhondt_allocation(local_votes, seats_each, ELECTORAL_THRESHOLD)
                for party, seats in allocation.items():
                    total_seats[party] = total_seats.get(party, 0) + seats

        # Normalize to exactly 350 seats
        current_total = sum(total_seats.values())
        if current_total != TOTAL_SEATS and current_total > 0:
            # Adjust largest party
            largest = max(total_seats, key=total_seats.get)
            total_seats[largest] += TOTAL_SEATS - current_total

        self.seat_distribution = total_seats

        # Form coalition
        self.coalition, self.coalition_stability = form_coalition(
            self.seat_distribution, self.parties, TOTAL_SEATS, self.rng
        )

        # Reset counters
        self.years_since_election = 0
        old_years = self.years_in_power

        # Check if governing party changed
        if self.election_history:
            old_coalition = self.election_history[-1].get("coalition", [])
            if self.coalition[0] != old_coalition[0] if old_coalition else True:
                self.years_in_power = 0

        result = {
            "year": state.year,
            "vote_shares": dict(self.vote_shares),
            "seats": dict(self.seat_distribution),
            "coalition": list(self.coalition),
            "stability": self.coalition_stability,
        }
        self.election_history.append(result)
        return result

    def step(self, state: SimulationState) -> GovernmentAction:
        """Run one year of the Spanish parliamentary system."""
        self.years_since_election += 1
        self.years_in_power += 1

        election_held = False

        # Check if election is due
        if self.years_since_election >= self.term_length:
            self._run_election(state)
            election_held = True

        # Check for early election (coalition collapse)
        elif self.coalition_stability < 0.3:
            # High chance of early election
            if self.rng.random() < 0.6:
                self._run_election(state)
                election_held = True
        elif self.coalition_stability < 0.4:
            if self.rng.random() < 0.2:
                self._run_election(state)
                election_held = True

        # Coalition stability drifts (can improve or worsen)
        stability_drift = self.rng.gauss(0, 0.03)
        # More parties = less stable
        coalition_size_penalty = len(self.coalition) * -0.02
        self.coalition_stability += stability_drift + coalition_size_penalty * 0.1
        self.coalition_stability = max(0.1, min(0.95, self.coalition_stability))

        # Generate government action based on coalition
        action = blend_party_policies(
            self.coalition, self.seat_distribution, self.parties
        )
        action.coalition_stability = self.coalition_stability
        action.election_held = election_held
        action.governing_parties = list(self.coalition)

        # Policy effectiveness reduced by instability
        # A fragile coalition can't fully implement its agenda
        effectiveness = 0.5 + self.coalition_stability * 0.5

        # Scale multipliers toward 1.0 based on effectiveness
        for field_name in ['tax_pressure_mult', 'gov_spending_mult',
                           'healthcare_spending_mult', 'education_spending_mult',
                           'defense_spending_mult', 'rd_spending_mult',
                           'social_protection_mult', 'infrastructure_mult',
                           'labor_regulation_mult', 'business_deregulation_mult',
                           'environmental_regulation_mult', 'immigration_openness_mult',
                           'decentralization_mult', 'corruption_effort_mult']:
            current = getattr(action, field_name)
            # Pull toward 1.0 (no change) based on ineffectiveness
            dampened = 1.0 + (current - 1.0) * effectiveness
            setattr(action, field_name, dampened)

        action.gdp_growth_tendency *= effectiveness
        action.inequality_tendency *= effectiveness

        coalition_str = " + ".join(self.coalition)
        action.description = (
            f"Year {state.year}: {coalition_str} government "
            f"(stability: {self.coalition_stability:.0%})"
            f"{' [ELECTION HELD]' if election_held else ''}"
        )

        return action

    def get_status(self) -> dict:
        return {
            "system": "Spanish Parliamentary (D'Hondt PR)",
            "coalition": self.coalition,
            "seats": self.seat_distribution,
            "vote_shares": {k: round(v, 3) for k, v in self.vote_shares.items()},
            "stability": round(self.coalition_stability, 2),
            "years_since_election": self.years_since_election,
            "years_in_power": self.years_in_power,
            "elections_held": len(self.election_history),
        }