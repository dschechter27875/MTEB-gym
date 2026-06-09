"""
ELOTracker: Maintains ELO ratings across models and computes leaderboard.

Algorithm:
  - Standard Elo with K=32
  - Bootstrap CI over verdict resampling
  - Win rate matrix for pairwise comparison table
  - Supports incremental updates (new verdicts → recalculate)

Reference: AlpacaEval uses Bradley-Terry; we use simpler Elo for now.
Future: swap in BT for more statistically principled estimates.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .judge import JudgeVerdict, Winner


K_FACTOR = 32
DEFAULT_RATING = 1000


@dataclass
class ModelStats:
    name: str
    rating: float = DEFAULT_RATING
    wins: int = 0
    losses: int = 0
    ties: int = 0
    games: int = 0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.games

    @property
    def display_rating(self) -> int:
        return int(round(self.rating))


class ELOTracker:
    """
    Maintains ELO ratings for all models in the gym.

    Args:
        k_factor: ELO K factor (higher = faster adaptation)
        bootstrap_n: Number of bootstrap samples for CI
        seed: Random seed
    """

    def __init__(
        self,
        k_factor: float = K_FACTOR,
        bootstrap_n: int = 1000,
        seed: int = 42,
    ):
        self.k = k_factor
        self.bootstrap_n = bootstrap_n
        self.rng = random.Random(seed)

        self._ratings: dict[str, float] = defaultdict(lambda: float(DEFAULT_RATING))
        self._verdicts: list[JudgeVerdict] = []
        self._pairwise: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "ties": 0}
        )

    def process_verdicts(self, verdicts: list[JudgeVerdict]) -> None:
        """Add new verdicts and update ELO ratings."""
        self._verdicts.extend(verdicts)
        self._recalculate_from_scratch()

    def _recalculate_from_scratch(self) -> None:
        """Replay all verdicts to get stable ratings (avoids order dependency)."""
        self._ratings = defaultdict(lambda: float(DEFAULT_RATING))
        self._pairwise = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

        # Shuffle for more stable convergence
        shuffled = self._verdicts[:]
        self.rng.shuffle(shuffled)

        for v in shuffled:
            self._apply_verdict(v)
            key = (v.model_a_name, v.model_b_name)
            if v.winner == Winner.A:
                self._pairwise[key]["wins"] += 1
            elif v.winner == Winner.B:
                self._pairwise[key]["losses"] += 1
            else:
                self._pairwise[key]["ties"] += 1

    def _apply_verdict(self, v: JudgeVerdict) -> None:
        ra = self._ratings[v.model_a_name]
        rb = self._ratings[v.model_b_name]

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        eb = 1.0 - ea

        sa = v.points_a
        sb = v.points_b

        self._ratings[v.model_a_name] = ra + self.k * (sa - ea)
        self._ratings[v.model_b_name] = rb + self.k * (sb - eb)

    def get_leaderboard(self, compute_ci: bool = True) -> list[ModelStats]:
        """
        Returns sorted leaderboard with optional bootstrap CIs.
        """
        model_names = list(self._ratings.keys())
        stats: dict[str, ModelStats] = {}

        for name in model_names:
            pairwise_totals = {"wins": 0, "losses": 0, "ties": 0}
            for (ma, mb), counts in self._pairwise.items():
                if ma == name:
                    pairwise_totals["wins"] += counts["wins"]
                    pairwise_totals["losses"] += counts["losses"]
                    pairwise_totals["ties"] += counts["ties"]
                elif mb == name:
                    pairwise_totals["wins"] += counts["losses"]
                    pairwise_totals["losses"] += counts["wins"]
                    pairwise_totals["ties"] += counts["ties"]

            total = sum(pairwise_totals.values())
            stats[name] = ModelStats(
                name=name,
                rating=self._ratings[name],
                wins=pairwise_totals["wins"],
                losses=pairwise_totals["losses"],
                ties=pairwise_totals["ties"],
                games=total,
            )

        if compute_ci and len(self._verdicts) > 0:
            cis = self._bootstrap_ci(model_names)
            for name, (lo, hi) in cis.items():
                stats[name].ci_lower = lo
                stats[name].ci_upper = hi

        return sorted(stats.values(), key=lambda s: s.rating, reverse=True)

    def _bootstrap_ci(
        self, model_names: list[str], alpha: float = 0.05
    ) -> dict[str, tuple[float, float]]:
        """Bootstrap confidence intervals via verdict resampling."""
        all_ratings: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.bootstrap_n):
            sample = self.rng.choices(self._verdicts, k=len(self._verdicts))
            boot_ratings: dict[str, float] = defaultdict(lambda: float(DEFAULT_RATING))

            shuffled = sample[:]
            self.rng.shuffle(shuffled)

            for v in shuffled:
                ra = boot_ratings[v.model_a_name]
                rb = boot_ratings[v.model_b_name]
                ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
                sa = v.points_a
                boot_ratings[v.model_a_name] = ra + self.k * (sa - ea)
                boot_ratings[v.model_b_name] = rb + self.k * ((1 - sa) - (1 - ea))

            for name in model_names:
                all_ratings[name].append(boot_ratings[name])

        cis = {}
        lower_p = int(alpha / 2 * self.bootstrap_n)
        upper_p = int((1 - alpha / 2) * self.bootstrap_n)
        for name, samples in all_ratings.items():
            sorted_s = sorted(samples)
            cis[name] = (sorted_s[lower_p], sorted_s[min(upper_p, len(sorted_s) - 1)])

        return cis

    def get_pairwise_matrix(self) -> dict[str, dict[str, float]]:
        """Win rate matrix for all model pairs."""
        models = list(self._ratings.keys())
        matrix: dict[str, dict[str, float]] = {m: {} for m in models}

        for ma in models:
            for mb in models:
                if ma == mb:
                    matrix[ma][mb] = 0.5
                    continue
                key = (ma, mb)
                counts = self._pairwise.get(key, {"wins": 0, "losses": 0, "ties": 0})
                total = counts["wins"] + counts["losses"] + counts["ties"]
                if total == 0:
                    matrix[ma][mb] = None
                else:
                    matrix[ma][mb] = (counts["wins"] + 0.5 * counts["ties"]) / total

        return matrix

    def save(self, path: str) -> None:
        """Save tracker state to JSON."""
        state = {
            "verdicts": [
                {
                    "query_id": v.query_id,
                    "query": v.query,
                    "winner": v.winner.value,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "model_a": v.model_a_name,
                    "model_b": v.model_b_name,
                    "position_bias_ok": v.position_bias_check,
                }
                for v in self._verdicts
            ],
            "leaderboard": [
                {
                    "name": s.name,
                    "rating": s.rating,
                    "wins": s.wins,
                    "losses": s.losses,
                    "ties": s.ties,
                    "games": s.games,
                    "win_rate": s.win_rate,
                    "ci_lower": s.ci_lower,
                    "ci_upper": s.ci_upper,
                }
                for s in self.get_leaderboard(compute_ci=False)
            ],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[ELO] Saved state to {path}")

    def load(self, path: str) -> None:
        """Load tracker state from JSON."""
        with open(path) as f:
            state = json.load(f)

        winner_map = {"A": Winner.A, "B": Winner.B, "tie": Winner.TIE}

        verdicts = []
        for v in state["verdicts"]:
            jv = JudgeVerdict(
                query_id=v["query_id"],
                query=v["query"],
                winner=winner_map[v["winner"]],
                confidence=v["confidence"],
                reasoning=v["reasoning"],
                model_a_name=v["model_a"],
                model_b_name=v["model_b"],
                position_bias_check=v.get("position_bias_ok"),
            )
            verdicts.append(jv)

        self._verdicts = []
        self.process_verdicts(verdicts)
        print(f"[ELO] Loaded {len(verdicts)} verdicts from {path}")
