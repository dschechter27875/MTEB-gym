"""
LLMJudge: Pairwise evaluation of retrieval results.

Design:
  - For retrieval: text LLM judges which result set better answers the query
  - Position bias mitigation: randomise A/B order, run twice, check consistency
  - Returns JudgeVerdict with winner, confidence, and raw reasoning

Judge prompt is calibrated for retrieval nuance:
  - Relevance (does it answer the query?)
  - Completeness (does it cover all aspects?)
  - Precision (are results on-topic vs noisy?)
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .retrieval_harness import MatchedPair, RetrievalResult


RETRIEVAL_JUDGE_PROMPT = """You are an expert information retrieval judge evaluating two search systems.

Query: "{query}"

System {label_a} returned these documents:
{docs_a}

System {label_b} returned these documents:
{docs_b}

Evaluate which system returned BETTER results for this query. Consider:
1. Relevance: How directly do the documents address the query?
2. Completeness: Do the documents cover the key aspects of what the user needs?
3. Precision: Are there irrelevant documents mixed in?


Respond in this EXACT JSON format (no markdown, no explanation outside JSON):
{{
  "winner": "{label_a}" or "{label_b}" or "tie",
  "confidence": "high" or "medium" or "low",
  "reasoning": "1-2 sentences explaining the decision"
}}"""


STS_JUDGE_PROMPT = """You are evaluating two semantic similarity systems.

Query text: "{query}"

System {label_a} found these semantically similar texts:
{docs_a}

System {label_b} found these semantically similar texts:
{docs_b}

Which system found texts that are MORE semantically similar to the query?

Respond in JSON:
{{
  "winner": "{label_a}" or "{label_b}" or "tie",
  "confidence": "high" or "medium" or "low",
  "reasoning": "1-2 sentences"
}}"""


class Winner(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"


@dataclass
class JudgeVerdict:
    query_id: str
    query: str
    winner: Winner
    confidence: str  # "high" | "medium" | "low"
    reasoning: str
    model_a_name: str
    model_b_name: str
    raw_response: str = ""
    position_bias_check: Optional[bool] = None  # True if consistent across flips

    @property
    def points_a(self) -> float:
        if self.winner == Winner.A:
            return 1.0
        elif self.winner == Winner.TIE:
            return 0.5
        return 0.0

    @property
    def points_b(self) -> float:
        return 1.0 - self.points_a


def _format_docs(result: RetrievalResult, max_docs: int = 5, max_chars: int = 300) -> str:
    lines = []
    for i, (text, score) in enumerate(
        zip(result.top_k_texts[:max_docs], result.top_k_scores[:max_docs])
    ):
        lines.append(f"  [{i+1}] (score={score:.3f}) {text[:max_chars]}...")
    return "\n".join(lines)


class LLMJudge:
    """
    Pairwise LLM judge for embedding model evaluation.

    Args:
        llm_client: Client with .chat(messages) -> str method
        flip_positions: Run each pair twice with A/B flipped to check position bias
        max_docs_shown: How many retrieved docs to show the judge (fewer = cheaper)
        model_name: Name of the judge LLM (for logging)
    """

    def __init__(
        self,
        llm_client: Any,
        flip_positions: bool = True,
        max_docs_shown: int = 5,
        model_name: str = "judge",
        seed: int = 42,
    ):
        self.llm = llm_client
        self.flip_positions = flip_positions
        self.max_docs_shown = max_docs_shown
        self.model_name = model_name
        self.rng = random.Random(seed)

    def judge_pair(self, pair: MatchedPair) -> JudgeVerdict:
        """Judge a single MatchedPair. Handles position-flip if configured."""

        verdict_forward = self._call_judge(
            pair=pair,
            a_first=True,
        )

        if not self.flip_positions:
            return verdict_forward

        # Run with A/B flipped to detect position bias
        verdict_flipped = self._call_judge(
            pair=pair,
            a_first=False,
        )

        consistent = verdict_forward.winner == verdict_flipped.winner
        verdict_forward.position_bias_check = consistent

        # If inconsistent, downgrade to tie (conservative)
        if not consistent and verdict_forward.confidence != "high":
            verdict_forward.winner = Winner.TIE
            verdict_forward.confidence = "low"
            verdict_forward.reasoning = (
                f"[Position bias detected] {verdict_forward.reasoning}"
            )

        return verdict_forward

    def judge_all(
        self,
        pairs: list[MatchedPair],
        show_progress: bool = True,
    ) -> list[JudgeVerdict]:
        """Judge all pairs and return verdicts."""
        verdicts = []

        iter_pairs = pairs
        if show_progress:
            try:
                from tqdm import tqdm
                iter_pairs = tqdm(pairs, desc=f"Judging [{self.model_name}]")
            except ImportError:
                pass

        for pair in iter_pairs:
            try:
                verdict = self.judge_pair(pair)
                verdicts.append(verdict)
            except Exception as e:
                print(f"[Judge] Error on {pair.query_id}: {e}")

        return verdicts

    def _call_judge(self, pair: MatchedPair, a_first: bool) -> JudgeVerdict:
        if a_first:
            res_first, res_second = pair.result_a, pair.result_b
            label_first, label_second = "A", "B"
        else:
            res_first, res_second = pair.result_b, pair.result_a
            label_first, label_second = "B", "A"

        docs_first = _format_docs(res_first, self.max_docs_shown)
        docs_second = _format_docs(res_second, self.max_docs_shown)

        prompt_template = (
            STS_JUDGE_PROMPT if pair.task_type == "sts" else RETRIEVAL_JUDGE_PROMPT
        )

        prompt = prompt_template.format(
            query=pair.query,
            label_a=label_first,
            label_b=label_second,
            docs_a=docs_first,
            docs_b=docs_second,
        )

        raw = self.llm.chat([{"role": "user", "content": prompt}])
        parsed = self._parse_response(raw, label_first, label_second)

        # Map back to canonical A/B names
        if not a_first:
            if parsed["winner"] == "A":
                parsed["winner"] = "B"
            elif parsed["winner"] == "B":
                parsed["winner"] = "A"

        winner_map = {"A": Winner.A, "B": Winner.B, "tie": Winner.TIE}
        return JudgeVerdict(
            query_id=pair.query_id,
            query=pair.query,
            winner=winner_map.get(parsed["winner"], Winner.TIE),
            confidence=parsed.get("confidence", "medium"),
            reasoning=parsed.get("reasoning", ""),
            model_a_name=pair.result_a.model_name,
            model_b_name=pair.result_b.model_name,
            raw_response=raw,
        )

    @staticmethod
    def _parse_response(raw: str, label_a: str, label_b: str) -> dict:
        """Robustly parse judge JSON response."""
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: extract winner with regex
            winner_match = re.search(
                rf'"winner"\s*:\s*"({label_a}|{label_b}|tie)"', raw
            )
            winner = winner_match.group(1) if winner_match else "tie"
            return {"winner": winner, "confidence": "low", "reasoning": "parse error"}

        winner = str(data.get("winner", "tie"))
        if winner not in (label_a, label_b, "tie"):
            winner = "tie"

        return {
            "winner": winner,
            "confidence": data.get("confidence", "medium"),
            "reasoning": data.get("reasoning", ""),
        }
