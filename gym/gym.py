"""
MTEBGym: Top-level orchestrator.

Usage:
    from gym import MTEBGym
    from sentence_transformers import SentenceTransformer

    gym = MTEBGym(
        model_a=SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct"),
        model_b=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        model_a_name="gte-Qwen2-7B",
        model_b_name="MiniLM-L6",
        llm_client=my_llm,          # any client with .chat(messages)->str
        output_dir="results/",
    )
    gym.run(corpus=my_corpus, n_queries=200)
    gym.leaderboard()               # prints ranked table
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .query_generator import QueryGenerator
from .retrieval_harness import RetrievalHarness
from .judge import LLMJudge
from .elo import ELOTracker


class MTEBGym:
    """
    Full MTEB Gym pipeline.

    Args:
        model_a: Reference embedding model (strong baseline)
        model_b: Challenger embedding model
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        llm_client: LLM client for query generation AND judging
                    Must have: .chat(messages: list[dict]) -> str
        output_dir: Directory to save intermediate results
        top_k: Docs to retrieve per query
        flip_judge: Randomise judge A/B order (reduces position bias)
        cache_embeddings: Cache corpus embeddings to disk
        docs_per_query: Docs shown to LLM when generating each query
    """

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
        llm_client: Optional[Any] = None,
        output_dir: str = "results",
        top_k: int = 10,
        flip_judge: bool = True,
        cache_embeddings: bool = True,
        docs_per_query: int = 5,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_gen = QueryGenerator(
            llm_client=llm_client,
            docs_per_query=docs_per_query,
            seed=seed,
        )
        self.harness = RetrievalHarness(
            model_a=model_a,
            model_b=model_b,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            top_k=top_k,
            cache_dir=str(self.output_dir / "cache") if cache_embeddings else None,
        )
        self.judge = LLMJudge(
            llm_client=llm_client,
            flip_positions=flip_judge,
            seed=seed,
        )
        self.elo = ELOTracker(seed=seed)

        self.model_a_name = model_a_name
        self.model_b_name = model_b_name

    def run(
        self,
        corpus: dict[str, str],
        n_queries: int = 200,
        corpus_name: str = "custom",
        task_type: str = "retrieval",
        resume: bool = True,
    ) -> None:
        """
        Full pipeline: generate queries → retrieve → judge → update ELO.

        Args:
            corpus: {doc_id: text} mapping
            n_queries: How many synthetic queries to generate
            corpus_name: Label for this corpus (shows in leaderboard)
            task_type: "retrieval" | "sts" | "clustering"
            resume: If True, skip query generation if cache exists
        """
        pairs_path = self.output_dir / f"pairs_{corpus_name}.jsonl"
        verdicts_path = self.output_dir / f"verdicts_{corpus_name}.json"

        # --- Step 1: Generate queries ---
        if resume and pairs_path.exists():
            print(f"[Gym] Loading cached pairs from {pairs_path}")
            raw_pairs = self.harness.load_pairs(str(pairs_path))
            # Reconstruct MatchedPair objects inline for judge
            from .retrieval_harness import MatchedPair, RetrievalResult
            pairs = []
            for p in raw_pairs:
                ra = p["result_a"]
                rb = p["result_b"]
                pairs.append(MatchedPair(
                    query=p["query"],
                    query_id=p["query_id"],
                    corpus_name=p["corpus_name"],
                    task_type=p["task_type"],
                    result_a=RetrievalResult(**ra),
                    result_b=RetrievalResult(**rb),
                ))
        else:
            print(f"\n[Gym] Step 1/3: Generating {n_queries} queries from '{corpus_name}'...")
            queries = self.query_gen.generate(
                corpus=corpus,
                n_queries=n_queries,
                corpus_name=corpus_name,
                task_type=task_type,
            )
            print(f"[Gym] Generated {len(queries)} queries")

            print(f"\n[Gym] Step 2/3: Retrieving with both models...")
            pairs = self.harness.run(queries=queries, corpus=corpus)
            self.harness.save_pairs(pairs, str(pairs_path))

        # --- Step 2: Judge ---
        if resume and verdicts_path.exists():
            print(f"[Gym] Loading cached verdicts from {verdicts_path}")
            self.elo.load(str(verdicts_path))
        else:
            print(f"\n[Gym] Step 3/3: Judging {len(pairs)} pairs...")
            verdicts = self.judge.judge_all(pairs)
            self.elo.process_verdicts(verdicts)
            self.elo.save(str(verdicts_path))

        print("\n[Gym] Done! Run .leaderboard() to see results.")

    def run_mteb_task(
        self,
        task_name: str,
        n_queries: int = 200,
        split: str = "test",
    ) -> None:
        """Convenience: run gym on a named MTEB task."""
        import mteb

        print(f"[Gym] Loading MTEB task: {task_name}")
        task = mteb.get_tasks(tasks=[task_name])[0]
        task.load_data()
        corpus_raw = task.corpus.get(split, {})
        corpus = {
            did: text
            for did, text in corpus_raw.items()
        }
        print(f"[Gym] Corpus size: {len(corpus)} docs")
        self.run(
            corpus=corpus,
            n_queries=n_queries,
            corpus_name=task_name,
            task_type="retrieval",
        )

    def leaderboard(self) -> None:
        """Print a formatted leaderboard table."""
        board = self.elo.get_leaderboard(compute_ci=True)

        header = f"{'Rank':<5} {'Model':<40} {'ELO':>6} {'CI ±':>8} {'W':>5} {'L':>5} {'T':>5} {'Win%':>7}"
        print("\n" + "=" * len(header))
        print("  MTEB GYM LEADERBOARD")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for rank, s in enumerate(board, 1):
            ci_half = (s.ci_upper - s.ci_lower) / 2 if s.ci_upper else 0
            print(
                f"{rank:<5} {s.name:<40} {s.display_rating:>6} "
                f"{f'±{ci_half:.0f}':>8} {s.wins:>5} {s.losses:>5} "
                f"{s.ties:>5} {s.win_rate*100:>6.1f}%"
            )

        print("=" * len(header) + "\n")

    def export_leaderboard_json(self, path: Optional[str] = None) -> dict:
        """Export leaderboard as JSON (for the web UI)."""
        board = self.elo.get_leaderboard(compute_ci=True)
        data = {
            "models": [
                {
                    "rank": i + 1,
                    "name": s.name,
                    "rating": s.display_rating,
                    "ci_lower": int(s.ci_lower) if s.ci_lower else None,
                    "ci_upper": int(s.ci_upper) if s.ci_upper else None,
                    "wins": s.wins,
                    "losses": s.losses,
                    "ties": s.ties,
                    "games": s.games,
                    "win_rate": round(s.win_rate * 100, 1),
                }
                for i, s in enumerate(board)
            ],
            "pairwise": self.elo.get_pairwise_matrix(),
            "total_verdicts": len(self.elo._verdicts),
        }

        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Gym] Exported leaderboard to {path}")

        return data
