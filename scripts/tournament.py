"""
Tournament: Round-robin evaluation across N embedding models.

Runs all N*(N-1)/2 pairwise matchups and produces a unified ELO leaderboard.

Usage:
    python scripts/tournament.py \
        --models "MiniLM-L6-v2" "mpnet-base-v2" "e5-large-v2" \
        --task NFCorpus \
        --n-queries 100 \
        --output results/tournament/
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "..")

from gym.elo import ELOTracker
from gym.judge import LLMJudge
from gym.query_generator import QueryGenerator
from gym.retrieval_harness import RetrievalHarness
from gym.clients import AnthropicClient, MockLLMClient


def load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def run_tournament(
    model_names: list[str],
    task_name: str,
    n_queries: int,
    output_dir: str,
    llm_client,
    use_mock: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load corpus once
    import mteb
    print(f"Loading MTEB task: {task_name}")
    task = mteb.get_tasks(tasks=[task_name])[0]
    task.load_data()
    corpus_raw = task.corpus.get("test", {})
    corpus = {
        did: (doc.get("title", "") + " " + doc.get("text", "")).strip()
        for did, doc in corpus_raw.items()
    }
    print(f"Corpus: {len(corpus)} docs")

    # Generate queries once (shared across all matchups)
    print(f"Generating {n_queries} queries...")
    qgen = QueryGenerator(llm_client=llm_client, docs_per_query=5)
    queries = qgen.generate(corpus=corpus, n_queries=n_queries, corpus_name=task_name)
    print(f"Generated {len(queries)} queries")

    # Pre-encode all models
    print("Pre-encoding all models...")
    models = {}
    for name in model_names:
        print(f"  Loading {name}...")
        models[name] = load_model(name)

    # Global ELO tracker
    global_elo = ELOTracker()
    judge = LLMJudge(llm_client=llm_client, flip_positions=True)

    # Run all pairs
    pairs_list = list(combinations(model_names, 2))
    print(f"\nRunning {len(pairs_list)} matchups...")

    for i, (name_a, name_b) in enumerate(pairs_list):
        print(f"\n[{i+1}/{len(pairs_list)}] {name_a} vs {name_b}")

        harness = RetrievalHarness(
            model_a=models[name_a],
            model_b=models[name_b],
            model_a_name=name_a,
            model_b_name=name_b,
            top_k=10,
            cache_dir=str(output_path / "cache"),
        )

        pairs = harness.run(queries=queries, corpus=corpus)
        verdicts = judge.judge_all(pairs)
        global_elo.process_verdicts(verdicts)

        matchup_key = f"{name_a}_vs_{name_b}".replace("/", "_").replace("-", "_")
        matchup_path = output_path / f"matchup_{matchup_key}.json"
        global_elo.save(str(matchup_path))

    # Final leaderboard
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    board = global_elo.get_leaderboard(compute_ci=True)
    for rank, s in enumerate(board, 1):
        ci = (s.ci_upper - s.ci_lower) / 2 if s.ci_upper else 0
        print(
            f"{rank}. {s.name:<40} ELO={s.display_rating:>5} ±{ci:.0f}  "
            f"W={s.wins} L={s.losses} T={s.ties}"
        )
    print("="*60)

    final_path = output_path / "final_leaderboard.json"
    global_elo.save(str(final_path))
    print(f"\nSaved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ])
    parser.add_argument("--task", default="NFCorpus")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--output", default="results/tournament")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API key needed)")
    args = parser.parse_args()

    llm = MockLLMClient() if args.mock else AnthropicClient()

    run_tournament(
        model_names=args.models,
        task_name=args.task,
        n_queries=args.n_queries,
        output_dir=args.output,
        llm_client=llm,
    )
