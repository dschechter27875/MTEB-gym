"""
validate.py: Compute Spearman correlation between MTEB Gym ELO rankings
and official MTEB benchmark scores.

Usage:
    python3 scripts/validate.py --results results/tournament_real/final_leaderboard.json

This tells you: do synthetic queries produce rankings that match MTEB ground truth?
Target: Spearman rho > 0.80 (Rohan got 0.86 with real human queries)
"""
import json
import argparse
from scipy.stats import spearmanr

# ── Official MTEB English Retrieval scores (from huggingface.co/spaces/mteb/leaderboard) ──
# Add/update models here as needed
MTEB_SCORES = {
    # Model name: MTEB English Retrieval average score
    "sentence-transformers/all-MiniLM-L6-v2":       41.95,
    "intfloat/multilingual-e5-small":                46.62,
    "BAAI/bge-large-en-v1.5":                        54.29,
    "intfloat/multilingual-e5-large-instruct":       57.97,
    "nomic-ai/nomic-embed-text-v1.5":                53.84,
    "mixedbread-ai/mxbai-embed-large-v1":            54.39,
    "jinaai/jina-embeddings-v2-base-en":             49.25,
    "intfloat/multilingual-e5-small":                46.62,
    "sentence-transformers/all-mpnet-base-v2":       43.81,
}


def load_leaderboard(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["leaderboard"]


def compute_spearman(leaderboard: list[dict]) -> None:
    gym_models = []
    gym_ratings = []
    mteb_scores = []

    print("\nModel comparison:")
    print(f"{'Model':<50} {'Gym ELO':>8} {'MTEB Score':>12}")
    print("-" * 72)

    for entry in sorted(leaderboard, key=lambda x: x["rating"], reverse=True):
        name = entry["name"]
        rating = entry["rating"]

        if name not in MTEB_SCORES:
            print(f"  WARNING: No MTEB score for {name} — skipping")
            continue

        mteb = MTEB_SCORES[name]
        gym_models.append(name)
        gym_ratings.append(rating)
        mteb_scores.append(mteb)
        print(f"{name:<50} {rating:>8.1f} {mteb:>12.2f}")

    if len(gym_models) < 3:
        print(f"\nNeed at least 3 models to compute correlation (have {len(gym_models)})")
        return

    rho, pvalue = spearmanr(gym_ratings, mteb_scores)

    print(f"\n{'='*72}")
    print(f"  Spearman ρ  = {rho:.4f}")
    print(f"  p-value     = {pvalue:.4f}")
    print(f"  n models    = {len(gym_models)}")
    print(f"{'='*72}")

    if rho >= 0.85:
        print("  ✅ Strong correlation — matches Rohan's benchmark (ρ=0.86)")
    elif rho >= 0.70:
        print("  🟡 Moderate correlation — promising but needs more models/queries")
    else:
        print("  ❌ Weak correlation — need more models or better judge")

    print(f"\n  Rohan's best (real human queries): ρ = 0.86")
    print(f"  Your result  (synthetic queries):  ρ = {rho:.2f}")
    delta = rho - 0.86
    print(f"  Difference:                        {delta:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/tournament_real/final_leaderboard.json",
        help="Path to final_leaderboard.json from tournament",
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results}")
    leaderboard = load_leaderboard(args.results)
    compute_spearman(leaderboard)


if __name__ == "__main__":
    main()
