"""
Example: Run MTEB Gym comparing two real embedding models on NFCorpus.

pip install mteb sentence-transformers anthropic tqdm
python scripts/run_example.py
"""
import sys
sys.path.insert(0, "..")

from gym import MTEBGym
from gym.clients import AnthropicClient, MockLLMClient

# ──────────────────────────────────────────────────────────────
# 1.  Pick your LLM for query generation + judging
# ──────────────────────────────────────────────────────────────
# Real usage (requires ANTHROPIC_API_KEY env var):
#   llm = AnthropicClient(model="claude-sonnet-4-6")

# For quick testing without API keys:
llm = AnthropicClient()

# ──────────────────────────────────────────────────────────────
# 2.  Load embedding models
# ──────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

model_a = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # reference
model_b = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # challenger

# ──────────────────────────────────────────────────────────────
# 3.  Build the gym
# ──────────────────────────────────────────────────────────────
gym = MTEBGym(
    model_a=model_a,
    model_b=model_b,
    model_a_name="MiniLM-L6-v2",
    model_b_name="mpnet-base-v2",
    llm_client=llm,
    output_dir="results/nfcorpus",
    top_k=10,
    flip_judge=True,
    cache_embeddings=True,
)

# ──────────────────────────────────────────────────────────────
# 4a. Run on a named MTEB task (loads corpus automatically)
# ──────────────────────────────────────────────────────────────
gym.run_mteb_task(task_name="NFCorpus", n_queries=200)

# ──────────────────────────────────────────────────────────────
# 4b. OR run on a custom corpus
# ──────────────────────────────────────────────────────────────
# corpus = {
#     "doc_0": "Transformers use self-attention to model long-range dependencies...",
#     "doc_1": "BERT pretrains on masked language modelling and next sentence prediction...",
#     # ... more docs
# }
# gym.run(corpus=corpus, n_queries=200, corpus_name="ml_papers")

# ──────────────────────────────────────────────────────────────
# 5.  View results
# ──────────────────────────────────────────────────────────────
gym.leaderboard()

# Export JSON for the leaderboard UI
gym.export_leaderboard_json("results/leaderboard.json")
