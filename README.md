# MTEB Gym 🏋️

Run fast, LLM-judged pairwise evaluations of any embedding model — no human labels required.
Based on the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068) by Muennighoff, KennethEnevoldsen, and orionw.

```
corpus → synthetic queries (LLM) → retrieve (Model A + B) → judge (LLM pairwise) → ELO
```

---

## Quickstart

```bash
pip install -e ".[full]"
```

```python
from sentence_transformers import SentenceTransformer
from gym import MTEBGym
from gym.clients import AnthropicClient

gym = MTEBGym(
    model_a=SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct"),
    model_b=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    model_a_name="gte-Qwen2-7B",
    model_b_name="MiniLM-L6",
    llm_client=AnthropicClient(model="claude-sonnet-4-6"),
    output_dir="results/",
)

# Run on a named MTEB task
gym.run_mteb_task("NFCorpus", n_queries=200)
gym.leaderboard()
```

Output:
```
================================================================
  MTEB GYM LEADERBOARD
================================================================
Rank  Model                                    ELO      CI ±     W     L     T   Win%
----------------------------------------------------------------------
1     mpnet-base-v2                           1013    ±  32    16    19   164   49.2%
2     MiniLM-L6-v2                             987    ±  32    19    16   164   50.8%
```

---

## Architecture

```
gym/
├── query_generator.py   # LLM → synthetic queries from corpus
├── retrieval_harness.py # Encode corpus + retrieve top-k with both models
├── judge.py             # Pairwise LLM judge (position-flip bias mitigation)
├── elo.py               # ELO ratings + bootstrap CI + pairwise matrix
├── gym.py               # Orchestrator
└── clients.py           # LLM adapters: Anthropic, OpenAI, Gemini, LiteLLM, Mock
```

### Query generation strategy

The query generator shows the LLM `k` random corpus docs and asks it to write a query that is *related but not directly answered* by those docs. This forces retrieval rather than shallow surface matching.

Key design choices (per the MTEB Gym discussion):
- **On-the-fly queries** (like LMSys) rather than fixed sets → captures diverse difficulty
- **Shared query sets** across model pairs for fair comparison
- Future: compare synthetic query rankings to real-world usage data

### Judge design

- Pairwise comparison: `(query, top-k from A, top-k from B)` → winner
- **Position-flip bias mitigation**: run each pair twice with A/B swapped; if inconsistent, downgrade to tie
- Confidence scoring: high/medium/low propagated to ELO update
- Prompt calibrated for retrieval nuance: relevance, completeness, precision

---

## Multi-model tournament

```bash
python scripts/tournament.py \
    --models "BAAI/bge-large-en-v1.5" "sentence-transformers/e5-large-v2" "Alibaba-NLP/gte-large-en-v1.5" \
    --task NFCorpus \
    --n-queries 200 \
    --output results/tournament/
```

---

## Leaderboard UI

```bash
cd leaderboard/
python -m http.server 8080
# → http://localhost:8080
```

Point it at your `leaderboard.json` (from `gym.export_leaderboard_json()`).

---

## Custom corpus

```python
corpus = {
    "arxiv_0001": "Attention is all you need. We propose a new network architecture...",
    "arxiv_0002": "BERT: Pre-training of Deep Bidirectional Transformers...",
    # ...
}
gym.run(corpus=corpus, n_queries=100, corpus_name="ml_papers")
```

Perfect for evaluating on internal/proprietary documents (customer service, legal, medical).

---

## Design notes

- **No ground-truth labels required** — the LLM judge replaces nDCG
- **ELO** with bootstrap CI (95%) via verdict resampling — swap for Bradley-Terry for more principled estimates
- **Caching** of corpus embeddings: re-run with new models without re-encoding the corpus
- **Resume-safe**: intermediate pairs and verdicts saved to JSONL/JSON

---

## Citation

```bibtex
@misc{mteb-gym-2025,
  title  = {MTEB Gym: LLM-as-Judge Offline Arena for Embedding Models},
  note   = {Based on MTEB Gym discussion: github.com/embeddings-benchmark/mteb/discussions/3068},
  year   = {2025}
}
```

Building on:
```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
