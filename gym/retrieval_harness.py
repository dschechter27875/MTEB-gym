"""
RetrievalHarness: Runs two embedding models on the same query set and
stores results side-by-side for judge evaluation.

Supports:
  - Any model with .encode(texts) -> np.ndarray  (sentence-transformers API)
  - mteb model wrappers
  - Batch processing with caching to avoid re-encoding large corpora
"""
from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .query_generator import GeneratedQuery


@dataclass
class RetrievalResult:
    query: str
    query_id: str
    model_name: str
    top_k_ids: list[str]
    top_k_texts: list[str]
    top_k_scores: list[float]
    corpus_name: str
    task_type: str


@dataclass
class MatchedPair:
    """A query with results from both Model A and Model B, ready for judging."""
    query: str
    query_id: str
    corpus_name: str
    task_type: str
    result_a: RetrievalResult
    result_b: RetrievalResult

    def to_dict(self) -> dict:
        return asdict(self)


class RetrievalHarness:
    """
    Encodes a corpus with two models, retrieves top-k for each query,
    returns MatchedPairs ready for the LLM judge.

    Args:
        model_a: Reference model (e.g. "Alibaba-NLP/gte-Qwen2-7B-instruct")
        model_b: Challenger model under evaluation
        top_k: Number of documents to retrieve per query
        cache_dir: If set, corpus embeddings are cached to disk by model+corpus hash
        batch_size: Encoding batch size
    """

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
        top_k: int = 10,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.top_k = top_k
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        queries: list[GeneratedQuery],
        corpus: dict[str, str],  # {doc_id: text}
        show_progress: bool = True,
    ) -> list[MatchedPair]:
        """
        Main entry point. Returns one MatchedPair per query.
        """
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]

        # Encode corpus with both models (with caching)
        print(f"[Harness] Encoding corpus ({len(doc_texts)} docs) with {self.model_a_name}...")
        emb_a = self._encode_corpus(self.model_a, doc_texts, self.model_a_name, corpus)

        print(f"[Harness] Encoding corpus with {self.model_b_name}...")
        emb_b = self._encode_corpus(self.model_b, doc_texts, self.model_b_name, corpus)

        # Encode queries
        query_texts = [q.query for q in queries]
        print(f"[Harness] Encoding {len(query_texts)} queries...")

        qemb_a = self._batch_encode(self.model_a, query_texts)
        qemb_b = self._batch_encode(self.model_b, query_texts)

        pairs: list[MatchedPair] = []

        iter_queries = enumerate(queries)
        if show_progress:
            try:
                from tqdm import tqdm
                iter_queries = tqdm(enumerate(queries), total=len(queries), desc="Retrieving")
            except ImportError:
                pass

        for i, gq in iter_queries:
            qid = f"q_{i:04d}"

            res_a = self._retrieve(
                query_emb=qemb_a[i],
                corpus_embs=emb_a,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                model_name=self.model_a_name,
                query=gq.query,
                query_id=qid,
                corpus_name=gq.corpus_name,
                task_type=gq.task_type,
            )
            res_b = self._retrieve(
                query_emb=qemb_b[i],
                corpus_embs=emb_b,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                model_name=self.model_b_name,
                query=gq.query,
                query_id=qid,
                corpus_name=gq.corpus_name,
                task_type=gq.task_type,
            )

            pairs.append(MatchedPair(
                query=gq.query,
                query_id=qid,
                corpus_name=gq.corpus_name,
                task_type=gq.task_type,
                result_a=res_a,
                result_b=res_b,
            ))

        return pairs

    def _retrieve(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray,
        doc_ids: list[str],
        doc_texts: list[str],
        model_name: str,
        query: str,
        query_id: str,
        corpus_name: str,
        task_type: str,
    ) -> RetrievalResult:
        scores = (corpus_embs @ query_emb) / (
            np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argsort(-scores)[: self.top_k]

        return RetrievalResult(
            query=query,
            query_id=query_id,
            model_name=model_name,
            top_k_ids=[doc_ids[j] for j in top_indices],
            top_k_texts=[doc_texts[j][:400] for j in top_indices],
            top_k_scores=[float(scores[j]) for j in top_indices],
            corpus_name=corpus_name,
            task_type=task_type,
        )

    def _batch_encode(self, model: Any, texts: list[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb = model.encode(batch)
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            all_embs.append(emb)
        return np.vstack(all_embs)

    def _encode_corpus(
        self,
        model: Any,
        doc_texts: list[str],
        model_name: str,
        corpus: dict[str, str],
    ) -> np.ndarray:
        if self.cache_dir:
            cache_key = hashlib.md5(
                (model_name + "".join(list(corpus.keys())[:20])).encode()
            ).hexdigest()[:12]
            cache_path = self.cache_dir / f"corpus_{model_name}_{cache_key}.npy"

            if cache_path.exists():
                print(f"[Harness] Loading cached corpus embeddings from {cache_path}")
                return np.load(str(cache_path))

        embs = self._batch_encode(model, doc_texts)

        if self.cache_dir and cache_path:
            np.save(str(cache_path), embs)
            print(f"[Harness] Cached corpus embeddings to {cache_path}")

        return embs

    def save_pairs(self, pairs: list[MatchedPair], path: str) -> None:
        """Save matched pairs to JSONL for later judge evaluation."""
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p.to_dict()) + "\n")
        print(f"[Harness] Saved {len(pairs)} pairs to {path}")

    @staticmethod
    def load_pairs(path: str) -> list[dict]:
        pairs = []
        with open(path) as f:
            for line in f:
                pairs.append(json.loads(line))
        return pairs
