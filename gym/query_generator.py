"""
QueryGenerator: Synthetic query generation from a corpus.

Strategy (per Niklas/orionw discussion):
  - Sample n random docs from the corpus
  - Ask an LLM to write a query that is RELATED but NOT directly answered
    by those docs (forces model to actually retrieve, not just regurgitate)
  - Optionally generate hard negatives alongside for richer judge context
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

QUERY_PROMPT = """You are constructing a retrieval benchmark.

You will be given {n_docs} document excerpts from a corpus.

Your task:
1. Write ONE search query that a real user might type.
2. The query must be RELATED to the topic of the documents, but NOT directly 
   answered by the excerpts provided — the ideal answer should be findable 
   elsewhere in the corpus.
3. The query should be natural, specific, and 5–20 words long.
4. Do NOT reference the documents themselves (no "according to the text above").

Documents:
{docs}

Respond with ONLY the query text, no explanation, no quotes."""

HARD_NEGATIVE_PROMPT = """Given this query:
  "{query}"

And these retrieved documents (which are NOT the best answer):
{docs}

Write ONE sentence explaining why these documents are relevant but imperfect 
for the query. This will help a judge understand nuanced relevance distinctions.
Respond with only the sentence."""


@dataclass
class GeneratedQuery:
    query: str
    source_doc_ids: list[str]
    corpus_name: str
    task_type: str  # "retrieval" | "sts" | "clustering"
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryGenerator:
    """
    Generates synthetic queries from a corpus using an LLM.

    Args:
        llm_client: Any client with a `.chat(messages) -> str` method.
                    Supports OpenAI, Anthropic, or any wrapper.
        docs_per_query: How many docs to show the LLM per query.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        llm_client: Any,
        docs_per_query: int = 5,
        seed: int = 42,
    ):
        self.llm = llm_client
        self.docs_per_query = docs_per_query
        self.rng = random.Random(seed)

    def generate(
        self,
        corpus: dict[str, str],  # {doc_id: text}
        n_queries: int = 100,
        corpus_name: str = "unknown",
        task_type: str = "retrieval",
        show_progress: bool = True,
    ) -> list[GeneratedQuery]:
        """
        Generate `n_queries` synthetic queries from `corpus`.

        Returns list of GeneratedQuery objects ready for retrieval harness.
        """
        doc_ids = list(corpus.keys())
        queries: list[GeneratedQuery] = []

        iter_range = range(n_queries)
        if show_progress:
            try:
                from tqdm import tqdm
                iter_range = tqdm(iter_range, desc=f"Generating queries [{corpus_name}]")
            except ImportError:
                pass

        for _ in iter_range:
            sampled_ids = self.rng.sample(doc_ids, min(self.docs_per_query, len(doc_ids)))
            sampled_docs = [corpus[did] for did in sampled_ids]

            doc_block = "\n\n".join(
                f"[Doc {i+1}]\n{text[:600]}" for i, text in enumerate(sampled_docs)
            )

            prompt = QUERY_PROMPT.format(n_docs=len(sampled_docs), docs=doc_block)

            try:
                raw = self.llm.chat([{"role": "user", "content": prompt}])
                query_text = self._clean(raw)
            except Exception as e:
                print(f"[QueryGenerator] LLM call failed: {e}")
                continue

            if query_text:
                queries.append(
                    GeneratedQuery(
                        query=query_text,
                        source_doc_ids=sampled_ids,
                        corpus_name=corpus_name,
                        task_type=task_type,
                    )
                )

        return queries

    def generate_from_mteb_task(
        self,
        task_name: str,
        n_queries: int = 100,
        split: str = "test",
    ) -> list[GeneratedQuery]:
        """
        Convenience: load corpus directly from an MTEB task and generate queries.

        Example:
            gen.generate_from_mteb_task("NFCorpus", n_queries=200)
        """
        import mteb

        task = mteb.get_tasks(tasks=[task_name])[0]
        task.load_data()
        corpus = task.corpus.get(split, {})
        doc_map = {
            did: (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for did, doc in corpus.items()
        }
        return self.generate(
            corpus=doc_map,
            n_queries=n_queries,
            corpus_name=task_name,
            task_type="retrieval",
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip().strip('"').strip("'")
        # Remove any "Query:" prefix the model might add
        text = re.sub(r"^(query|question|search query)[\s:]+", "", text, flags=re.IGNORECASE)
        return text.strip()
