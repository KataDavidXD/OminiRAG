from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class HotpotQAAPI:
    """Loader for the HotpotQA dataset (parquet format from HuggingFace).

    Reads the ``distractor`` split by default. Each row contains:
      id, question, answer, type (comparison|bridge), level (easy|medium|hard),
      supporting_facts (dict: title -> list[int]), context (dict: title -> list[str])
    """

    def __init__(self, root_dir: str, split: str = "distractor"):
        self.root_dir = Path(root_dir)
        self.split = split
        self._split_dir = self.root_dir / split

        if not self._split_dir.exists():
            raise ValueError(f"Split directory not found: {self._split_dir}")

        self._df_cache: Optional[pd.DataFrame] = None
        self._items_cache: Optional[List[dict]] = None

    def _load_dataframe(self) -> pd.DataFrame:
        if self._df_cache is not None:
            return self._df_cache

        parquet_files = sorted(self._split_dir.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self._split_dir}")

        dfs = [pd.read_parquet(p) for p in parquet_files]
        self._df_cache = pd.concat(dfs, ignore_index=True)
        return self._df_cache

    def _row_to_item(self, row: pd.Series) -> dict:
        context = row.get("context")
        if isinstance(context, dict):
            context_titles = list(context.get("title", []))
            context_sentences = list(context.get("sentences", []))
        else:
            context_titles = []
            context_sentences = []

        supporting_facts = row.get("supporting_facts")
        if isinstance(supporting_facts, dict):
            sf_titles = list(supporting_facts.get("title", []))
            sf_sent_ids = list(supporting_facts.get("sent_id", []))
        else:
            sf_titles = []
            sf_sent_ids = []

        context_text = ""
        for title, sents in zip(context_titles, context_sentences):
            if isinstance(sents, list):
                context_text += f"[{title}] " + " ".join(sents) + "\n"

        return {
            "id": str(row["id"]),
            "question": row["question"],
            "answer": row["answer"],
            "type": row.get("type", "unknown"),
            "level": row.get("level", "unknown"),
            "supporting_facts_titles": sf_titles,
            "supporting_facts_sent_ids": sf_sent_ids,
            "context_titles": context_titles,
            "context_sentences": context_sentences,
            "context_text": context_text,
            "context_length": len(context_text),
        }

    def load_items(self, subset: str = "validation") -> List[dict]:
        if self._items_cache is not None:
            return self._items_cache

        df = self._load_dataframe()

        items = [self._row_to_item(row) for _, row in df.iterrows()]
        self._items_cache = items
        return items

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        replace: bool = False,
    ) -> List[dict]:
        items = self.load_items()
        rng = random.Random(seed)

        if replace:
            return [rng.choice(items) for _ in range(n)]
        if n > len(items):
            raise ValueError(f"Requested n={n}, but only {len(items)} items available.")
        return rng.sample(items, n)

    def get_stats(self) -> Dict[str, int]:
        items = self.load_items()
        stats: Dict[str, int] = {}
        for item in items:
            key = f"{item['type']}_{item['level']}"
            stats[key] = stats.get(key, 0) + 1
        return dict(sorted(stats.items()))

    def to_dataframe(self, items: Optional[List[dict]] = None) -> pd.DataFrame:
        if items is None:
            items = self.load_items()
        return pd.DataFrame(items)


if __name__ == "__main__":
    import os

    root_dir = os.environ.get(
        "HOTPOTQA_ROOT",
        str(
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "datasets--hotpotqa--hotpot_qa"
            / "snapshots"
            / "1908d6afbbead072334abe2965f91bd2709910ab"
        ),
    )
    api = HotpotQAAPI(root_dir)
    print("Stats:", api.get_stats())
    items = api.sample(5, seed=42)
    for item in items:
        print(f"  [{item['type']:10s} {item['level']:6s}] {item['question'][:80]}")
