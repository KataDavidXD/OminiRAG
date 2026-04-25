from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path
from typing import Dict, List, Optional


_KNOWN_SUBSETS = {
    "asqa": "asqa_eval_gtr_top100.json",
    "qampari": "qampari_eval_gtr_top100.json",
    "eli5": "eli5_eval_bm25_top100.json",
}


class ALCEAPI:
    """Loader for the ALCE benchmark dataset (princeton-nlp/ALCE-data).

    The HF download is a single tar archive containing JSON files for each
    subset (ASQA, QAMPARI, ELI5).  Each JSON is a list of dicts with keys:
      sample_id, question, answer, docs (list of 100 docs), qa_pairs, ...
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self._tar_path = self.root_dir / "ALCE-data.tar"

        if not self._tar_path.exists():
            raise ValueError(f"ALCE-data.tar not found: {self._tar_path}")

        self._cache: Dict[str, List[dict]] = {}

    def available_subsets(self) -> List[str]:
        return list(_KNOWN_SUBSETS.keys())

    def _extract_json(self, member_name: str) -> List[dict]:
        with tarfile.open(self._tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(member_name):
                    f = tar.extractfile(member)
                    if f is None:
                        raise ValueError(f"Cannot extract {member_name}")
                    return json.load(f)
        raise ValueError(f"File {member_name} not found in {self._tar_path}")

    def _normalize_item(self, raw: dict, subset: str, idx: int) -> dict:
        docs = raw.get("docs", [])
        qa_pairs = raw.get("qa_pairs", [])

        short_answers = []
        for qp in qa_pairs:
            short_answers.extend(qp.get("short_answers", []))

        return {
            "id": raw.get("sample_id", f"{subset}::{idx}"),
            "subset": subset,
            "question": raw.get("question", ""),
            "answer": raw.get("answer", ""),
            "short_answers": short_answers,
            "docs": docs,
            "n_docs": len(docs),
            "qa_pairs": qa_pairs,
            "n_qa_pairs": len(qa_pairs),
        }

    def load_subset(self, subset: str) -> List[dict]:
        if subset in self._cache:
            return self._cache[subset]

        if subset not in _KNOWN_SUBSETS:
            raise ValueError(
                f"Unknown subset: {subset}. Available: {list(_KNOWN_SUBSETS.keys())}"
            )

        filename = _KNOWN_SUBSETS[subset]
        raw_items = self._extract_json(filename)
        items = [
            self._normalize_item(raw, subset, i) for i, raw in enumerate(raw_items)
        ]
        self._cache[subset] = items
        return items

    def load_all(self, subsets: Optional[List[str]] = None) -> Dict[str, List[dict]]:
        target = subsets or list(_KNOWN_SUBSETS.keys())
        return {s: self.load_subset(s) for s in target}

    def sample(
        self,
        n: int,
        subset: str = "asqa",
        seed: Optional[int] = None,
        replace: bool = False,
    ) -> List[dict]:
        items = self.load_subset(subset)
        rng = random.Random(seed)

        if replace:
            return [rng.choice(items) for _ in range(n)]
        if n > len(items):
            raise ValueError(f"Requested n={n}, but only {len(items)} items in {subset}.")
        return rng.sample(items, n)

    def get_stats(self, subsets: Optional[List[str]] = None) -> Dict[str, int]:
        all_data = self.load_all(subsets)
        return {subset: len(items) for subset, items in all_data.items()}


if __name__ == "__main__":
    import os

    root_dir = os.environ.get(
        "ALCE_ROOT",
        str(
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "datasets--princeton-nlp--ALCE-data"
            / "snapshots"
            / "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
        ),
    )
    api = ALCEAPI(root_dir)
    print("Stats:", api.get_stats())
    items = api.sample(3, subset="asqa", seed=42)
    for item in items:
        print(f"  [{item['subset']:10s}] {item['question'][:80]}  (docs={item['n_docs']})")
