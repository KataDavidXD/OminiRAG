from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

import pandas as pd


class UltraDomainAPI:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self._domain_to_path = self._discover_domains()

        # 已加载样本缓存
        self._cache: Dict[str, List[dict]] = {}

        # 每个 domain 的行数缓存
        self._stats_cache: Dict[str, int] = {}

        # 所有 domain 合并池缓存（给 uniform 用）
        self._pooled_cache: Dict[tuple[str, ...], List[dict]] = {}

    def _discover_domains(self) -> Dict[str, Path]:
        domain_to_path = {}
        for path in self.root_dir.glob("*.jsonl"):
            if path.suffix != ".jsonl":
                continue
            domain_to_path[path.stem] = path

        if not domain_to_path:
            raise ValueError(f"No jsonl files found under: {self.root_dir}")

        return dict(sorted(domain_to_path.items()))

    @property
    def available_domains(self) -> List[str]:
        return list(self._domain_to_path.keys())

    def _normalize_domains(self, target_domains: Optional[Sequence[str]]) -> List[str]:
        if target_domains is None:
            return self.available_domains

        if isinstance(target_domains, str):
            target_domains = [target_domains]

        normalized = []
        for d in target_domains:
            if d not in self._domain_to_path:
                raise ValueError(
                    f"Unknown domain: {d}. Available: {self.available_domains}"
                )
            normalized.append(d)

        if not normalized:
            raise ValueError("target_domains is empty after normalization.")

        return normalized

    def _count_domain_rows(self, domain: str) -> int:
        if domain in self._stats_cache:
            return self._stats_cache[domain]

        path = self._domain_to_path[domain]
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1

        self._stats_cache[domain] = count
        return count

    def get_domain_stats(self) -> Dict[str, int]:
        return {domain: self._count_domain_rows(domain) for domain in self.available_domains}

    def load_domain(self, domain: str) -> List[dict]:
        if domain not in self._domain_to_path:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {self.available_domains}"
            )

        if domain in self._cache:
            return self._cache[domain]

        path = self._domain_to_path[domain]
        rows: List[dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["_domain"] = domain
                item["_local_id"] = i
                rows.append(item)

        self._cache[domain] = rows
        self._stats_cache[domain] = len(rows)
        return rows

    def load_domains(
        self, target_domains: Optional[Sequence[str]] = None
    ) -> Dict[str, List[dict]]:
        domains = self._normalize_domains(target_domains)
        return {domain: self.load_domain(domain) for domain in domains}

    def _get_pooled_rows(self, domains: List[str]) -> List[dict]:
        key = tuple(sorted(domains))
        if key in self._pooled_cache:
            return self._pooled_cache[key]

        pooled = []
        for domain in domains:
            pooled.extend(self.load_domain(domain))
        self._pooled_cache[key] = pooled
        return pooled

    def sample(
        self,
        n: int,
        target_domains: Optional[Sequence[str]] = None,
        strategy: str = "uniform",
        seed: Optional[int] = None,
        replace: bool = False,
        standardize: bool = True,
    ) -> List[dict]:
        if n <= 0:
            raise ValueError("n must be > 0")

        rng = random.Random(seed)
        domains = self._normalize_domains(target_domains)

        if strategy == "uniform":
            samples = self._sample_uniform(
                domains=domains,
                n=n,
                rng=rng,
                replace=replace,
            )
        elif strategy == "balanced":
            samples = self._sample_balanced(
                domains=domains,
                n=n,
                rng=rng,
                replace=replace,
            )
        else:
            raise ValueError("strategy must be one of: ['uniform', 'balanced']")

        if standardize:
            return [self._to_standard_item(x) for x in samples]
        return samples

    def _sample_uniform(
        self,
        domains: List[str],
        n: int,
        rng: random.Random,
        replace: bool,
    ) -> List[dict]:
        pool = self._get_pooled_rows(domains)

        if not pool:
            return []

        if replace:
            return [rng.choice(pool) for _ in range(n)]

        if n > len(pool):
            raise ValueError(
                f"Requested n={n}, but only {len(pool)} samples available without replacement."
            )

        return rng.sample(pool, n)

    def _sample_balanced(
        self,
        domains: List[str],
        n: int,
        rng: random.Random,
        replace: bool,
    ) -> List[dict]:
        if not domains:
            return []

        base = n // len(domains)
        rem = n % len(domains)
        result = []

        for idx, domain in enumerate(domains):
            k = base + (1 if idx < rem else 0)
            rows = self.load_domain(domain)

            if not rows or k == 0:
                continue

            if replace:
                result.extend(rng.choice(rows) for _ in range(k))
            else:
                if k > len(rows):
                    raise ValueError(
                        f"Domain '{domain}' only has {len(rows)} samples, cannot sample {k} without replacement."
                    )
                result.extend(rng.sample(rows, k))

        rng.shuffle(result)
        return result

    def _to_standard_item(self, item: dict) -> dict:
        # 更贴近 UltraDomain 实际 schema
        query = (
            item.get("input")
            or item.get("question")
            or item.get("query")
            or item.get("prompt")
            or item.get("instruction")
            or ""
        )

        # UltraDomain 常见是 answers: List[str]
        answers = item.get("answers")
        if isinstance(answers, list):
            answer = answers[0] if answers else None
        else:
            answer = (
                item.get("answer")
                or item.get("output")
                or item.get("response")
                or item.get("gold")
            )

        return {
            "id": f"{item.get('_domain', item.get('label', 'unknown'))}::{item.get('_local_id', -1)}",
            "domain": item.get("_domain", item.get("label", "unknown")),
            "query": query,
            "answer": answer,
            "answers": answers if isinstance(answers, list) else None,
            "context": item.get("context"),
            "context_id": item.get("context_id"),
            "label": item.get("label"),
            "length": item.get("length"),
            "title": (item.get("meta") or {}).get("title") if isinstance(item.get("meta"), dict) else None,
            "authors": (item.get("meta") or {}).get("authors") if isinstance(item.get("meta"), dict) else None,
            "raw": item,
        }

    def sample_batch(
        self,
        n: int,
        target_domains: Optional[Sequence[str]] = None,
        strategy: str = "uniform",
        seed: Optional[int] = None,
        replace: bool = False,
        standardize: bool = True,
    ) -> dict:
        normalized_domains = self._normalize_domains(target_domains)
        items = self.sample(
            n=n,
            target_domains=normalized_domains,
            strategy=strategy,
            seed=seed,
            replace=replace,
            standardize=standardize,
        )
        return {
            "sampling_strategy": strategy,
            "target_domains": normalized_domains,
            "seed": seed,
            "n": n,
            "replace": replace,
            "items": items,
        }

    def to_dataframe(
        self,
        items: List[dict],
        flatten: bool = True,
        drop_raw: bool = False,
    ) -> pd.DataFrame:
        if not items:
            return pd.DataFrame()

        if flatten:
            df = pd.json_normalize(items, sep=".")
        else:
            df = pd.DataFrame(items)

        if drop_raw and "raw" in df.columns:
            df = df.drop(columns=["raw"])

        return df


if __name__ == "__main__":
    '''
    uv tool install hf
    hf download TommyChien/UltraDomain --repo-type=dataset
    -> find root_dir
    '''

    root_dir = r"C:\Users\Administrator\.cache\huggingface\hub\datasets--TommyChien--UltraDomain\snapshots\aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"
    api = UltraDomainAPI(root_dir)

    print(api.available_domains)
    print(api.get_domain_stats())

    '''
    ['agriculture', 'art', 'biography', 'biology', 'cooking', 'cs', 'fiction', 'fin', 'health', 'history', 'legal', 'literature', 'mathematics', 'mix', 'music', 'philosophy', 'physics', 'politics', 'psychology', 'technology']
    {'agriculture': 100, 'art': 200, 'biography': 180, 'biology': 220, 'cooking': 120, 'cs': 100, 'fiction': 220, 'fin': 345, 'health': 180, 'history': 180, 'legal': 438, 'literature': 180, 'mathematics': 160, 'mix': 130, 'music': 200, 'philosophy': 200, 'physics': 160, 'politics': 180, 'psychology': 200, 'technology': 240}
    '''

    samples = api.sample(
        n=20,
        target_domains=["physics", "cs", "mathematics"],
        strategy="balanced",
        seed=42,
        standardize=True,
    )

    df = api.to_dataframe(samples, flatten=True, drop_raw=True)
    print(df.head())
    print(df.columns.tolist())

    '''
    ['id', 'domain', 'query', 'answer', 'answers', 'context', 'context_id', 'label', 'length', 'title', 'authors', 'raw.input', 'raw.answers', 'raw.context', 'raw.length', 'raw.context_id', 'raw._id', 'raw.label', 'raw.meta.title', 'raw.meta.authors', 'raw._domain', 'raw._local_id']


    id       domain                                              query  ...    raw.meta.authors  raw._domain raw._local_id
    0  mathematics::7  mathematics  How does the concept of dimensions extend beyo...  ...       Theoni Pappas  mathematics             7
    '''

    #df.to_csv("ultradomain_samples.csv", index=False, encoding="utf-8-sig")

    def preview_row(row, max_str=80):
        out = {}
        for k, v in row.items():
            if isinstance(v, str):
                out[k] = v[:max_str] + ("..." if len(v) > max_str else "")
            else:
                out[k] = v
        return out

    row = df.iloc[0].to_dict()

    from pprint import pprint
    pprint(preview_row(row))

    '''
        {'answer': 'Beyond the third dimension, concepts extend into higher dimensions '
            'such as the f...',
    'answers': ['Beyond the third dimension, concepts extend into higher '
                'dimensions such as the fourth dimension, which includes time as '
                'a dimension in physics or the hypercube in geometry. This '
                'understanding challenges traditional spatial perceptions and '
                'opens up new possibilities for modeling complex phenomena and '
                'data structures.'],
    'authors': 'Theoni Pappas',
    'context': ' \n'
                '# THE  \n'
                'MAGIC  \n'
                'OF  \n'
                'MATHEMATICS\n'
                '\n'
                'Discovering the Spell of Mathematics\n'
                '\n'
                '**THEO...',
    'context_id': '49462ff5255c3b8026ea4713e0f8ae5a',
    'domain': 'mathematics',
    'id': 'mathematics::7',
    'label': 'mathematics',
    'length': 84508,
    'query': 'How does the concept of dimensions extend beyond the third '
            'dimension, and what i...',
    'raw._domain': 'mathematics',
    'raw._id': '52c93333be72df7be91bac0a405fd808',
    'raw._local_id': 7,
    'raw.answers': ['Beyond the third dimension, concepts extend into higher '
                    'dimensions such as the fourth dimension, which includes time '
                    'as a dimension in physics or the hypercube in geometry. This '
                    'understanding challenges traditional spatial perceptions and '
                    'opens up new possibilities for modeling complex phenomena '
                    'and data structures.'],
    'raw.context': ' \n'
                    '# THE  \n'
                    'MAGIC  \n'
                    'OF  \n'
                    'MATHEMATICS\n'
                    '\n'
                    'Discovering the Spell of Mathematics\n'
                    '\n'
                    '**THEO...',
    'raw.context_id': '49462ff5255c3b8026ea4713e0f8ae5a',
    'raw.input': 'How does the concept of dimensions extend beyond the third '
                'dimension, and what i...',
    'raw.label': 'mathematics',
    'raw.length': 84508,
    'raw.meta.authors': 'Theoni Pappas',
    'raw.meta.title': 'The Magic of Mathematics',
    'title': 'The Magic of Mathematics'}
    '''
