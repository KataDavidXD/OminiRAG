from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


class FreshWikiAPI:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.json_dir = self.root_dir / "json"
        self.txt_dir = self.root_dir / "txt"
        self.topic_csv = self.root_dir / "topic_list.csv"

        if not self.json_dir.exists():
            raise ValueError(f"json directory not found: {self.json_dir}")
        if not self.txt_dir.exists():
            raise ValueError(f"txt directory not found: {self.txt_dir}")
        if not self.topic_csv.exists():
            raise ValueError(f"topic_list.csv not found: {self.topic_csv}")

        self._topic_meta: Optional[Dict[str, dict]] = None
        self._docs_cache: Optional[List[dict]] = None
        self._json_cache: Dict[str, dict] = {}
        self._txt_cache: Dict[str, str] = {}

    def _topic_to_stem(self, topic: str) -> str:
        return topic.replace(" ", "_").replace("/", "_")

    def _load_topic_meta(self) -> Dict[str, dict]:
        if self._topic_meta is not None:
            return self._topic_meta

        df = pd.read_csv(self.topic_csv)
        meta = {}

        for _, row in df.iterrows():
            topic = row["topic"]
            stem = self._topic_to_stem(topic)

            predicted_scores = row.get("predicted_scores")
            if isinstance(predicted_scores, str):
                try:
                    predicted_scores = ast.literal_eval(predicted_scores)
                except Exception:
                    predicted_scores = predicted_scores

            meta[stem] = {
                "topic": topic,
                "url": row.get("url"),
                "predicted_class": row.get("predicted_class"),
                "predicted_scores": predicted_scores,
            }

        self._topic_meta = meta
        return meta

    def _load_json_doc(self, stem: str) -> dict:
        if stem in self._json_cache:
            return self._json_cache[stem]

        path = self.json_dir / f"{stem}.json"
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        self._json_cache[stem] = obj
        return obj

    def _load_txt_doc(self, stem: str) -> str:
        if stem in self._txt_cache:
            return self._txt_cache[stem]

        path = self.txt_dir / f"{stem}.txt"
        text = path.read_text(encoding="utf-8", errors="replace")
        self._txt_cache[stem] = text
        return text

    def _build_document(self, stem: str) -> dict:
        meta = self._load_topic_meta().get(stem, {})
        json_obj = self._load_json_doc(stem)
        txt_text = self._load_txt_doc(stem)

        return {
            "id": stem,
            "topic": meta.get("topic", stem.replace("_", " ")),
            "title": json_obj.get("title", stem),
            "url": json_obj.get("url", meta.get("url")),
            "summary": json_obj.get("summary"),
            "text": txt_text,
            "predicted_class": meta.get("predicted_class"),
            "predicted_scores": meta.get("predicted_scores"),
            "content": json_obj.get("content", []),
        }

    def load_documents(self) -> List[dict]:
        if self._docs_cache is not None:
            return self._docs_cache

        stems = sorted(p.stem for p in self.json_dir.glob("*.json"))
        docs = [self._build_document(stem) for stem in stems]
        self._docs_cache = docs
        return docs

    def sample_documents(
        self,
        n: int,
        seed: Optional[int] = None,
        replace: bool = False,
    ) -> List[dict]:
        docs = self.load_documents()
        rng = random.Random(seed)

        if replace:
            return [rng.choice(docs) for _ in range(n)]

        if n > len(docs):
            raise ValueError(f"Requested n={n}, but only {len(docs)} docs available.")

        return rng.sample(docs, n)

    def extract_sections(self, doc: dict) -> List[dict]:
        sections = []
        content = doc.get("content", [])

        for i, sec in enumerate(content):
            section_title = sec.get("section_title", f"section_{i}")
            section_content = sec.get("section_content", [])

            sentences = []
            for item in section_content:
                sent = item.get("sentence", "")
                if sent:
                    sentences.append(sent)

            section_text = " ".join(sentences)

            sections.append({
                "doc_id": doc["id"],
                "topic": doc["topic"],
                "section_index": i,
                "section_title": section_title,
                "section_text": section_text,
                "n_sentences": len(sentences),
            })

        return sections

    def sample_sections(
        self,
        n_docs: int,
        seed: Optional[int] = None,
    ) -> List[dict]:
        docs = self.sample_documents(n=n_docs, seed=seed, replace=False)
        sections = []
        for doc in docs:
            sections.extend(self.extract_sections(doc))
        return sections

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1200,
        overlap: int = 150,
    ) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap

        return chunks

    def sample_chunks(
        self,
        n_docs: int,
        chunk_size: int = 1200,
        overlap: int = 150,
        seed: Optional[int] = None,
    ) -> List[dict]:
        docs = self.sample_documents(n=n_docs, seed=seed, replace=False)
        all_chunks = []

        for doc in docs:
            chunks = self.chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc['id']}::chunk_{idx}",
                    "doc_id": doc["id"],
                    "topic": doc["topic"],
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "url": doc["url"],
                    "predicted_class": doc["predicted_class"],
                })

        return all_chunks

    def get_doc_stats(self) -> pd.DataFrame:
        docs = self.load_documents()
        rows = []

        for doc in docs:
            rows.append({
                "id": doc["id"],
                "topic": doc["topic"],
                "predicted_class": doc["predicted_class"],
                "text_length": len(doc.get("text", "")),
                "n_sections": len(doc.get("content", [])),
                "url": doc.get("url"),
            })

        return pd.DataFrame(rows)

    def to_dataframe(self, items: List[dict], drop_content: bool = False) -> pd.DataFrame:
        df = pd.json_normalize(items, sep=".")
        if drop_content and "content" in df.columns:
            df = df.drop(columns=["content"])
        return df

if __name__ == "__main__":  
    
    '''
    uv tool install hf
    hf download EchoShao8899/FreshWiki --repo-type=dataset
    -> find root_dir
    '''

    root_dir = r"C:\Users\Administrator\.cache\huggingface\hub\datasets--EchoShao8899--FreshWiki\snapshots\03f2f8abbe54c78e834f70783de105129c07e18e"
    api = FreshWikiAPI(root_dir)
    df_stats = api.get_doc_stats()
    print(df_stats.head())

    docs = api.sample_documents(n=5, seed=42)
    df_docs = api.to_dataframe(docs, drop_content=True)
    print(df_docs.columns.tolist())
    '''
    对每个 Wikipedia 主题条目质量等级的“预测结果”，用的是 Wikipedia 常见的文章质量分级体系。Wikipedia 的这套等级通常包括 Stub, Start, C, B, GA, FA。
    对应关系大致是：

    Stub：很短、很初级，信息很少
    Start：已经不是纯 stub，但还比较初步
    C：中等，已经有一定结构和内容，但还不够完整
    B：质量较好，基本满足较核心的编辑标准
    GA：Good Article
    FA：Featured Article，最高档之一，通常代表非常成熟、全面、写得好的页面
    
    '''
    
    def preview_row(row, max_str=80):
        out = {}
        for k, v in row.items():
            if isinstance(v, str):
                out[k] = v[:max_str] + ("..." if len(v) > max_str else "")
            else:
                out[k] = v
        return out

    row = df_docs.iloc[0].to_dict()

    from pprint import pprint
    pprint(preview_row(row))
    '''
    'id': 'Sardar_(2022_film)',
    'predicted_class': 'GA',
    'predicted_scores.B': 0.15871378853219237,
    'predicted_scores.C': 0.16826552099778325,
    'predicted_scores.FA': 0.11287859359808501,
    'predicted_scores.GA': 0.5428253123823522,
    'predicted_scores.Start': 0.012795357209396343,
    'predicted_scores.Stub': 0.0045214272801910186,
    'summary': 'Sardar (transl.\u2009Chief) is a 2022 Indian Tamil-language spy '
                'action-thriller film ...',
    'text': 'Sardar_(2022_film)\n'
            '\n'
            'Sardar (transl.\u2009Chief) is a 2022 Indian Tamil-language spy a...',
    'title': 'Sardar_(2022_film)',
    'topic': 'Sardar (2022 film)',
    'url': 'https://en.wikipedia.org/wiki/Sardar_(2022_film)'}
    '''
