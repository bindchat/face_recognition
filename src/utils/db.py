from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import json
import numpy as np


@dataclass
class FaceEntry:
    name: str
    embedding: np.ndarray


class FaceDatabase:
    def __init__(self, db_path: str = "data/faces_db.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._name_to_embeddings: Dict[str, List[np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, emb_list in data.items():
            self._name_to_embeddings[name] = [np.array(e, dtype=np.float32) for e in emb_list]

    def _save(self) -> None:
        serializable = {name: [e.tolist() for e in emb_list] for name, emb_list in self._name_to_embeddings.items()}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def add(self, name: str, embeddings: np.ndarray) -> None:
        if name not in self._name_to_embeddings:
            self._name_to_embeddings[name] = []
        for emb in embeddings:
            self._name_to_embeddings[name].append(np.asarray(emb, dtype=np.float32))
        self._save()

    def list_names(self) -> List[str]:
        return list(self._name_to_embeddings.keys())

    def search(self, query_embeddings: np.ndarray, threshold: float = 0.35) -> Tuple[List[str], List[float]]:
        if query_embeddings.size == 0 or not self._name_to_embeddings:
            return ["unknown"] * len(query_embeddings), [1.0] * len(query_embeddings)
        labels: List[str] = []
        scores: List[float] = []
        # Precompute centroids for each identity
        name_to_centroid = {
            name: np.mean(np.stack(emb_list, axis=0), axis=0)
            for name, emb_list in self._name_to_embeddings.items() if emb_list
        }
        # Normalize centroids and queries
        for name in name_to_centroid:
            c = name_to_centroid[name]
            norm = np.linalg.norm(c) + 1e-8
            name_to_centroid[name] = c / norm
        queries = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        centroids = np.stack(list(name_to_centroid.values()), axis=0)
        names = list(name_to_centroid.keys())
        # Cosine distance
        sims = queries @ centroids.T  # (N, M)
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(sims.shape[0]), best_idx]
        for i in range(len(queries)):
            sim = float(best_sim[i])
            if sim >= 1.0 - threshold:  # translate threshold in cosine distance terms
                labels.append(names[int(best_idx[i])])
                scores.append(sim)
            else:
                labels.append("unknown")
                scores.append(sim)
        return labels, scores
