from __future__ import annotations

"""
这个文件是一个很简单的“人脸小数据库”。
- 我们用“名字”当作键，把很多条“数字指纹”（向量）存起来；
- 可以把新的向量加进去；
- 也可以拿一堆向量来问：最像的是谁？
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import json
import numpy as np


@dataclass
class FaceEntry:
    """一条人脸记录：包含名字和一条向量。"""

    name: str
    embedding: np.ndarray


class FaceDatabase:
    """一个用 JSON 文件保存的简易数据库，专门放“人名 → 多个向量”。"""

    def __init__(self, db_path: str = "data/faces_db.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._name_to_embeddings: Dict[str, List[np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        """从硬盘读取 JSON 文件，恢复成内存里的字典结构。"""
        if not os.path.exists(self.db_path):
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, emb_list in data.items():
            self._name_to_embeddings[name] = [np.array(e, dtype=np.float32) for e in emb_list]

    def _save(self) -> None:
        """把内存里的数据变成 JSON 字符串，写回硬盘。"""
        serializable = {name: [e.tolist() for e in emb_list] for name, emb_list in self._name_to_embeddings.items()}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def add(self, name: str, embeddings: np.ndarray) -> None:
        """把这个人的很多条向量都存起来。"""
        if name not in self._name_to_embeddings:
            self._name_to_embeddings[name] = []
        for emb in embeddings:
            self._name_to_embeddings[name].append(np.asarray(emb, dtype=np.float32))
        self._save()

    def list_names(self) -> List[str]:
        """列出数据库里都有谁（有哪些人名）。"""
        return list(self._name_to_embeddings.keys())

    def search(self, query_embeddings: np.ndarray, threshold: float = 0.35) -> Tuple[List[str], List[float]]:
        """给出一堆“要认的人”的向量，返回最像的名字和相似分数。

        原理（简化版）：
        - 对每个人，把他所有向量求“平均”（得到一个“中心”）。
        - 把“要认的人”的向量和每个“中心”做余弦相似度对比。
        - 分数最高的那个就是最像的。如果分数不够高，就说是“unknown”。
        """

        if query_embeddings.size == 0 or not self._name_to_embeddings:
            # 如果没有要查的，或者数据库是空的，就都当作 unknown
            return ["unknown"] * len(query_embeddings), [1.0] * len(query_embeddings)

        labels: List[str] = []
        scores: List[float] = []

        # 1) 先为每个人算一个“中心向量”（很多条向量的平均）
        name_to_centroid = {
            name: np.mean(np.stack(emb_list, axis=0), axis=0)
            for name, emb_list in self._name_to_embeddings.items() if emb_list
        }

        # 2) 把每个“中心”和“要查的向量”都做归一化，便于用余弦相似度
        for name in name_to_centroid:
            c = name_to_centroid[name]
            norm = np.linalg.norm(c) + 1e-8
            name_to_centroid[name] = c / norm
        queries = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)

        # 组成矩阵，方便一次性计算所有相似度
        centroids = np.stack(list(name_to_centroid.values()), axis=0)
        names = list(name_to_centroid.keys())

        # 3) 计算 (N, M) 的相似度矩阵，N 是要查的人数，M 是数据库里的人数
        sims = queries @ centroids.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(sims.shape[0]), best_idx]

        # 4) 如果分数够高（>= 1 - threshold），就认成那个人；否则 unknown
        for i in range(len(queries)):
            sim = float(best_sim[i])
            if sim >= 1.0 - threshold:  # 把阈值从“距离”换算到了“相似度”
                labels.append(names[int(best_idx[i])])
                scores.append(sim)
            else:
                labels.append("unknown")
                scores.append(sim)
        return labels, scores
