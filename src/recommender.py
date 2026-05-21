"""Unified recommendation interface."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_pickle(name: str):
    with open(ARTIFACTS_DIR / name, "rb") as f:
        return pickle.load(f)


def artifacts_available() -> bool:
    required = ["anime_meta.parquet", "mappings.pkl", "user_history.pkl"]
    return all((ARTIFACTS_DIR / r).exists() for r in required)


@dataclass
class Catalog:
    meta: pd.DataFrame
    user_id_to_idx: dict
    idx_to_user_id: dict
    anime_id_to_idx: dict
    idx_to_anime_id: dict
    user_history: dict

    @classmethod
    def load(cls):
        meta = pd.read_parquet(ARTIFACTS_DIR / "anime_meta.parquet").set_index("anime_id")
        mappings = _load_pickle("mappings.pkl")
        user_history = _load_pickle("user_history.pkl")
        return cls(
            meta=meta,
            user_id_to_idx=mappings["user_id_to_idx"],
            idx_to_user_id=mappings["idx_to_user_id"],
            anime_id_to_idx=mappings["anime_id_to_idx"],
            idx_to_anime_id=mappings["idx_to_anime_id"],
            user_history=user_history,
        )

    def demo_users(self):
        path = ARTIFACTS_DIR / "demo_users.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        ranked = sorted(self.user_history.items(), key=lambda kv: -len(kv[1]))
        return [uid for uid, _ in ranked[:10]]

    def format_results(self, anime_ids, scores, reason_template=None):
        rows = []
        for aid, score in zip(anime_ids, scores):
            if aid not in self.meta.index:
                continue
            row = self.meta.loc[aid]
            rows.append({
                "anime_id": int(aid),
                "title": row.get("name", row.get("title", "?")),
                "genres": row.get("genre", row.get("genres", "")),
                "type": row.get("type", ""),
                "episodes": row.get("episodes", ""),
                "global_rating": row.get("rating", float("nan")),
                "members": row.get("members", 0),
                "score": float(score),
                "reason": reason_template.format(**row.to_dict()) if reason_template else "",
            })
        return rows


class BaseRecommender:
    name = "base"
    explanation = ""

    def __init__(self, catalog):
        self.catalog = catalog

    def recommend(self, user_id, k=10):
        raise NotImplementedError


def _topk_with_mask(scores, mask_idx, k):
    """Mask given indices to -inf, return top-k indices ordered."""
    for i in mask_idx:
        scores[i] = -np.inf
    n = len(scores)
    cap = min(k * 2, n - 1)
    top_idx = np.argpartition(-scores, cap)[: cap + 1]
    return top_idx[np.argsort(-scores[top_idx])][:k]


class PopularityRecommender(BaseRecommender):
    name = "Popularity (Baseline)"
    explanation = "推薦全站最熱門 (members 數最多) 且使用者尚未看過的作品。"

    def recommend(self, user_id, k=10):
        seen = self.catalog.user_history.get(user_id, set())
        meta = self.catalog.meta
        ranked = meta[~meta.index.isin(seen)].sort_values("members", ascending=False).head(k)
        return self.catalog.format_results(
            ranked.index.tolist(),
            ranked["members"].astype(float).tolist(),
        )


class ContentBasedRecommender(BaseRecommender):
    name = "Content-Based"
    explanation = "用 Genre multi-hot + Type one-hot 表示每部作品,以 cosine similarity 找出與你高評分作品最相似的新作品。"

    def __init__(self, catalog):
        super().__init__(catalog)
        self._sim = sparse.load_npz(ARTIFACTS_DIR / "content_sim.npz")

    def recommend(self, user_id, k=10):
        seen_anime_ids = self.catalog.user_history.get(user_id, set())
        if not seen_anime_ids:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        seen_idx = [self.catalog.anime_id_to_idx[a] for a in seen_anime_ids
                    if a in self.catalog.anime_id_to_idx]
        if not seen_idx:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        sims = np.asarray(self._sim[seen_idx].sum(axis=0)).ravel()
        top_idx = _topk_with_mask(sims, seen_idx, k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, sims[top_idx])


class UserCFRecommender(BaseRecommender):
    name = "User-Based CF"
    explanation = "找出評分習慣與你最相似的 K 位使用者,推薦他們高評分但你還沒看過的作品。"

    def __init__(self, catalog):
        super().__init__(catalog)
        bundle = _load_pickle("user_cf_model.pkl")
        self._user_item = bundle["user_item"]
        self._neighbors_idx = bundle["neighbors_idx"]
        self._neighbors_sim = bundle["neighbors_sim"]

    def recommend(self, user_id, k=10):
        if user_id not in self.catalog.user_id_to_idx:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        u = self.catalog.user_id_to_idx[user_id]
        nb_idx = self._neighbors_idx[u]
        nb_sim = self._neighbors_sim[u]
        scores = np.asarray(nb_sim @ self._user_item[nb_idx].toarray()).ravel()
        seen = self.catalog.user_history.get(user_id, set())
        mask = [self.catalog.anime_id_to_idx[a] for a in seen if a in self.catalog.anime_id_to_idx]
        top_idx = _topk_with_mask(scores, mask, k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])


class SVDRecommender(BaseRecommender):
    name = "SVD (Matrix Factorization)"
    explanation = "把 user-item 評分矩陣分解成潛在因子,用 user 與 item 的潛在向量內積預測未看過作品的評分。"

    def __init__(self, catalog):
        super().__init__(catalog)
        bundle = _load_pickle("svd_model.pkl")
        self.user_factors = bundle["user_factors"]
        self.item_factors = bundle["item_factors"]
        self.user_bias = bundle["user_bias"]
        self.item_bias = bundle["item_bias"]
        self.global_mean = bundle["global_mean"]

    def recommend(self, user_id, k=10):
        if user_id not in self.catalog.user_id_to_idx:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        u = self.catalog.user_id_to_idx[user_id]
        scores = (self.global_mean + self.user_bias[u] + self.item_bias
                  + self.item_factors @ self.user_factors[u])
        seen = self.catalog.user_history.get(user_id, set())
        mask = [self.catalog.anime_id_to_idx[a] for a in seen if a in self.catalog.anime_id_to_idx]
        top_idx = _topk_with_mask(scores, mask, k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])


ALGO_REGISTRY = {
    "Popularity (Baseline)": PopularityRecommender,
    "Content-Based": ContentBasedRecommender,
    "User-Based CF": UserCFRecommender,
    "SVD (Matrix Factorization)": SVDRecommender,
}


def get_recommender(name, catalog):
    return ALGO_REGISTRY[name](catalog)


def available_algorithms():
    available = ["Popularity (Baseline)"]
    if (ARTIFACTS_DIR / "content_sim.npz").exists():
        available.append("Content-Based")
    if (ARTIFACTS_DIR / "user_cf_model.pkl").exists():
        available.append("User-Based CF")
    if (ARTIFACTS_DIR / "svd_model.pkl").exists():
        available.append("SVD (Matrix Factorization)")
    return available


def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ColdStartRecommender:
    """Generate recommendations for a brand-new user from a ratings dict."""

    def __init__(self, catalog):
        self.catalog = catalog
        self._content_sim = None
        self._user_cf = None
        self._svd = None

    def _load_content(self):
        if self._content_sim is None:
            self._content_sim = sparse.load_npz(ARTIFACTS_DIR / "content_sim.npz")
        return self._content_sim

    def _load_user_cf(self):
        if self._user_cf is None:
            self._user_cf = _load_pickle("user_cf_model.pkl")
        return self._user_cf

    def _load_svd(self):
        if self._svd is None:
            self._svd = _load_pickle("svd_model.pkl")
        return self._svd

    def recommend(self, ratings, algo, k=10):
        if not ratings:
            return self._popularity(ratings, k)
        if algo == "Popularity (Baseline)":
            return self._popularity(ratings, k)
        if algo == "Content-Based":
            return self._content(ratings, k)
        if algo == "User-Based CF":
            return self._user_cf_recommend(ratings, k)
        if algo == "SVD (Matrix Factorization)":
            return self._svd_recommend(ratings, k)
        raise ValueError("Unknown algorithm: " + str(algo))

    def _mask_rated_ids(self, ratings):
        return [self.catalog.anime_id_to_idx[a] for a in ratings
                if a in self.catalog.anime_id_to_idx]

    def _popularity(self, ratings, k):
        rated = set(ratings.keys())
        meta = self.catalog.meta
        ranked = meta[~meta.index.isin(rated)].sort_values("members", ascending=False).head(k)
        return self.catalog.format_results(
            ranked.index.tolist(),
            ranked["members"].astype(float).tolist(),
        )

    def _content(self, ratings, k):
        sim = self._load_content()
        n_items = sim.shape[0]
        scores = np.zeros(n_items, dtype=np.float32)
        used = False
        for aid, r in ratings.items():
            if aid not in self.catalog.anime_id_to_idx:
                continue
            idx = self.catalog.anime_id_to_idx[aid]
            weight = (r - 5.0) / 5.0
            row = np.asarray(sim[idx].todense()).ravel().astype(np.float32)
            scores += weight * row
            used = True
        if not used:
            return self._popularity(ratings, k)
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])

    def _user_cf_recommend(self, ratings, k, n_neighbors=30):
        bundle = self._load_user_cf()
        user_item = bundle["user_item"]
        n_users, n_items = user_item.shape

        new_vec = np.zeros(n_items, dtype=np.float32)
        for aid, r in ratings.items():
            if aid in self.catalog.anime_id_to_idx:
                new_vec[self.catalog.anime_id_to_idx[aid]] = r
        if not np.any(new_vec):
            return self._popularity(ratings, k)
        new_norm = float(np.linalg.norm(new_vec))
        new_vec_unit = new_vec / new_norm

        dots = np.asarray(user_item @ new_vec_unit).ravel()
        sq = user_item.multiply(user_item).sum(axis=1)
        user_norms = np.sqrt(np.asarray(sq).ravel())
        user_norms[user_norms == 0] = 1.0
        sims_to_users = dots / user_norms

        n_neighbors = min(n_neighbors, n_users)
        top_users = np.argpartition(-sims_to_users, n_neighbors - 1)[:n_neighbors]
        top_sims = sims_to_users[top_users]
        order = np.argsort(-top_sims)
        top_users = top_users[order]
        top_sims = top_sims[order]

        scores = np.asarray(top_sims @ user_item[top_users].toarray()).ravel()
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])

    def _svd_recommend(self, ratings, k):
        svd = self._load_svd()
        rated_idx = []
        rated_targets = []
        for aid, r in ratings.items():
            if aid in self.catalog.anime_id_to_idx:
                rated_idx.append(self.catalog.anime_id_to_idx[aid])
                rated_targets.append(r)
        if not rated_idx:
            return self._popularity(ratings, k)

        rated_idx_np = np.asarray(rated_idx)
        target = (np.asarray(rated_targets, dtype=np.float32)
                  - svd["global_mean"]
                  - svd["item_bias"][rated_idx_np])
        A = svd["item_factors"][rated_idx_np]

        n_factors = A.shape[1]
        lam = 0.1
        gram = A.T @ A + lam * np.eye(n_factors, dtype=np.float32)
        rhs = A.T @ target
        u_factor = np.linalg.solve(gram, rhs).astype(np.float32)

        scores = (svd["global_mean"]
                  + svd["item_bias"]
                  + svd["item_factors"] @ u_factor)
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])
