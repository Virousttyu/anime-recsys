"""Unified recommendation interface — classical, multimodal, and LLM-augmented recommenders."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_pickle(name):
    with open(ARTIFACTS_DIR / name, "rb") as f:
        return pickle.load(f)


def artifacts_available():
    required = ["anime_meta.parquet", "mappings.pkl", "user_history.pkl"]
    return all((ARTIFACTS_DIR / r).exists() for r in required)


def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _topk_with_mask(scores, mask_idx, k):
    for i in mask_idx:
        scores[i] = -np.inf
    n = len(scores)
    cap = min(k * 2, n - 1)
    top_idx = np.argpartition(-scores, cap)[: cap + 1]
    return top_idx[np.argsort(-scores[top_idx])][:k]


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
                "synopsis": row.get("synopsis", ""),
                "score": float(score),
                "reason": reason_template.format(**row.to_dict()) if reason_template else "",
            })
        return rows


# ===========================================================================
# Base + classic recommenders
# ===========================================================================

class BaseRecommender:
    name = "base"
    explanation = ""

    def __init__(self, catalog):
        self.catalog = catalog

    def recommend(self, user_id, k=10):
        raise NotImplementedError


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
    explanation = "Genre multi-hot + Type one-hot, cosine similarity, 推薦與你高評分作品最相似的作品。"

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
    explanation = "KNN on user-item rating matrix。找評分習慣相似的 K 位使用者,推薦他們高評分但你未看過的作品。"

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
    explanation = "Truncated SVD,user 與 item 的潛在向量內積預測未看過作品的評分。"

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


# ===========================================================================
# NEW: Multimodal Recommender (Text + Structural)
# ===========================================================================

class MultimodalRecommender(BaseRecommender):
    name = "Multimodal (Text+Structural)"
    explanation = (
        "多模態混合推薦:結合 sentence-transformers 對 synopsis 的語意嵌入 (text 模態) "
        "與 genre/type 結構化特徵 (structural 模態),加權合併後排序。"
    )

    def __init__(self, catalog, text_weight=0.6):
        super().__init__(catalog)
        self._sim = sparse.load_npz(ARTIFACTS_DIR / "content_sim.npz")
        self._text_emb = np.load(ARTIFACTS_DIR / "text_embeddings.npy")
        self.text_weight = float(text_weight)

    def recommend(self, user_id, k=10):
        seen_anime_ids = self.catalog.user_history.get(user_id, set())
        if not seen_anime_ids:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        seen_idx = [self.catalog.anime_id_to_idx[a] for a in seen_anime_ids
                    if a in self.catalog.anime_id_to_idx]
        if not seen_idx:
            return PopularityRecommender(self.catalog).recommend(user_id, k)
        scores = self._combined_scores(seen_idx)
        top_idx = _topk_with_mask(scores, seen_idx, k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])

    def _combined_scores(self, seen_idx):
        # Text modality: build user vector by averaging seen items' embeddings
        u_vec = self._text_emb[seen_idx].mean(axis=0)
        n = np.linalg.norm(u_vec)
        if n > 0:
            u_vec = u_vec / n
        text_sim = self._text_emb @ u_vec  # (N_ITEMS,)
        # Structural modality
        struct_sim = np.asarray(self._sim[seen_idx].sum(axis=0)).ravel().astype(np.float32)
        # Weighted combination
        w = self.text_weight
        return (w * _norm01(text_sim) + (1 - w) * _norm01(struct_sim)).astype(np.float32)


# ===========================================================================
# NEW: Ensemble for Top Picks (cross-model consensus)
# ===========================================================================

def ensemble_top_picks(catalog, user_id, k=3, pool=30, available_models=None):
    """Cross-model agreement: anime ranked in multiple recommenders' Top-N
    get higher consensus scores via reciprocal-rank fusion.
    """
    if available_models is None:
        available_models = ALGO_REGISTRY
    scores = {}  # anime_id -> aggregated reciprocal rank
    counts = {}  # anime_id -> # models that included it
    for name, cls in available_models.items():
        if name == "Popularity (Baseline)":
            continue  # exclude baseline to avoid bias toward popular-only items
        try:
            rec = cls(catalog).recommend(user_id, k=pool)
        except Exception:
            continue
        for rank, item in enumerate(rec):
            aid = item["anime_id"]
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (rank + 1.0)
            counts[aid] = counts.get(aid, 0) + 1
    if not scores:
        return PopularityRecommender(catalog).recommend(user_id, k)
    # Sort: prefer items found in MORE models, then by aggregated reciprocal rank
    ordered = sorted(scores.keys(), key=lambda a: (-counts[a], -scores[a]))[:k]
    return catalog.format_results(ordered, [scores[a] for a in ordered])


# ===========================================================================
# NEW: Discovery Recommender (Tinder-mode 探索式推薦)
# ===========================================================================

class DiscoveryRecommender:
    """Interactive exploration: build a candidate pool from a seed anime or
    a set of genre tags, then serve one anime at a time with deboost-learning
    from user rejections."""

    def __init__(self, catalog):
        self.catalog = catalog
        self._text_emb = np.load(ARTIFACTS_DIR / "text_embeddings.npy")
        try:
            self._sim = sparse.load_npz(ARTIFACTS_DIR / "content_sim.npz")
        except Exception:
            self._sim = None

    def pool_from_seed(self, seed_anime_id, pool_size=50):
        if seed_anime_id not in self.catalog.anime_id_to_idx:
            return []
        seed_idx = self.catalog.anime_id_to_idx[seed_anime_id]
        text_sim = self._text_emb @ self._text_emb[seed_idx]
        if self._sim is not None:
            struct_sim = np.asarray(self._sim[seed_idx].todense()).ravel().astype(np.float32)
            combined = 0.6 * _norm01(text_sim) + 0.4 * _norm01(struct_sim)
        else:
            combined = _norm01(text_sim)
        combined[seed_idx] = -np.inf
        pool_size = min(pool_size, len(combined) - 1)
        top_idx = np.argpartition(-combined, pool_size)[:pool_size]
        top_idx = top_idx[np.argsort(-combined[top_idx])]
        return [(self.catalog.idx_to_anime_id[i], float(combined[i])) for i in top_idx]

    def pool_from_tags(self, genre_tags, pool_size=50):
        if not genre_tags:
            return []
        tags_lower = [t.lower() for t in genre_tags]
        meta = self.catalog.meta
        def matches(g):
            if not isinstance(g, str):
                return 0
            gl = g.lower()
            return sum(1 for t in tags_lower if t in gl)
        match_counts = meta["genre"].apply(matches)
        matched = meta[match_counts > 0].copy()
        if matched.empty:
            return []
        matched["_match_count"] = match_counts[match_counts > 0]
        # Rank by # tags matched then by members (popularity)
        matched = matched.sort_values(
            ["_match_count", "members"], ascending=[False, False]
        ).head(pool_size)
        return [(int(a), float(matched.loc[a, "_match_count"])) for a in matched.index]

    def next_card(self, pool, shown, rejected, deboost=0.4):
        """Given pool (list of (anime_id, base_score)), pick next card.
        - shown : set of anime_ids already presented
        - rejected : set of anime_ids the user said 不感興趣
        Returns next anime_id (or None if pool exhausted).
        """
        remaining = [(a, s) for a, s in pool if a not in shown]
        if not remaining:
            return None
        if not rejected:
            return remaining[0][0]
        # Compute deboost: each remaining anime's similarity to the rejected centroid
        rejected_idx = [self.catalog.anime_id_to_idx[a] for a in rejected
                        if a in self.catalog.anime_id_to_idx]
        if not rejected_idx:
            return remaining[0][0]
        reject_vec = self._text_emb[rejected_idx].mean(axis=0)
        n = np.linalg.norm(reject_vec)
        if n > 0:
            reject_vec = reject_vec / n
        # Score each remaining = base_score - deboost * similarity_to_rejected
        rem_anime_ids = [a for a, _ in remaining]
        rem_idx = [self.catalog.anime_id_to_idx[a] for a in rem_anime_ids
                   if a in self.catalog.anime_id_to_idx]
        if len(rem_idx) != len(rem_anime_ids):
            # Fallback
            return remaining[0][0]
        sim_to_rejected = self._text_emb[rem_idx] @ reject_vec
        base_scores = np.array([s for _, s in remaining], dtype=np.float32)
        adjusted = base_scores - deboost * sim_to_rejected
        best = int(np.argmax(adjusted))
        return rem_anime_ids[best]


# ===========================================================================
# Registry + factory
# ===========================================================================

ALGO_REGISTRY = {
    "Popularity (Baseline)": PopularityRecommender,
    "Content-Based": ContentBasedRecommender,
    "User-Based CF": UserCFRecommender,
    "SVD (Matrix Factorization)": SVDRecommender,
    "Multimodal (Text+Structural)": MultimodalRecommender,
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
    if ((ARTIFACTS_DIR / "text_embeddings.npy").exists()
            and (ARTIFACTS_DIR / "content_sim.npz").exists()):
        available.append("Multimodal (Text+Structural)")
    return available


def multimodal_available():
    return (ARTIFACTS_DIR / "text_embeddings.npy").exists()


def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# Cold-start support
# ===========================================================================

class ColdStartRecommender:
    """Generate recommendations for a brand-new user from a ratings dict."""

    def __init__(self, catalog):
        self.catalog = catalog
        self._content_sim = None
        self._user_cf = None
        self._svd = None
        self._text_emb = None

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

    def _load_text(self):
        if self._text_emb is None:
            self._text_emb = np.load(ARTIFACTS_DIR / "text_embeddings.npy")
        return self._text_emb

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
        if algo == "Multimodal (Text+Structural)":
            return self._multimodal(ratings, k)
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
        scores = np.zeros(sim.shape[0], dtype=np.float32)
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
        nv = new_vec / float(np.linalg.norm(new_vec))
        dots = np.asarray(user_item @ nv).ravel()
        sq = user_item.multiply(user_item).sum(axis=1)
        unorms = np.sqrt(np.asarray(sq).ravel())
        unorms[unorms == 0] = 1.0
        sims = dots / unorms
        n_neighbors = min(n_neighbors, n_users)
        top_users = np.argpartition(-sims, n_neighbors - 1)[:n_neighbors]
        top_sims = sims[top_users]
        order = np.argsort(-top_sims)
        top_users, top_sims = top_users[order], top_sims[order]
        scores = np.asarray(top_sims @ user_item[top_users].toarray()).ravel()
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])

    def _svd_recommend(self, ratings, k):
        svd = self._load_svd()
        rated_idx, rated_targets = [], []
        for aid, r in ratings.items():
            if aid in self.catalog.anime_id_to_idx:
                rated_idx.append(self.catalog.anime_id_to_idx[aid])
                rated_targets.append(r)
        if not rated_idx:
            return self._popularity(ratings, k)
        rated_idx_np = np.asarray(rated_idx)
        target = (np.asarray(rated_targets, dtype=np.float32)
                  - svd["global_mean"] - svd["item_bias"][rated_idx_np])
        A = svd["item_factors"][rated_idx_np]
        n_factors = A.shape[1]
        gram = A.T @ A + 0.1 * np.eye(n_factors, dtype=np.float32)
        u = np.linalg.solve(gram, A.T @ target).astype(np.float32)
        scores = svd["global_mean"] + svd["item_bias"] + svd["item_factors"] @ u
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])

    def _multimodal(self, ratings, k):
        emb = self._load_text()
        sim = self._load_content()
        rated_idx, weights = [], []
        for aid, r in ratings.items():
            if aid in self.catalog.anime_id_to_idx:
                rated_idx.append(self.catalog.anime_id_to_idx[aid])
                weights.append((r - 5.0) / 5.0)
        if not rated_idx:
            return self._popularity(ratings, k)
        w = np.array(weights, dtype=np.float32)
        # Text modality: weighted average user vector
        u_vec = (emb[rated_idx] * w[:, None]).sum(axis=0)
        if np.linalg.norm(u_vec) > 1e-9:
            u_vec = u_vec / np.linalg.norm(u_vec)
        text_sim = emb @ u_vec
        # Structural modality: weighted sum
        struct_sim = np.zeros(sim.shape[0], dtype=np.float32)
        for i, weight in zip(rated_idx, w):
            row = np.asarray(sim[i].todense()).ravel().astype(np.float32)
            struct_sim += weight * row
        scores = 0.6 * _norm01(text_sim) + 0.4 * _norm01(struct_sim)
        top_idx = _topk_with_mask(scores, self._mask_rated_ids(ratings), k)
        anime_ids = [self.catalog.idx_to_anime_id[i] for i in top_idx]
        return self.catalog.format_results(anime_ids, scores[top_idx])
