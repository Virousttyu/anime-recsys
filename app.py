"""Streamlit Web Demo — Anime Recommender (期末版).

Run with:
    streamlit run app.py

期末新增功能:
  • Multimodal (Text+Structural) 演算法可選
  • Feature A: 🏆 三大首推 (Ensemble cross-model consensus + LLM 推薦詞)
  • Feature B: 🎲 探索式推薦 (Tinder mode + 即時 deboost 學習)
  • Groq LLM 整體推薦解釋 (推薦結果分頁)
  • 家庭友善模式 (側邊欄開關,過濾 Hentai/Ecchi)
  • 關於本系統 分頁更新為期末架構
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.recommender import (
    ARTIFACTS_DIR,
    Catalog,
    ColdStartRecommender,
    DiscoveryRecommender,
    artifacts_available,
    available_algorithms,
    ensemble_top_picks,
    get_recommender,
    load_metrics,
    multimodal_available,
)
from src import llm

st.set_page_config(page_title="Anime Recommender (期末)", layout="wide", page_icon=":clapper:")

UNSAFE_TAGS = ("Hentai", "Ecchi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_safe(genres_str):
    if not isinstance(genres_str, str):
        return True
    g = genres_str.lower()
    return not any(t.lower() in g for t in UNSAFE_TAGS)


def _filter_safe(results):
    return [r for r in results if _is_safe(r.get("genres", ""))]


def _pad_with_popularity(results, k, _catalog, safe_only=True):
    """若 results 不足 k 個 (例如被家庭友善模式過濾掉很多),
    用 Popularity baseline 的安全作品補滿,確保 UI 一定有 k 個項目。"""
    if len(results) >= k:
        return results[:k]
    have_ids = {r["anime_id"] for r in results}
    meta = _catalog.meta
    pool = meta[~meta.index.isin(have_ids)]
    if safe_only:
        pool = pool[pool["genre"].apply(_is_safe)]
    need = k - len(results)
    pool = pool.sort_values("members", ascending=False).head(need)
    padded = _catalog.format_results(
        pool.index.tolist(),
        pool["members"].astype(float).tolist(),
    )
    return results + padded


def _render_results_table(results):
    df = pd.DataFrame(results)
    if df.empty:
        st.warning("沒有產生任何推薦。")
        return
    cols = [c for c in ["title", "genres", "type", "episodes", "global_rating", "members", "score"]
            if c in df.columns]
    st.dataframe(
        df[cols].rename(columns={
            "title": "作品名稱", "genres": "類型", "type": "形式",
            "episodes": "集數", "global_rating": "全站均分",
            "members": "觀看人數", "score": "推薦分數",
        }),
        width="stretch", hide_index=True,
    )


def _top_pick_card(col, rank, item, user_likes_titles):
    """Feature A: 渲染一張首推卡片 (含 LLM 推薦詞)。"""
    medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else "🏆"
    with col:
        with st.container(border=True):
            st.markdown(f"### {medal} {item['title']}")
            st.caption(f"**{item.get('genres','')}** · {item.get('type','')} · 全站 {item.get('global_rating','?')} ⭐")
            # LLM pitch
            cache_key = f"pitch::{item['anime_id']}::{','.join(user_likes_titles[:5])}"
            if "pitch_cache" not in st.session_state:
                st.session_state.pitch_cache = {}
            if cache_key not in st.session_state.pitch_cache:
                if llm.is_configured():
                    with st.spinner("LLM 撰寫推薦詞中..."):
                        pitch = llm.top_pick_pitch(
                            item["title"], item.get("genres", ""),
                            item.get("synopsis", ""), user_likes_titles,
                        )
                else:
                    pitch = ""
                st.session_state.pitch_cache[cache_key] = pitch
            pitch = st.session_state.pitch_cache.get(cache_key, "")
            if pitch:
                st.markdown(f"💬 _{pitch}_")
            elif not llm.is_configured():
                st.caption("(設定 Groq API key 即可顯示 LLM 推薦詞)")


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

st.title("🎬 Anime Recommender System")
st.caption("行為預測與推薦系統 · 期末 Demo — 多模態 + LLM + 探索式互動")

if not artifacts_available():
    st.error(
        "⚠️ 找不到必要的模型檔案。\n\n"
        f"請確認下列檔案已放到 `{ARTIFACTS_DIR}`:\n"
        "- anime_meta.parquet / mappings.pkl / user_history.pkl\n\n"
        "由 Colab notebooks 01–04 產生。"
    )
    st.stop()


@st.cache_resource(show_spinner="載入模型中...")
def load_catalog():
    return Catalog.load()


@st.cache_resource(show_spinner="準備推薦器...")
def load_recommender(name, _catalog):
    return get_recommender(name, _catalog)


@st.cache_resource(show_spinner="準備探索式推薦...")
def load_discovery(_catalog):
    return DiscoveryRecommender(_catalog)


catalog = load_catalog()
algos = available_algorithms()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Demo 設定")
demo_user_ids = catalog.demo_users()
user_choice = st.sidebar.selectbox(
    "選擇 Demo 使用者 (user_id)",
    options=demo_user_ids,
    format_func=lambda uid: f"User #{uid} · {len(catalog.user_history.get(uid, set()))} 部已看過",
)
algo_choice = st.sidebar.selectbox(
    "選擇推薦演算法",
    options=algos,
    index=algos.index("Multimodal (Text+Structural)") if "Multimodal (Text+Structural)" in algos else 0,
)
k = st.sidebar.slider("推薦數量 (Top-K)", 5, 20, 10, 1)
family_safe = st.sidebar.checkbox("👨‍👩‍👧 家庭友善模式 (過濾 Hentai/Ecchi)", value=True)

st.sidebar.markdown("---")
if llm.is_configured():
    st.sidebar.success("🤖 Groq LLM 已連線")
else:
    st.sidebar.warning("🤖 Groq LLM 未設定\n\n參考 `.streamlit/secrets.toml.example`")

if multimodal_available():
    st.sidebar.caption("✅ Multimodal 模態:text + structural")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_reco, tab_history, tab_discovery, tab_custom, tab_metrics, tab_about = st.tabs([
    "📋 推薦結果",
    "📚 觀看紀錄",
    "🎲 探索式推薦",
    "✨ 自訂使用者 (冷啟動)",
    "📊 模型評估比較",
    "ℹ️ 關於本系統",
])


# ====================== Tab 1: 推薦結果 (含 Feature A) ======================

with tab_reco:
    recommender = load_recommender(algo_choice, catalog)

    # 取使用者最喜歡的作品名 (給 LLM 當 context)
    history_set = catalog.user_history.get(int(user_choice), set())
    user_likes = []
    if history_set:
        h_df = catalog.meta.loc[catalog.meta.index.isin(history_set)].copy()
        if "rating" in h_df.columns:
            h_df = h_df.sort_values("rating", ascending=False)
        user_likes = h_df.head(5).get("name", pd.Series()).astype(str).tolist()

    # ============ Feature A: 三大首推 (Cross-model Ensemble) ============
    st.subheader("🏆 三大首推")
    st.caption("跨多個演算法 (Content-Based / User-CF / SVD / Multimodal) 的共識推薦,先看這三部最不會錯。")
    with st.spinner("計算跨模型共識中..."):
        # 家庭友善時要更大的候選池才能保證 filter 後還有 3 個
        picks = ensemble_top_picks(
            catalog, int(user_choice),
            k=15 if family_safe else 3,
            pool=50 if family_safe else 30,
        )
    if family_safe:
        picks = _filter_safe(picks)
        picks = _pad_with_popularity(picks, 3, catalog, safe_only=True)
    cols = st.columns(3)
    for i, p in enumerate(picks[:3]):
        _top_pick_card(cols[i], i, p, user_likes)

    st.markdown("---")

    # ============ 全演算法 Top-K 列表 ============
    st.subheader(f"📋 演算法: {algo_choice} — Top {k}")
    st.info(recommender.explanation)
    with st.spinner("計算推薦中..."):
        # 家庭友善時要要更大的候選池
        results = recommender.recommend(int(user_choice), k=k * (5 if family_safe else 1))
    if family_safe:
        results = _filter_safe(results)
        results = _pad_with_popularity(results, k, catalog, safe_only=True)
    else:
        results = results[:k]
    if not results:
        st.warning("沒有產生任何推薦。")
    else:
        _render_results_table(results)

        # ============ LLM 整體解釋 ============
        if llm.is_configured() and user_likes:
            with st.expander("🤖 LLM 解讀:為什麼這份清單適合你?", expanded=False):
                cache_key = f"top10::{user_choice}::{algo_choice}::{family_safe}"
                if "top10_cache" not in st.session_state:
                    st.session_state.top10_cache = {}
                if cache_key not in st.session_state.top10_cache:
                    with st.spinner("LLM 分析中..."):
                        text = llm.top10_explanation(
                            [r["title"] for r in results], user_likes,
                        )
                    st.session_state.top10_cache[cache_key] = text
                exp = st.session_state.top10_cache.get(cache_key, "")
                if exp:
                    st.write(exp)
                else:
                    st.caption("(LLM 暫時無法回應)")


# ====================== Tab 2: 觀看紀錄 ======================

with tab_history:
    st.subheader(f"User #{user_choice} 的觀看紀錄")
    seen = sorted(catalog.user_history.get(int(user_choice), set()))
    if seen:
        seen_df = catalog.meta.loc[catalog.meta.index.isin(seen)].copy()
        st.write(f"共 {len(seen_df)} 部作品")
        out = seen_df.reset_index()
        cols = [c for c in ["anime_id", "name", "genre", "type", "rating"] if c in out.columns]
        st.dataframe(out[cols], width="stretch", hide_index=True)
    else:
        st.info("此使用者尚無觀看紀錄。")


# ====================== Tab 3: Feature B 探索式推薦 ======================

with tab_discovery:
    st.subheader("🎲 探索式推薦 (Discovery Loop)")
    st.markdown(
        "**從一個起點開始,系統一次只推一部給你看**。已看過 → 下一部;不感興趣 → 系統會 deboost 相似作品再推下一部;"
        "找到喜歡的就停在那裡。這個分頁展示**即時負回饋學習** (interactive negative feedback)。"
    )

    # 初始化 session state
    if "disc_pool" not in st.session_state:
        st.session_state.disc_pool = []
        st.session_state.disc_shown = set()
        st.session_state.disc_rejected = set()
        st.session_state.disc_current = None
        st.session_state.disc_chosen = None
        st.session_state.disc_seed_label = ""

    # 起始方式選擇
    mode = st.radio(
        "選擇起始方式:",
        ("從某部作品開始(找相似)", "從類型標籤開始"),
        key="disc_mode",
        horizontal=True,
    )

    discovery = load_discovery(catalog)

    if mode == "從某部作品開始(找相似)":
        name_col = "name" if "name" in catalog.meta.columns else "title"
        all_titles_map = {str(catalog.meta.loc[aid, name_col]): aid
                          for aid in catalog.meta.index
                          if pd.notna(catalog.meta.loc[aid, name_col])}
        seed_title = st.selectbox(
            "🔍 選一部 seed 作品 (找相似)",
            options=[""] + sorted(all_titles_map.keys()),
            index=0,
        )
        if st.button("🎬 開始探索", type="primary", disabled=(not seed_title)):
            seed_id = all_titles_map[seed_title]
            pool = discovery.pool_from_seed(seed_id, pool_size=50)
            if family_safe:
                pool = [(a, s) for a, s in pool
                        if _is_safe(str(catalog.meta.loc[a, "genre"]) if a in catalog.meta.index else "")]
            st.session_state.disc_pool = pool
            st.session_state.disc_shown = set()
            st.session_state.disc_rejected = set()
            st.session_state.disc_chosen = None
            st.session_state.disc_seed_label = f"以《{seed_title}》為相似起點"
            st.session_state.disc_current = discovery.next_card(
                pool, st.session_state.disc_shown, st.session_state.disc_rejected,
            )
            st.rerun()
    else:
        # Genre tag mode
        all_genres = set()
        for g in catalog.meta["genre"].dropna():
            for t in str(g).split(","):
                t = t.strip()
                if t and (not family_safe or t not in UNSAFE_TAGS):
                    all_genres.add(t)
        chosen_tags = st.multiselect(
            "🏷️ 選擇想看的類型標籤 (多選)",
            options=sorted(all_genres),
            placeholder="例如:Action, Mecha, Psychological",
        )
        if st.button("🎬 開始探索", type="primary", disabled=(len(chosen_tags) == 0)):
            pool = discovery.pool_from_tags(chosen_tags, pool_size=50)
            if family_safe:
                pool = [(a, s) for a, s in pool
                        if _is_safe(str(catalog.meta.loc[a, "genre"]) if a in catalog.meta.index else "")]
            st.session_state.disc_pool = pool
            st.session_state.disc_shown = set()
            st.session_state.disc_rejected = set()
            st.session_state.disc_chosen = None
            st.session_state.disc_seed_label = f"以類型「{' + '.join(chosen_tags)}」起點"
            st.session_state.disc_current = discovery.next_card(
                pool, st.session_state.disc_shown, st.session_state.disc_rejected,
            )
            st.rerun()

    st.markdown("---")

    # 顯示當前卡片
    if st.session_state.disc_chosen is not None:
        chosen_id = st.session_state.disc_chosen
        if chosen_id in catalog.meta.index:
            row = catalog.meta.loc[chosen_id]
            st.success(f"🎉 你找到了:**{row['name']}** ({row.get('type','?')}, 全站 {row.get('rating','?')}⭐)")
            with st.expander("劇情簡介", expanded=True):
                st.write(str(row.get("synopsis", "(無 synopsis)"))[:1000])
            st.caption(f"統計:已看過 {len(st.session_state.disc_shown)} 部、不感興趣 {len(st.session_state.disc_rejected)} 部")
        if st.button("🔄 重新開始探索"):
            for k_ in ("disc_pool", "disc_shown", "disc_rejected",
                      "disc_current", "disc_chosen", "disc_seed_label"):
                st.session_state.pop(k_, None)
            st.rerun()
    elif st.session_state.disc_current is not None:
        current_id = st.session_state.disc_current
        if current_id in catalog.meta.index:
            row = catalog.meta.loc[current_id]
            with st.container(border=True):
                st.markdown(f"### 🎴 《{row['name']}》")
                st.caption(
                    f"**{row.get('genre','')}** · {row.get('type','')} · "
                    f"{row.get('episodes','?')} 集 · 全站 {row.get('rating','?')} ⭐ · "
                    f"{row.get('members',0):,} 人看過"
                )
                # LLM 短評
                if llm.is_configured():
                    pitch_key = f"disc_pitch::{current_id}::{st.session_state.disc_seed_label}"
                    if "disc_pitch_cache" not in st.session_state:
                        st.session_state.disc_pitch_cache = {}
                    if pitch_key not in st.session_state.disc_pitch_cache:
                        with st.spinner("LLM 短評中..."):
                            pitch = llm.discovery_pitch(
                                row["name"], row.get("genre", ""),
                                str(row.get("synopsis", "")),
                                st.session_state.disc_seed_label,
                            )
                        st.session_state.disc_pitch_cache[pitch_key] = pitch
                    pitch = st.session_state.disc_pitch_cache.get(pitch_key, "")
                    if pitch:
                        st.markdown(f"💬 _{pitch}_")
                with st.expander("劇情簡介"):
                    st.write(str(row.get("synopsis", "(無 synopsis)"))[:800])

            st.markdown("**你對這部的回應:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ 看過了,下一部", width="stretch"):
                    st.session_state.disc_shown.add(current_id)
                    st.session_state.disc_current = discovery.next_card(
                        st.session_state.disc_pool,
                        st.session_state.disc_shown,
                        st.session_state.disc_rejected,
                    )
                    st.rerun()
            with c2:
                if st.button("⭐ 沒看過,就是這個!", type="primary", width="stretch"):
                    st.session_state.disc_chosen = current_id
                    st.rerun()
            with c3:
                if st.button("🚫 不感興趣,下一部", width="stretch"):
                    st.session_state.disc_shown.add(current_id)
                    st.session_state.disc_rejected.add(current_id)
                    st.session_state.disc_current = discovery.next_card(
                        st.session_state.disc_pool,
                        st.session_state.disc_shown,
                        st.session_state.disc_rejected,
                    )
                    st.rerun()
            st.caption(
                f"📊 進度:已看過 {len(st.session_state.disc_shown)} 部、"
                f"不感興趣 {len(st.session_state.disc_rejected)} 部 · "
                f"候選池 {len(st.session_state.disc_pool)} 部 · "
                f"{st.session_state.disc_seed_label}"
            )
    elif st.session_state.disc_pool:
        st.warning("候選池都用完了!請重新開始探索。")
        if st.button("🔄 重新選起點"):
            for k_ in ("disc_pool", "disc_shown", "disc_rejected",
                      "disc_current", "disc_chosen", "disc_seed_label"):
                st.session_state.pop(k_, None)
            st.rerun()
    else:
        st.info("👆 選一個起始方式並按「開始探索」。")


# ====================== Tab 4: 自訂使用者 (期中既有) ======================

with tab_custom:
    st.subheader("自訂使用者 ‒ 冷啟動推薦 (Cold-Start)")
    st.markdown(
        "從零開始建立一位使用者 — 選 5 部以上動漫並評分,系統即時計算推薦。"
        "**展示新使用者上線場景**。"
    )
    name_col = "name" if "name" in catalog.meta.columns else "title"
    title_series = catalog.meta[name_col].dropna()
    title_to_id = {str(t): aid for aid, t in title_series.items()}
    all_titles = sorted(title_to_id.keys())
    if "custom_ratings" not in st.session_state:
        st.session_state.custom_ratings = {}
    if "custom_results" not in st.session_state:
        st.session_state.custom_results = None
    selected_titles = st.multiselect(
        "🔍 搜尋並選擇你看過的動漫 (建議 5‒10 部)",
        options=all_titles, max_selections=20,
        placeholder="輸入動漫名稱關鍵字...",
    )
    selected_ids = {title_to_id[t] for t in selected_titles}
    st.session_state.custom_ratings = {
        aid: r for aid, r in st.session_state.custom_ratings.items() if aid in selected_ids
    }
    if selected_titles:
        st.markdown("**為每部作品評分 (1‒10,5 是中性,8 以上代表喜歡)**")
        rate_cols = st.columns(2)
        for i, title in enumerate(selected_titles):
            anime_id = title_to_id[title]
            default = st.session_state.custom_ratings.get(anime_id, 8)
            with rate_cols[i % 2]:
                st.session_state.custom_ratings[anime_id] = st.slider(
                    title, 1, 10, default, key=f"rate_{anime_id}",
                )
        st.markdown("---")
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            custom_algo = st.selectbox("推薦演算法", options=algos, key="custom_algo")
        with col_b:
            custom_k = st.slider("推薦數量", 5, 20, 10, key="custom_k")
        with col_c:
            st.markdown("")
            st.markdown("")
            go = st.button("✨ 產生推薦", type="primary", width="stretch")
        if go:
            if len(st.session_state.custom_ratings) == 0:
                st.warning("請至少對 1 部作品評分。")
            else:
                with st.spinner("從你的評分推算推薦中..."):
                    cs = ColdStartRecommender(catalog)
                    res = cs.recommend(
                        st.session_state.custom_ratings, custom_algo,
                        custom_k * (5 if family_safe else 1),
                    )
                    if family_safe:
                        res = _filter_safe(res)
                        res = _pad_with_popularity(res, custom_k, catalog, safe_only=True)
                    else:
                        res = res[:custom_k]
                    st.session_state.custom_results = {
                        "algo": custom_algo, "k": custom_k, "results": res,
                    }
        if st.session_state.custom_results:
            res = st.session_state.custom_results
            st.markdown(f"### 推薦結果 — 演算法:{res['algo']}")
            if not res["results"]:
                st.warning("沒有產生任何推薦。")
            else:
                _render_results_table(res["results"])
                with st.expander("💡 你輸入的評分"):
                    inputs = [{"作品": catalog.meta.loc[aid, name_col], "你的評分": r}
                              for aid, r in st.session_state.custom_ratings.items()
                              if aid in catalog.meta.index]
                    if inputs:
                        st.dataframe(pd.DataFrame(inputs).sort_values("你的評分", ascending=False),
                                     width="stretch", hide_index=True)
    else:
        st.info("👆 請先在上方搜尋並選擇你看過的動漫。")


# ====================== Tab 5: 模型評估 ======================

with tab_metrics:
    st.subheader("離線評估指標 (Precision@10 / Recall@10 / NDCG@10)")
    metrics = load_metrics()
    img_path = ARTIFACTS_DIR / "metrics_comparison.png"
    if metrics is None and not img_path.exists():
        st.info("尚未產生評估數據。請在 Colab 執行 `notebooks/03_evaluate.ipynb`。")
    else:
        if metrics is not None:
            st.dataframe(pd.DataFrame(metrics).T, width="stretch")
        if img_path.exists():
            st.image(str(img_path), caption="模型比較條形圖")


# ====================== Tab 6: About ======================

with tab_about:
    st.subheader("關於本系統 (期末版)")
    st.markdown(
        """
**主題**:基於 MyAnimeList (hernan4444 2020 版) 公開資料集的個人化動漫推薦系統。

**期末新增重點 (對比期中)**:
1. **多模態推薦**:結構化 (genre/type) + 文字語意 (synopsis 經 sentence-transformers 嵌入) 加權混合
2. **LLM 整合 (Groq + Llama 3.1)**:Top Picks 卡片副本、整體清單解釋、探索式推薦短評
3. **🏆 三大首推 (Feature A)**:跨多模型的 reciprocal-rank fusion,給最值得優先看的 3 部
4. **🎲 探索式推薦 (Feature B)**:Tinder-style 單張瀏覽 + 即時 deboost 學習負回饋

**訓練 / 服務分離架構**:
- 訓練、嵌入、評估、產出 artifacts → Google Colab
- Streamlit Demo → 本地端,不重訓,只 inference + LLM API 呼叫

**演算法清單**:
- Popularity baseline:全站熱門
- Content-Based:Genre multi-hot + Type one-hot,cosine similarity
- User-Based CF:KNN (k=30) on user-item rating matrix
- SVD:Truncated SVD,latent factors = 50
- **Multimodal (Text+Structural) [期末新增]**:sentence-transformers (all-MiniLM-L6-v2) 384-D 文字嵌入 + 結構化特徵,權重 0.6/0.4

**評估方式**:Precision@10 / Recall@10 / NDCG@10,以 Popularity 為對照組,於隨機抽樣 1000 位 user 上計算。
        """
    )
