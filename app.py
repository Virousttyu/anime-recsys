"""Streamlit Web Demo for the Anime Recommender System.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import (
    ARTIFACTS_DIR,
    Catalog,
    ColdStartRecommender,
    artifacts_available,
    available_algorithms,
    get_recommender,
    load_metrics,
)

st.set_page_config(page_title="Anime Recommender", layout="wide", page_icon=":clapper:")


# ---------------------------------------------------------------------------
# Bootstrapping & error states
# ---------------------------------------------------------------------------

st.title("🎬 Anime Recommender System")
st.caption("行為預測與推薦系統 · 期中報告 Demo")

if not artifacts_available():
    st.error(
        "⚠️ 找不到必要的模型檔案。\n\n"
        f"請確認下列檔案已放到資料夾 `{ARTIFACTS_DIR}` :\n"
        "- anime_meta.parquet\n"
        "- mappings.pkl\n"
        "- user_history.pkl\n\n"
        "這些檔案應由 Colab notebook (`02_train_models.ipynb`) 產生後下載到本地。"
    )
    st.stop()


@st.cache_resource(show_spinner="載入模型中...")
def load_catalog():
    return Catalog.load()


@st.cache_resource(show_spinner="準備推薦器...")
def load_recommender(name, _catalog):
    return get_recommender(name, _catalog)


catalog = load_catalog()
algos = available_algorithms()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Demo 設定")

demo_user_ids = catalog.demo_users()
user_choice = st.sidebar.selectbox(
    "選擇 Demo 使用者 (user_id)",
    options=demo_user_ids,
    format_func=lambda uid: f"User #{uid}  ·  {len(catalog.user_history.get(uid, set()))} 部已看過",
)

algo_choice = st.sidebar.selectbox("選擇推薦演算法", options=algos)
k = st.sidebar.slider("推薦數量 (Top-K)", min_value=5, max_value=20, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 在 Colab 重新訓練模型後,把 `artifacts/` 整個資料夾覆蓋到本專案下,"
    "點 Streamlit 右上角的 'Rerun' 即可載入新模型。"
)


# ---------------------------------------------------------------------------
# Main: recommendations
# ---------------------------------------------------------------------------

tab_reco, tab_history, tab_custom, tab_metrics, tab_about = st.tabs(
    ["📋 推薦結果", "📚 使用者觀看紀錄", "✨ 自訂使用者 (冷啟動)", "📊 模型評估比較", "ℹ️ 關於本系統"]
)


def _render_results_table(results):
    df = pd.DataFrame(results)
    if df.empty:
        st.warning("沒有產生任何推薦。")
        return
    display_cols = [c for c in
                    ["title", "genres", "type", "episodes", "global_rating", "members", "score"]
                    if c in df.columns]
    st.dataframe(
        df[display_cols].rename(columns={
            "title": "作品名稱", "genres": "類型", "type": "形式",
            "episodes": "集數", "global_rating": "全站均分",
            "members": "觀看人數", "score": "推薦分數",
        }),
        width="stretch",
        hide_index=True,
    )


with tab_reco:
    st.subheader(f"演算法: {algo_choice}")
    recommender = load_recommender(algo_choice, catalog)
    st.info(recommender.explanation)
    with st.spinner("計算推薦中..."):
        results = recommender.recommend(int(user_choice), k=k)
    if not results:
        st.warning("沒有產生任何推薦。請嘗試其他使用者。")
    else:
        _render_results_table(results)


with tab_history:
    st.subheader(f"User #{user_choice} 的觀看紀錄")
    seen = sorted(catalog.user_history.get(int(user_choice), set()))
    if seen:
        seen_df = catalog.meta.loc[catalog.meta.index.isin(seen)].copy()
        st.write(f"共 {len(seen_df)} 部作品")
        cols = ["anime_id", "name", "genre", "type", "rating"]
        out = seen_df.reset_index()
        st.dataframe(
            out[[c for c in cols if c in out.columns]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("此使用者尚無觀看紀錄。")


with tab_custom:
    st.subheader("自訂使用者 ‒ 冷啟動推薦 (Cold-Start)")
    st.markdown(
        "從零開始建立一位使用者 — 在下面選 5 部以上你看過的動漫並評分,"
        "系統會即時計算並推薦你可能會喜歡的下一部作品。"
        "**這是推薦系統最常見的「新使用者上線」場景**。"
    )

    name_col = "name" if "name" in catalog.meta.columns else "title"
    title_series = catalog.meta[name_col].dropna()
    title_to_id = {str(title): aid for aid, title in title_series.items()}
    all_titles = sorted(title_to_id.keys())

    if "custom_ratings" not in st.session_state:
        st.session_state.custom_ratings = {}
    if "custom_results" not in st.session_state:
        st.session_state.custom_results = None

    selected_titles = st.multiselect(
        "🔍 搜尋並選擇你看過的動漫 (建議 5‒10 部)",
        options=all_titles,
        max_selections=20,
        placeholder="輸入動漫名稱關鍵字...",
        help="可以混搭你喜歡和討厭的作品,以提升推薦準確度。",
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
                    st.session_state.custom_results = {
                        "algo": custom_algo,
                        "k": custom_k,
                        "results": cs.recommend(
                            st.session_state.custom_ratings, custom_algo, custom_k
                        ),
                    }

        if st.session_state.custom_results:
            res = st.session_state.custom_results
            st.markdown(f"### 推薦結果 — 演算法:{res['algo']}")
            if not res["results"]:
                st.warning("沒有產生任何推薦,請增加評分作品數量再試。")
            else:
                _render_results_table(res["results"])
                with st.expander("💡 你輸入的評分"):
                    inputs = [
                        {"作品": catalog.meta.loc[aid, name_col], "你的評分": r}
                        for aid, r in st.session_state.custom_ratings.items()
                        if aid in catalog.meta.index
                    ]
                    if inputs:
                        st.dataframe(
                            pd.DataFrame(inputs).sort_values("你的評分", ascending=False),
                            width="stretch",
                            hide_index=True,
                        )
    else:
        st.info("👆 請先在上方搜尋並選擇你看過的動漫,即可開始評分。")


with tab_metrics:
    st.subheader("離線評估指標 (Precision@10 / Recall@10 / NDCG@10)")
    metrics = load_metrics()
    img_path = ARTIFACTS_DIR / "metrics_comparison.png"
    if metrics is None and not img_path.exists():
        st.info("尚未產生評估數據。請在 Colab 執行 `notebooks/03_evaluate.ipynb`。")
    else:
        if metrics is not None:
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df, width="stretch")
        if img_path.exists():
            st.image(str(img_path), caption="模型比較條形圖")


with tab_about:
    st.subheader("關於本系統")
    st.markdown(
        """
**主題**:基於 MyAnimeList 公開資料集的個人化動漫推薦系統。

**訓練與服務分離架構**:
- 訓練、評估、產出模型 → Google Colab (免費 GPU/RAM、避免 Windows 套件相容問題)
- Streamlit Demo → 本地端 (只 inference,不重訓)

**演算法**:
- Popularity baseline:推薦全站最熱門作品
- Content-Based:Genre multi-hot + Type one-hot,cosine similarity (CooperUnion 資料集無 synopsis 欄位)
- User-Based CF:KNN (k=30) on user-item rating matrix
- SVD:Truncated SVD on centered rating matrix,latent factors = 50,本地端以純 numpy inference

**冷啟動 (Cold-Start) 支援**:本系統提供「自訂使用者」模式,允許全新使用者即時評分後立即推薦。
SVD 採用 fold-in 技巧 (ridge regression 解出臨時 user factor);User-CF 即時計算新使用者與所有現存使用者的 cosine similarity。

**評估方式**:Precision@10、Recall@10、NDCG@10,以 Popularity 為對照組,於 1000 位隨機抽樣 user 上計算。

**期末擴展計畫**:加入 Neural CF / Transformer 序列模型、做即時評分推薦、部署到雲端。
        """
    )
