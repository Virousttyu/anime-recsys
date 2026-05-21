# Artifacts

這個資料夾存放從 Colab 訓練後下載的模型與資料檔案。Streamlit 啟動時會從這裡讀取。

## 必要檔案 (跑完 Colab 02 後產生)

| 檔案 | 內容 | 大小估計 |
|---|---|---|
| `anime_meta.parquet` | 動漫 metadata (anime_id, name, genre, type, episodes, rating, members) | ~1 MB |
| `mappings.pkl` | user_id ↔ idx、anime_id ↔ idx 對照表 | ~3 MB |
| `user_history.pkl` | dict[user_id → set(anime_id)],已看過清單 | ~10 MB |
| `demo_users.json` | 預先挑好的 10 個 demo user_id | <1 KB |
| `content_sim.npz` | Content-Based 模型:item-item 相似度稀疏矩陣 | ~20 MB |
| `user_cf_model.pkl` | User-CF 模型:user-item matrix + KNN 鄰居 | ~30 MB |
| `svd_model.pkl` | SVD 模型:user/item factors + biases | ~5 MB |

## 03_evaluate 跑完後額外產生

| 檔案 | 內容 |
|---|---|
| `metrics.json` | 三模型 + Popularity baseline 的 Precision@10 / Recall@10 / NDCG@10 |
| `metrics_comparison.png` | 模型比較條形圖 (簡報直接使用) |

## 沒有所有檔案會怎樣?

`src/recommender.py` 設計成**漸進可用 (graceful degradation)**:
- 只要有 `anime_meta`、`mappings`、`user_history` 就能跑 Popularity baseline
- 各別演算法的 .pkl/.npz 缺一個,UI 上對應選項會自動隱藏
- 全部缺時 Streamlit 啟動畫面會顯示明確錯誤訊息
