# Colab Notebooks

期末改版:依序在 Google Colab 上執行下列**四本** notebook,訓練流程全部在 Colab 完成,本地端不需要重訓。

| 順序 | Notebook | 目的 | 估計時間 |
|---|---|---|---|
| 1 | `01_eda.ipynb` | 下載 hernan4444 資料集、EDA、切分 train/test (含 synopsis) | 5‒10 分鐘 |
| 2 | `02_train_models.ipynb` | 訓練 Content-Based / User-CF / SVD 三個傳統模型 | 10‒20 分鐘 |
| 3 | `03_evaluate.ipynb` | Precision@10 / Recall@10 / NDCG@10、產生條形圖 | 3‒5 分鐘 |
| 4 | `04_multimodal_embeddings.ipynb` | **(期末新增)** 用 sentence-transformers 對 synopsis 做語意嵌入 | 3‒8 分鐘 |

## 在 Colab 跑的方法

1. 在 Colab 開啟 https://colab.research.google.com
2. File → Upload notebook → 選擇本資料夾中的 `.ipynb`
3. 第一個 cell 會掛載 Google Drive,把產出檔案存到 `/content/drive/MyDrive/anime-recsys/artifacts/`
4. 點選 Runtime → Run all
5. 跑完後在 Google Drive 中找到 `artifacts/` 資料夾,**整個下載到本地專案的 `artifacts/` 取代原本的內容**

## 期末改版差異

- **資料集從 CooperUnion 改成 hernan4444 的 2020 版** ── 因為需要 synopsis 欄位給多模態文字嵌入用
- **多了 Notebook 04** ── 產生 384 維文字語意向量 (text_embeddings.npy)
- Notebook 02、03 **完全不用改**,因為 Notebook 01 已經把欄位名稱對齊舊 pipeline

## Kaggle API 設定 (只做一次)

1. 在 https://www.kaggle.com 註冊帳號
2. Account → Settings → API → Create New API Token
3. 會下載一個 `kaggle.json` 檔案
4. 在 `01_eda.ipynb` 第一段會有一個檔案上傳的 cell,選這個 `kaggle.json` 上傳即可

之後其他 Notebook 都會自動讀取已下載的資料,不用再上傳 kaggle.json。
