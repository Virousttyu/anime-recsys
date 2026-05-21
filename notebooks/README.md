# Colab Notebooks

依序在 Google Colab 上執行下列三本 notebook,訓練流程全部在 Colab 完成,本地端不需要重訓。

| 順序 | Notebook | 目的 | 估計時間 |
|---|---|---|---|
| 1 | `01_eda.ipynb` | 下載 Kaggle 資料、EDA、切分 train/test | 5‒10 分鐘 |
| 2 | `02_train_models.ipynb` | 訓練 Content-Based / User-CF / SVD 並輸出 artifacts | 10‒20 分鐘 |
| 3 | `03_evaluate.ipynb` | 評估 Precision@10 / Recall@10 / NDCG@10、產生條形圖 | 3‒5 分鐘 |

## 在 Colab 跑的方法

1. 在 Colab 開啟 https://colab.research.google.com
2. File → Upload notebook → 選擇本資料夾中的 `.ipynb`
3. 第一個 cell 會掛載 Google Drive,把產出檔案存到 `/content/drive/MyDrive/anime-recsys/artifacts/`
4. 點選 Runtime → Run all
5. 跑完後在 Google Drive 中找到 `artifacts/` 資料夾,**整個下載到本地專案的 `artifacts/` 取代原本的空資料夾**

## Kaggle API 設定 (只做一次)

1. 在 https://www.kaggle.com 註冊帳號
2. Account → Settings → API → Create New API Token
3. 會下載一個 `kaggle.json` 檔案
4. 在 `01_eda.ipynb` 第一段會有一個檔案上傳的 cell,選這個 `kaggle.json` 上傳即可

之後其他 Notebook 都會自動讀取已下載的資料,不用再上傳 kaggle.json。
