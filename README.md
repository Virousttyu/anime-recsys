# Anime Recommender System

> 行為預測與推薦系統 ‒ 期中報告專題
> 主題：基於 MyAnimeList 公開資料集的個人化動漫推薦系統

## 專案概念

本系統解決動漫觀眾「作品太多挑不出來」的困境。傳統的「熱門排行」缺乏個人化，「相似作品」推薦只看單一作品。我們建構一套個人化推薦系統，輸入使用者過往的評分後，輸出 Top-10 推薦清單，並比較三種傳統推薦演算法的效果差異。

## 訓練／服務分離架構

| 環境 | 負責 | 為什麼 |
|---|---|---|
| Google Colab | 資料下載、EDA、模型訓練、評估 | 免費 RAM、避免 Windows 套件安裝問題 (scikit-surprise 等) |
| 本地端 | Streamlit Web Demo、推薦服務 | 不需重訓，只載入訓練好的模型檔案 |

訓練產出的 `.pkl` / `.npz` / `.json` 檔案從 Colab 下載到本地的 `artifacts/` 資料夾後，Streamlit 即可上線。

## 目錄結構

```
anime-recsys/
├── app.py                  # Streamlit 入口（本地端執行）
├── requirements.txt        # 本地端套件
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── recommender.py      # 統一推薦介面：載入 artifacts、產生 Top-K
│   └── ui_helpers.py       # Streamlit UI 工具函式
├── notebooks/              # 在 Colab 執行的 Notebook
│   ├── 01_eda.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_evaluate.ipynb
└── artifacts/              # 從 Colab 下載過來的模型與資料
    ├── anime_meta.parquet
    ├── mappings.pkl
    ├── content_sim.npz
    ├── user_cf_model.pkl
    ├── svd_model.pkl
    ├── user_history.pkl
    ├── demo_users.json
    ├── metrics.json
    └── metrics_comparison.png
```

## Setup (給組員 / 給老師)

### 0. 環境需求
- Python 3.10 或以上
- Windows / macOS / Linux 皆可（本地端不需 GPU）

### 1. 取得程式碼
```bash
git clone <your-repo-url>
cd anime-recsys
```

### 2. (建議) 建立虛擬環境
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. 安裝套件
```bash
pip install -r requirements.txt
```

### 4. 取得訓練好的模型 (artifacts)
方案 A ‒ 直接從我們的 Google Drive 下載解壓到 `artifacts/`（推薦給老師）。
方案 B ‒ 自己在 Colab 跑：依序執行 `notebooks/01_eda.ipynb`、`02_train_models.ipynb`、`03_evaluate.ipynb`，下載產出檔案到 `artifacts/`。

### 5. 啟動 Streamlit
```bash
streamlit run app.py
```
瀏覽器會自動打開 `http://localhost:8501`。

## Demo 使用方式

1. 左側選擇一位 Demo User（從資料集中挑出有完整評分歷史的使用者）
2. 選擇推薦演算法（Popularity / Content-Based / User-CF / SVD）
3. 看畫面中央的 Top-10 推薦結果
4. 切到「評估比較」分頁可看到三模型在離線資料上的 Precision@10 / Recall@10 / NDCG@10 條形圖

## 期中評分對應

| Rubric 項目 | 比例 | 對應位置 |
|---|---|---|
| 問題定義清晰度 | 30% | 簡報前 5 頁 + README |
| 模型設計合理性 | 25% | `notebooks/02_train_models.ipynb` + 簡報 |
| 系統架構完整性 | 20% | 本 README 的架構圖 + 程式碼結構 |
| Prototype 完成度 | 15% | `app.py` + Demo 影片 |
| 簡報與表達能力 | 10% | 簡報排練 + 分工 |

## 開發筆記

- 訓練只在 Colab 上做，本地端不需要 GPU、也不需要裝 scikit-surprise（在 Windows 上常出問題）
- artifacts 檔案大小總計約 30–80 MB，可放在 Google Drive 共享給組員
- Streamlit 啟動約 5 秒（載入 .pkl 一次性開銷）
