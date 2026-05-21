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

訓練產出的 `.pkl` / `.npz` / `.parquet` 檔案從 Colab 下載到本地的 `artifacts/` 資料夾後，Streamlit 即可上線。

## 目錄結構

```
anime-recsys/
├── app.py                  # Streamlit 入口（本地端執行）
├── requirements.txt        # 本地端套件
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── recommender.py      # 統一推薦介面：四種演算法 + 冷啟動邏輯
├── notebooks/              # 在 Colab 執行的 Notebook
│   ├── README.md
│   ├── 01_eda.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_evaluate.ipynb
└── artifacts/              # 從 Colab 下載過來的模型與資料
    ├── README.md
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

> 註：`artifacts/` 內的模型檔案（`.pkl` / `.npz` / `.parquet`）因體積較大（約 30–80 MB）未納入 git 版控，需另外取得，見下方 Setup 步驟 4。

## Setup (給組員 / 給老師)

### 0. 環境需求
- Python 3.10 或以上
- Windows / macOS / Linux 皆可（本地端不需 GPU）

### 1. 取得程式碼
```bash
git clone https://github.com/Virousttyu/anime-recsys.git
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

`artifacts/` 內的模型檔案需另外取得，以下二擇一：

**方案 A（最快）— 直接下載我們訓練好的 artifacts**

點下列 Google Drive 連結，下載整個資料夾（Google Drive 會自動打包成 zip），解壓後把裡面所有檔案放進專案的 `artifacts/` 資料夾：

> 📂 Google Drive：https://drive.google.com/drive/folders/1Toe0bj6tEPwrx8V-HVid48t-nk9Zmw6_?usp=sharing

**方案 B — 自己在 Colab 重新訓練**

依序在 Google Colab 開啟並執行 `notebooks/` 內的三本 notebook：

1. `01_eda.ipynb` — 下載 Kaggle 資料、EDA、切分 train/test
2. `02_train_models.ipynb` — 訓練 Content-Based / User-CF / SVD 三個模型
3. `03_evaluate.ipynb` — 計算評估指標、產生比較圖

跑完後把 Colab 產出的檔案下載到本地的 `artifacts/`。詳細步驟見 `notebooks/README.md`。

### 5. 啟動 Streamlit
```bash
streamlit run app.py
```
瀏覽器會自動打開 `http://localhost:8501`。

## Demo 使用方式

Streamlit 介面共有 5 個分頁：

1. **推薦結果** — 從左側邊欄選一位 Demo User 與演算法，看 Top-K 推薦清單
2. **使用者觀看紀錄** — 檢視該 User 過去評分過的所有作品
3. **自訂使用者 (冷啟動)** — 自己搜尋並選 5–10 部動漫評分，系統即時產生推薦，展示新使用者上線的 cold-start 場景
4. **模型評估比較** — 三模型 + Popularity baseline 的 Precision@10 / Recall@10 / NDCG@10 數字與條形圖
5. **關於本系統** — 架構與演算法說明

左側邊欄可切換 Demo User、推薦演算法（Popularity / Content-Based / User-CF / SVD）與推薦數量 Top-K。

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
