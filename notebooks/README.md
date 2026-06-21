# Colab Notebooks

All training, embedding generation, and offline evaluation runs in Google Colab. The local serving layer never re-trains; it only loads the artifacts produced here.

| Order | Notebook | Purpose | Wall time (Colab CPU) |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Download the dataset, run EDA, filter sparse users/items, per-user 80/20 split, persist `anime_meta`, `mappings`, `user_history`, train/test parquets | 5–10 min |
| 2 | `02_train_models.ipynb` | Train Content-Based, User-Based CF (KNN), and SVD (truncated MF) | 10–20 min |
| 3 | `03_evaluate.ipynb` | Precision@10 / Recall@10 / NDCG@10 across all five algorithms, render comparison bar chart | 3–5 min |
| 4 | `04_multimodal_embeddings.ipynb` | Encode each anime's synopsis with `sentence-transformers/all-MiniLM-L6-v2` to 384-D L2-normalized vectors | 3–8 min |

## How to run

1. Open https://colab.research.google.com
2. File → Upload notebook → pick the `.ipynb`
3. The first cell mounts Google Drive and writes artifacts to `/content/drive/MyDrive/anime-recsys/artifacts/`
4. Runtime → Run all
5. After all four notebooks finish, download the entire `artifacts/` folder from Drive into the local project's `artifacts/`

## Kaggle API setup (one-time)

1. Create a Kaggle account at https://www.kaggle.com
2. Account → Settings → API → Create New API Token
3. A `kaggle.json` file is downloaded
4. `01_eda.ipynb` has a cell that prompts you to upload `kaggle.json` on first run. After that, subsequent notebooks reuse the cached credentials and the downloaded dataset.

## Implementation notes

- Notebooks 02 and 03 read column names produced by Notebook 01; if you swap to a different dataset, only Notebook 01 needs editing.
- Notebook 04 produces a 384-D float32 `text_embeddings.npy` (~20 MB for ~14K anime). This file is gitignored and shared via Google Drive.
- All four notebooks are idempotent: re-running overwrites the previous artifacts. The serving layer (`streamlit run app.py`) picks up new artifacts on next "Rerun".
