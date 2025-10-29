# Movie Recommender - Interactive + Analysis (User & Group)

Personalized and group movie recommendations using **user-based collaborative filtering** (Pearson kNN). Works with **MovieLens 1M** (`ratings.dat`, `movies.dat`) or **100K** (`u.data`, `u.item`). If no data is found, the script generates a small **synthetic** dataset so it still runs.

## Features
- **Interactive CLI**: enter any **user id** → see **nearest neighbors** and **Top-N** movie recommendations.
- **Analysis**: EDA, sparsity, user/item distributions, CF **RMSE/MAE** (per-user test split), neighbor similarity stats, catalog & genre coverage of Top-N.
- **Group rounds**: optionally enter a list of user ids to get **sequential group** recommendations with fairness weighting.
- **Robust loading**: supports MovieLens 1M → 100K → synthetic fallback.
- **Jupyter/Notebook safe**: ignores hidden kernel args.

## Data
- Downloaded from https://grouplens.org/datasets/movielens/.
- Place MovieLens files under `./data/`:
- **1M**: `ratings.dat`, `movies.dat` (double-colon `::` separated).
- **100K**: `u.data`, `u.item`.

## Follow prompts:

- Enter a user id from your dataset (e.g., 348).
- Optionally enter group ids (e.g., 234,456,745) to run sequential rounds.
- Results also save to ./outputs/.

## Requirements
Create a virtual env (optional) and install deps:
```bash
pip install -r requirements.txt
