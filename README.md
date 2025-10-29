# Movie Recommender — Interactive + Analysis

Personalized and group movie recommendations using user-based collaborative filtering (Pearson kNN), plus dataset-wide analysis. Works with **MovieLens 1M** (`ratings.dat`, `movies.dat`) or **100K** (`u.data`, `u.item`). Falls back to a tiny synthetic dataset if files aren’t found.

## Features
- **Interactive CLI**: enter any user id; get Top-N and nearest neighbors; optionally run **sequential group** recommendations with fairness (coverage-based weighting).
- **Analysis**: EDA, sparsity, user/item distributions, CF **RMSE/MAE** (per-user test split), neighbor similarity stats, catalog & genre coverage of Top-N.

## Data
Place MovieLens files under `./data/`:
- **1M**: `ratings.dat`, `movies.dat` (double-colon `::` separated).
- **100K**: `u.data`, `u.item`.

## Quickstart
```bash
pip install -r requirements.txt
python recsys_project.py --data_dir data
