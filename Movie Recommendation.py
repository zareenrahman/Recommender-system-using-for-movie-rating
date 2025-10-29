#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# file: recsys_project.py
from __future__ import annotations
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd


# =========================
# Data loading (1M → 100K → synthetic)
# =========================
def load_movielens_1m_if_available(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[None, None]:
    """
    Load MovieLens 1M if ratings.dat/movies.dat are present (UserID::MovieID::Rating::Timestamp).
    Returns (ratings_df, items_df) or (None, None) if files missing.
    """
    rpath = os.path.join(data_dir, "ratings.dat")
    mpath = os.path.join(data_dir, "movies.dat")
    if os.path.isfile(rpath) and os.path.isfile(mpath):
        ratings = pd.read_csv(
            rpath, sep="::", engine="python",
            names=["user_id", "item_id", "rating", "timestamp"], header=None,
        )
        items = pd.read_csv(
            mpath, sep="::", engine="python",
            names=["item_id", "title", "genres"], header=None, encoding="latin-1",
        )[["item_id", "title"]]
        return ratings, items
    return None, None


def load_movielens_100k_if_available(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[None, None]:
    """
    Load MovieLens 100K if u.data/u.item exist (tab/pipe separated).
    """
    udata = os.path.join(data_dir, "u.data")
    uitem = os.path.join(data_dir, "u.item")
    if os.path.isfile(udata) and os.path.isfile(uitem):
        ratings = pd.read_csv(
            udata, sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"], engine="python",
        )
        items = pd.read_csv(
            uitem, sep="|", encoding="latin-1",
            header=None, engine="python",
        ).rename(columns={0: "item_id", 1: "title"})[["item_id", "title"]]
        return ratings, items
    return None, None


def make_synthetic_dataset(n_users: int = 30, n_items: int = 60, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Small synthetic fallback so the pipeline runs anywhere.
    """
    rng = random.Random(seed)
    rows = []
    user_latent = {u + 1: rng.uniform(-1, 1) for u in range(n_users)}
    item_latent = {i + 1: rng.uniform(-1, 1) for i in range(n_items)}
    for u in range(1, n_users + 1):
        rated = rng.sample(range(1, n_items + 1), k=max(5, int(0.4 * n_items)))
        for i in rated:
            mu = 3.5 + 1.5 * (user_latent[u] * item_latent[i])
            r = min(5, max(1, round(rng.gauss(mu, 0.8))))
            rows.append((u, i, r, 0))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    items = pd.DataFrame({"item_id": list(range(1, n_items + 1)),
                          "title": [f"Item {i}" for i in range(1, n_items + 1)]})
    return ratings, items


def load_data(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Order: 1M → 100K → synthetic. Returns (ratings, items, is_synthetic).
    """
    r1m, m1m = load_movielens_1m_if_available(data_root)
    if r1m is not None:
        return r1m, m1m, False
    r100k, m100k = load_movielens_100k_if_available(data_root)
    if r100k is not None:
        return r100k, m100k, False
    r, m = make_synthetic_dataset()
    return r, m, True


# =========================
# In-memory structures
# =========================
@dataclass
class UIData:
    by_user: Dict[int, Dict[int, float]]
    by_item: Dict[int, Dict[int, float]]
    user_means: Dict[int, float]
    all_users: Set[int]
    all_items: Set[int]


def build_ui(ratings: pd.DataFrame) -> UIData:
    """
    Build user→{item:rating}, item→{user:rating}, and user means.
    """
    by_user: Dict[int, Dict[int, float]] = defaultdict(dict)
    by_item: Dict[int, Dict[int, float]] = defaultdict(dict)
    for row in ratings.itertuples(index=False):
        u = int(row.user_id); i = int(row.item_id); r = float(row.rating)
        by_user[u][i] = r
        by_item[i][u] = r
    user_means = {u: (sum(items.values()) / len(items)) if items else 0.0 for u, items in by_user.items()}
    return UIData(by_user, by_item, user_means, set(by_user.keys()), set(by_item.keys()))


# =========================
# Similarity + Predictions
# =========================
def pearson_similarity(u_ratings: Dict[int, float], v_ratings: Dict[int, float],
                       u_mean: float, v_mean: float, min_overlap: int = 2, shrinkage: float = 10.0) -> float:
    """
    Centered Pearson with overlap-based shrinkage.
    Returns 0 if overlap too small or variance ~0.
    """
    common = set(u_ratings.keys()) & set(v_ratings.keys())
    n = len(common)
    if n < min_overlap:
        return 0.0
    num = den_u = den_v = 0.0
    for i in common:
        du = u_ratings[i] - u_mean
        dv = v_ratings[i] - v_mean
        num += du * dv
        den_u += du * du
        den_v += dv * dv
    if den_u <= 1e-12 or den_v <= 1e-12:
        return 0.0
    rho = num / math.sqrt(den_u * den_v)
    return (n / (n + shrinkage)) * rho  # why: stabilize small-overlap noise


def top_k_neighbors(u: int, data: UIData, k: int = 25) -> List[Tuple[int, float]]:
    """
    Top-k positive-similarity neighbors by Pearson.
    """
    sims: List[Tuple[int, float]] = []
    u_r = data.by_user.get(u, {})
    for v in data.all_users:
        if v == u:
            continue
        s = pearson_similarity(u_r, data.by_user[v], data.user_means.get(u, 0.0), data.user_means.get(v, 0.0))
        if s > 0:
            sims.append((v, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def predict_user_scores(u: int, data: UIData, k: int = 25) -> Dict[int, float]:
    """
    Mean-centered kNN deviations; returns predictions for items the user hasn't rated.
    """
    neighbors = top_k_neighbors(u, data, k=k)
    seen = set(data.by_user.get(u, {}).keys())
    candidate_items = data.all_items - seen
    numer: Dict[int, float] = defaultdict(float)
    denom: Dict[int, float] = defaultdict(float)
    for v, sim in neighbors:
        v_mean = data.user_means[v]
        for i, rv in data.by_user[v].items():
            if i in seen:
                continue
            numer[i] += sim * (rv - v_mean)
            denom[i] += abs(sim)
    u_mean = data.user_means.get(u, 0.0)
    return {i: (u_mean + numer[i] / denom[i]) for i in candidate_items if denom[i] > 1e-12}


def top_n_for_user(u: int, data: UIData, items: pd.DataFrame, n: int = 10, k: int = 25) -> pd.DataFrame:
    """
    Return a Top-N DataFrame: item_id, title, score.
    """
    preds = predict_user_scores(u, data, k=k)
    if not preds:
        return pd.DataFrame(columns=["item_id", "title", "score"])
    df = pd.DataFrame([{"item_id": i, "score": s} for i, s in preds.items()])
    return df.merge(items, on="item_id", how="left").sort_values("score", ascending=False).head(n)[
        ["item_id", "title", "score"]
    ]


# =========================
# Group + Sequential
# =========================
def score_items_for_group_dict(group: List[int], data: UIData, k: int = 25) -> Dict[int, Dict[int, float]]:
    """
    item_id → {user_id: predicted_score} for given group (missing preds omitted).
    """
    per_user_preds = {u: predict_user_scores(u, data, k=k) for u in group}
    all_items = set().union(*[set(p.keys()) for p in per_user_preds.values()])
    res: Dict[int, Dict[int, float]] = {}
    for i in all_items:
        res[i] = {u: per_user_preds[u][i] for u in group if i in per_user_preds[u]}
    return res


def rank_group_weighted(scores: Dict[int, Dict[int, float]], items: pd.DataFrame,
                        group: List[int], member_weights: List[float]) -> pd.DataFrame:
    """
    Weighted average over available members only; renormalize weights over present members.
    """
    rows = []
    for i, u2s in scores.items():
        num = den = 0.0
        for w, u in zip(member_weights, group):
            if u in u2s:
                num += w * u2s[u]
                den += w
        if den > 0:
            mu = num / den
            std = float(np.std(list(u2s.values()))) if len(u2s) > 1 else 0.0
            rows.append((i, mu, std))
    df = pd.DataFrame(rows, columns=["item_id", "score", "disagreement"]).merge(items, on="item_id", how="left")
    return df.sort_values(["score", "disagreement"], ascending=[False, True])


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert coverage to weights that favor lower-coverage users.
    """
    z = (x - x.min()) / max(1e-12, (x.max() - x.min()))
    z = -z / max(1e-6, temperature)  # why: invert to upweight under-served members
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def sequential_group_recs(group: List[int], data: UIData, items: pd.DataFrame, rounds: int = 3,
                          k: int = 25, alpha: float = 1.0, topn: int = 10) -> List[pd.DataFrame]:
    """
    Multi-round group recs with adaptive fairness via coverage-based weights.
    """
    coverage = {u: 0.0 for u in group}
    already: Set[int] = set()
    outputs: List[pd.DataFrame] = []
    for _ in range(rounds):
        full_scores = score_items_for_group_dict(group, data, k=k)
        if already:
            full_scores = {i: u2s for i, u2s in full_scores.items() if i not in already}
        cov_vec = np.array([coverage[u] for u in group], dtype=float)
        weights = softmax(alpha * cov_vec)
        df_weighted = rank_group_weighted(full_scores, items, group, member_weights=list(weights))
        top_df = df_weighted.head(topn).reset_index(drop=True)
        outputs.append(top_df)
        for _, row in top_df.iterrows():
            i = int(row["item_id"])
            already.add(i)
            for u in group:
                if u in full_scores[i]:
                    coverage[u] += full_scores[i][u]
    return outputs


# =========================
# IO helpers
# =========================
def ensure_outputs_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)


# =========================
# Interactive UX
# =========================
def prompt_int(prompt: str) -> int | None:
    """
    Read an int from input; return None on empty/q.
    """
    s = input(prompt).strip()
    if s == "" or s.lower() == "q":
        return None
    try:
        return int(s)
    except ValueError:
        print("Please enter a valid integer (or 'q' to quit).")
        return prompt_int(prompt)


def prompt_int_list(prompt: str) -> List[int]:
    """
    Parse comma/space-separated ints; empty → [].
    """
    s = input(prompt).strip()
    if not s:
        return []
    parts = [p for chunk in s.split(",") for p in chunk.split()]
    out: List[int] = []
    for p in parts:
        if p.lower() == "q":
            continue
        try:
            out.append(int(p))
        except ValueError:
            print(f"Ignored non-integer token: {p}")
    return out


def interactive_session(ui: UIData, items: pd.DataFrame, k: int, topn: int, rounds: int, alpha: float):
    """
    Loop: ask for a user id → show personal recs; optionally ask for a group → sequential recs.
    """
    print("\nInteractive mode. Enter user ids from your dataset to check recommendations. Type 'q' to exit.\n")
    while True:
        u = prompt_int("User id to inspect (or 'q' to quit): ")
        if u is None:
            break
        if u not in ui.all_users:
            print("User not found in data. Try another id.")
            continue

        # Individual
        neigh = top_k_neighbors(u, ui, k=k)
        print(f"\nMost similar users to {u} (top {min(10, len(neigh))}):")
        for v, s in neigh[:10]:
            print(f"  user {v:>6}  sim={s:.3f}")
        df_top = top_n_for_user(u, ui, items, n=topn, k=k)
        print(f"\nTop-{topn} for user {u}:\n{df_top.to_string(index=False)}\n")

        # Group (optional)
        group = prompt_int_list("Enter group user ids (space/comma separated) to run group rounds, or press Enter to skip: ")
        group = [g for g in group if g in ui.all_users and g != u]
        if group:
            # why: include the inspected user by default to make a realistic group
            if u not in group:
                group = [u] + group
            group = list(dict.fromkeys(group))  # keep order, drop dups
            print(f"\nSequential group recs for group {group} (rounds={rounds})")
            seq = sequential_group_recs(group, ui, items, rounds=rounds, k=k, alpha=alpha, topn=topn)
            out_dir = ensure_outputs_dir()
            for r, df in enumerate(seq, start=1):
                path = os.path.join(out_dir, f"round{r}_topn_group_{'-'.join(map(str, group))}.csv")
                save_df(df, path)
                print(f"Round {r} top-{topn}:\n{df.head(10).to_string(index=False)}\nSaved → {path}\n")

        cont = input("Check another user? (y/n): ").strip().lower()
        if cont not in ("y", "yes"):
            break
    print("\nDone.")


# =========================
# Entry point
# =========================
def main():
    parser = argparse.ArgumentParser(description="Movie Recommender: interactive user & group recommendations")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with MovieLens files (1M: ratings.dat/movies.dat or 100K: u.data/u.item).")
    parser.add_argument("--k", type=int, default=25, help="k nearest neighbors.")
    parser.add_argument("--topn", type=int, default=10, help="Top-N recommendations to display.")
    parser.add_argument("--rounds", type=int, default=3, help="Sequential rounds for group recommendations.")
    parser.add_argument("--alpha_seq", type=float, default=1.0, help="Fairness strength across rounds.")
    args, _ = parser.parse_known_args()  # why: ignore Jupyter's extra -f arg

    ratings, items, is_synth = load_data(args.data_dir)
    if is_synth:
        print("MovieLens files not found. Using synthetic data so the pipeline runs end-to-end.")
    ui = build_ui(ratings)

    interactive_session(ui, items, k=args.k, topn=args.topn, rounds=args.rounds, alpha=args.alpha_seq)


if __name__ == "__main__":
    main()


# In[ ]:




