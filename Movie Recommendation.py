import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

# Load datasets (Ensure 'ratings.csv' and 'movies.csv' are in the same directory)
try:
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: Required dataset files (ratings.csv, movies.csv) not found.")
    exit()

# Prepare data mappings
user_ratings = {user: dict(zip(group['movieId'], group['rating'])) for user, group in ratings_df.groupby('userId')}
movie_genres = {row['movieId']: row['genres'].split('|') for _, row in movies_df.iterrows()}
movie_titles = movies_df.set_index('movieId')['title'].to_dict()

# Similarity Functions
def pearson_similarity(user1, user2):
    common_movies = set(user1.keys()).intersection(user2.keys())
    if len(common_movies) == 0:
        return 0
    user1_ratings = [user1[movie] for movie in common_movies]
    user2_ratings = [user2[movie] for movie in common_movies]
    numerator = sum((user1[movie] - np.mean(user1_ratings)) * (user2[movie] - np.mean(user2_ratings)) for movie in common_movies)
    denominator = np.sqrt(sum((user1[movie] - np.mean(user1_ratings)) ** 2 for movie in common_movies)) * \
                  np.sqrt(sum((user2[movie] - np.mean(user2_ratings)) ** 2 for movie in common_movies))
    return numerator / denominator if denominator != 0 else 0

def jaccard_similarity(user1_ratings, user2_ratings):
    """Calculate the Jaccard similarity based on genres rated by two users."""
    user1_genres = set(
        g for movie in user1_ratings.keys() for g in movie_genres.get(movie, [])
    )
    user2_genres = set(
        g for movie in user2_ratings.keys() for g in movie_genres.get(movie, [])
    )
    intersection = len(user1_genres.intersection(user2_genres))
    union = len(user1_genres.union(user2_genres))
    return intersection / union if union != 0 else 0

# Helper Functions
def get_similar_users(user_id, k=5, similarity_func=pearson_similarity):
    """Find the top K similar users based on the similarity function."""
    similarities = []
    for other_user_id, other_user_ratings in user_ratings.items():
        if other_user_id != user_id:
            sim_score = similarity_func(user_ratings[user_id], other_user_ratings)
            if not np.isnan(sim_score):
                similarities.append((other_user_id, sim_score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

# Recommendation Functions
def compute_recommendations(user_id, k=5):
    """Compute recommendations for a single user."""
    similar_users = get_similar_users(user_id, k)
    weighted_ratings = defaultdict(float)
    similarity_sum = defaultdict(float)
    user_mean_rating = np.mean(list(user_ratings[user_id].values()))

    for other_user, similarity in similar_users:
        other_user_mean = np.mean(list(user_ratings[other_user].values()))
        for movie_id, rating in user_ratings[other_user].items():
            if movie_id not in user_ratings[user_id]:
                weighted_ratings[movie_id] += similarity * (rating - other_user_mean)
                similarity_sum[movie_id] += abs(similarity)

    recommendations = {
        movie_id: user_mean_rating + (weighted_sum / similarity_sum[movie_id])
        for movie_id, weighted_sum in weighted_ratings.items() if similarity_sum[movie_id] > 0
    }
    return recommendations

def group_recommendations_avg(user_ids, k=10):
    """Generate group recommendations using the average method."""
    all_recommendations = defaultdict(list)
    for user_id in user_ids:
        user_recommendations = compute_recommendations(user_id, k)
        for movie_id, score in user_recommendations.items():
            all_recommendations[movie_id].append(score)
    avg_recommendations = {movie: np.mean(scores) for movie, scores in all_recommendations.items()}
    return sorted(avg_recommendations.items(), key=lambda x: x[1], reverse=True)[:k]

# Menu-Driven Interface
while True:
    print("\n--- Movie Recommendation System ---")
    print("1. Compute Pearson Similarity between Users")
    print("2. Compute Jaccard Similarity between Users")
    print("3. Get Group Recommendations (Average Method)")
    print("4. Exit")

    choice = input("Enter your choice: ").strip()
    if choice == "1":
        user1_id = int(input("Enter the first user ID: "))
        user2_id = int(input("Enter the second user ID: "))
        if user1_id in user_ratings and user2_id in user_ratings:
            score = pearson_similarity(user_ratings[user1_id], user_ratings[user2_id])
            print(f"Pearson Similarity between User {user1_id} and User {user2_id}: {score:.4f}")
        else:
            print("One or both users not found in the dataset.")

    elif choice == "2":
        try:
            user1_id = int(input("Enter the first user ID: ").strip())
            user2_id = int(input("Enter the second user ID: ").strip())
            if user1_id in user_ratings and user2_id in user_ratings:
                score = jaccard_similarity(user_ratings[user1_id],
                                           user_ratings[user2_id])
                print(
                    f"Jaccard Similarity between User {user1_id} and User {user2_id}: {score:.4f}")
            else:
                print("One or both users not found in the dataset.")
        except ValueError:
            print("Invalid input. Please enter numeric user IDs.")

    elif choice == "3":
        user_ids = list(map(int, input("Enter user IDs (comma-separated): ").split(',')))
        recommendations = group_recommendations_avg(user_ids)
        print("\nGroup Recommendations (Average Method):")
        for movie_id, score in recommendations:
            print(f"{movie_titles.get(movie_id, 'Unknown')} (Score: {score:.2f})")
    elif choice == "4":
        print("Exiting the system.")
        break
    else:
        print("Invalid choice. Please try again.")
