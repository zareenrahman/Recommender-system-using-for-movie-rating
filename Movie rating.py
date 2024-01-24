import pandas as pd

# Load data from CSV files
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge ratings and movies based on 'movieId'
merged_data = pd.merge(ratings, movies, on='movieId')

# Create a dictionary to store user ratings for each item
user_ratings_dict = {}
for _, row in merged_data.iterrows():
    user_id, movie_id, rating = row['userId'], row['movieId'], row['rating']
    if user_id not in user_ratings_dict:
        user_ratings_dict[user_id] = {}
    user_ratings_dict[user_id][movie_id] = rating

# Function to get group recommendations using the average method
def get_group_recommendations_average(group_user_ids):
    group_recommendations = {}

    # Calculate average ratings for each movie in the group
    for user_id in group_user_ids:
        user_rated_items = user_ratings_dict.get(user_id, {})
        for item in user_rated_items:
            if item not in group_recommendations:
                group_recommendations[item] = [user_ratings_dict[user_id][item]]
            else:
                group_recommendations[item].append(user_ratings_dict[user_id][item])

    # Calculate the average score for each movie in the aggregated recommendations
    group_average_recommendations = {movie_id: sum(ratings_list) / len(group_user_ids) for movie_id, ratings_list in group_recommendations.items()}

    # Sort movies by average rating in descending order
    sorted_group_recommendations = sorted(group_average_recommendations.items(), key=lambda x: x[1], reverse=True)

    return sorted_group_recommendations

# Function to get group recommendations using the least misery method
def get_group_recommendations_least_misery(group_user_ids):
    group_recommendations = {}

    # Calculate least misery ratings for each movie in the group
    for user_id in group_user_ids:
        user_rated_items = user_ratings_dict.get(user_id, {})
        for item in user_rated_items:
            if item not in group_recommendations:
                group_recommendations[item] = [user_ratings_dict[user_id][item]]
            else:
                group_recommendations[item].append(user_ratings_dict[user_id][item])

    # Calculate the least misery score for each movie in the aggregated recommendations
    group_least_misery_recommendations = {movie_id: min(ratings_list) for movie_id, ratings_list in group_recommendations.items()}

    # Sort movies by least misery rating in descending order
    sorted_group_recommendations = sorted(group_least_misery_recommendations.items(), key=lambda x: x[1], reverse=True)

    return sorted_group_recommendations

# Example: List of user IDs in the group
group_user_ids = [1, 2, 3]

# Get group recommendations using the average method
group_recommendations_average = get_group_recommendations_average(group_user_ids)

# Display the top 10 recommended movies with their names for the average method
top_10_recommendations_average = group_recommendations_average[:10]
print("\nTop 10 Recommendations (Average Method):")
for movie_id, avg_rating in top_10_recommendations_average:
    movie_info = movies[movies['movieId'] == movie_id]
    movie_title = movie_info['title'].values[0] if not movie_info.empty else f"Movie ID {movie_id}"
    print(f"{movie_title}: Average Score {avg_rating:.1f}")

# Get group recommendations using the least misery method
group_recommendations_least_misery = get_group_recommendations_least_misery(group_user_ids)

# Display the top 10 recommended movies with their names for the least misery method
top_10_recommendations_least_misery = group_recommendations_least_misery[:10]
print("\nTop 10 Recommendations (Least Misery Method):")
for movie_id, misery_rating in top_10_recommendations_least_misery:
    movie_info = movies[movies['movieId'] == movie_id]
    movie_title = movie_info['title'].values[0] if not movie_info.empty else f"Movie ID {movie_id}"
    print(f"{movie_title}: Least Misery Rating {misery_rating:.1f}")