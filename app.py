import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class MovieRecommenderML:
    def __init__(self):
        """
        Initialize the movie recommender with sample data and ML setup.
        """
        # Sample movie data
        self.movies_data = pd.DataFrame([
            {"movieId": 1, "title": "Inception", "genres": "Action|Sci-Fi"},
            {"movieId": 2, "title": "The Matrix", "genres": "Sci-Fi|Action"},
            {"movieId": 3, "title": "Interstellar", "genres": "Sci-Fi|Drama"},
            {"movieId": 4, "title": "Dark Knight", "genres": "Action|Drama"},
            {"movieId": 5, "title": "Pulp Fiction", "genres": "Crime|Drama"}
        ])
        
        # Ratings storage
        self.ratings_data = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    
    def collect_initial_ratings(self, user_id):
        """
        Collect initial movie ratings from the user.
        
        Args:
            user_id (int): ID of the user rating movies
        """
        print("\nPlease rate the following movies (1-5):")
        user_movie_ratings = []
        
        for _, movie in self.movies_data.iterrows():
            while True:
                try:
                    rating = int(input(f"Rate '{movie['title']}' (1-5, or 0 if you haven't seen it): "))
                    if 0 <= rating <= 5:
                        if rating > 0:
                            user_movie_ratings.append({
                                'userId': user_id,
                                'movieId': movie['movieId'],
                                'rating': rating
                            })
                        break
                    else:
                        print("Please enter a rating between 0 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # Add user's ratings to ratings dataframe
        user_ratings_df = pd.DataFrame(user_movie_ratings)
        self.ratings_data = pd.concat([self.ratings_data, user_ratings_df], ignore_index=True)
        
        return user_ratings_df
    
    def prepare_rating_matrix(self):
        """
        Create a user-item rating matrix.
        """
        # Pivot table of ratings
        self.rating_matrix = self.ratings_data.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        # Normalize ratings if possible
        if not self.rating_matrix.empty:
            self.scaler = StandardScaler()
            self.normalized_matrix = self.scaler.fit_transform(self.rating_matrix)
        else:
            self.normalized_matrix = np.array([])
    
    def find_similar_users(self, user_id, top_n=3):
        """
        Find most similar users based on rating patterns.
        """
        # If not enough ratings, return all users
        if len(self.ratings_data) < 5:
            return []
        
        # Prepare rating matrix
        self.prepare_rating_matrix()
        
        # If matrix is empty or user not in matrix, return empty list
        if self.normalized_matrix.size == 0 or user_id not in self.rating_matrix.index:
            return []
        
        # Get user's index in matrix
        user_index = self.rating_matrix.index.get_loc(user_id)
        
        # Compute cosine similarity
        user_vector = self.normalized_matrix[user_index].reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.normalized_matrix)[0]
        
        # Sort and get top similar users (exclude self)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        return [self.rating_matrix.index[idx] for idx in similar_indices]
    
    def recommend_movies(self, user_id, n_recommendations=3):
        """
        Generate movie recommendations for a user.
        """
        # Prepare rating matrix
        self.prepare_rating_matrix()
        
        # Get user's existing ratings
        user_rated_movies = set(
            self.ratings_data[self.ratings_data['userId'] == user_id]['movieId']
        )
        
        # If user has no ratings, cannot recommend
        if len(user_rated_movies) == 0:
            return pd.DataFrame(columns=['movieId', 'title', 'genres'])
        
        # Find similar users
        similar_users = self.find_similar_users(user_id)
        
        # Collect candidate movies
        candidate_movies = []
        for similar_user in similar_users:
            # Get similar user's highly rated movies not rated by target user
            similar_user_ratings = self.ratings_data[
                (self.ratings_data['userId'] == similar_user) & 
                (self.ratings_data['rating'] >= 4) & 
                (~self.ratings_data['movieId'].isin(user_rated_movies))
            ]
            candidate_movies.extend(similar_user_ratings['movieId'].tolist())
        
        # If no candidates, recommend unseen movies
        if not candidate_movies:
            unseen_movies = set(self.movies_data['movieId']) - user_rated_movies
            candidate_movies = list(unseen_movies)
        
        # If still no candidates, return empty DataFrame
        if not candidate_movies:
            return pd.DataFrame(columns=['movieId', 'title', 'genres'])
        
        # Count and rank candidate movies
        movie_scores = pd.Series(candidate_movies).value_counts()
        top_movie_ids = movie_scores.head(n_recommendations).index.tolist()
        
        # Return recommended movies
        return self.movies_data[self.movies_data['movieId'].isin(top_movie_ids)]

def main():
    # Initialize recommender
    recommender = MovieRecommenderML()
    
    while True:
        try:
            # Get user input
            user_id = int(input("Enter your user ID (1-5): "))
            
            # Collect initial ratings
            initial_ratings = recommender.collect_initial_ratings(user_id)
            
            # Get recommendations
            recommendations = recommender.recommend_movies(user_id)
            
            # If no recommendations, inform user
            if recommendations.empty:
                print("\nNot enough data to generate recommendations. Please rate more movies.")
            else:
                print("\nRecommended Movies:")
                print(recommendations[['title', 'genres']])
            
            # Option to rate more movies
            rate_movie = input("\nWould you like to rate another movie? (yes/no): ").lower()
            if rate_movie == 'yes':
                movie_id = int(input("Enter movie ID to rate: "))
                rating = float(input("Enter your rating (1-5): "))
                # Add the new rating
                new_rating = pd.DataFrame({
                    'userId': [user_id],
                    'movieId': [movie_id],
                    'rating': [rating]
                })
                recommender.ratings_data = pd.concat([recommender.ratings_data, new_rating], ignore_index=True)
            
            # Continue or exit
            continue_choice = input("\nContinue? (yes/no): ").lower()
            if continue_choice != 'yes':
                break
        
        except Exception as e:
            print(f"An error occurred: {e}")
    
    print("Thank you for using the movie recommender!")

if __name__ == "__main__":
    main()