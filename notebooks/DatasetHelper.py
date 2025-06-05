import pandas as pd
import numpy as np
import requests
import zipfile
import os
from io import StringIO

def download_movielens_data(dataset_size='small'):
    """
    Download MovieLens dataset
    
    Args:
        dataset_size (str): 'small' (100k ratings) or 'medium' (1M ratings) or 'large' (25M ratings)
    
    Returns:
        tuple: (ratings_df, movies_df)
    """
    
    # Dataset URLs
    urls = {
        'small': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'medium': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'large': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    }
    
    if dataset_size not in urls:
        raise ValueError("dataset_size must be 'small', 'medium', or 'large'")
    
    url = urls[dataset_size]
    zip_filename = f'movielens_{dataset_size}.zip'
    
    print(f"Downloading MovieLens {dataset_size} dataset...")
    
    # Download the dataset
    response = requests.get(url)
    with open(zip_filename, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('movielens_data')
    
    # Clean up zip file
    os.remove(zip_filename)
    
    print("Dataset downloaded and extracted successfully!")
    
    return load_movielens_data(dataset_size)

def load_movielens_data(dataset_size='small'):
    """
    Load MovieLens data from local files
    
    Args:
        dataset_size (str): 'small', 'medium', or 'large'
    
    Returns:
        tuple: (ratings_df, movies_df)
    """
    
    # Determine folder structure based on dataset size
    if dataset_size == 'small':
        folder_name = 'ml-latest-small'
        ratings_file = 'ratings.csv'
        movies_file = 'movies.csv'
        separator = ','
    elif dataset_size == 'medium':
        folder_name = 'ml-1m'
        ratings_file = 'ratings.dat'
        movies_file = 'movies.dat'
        separator = '::'
    elif dataset_size == 'large':
        folder_name = 'ml-25m'
        ratings_file = 'ratings.csv'
        movies_file = 'movies.csv'
        separator = ','
    else:
        raise ValueError("dataset_size must be 'small', 'medium', or 'large'")
    
    base_path = f'movielens_data/{folder_name}'
    
    print(f"Loading MovieLens {dataset_size} dataset from local files...")
    
    # Load ratings data
    if separator == '::':
        # For .dat files (ml-1m)
        ratings_df = pd.read_csv(
            f'{base_path}/{ratings_file}',
            sep=separator,
            names=['userId', 'movieId', 'rating', 'timestamp'],
            engine='python'
        )
        movies_df = pd.read_csv(
            f'{base_path}/{movies_file}',
            sep=separator,
            names=['movieId', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
    else:
        # For .csv files
        ratings_df = pd.read_csv(f'{base_path}/{ratings_file}')
        movies_df = pd.read_csv(f'{base_path}/{movies_file}')
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    print(f"Users: {ratings_df['userId'].nunique()}")
    print(f"Movies: {ratings_df['movieId'].nunique()}")
    print(f"Rating range: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    
    return ratings_df, movies_df

def load_movielens_from_files(ratings_path, movies_path):
    """
    Load MovieLens data from custom file paths
    
    Args:
        ratings_path (str): Path to ratings file
        movies_path (str): Path to movies file
    
    Returns:
        tuple: (ratings_df, movies_df)
    """
    
    print("Loading MovieLens data from custom files...")
    
    # Try to detect file format
    if ratings_path.endswith('.dat'):
        # Assume it's ml-1m format
        ratings_df = pd.read_csv(
            ratings_path,
            sep='::',
            names=['userId', 'movieId', 'rating', 'timestamp'],
            engine='python'
        )
        movies_df = pd.read_csv(
            movies_path,
            sep='::',
            names=['movieId', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
    else:
        # Assume CSV format
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    return ratings_df, movies_df

def sample_data_for_testing(ratings_df, movies_df, n_users=1000, n_movies=500, n_ratings=5000):
    """
    Sample a subset of the data for faster testing
    
    Args:
        ratings_df: Full ratings dataframe
        movies_df: Full movies dataframe
        n_users: Number of users to keep
        n_movies: Number of movies to keep
        n_ratings: Max number of ratings to keep
    
    Returns:
        tuple: (sampled_ratings_df, sampled_movies_df)
    """
    
    print(f"Sampling data: {n_users} users, {n_movies} movies, max {n_ratings} ratings")
    
    # Get top users and movies by number of ratings
    top_users = ratings_df['userId'].value_counts().head(n_users).index
    top_movies = ratings_df['movieId'].value_counts().head(n_movies).index
    
    # Filter ratings
    sampled_ratings = ratings_df[
        (ratings_df['userId'].isin(top_users)) & 
        (ratings_df['movieId'].isin(top_movies))
    ].head(n_ratings)
    
    # Filter movies
    used_movies = sampled_ratings['movieId'].unique()
    sampled_movies = movies_df[movies_df['movieId'].isin(used_movies)]
    
    print(f"Sampled data: {len(sampled_ratings)} ratings, {len(sampled_movies)} movies")
    return sampled_ratings, sampled_movies

# Example usage functions
def get_movielens_data(size='small', sample=False):
    """
    Convenience function to get MovieLens data
    
    Args:
        size (str): 'small', 'medium', or 'large'
        sample (bool): Whether to sample the data for faster processing
    
    Returns:
        tuple: (ratings_df, movies_df)
    """
    
    try:
        # Try to load from existing files first
        ratings_df, movies_df = load_movielens_data(size)
    except FileNotFoundError:
        # Download if not found
        ratings_df, movies_df = download_movielens_data(size)
    
    if sample:
        ratings_df, movies_df = sample_data_for_testing(ratings_df, movies_df)
    
    return ratings_df, movies_df

# # Alternative: Load from uploaded files
# def load_from_uploaded_files():
#     """
#     Load MovieLens data from files uploaded to the conversation
#     This function assumes you've uploaded ratings.csv and movies.csv files
#     """
#     try:
#         # Read uploaded files
#         ratings_data = window.fs.readFile('ratings.csv', {'encoding': 'utf8'})
#         movies_data = window.fs.readFile('movies.csv', {'encoding': 'utf8'})
        
#         # Parse CSV data
#         ratings_df = pd.read_csv(StringIO(ratings_data))
#         movies_df = pd.read_csv(StringIO(movies_data))
        
#         print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies from uploaded files")
#         return ratings_df, movies_df
    
#     except Exception as e:
#         print(f"Error loading uploaded files: {e}")
#         print("Please make sure you've uploaded 'ratings.csv' and 'movies.csv' files")
#         return None, None

# # Demo usage with real data
# if __name__ == "__main__":
#     # Option 1: Download and use MovieLens small dataset
#     print("=== Option 1: Download MovieLens dataset ===")
#     ratings_df, movies_df = get_movielens_data(size='small', sample=False)
    
#     # Display basic statistics
#     print(f"\nDataset Statistics:")
#     print(f"Number of ratings: {len(ratings_df):,}")
#     print(f"Number of movies: {len(movies_df):,}")
#     print(f"Number of users: {ratings_df['userId'].nunique():,}")
#     print(f"Average rating: {ratings_df['rating'].mean():.2f}")
#     print(f"Rating distribution:")
#     print(ratings_df['rating'].value_counts().sort_index())
    
#     # Show sample data
#     print(f"\nSample ratings:")
#     print(ratings_df.head())
#     print(f"\nSample movies:")
#     print(movies_df.head())
    
    # # Option 2: Use sampled version for faster processing
    # print("\n=== Option 2: Sampled version ===")
    # sample_ratings, sample_movies = get_movielens_data(size='small', sample=True)
    # print(f"Sampled: {len(sample_ratings)} ratings, {len(sample_movies)} movies")