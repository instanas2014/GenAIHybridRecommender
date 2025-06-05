from sklearn.preprocessing import LabelEncoder
from OpenAIClientHelper import OpenAIClient
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import openai
from pathlib import Path
import pickle
import json
import os

env_api_key = os.getenv("OPEN_AI_KEY") 
#please run EXPORT OPEN_AI_KEY="<your key>" in bash or SET OPEN_AI_KEY="<your key>" in windows cmd
#if you are using Conda env , do conda env config vars set OPEN_AI_KEY="<your key>"

if env_api_key is None:
    raise ValueError("OPEN_API_KEY environment variable not set.")

# Use your api_key here
print(f"Using API Key: {env_api_key[:5]}...") # Print only first 5 chars for security


# Deep Learning Collaborative Filtering Model
class DeepCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64]):
        super(DeepCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural MF layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        output = self.mlp(concat_emb)
        
        return output.squeeze()

class MovieDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]



class HybridPersonalizationSystem:
    def __init__(self, model_path=None):
        self.api_key = env_api_key
        self.deep_model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.openai_client = OpenAIClient(api_key =self.api_key, use_mock=False) # MockOpenAIResponse #OpenAIClient(api_key="your-api-key-here", use_mock=False) for actual API response
        self.movie_features = {}
        self.user_profiles = {}
        self.model_metadata = {}
        
    def prepare_data(self, ratings_df, movies_df):
        """Prepare and encode data for training"""
        print("Preparing data...")
        
        # Encode users and items
        ratings_df['user_id_encoded'] = self.user_encoder.fit_transform(ratings_df['userId'])
        ratings_df['item_id_encoded'] = self.item_encoder.fit_transform(ratings_df['movieId'])
        
        # Normalize ratings to 0-1 scale
        ratings_df['rating_normalized'] = ratings_df['rating'] / 5.0
        
        # Store movie features for LLM context
        for _, movie in movies_df.iterrows():
            self.movie_features[movie['movieId']] = {
                'title': movie['title'],
                'genres': movie['genres'].split('|') if 'genres' in movie else [],
                'rating': ratings_df[ratings_df.movieId==movie['movieId']].groupby('movieId')['rating'].mean().values[0] #get only the value and ignore the movieId index so that is serializable
            }
        
        # Store metadata for saving/loading
        self.model_metadata = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
            'embedding_dim': 50,
            'hidden_dims': [128, 64]
        }

        
        return ratings_df
    
    def train_deep_model(self, ratings_df, epochs=50, lr=0.001):
        """Train the deep learning collaborative filtering model"""
        print("Training deep learning model...")
        
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.item_encoder.classes_)
        
        # Initialize model
        self.deep_model = DeepCF(num_users, num_items)
        optimizer = optim.Adam(self.deep_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare training data
        train_data, val_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        train_dataset = MovieDataset(
            train_data['user_id_encoded'].values,
            train_data['item_id_encoded'].values,
            train_data['rating_normalized'].values
        )
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        
        # Training loop
        self.deep_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for user_ids, item_ids, ratings in train_loader:
                optimizer.zero_grad()
                predictions = self.deep_model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def get_deep_predictions(self, user_id, candidate_items, top_k=10):
        """Get top-k predictions from deep learning model"""
        if user_id not in self.user_encoder.classes_:
            # Handle cold start - return random sample
            return np.random.choice(candidate_items, min(top_k, len(candidate_items)), replace=False)
        
        user_encoded = self.user_encoder.transform([user_id])[0]
        
        # Get predictions for all candidate items
        predictions = []
        self.deep_model.eval()
        
        with torch.no_grad():
            for item_id in candidate_items:
                if item_id in self.item_encoder.classes_:
                    item_encoded = self.item_encoder.transform([item_id])[0]
                    user_tensor = torch.LongTensor([user_encoded])
                    item_tensor = torch.LongTensor([item_encoded])
                    pred = self.deep_model(user_tensor, item_tensor).item()
                    predictions.append((item_id, pred))
        
        # Sort by prediction score and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:top_k]]
    
    def build_user_context(self, user_id, recent_ratings=None, current_time=None, session_context=None):
        """Build contextual information for LLM"""
        context = {
            'user_id': user_id,
            'recent_activity': recent_ratings or [],
            'time_context': current_time or 'evening',
            'session_info': session_context or {}
        }
        return context
    
    def llm_adjust_recommendations(self, user_context, deep_recommendations, adjustment_factors=None):
        """Use LLM to adjust deep learning recommendations"""
        
        # Build movie context for LLM
        movie_context = []
        for movie_id in deep_recommendations:
            if movie_id in self.movie_features:
                movie_info = self.movie_features[movie_id]
                movie_context.append({
                    'id': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres']
                })
        
        # Create LLM prompt
        prompt = f"""
        You are a movie recommendation system. Given the following context, adjust the recommendation scores:
        
        User Context:
        - Recent activity: {user_context.get('recent_activity', [])}
        - Current time: {user_context.get('time_context', 'unknown')}
        - Session info: {user_context.get('session_info', {})}
        
        Deep Learning Recommendations (in order):
        {json.dumps(movie_context, indent=2)}
        
        Consider factors like:
        1. Genre preferences from recent activity
        2. Time of day appropriateness
        3. Mood and context
        4. Diversity in recommendations
        
        Provide adjusted scores (0-1) for each movie and brief reasoning.
        Return JSON format: {{"adjusted_scores": [score1, score2, ...], "reasoning": "explanation"}}
        """
        
        try:
            # Get LLM response (using mock client here)
            response = self.openai_client.chat_completions_create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # Parse LLM response
            llm_output = json.loads(response.choices[0].message.content)
            adjusted_scores = llm_output.get('adjusted_scores', [1.0] * len(deep_recommendations))
            reasoning = llm_output.get('reasoning', 'No specific adjustments made')
            
            # Combine with original recommendations
            final_recommendations = []
            for i, (movie_id, score) in enumerate(zip(deep_recommendations, adjusted_scores)):
                final_recommendations.append({
                    'movie_id': movie_id,
                    'title': self.movie_features.get(movie_id, {}).get('title', 'Unknown'),
                    'genres': self.movie_features.get(movie_id, {}).get('genres', []),
                    'rating': self.movie_features.get(movie_id, {}).get('rating', ''),
                    'deep_score': 0.8 - (i * 0.1),  # Simulated original scores
                    'adjusted_score': score,
                    'final_score': (0.8 - (i * 0.1)) * 0.4 + score * 0.6 # Weighted combination
                })
            
            # Re-sort by final score
            final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            
            return final_recommendations, reasoning
            
        except Exception as e:
            print(f"LLM adjustment failed: {e}")
            # Fallback to original recommendations
            fallback_recs = []
            for i, movie_id in enumerate(deep_recommendations):
                fallback_recs.append({
                    'movie_id': movie_id,
                    'title': self.movie_features.get(movie_id, {}).get('title', 'Unknown'),
                    'genres': self.movie_features.get(movie_id, {}).get('genres', []),
                    'rating': self.movie_features.get(movie_id, {}).get('rating', []),
                    'deep_score': 0.8 - (i * 0.1),
                    'adjusted_score': 0.8 - (i * 0.1),
                    'final_score': 0.8 - (i * 0.1)
                })
            return fallback_recs, "Using original deep learning recommendations"
    
    def get_personalized_recommendations(self, user_id, context=None, top_k=5):
        """Main method to get personalized recommendations"""
        print(f"\nGenerating recommendations for user {user_id}...")
        
        # Step 1: Get candidate items (in practice, this might be filtered by business rules)
        candidate_items = list(self.movie_features.keys())[:100]  # Sample for demo
        
        # Step 2: Get deep learning predictions
        deep_recs = self.get_deep_predictions(user_id, candidate_items, top_k=top_k*2)
        print(f"Deep learning top picks: {deep_recs[:top_k]}")
        
        # Step 3: Build user context
        user_context = context or self.build_user_context(user_id)
        
        # Step 4: LLM adjustment
        final_recs, reasoning = self.llm_adjust_recommendations(user_context, deep_recs[:top_k])
        
        print(f"LLM Reasoning: {reasoning}")
        print("\nFinal Recommendations:")
        for i, rec in enumerate(final_recs[:top_k], 1):
            print(f"{i}. {rec['title']} (Genres: {', '.join(rec['genres'])[:50]})")
            print(f" Rating: {rec['rating']:.3f}")
            print(f"   Deep Score: {rec['deep_score']:.3f}, Adjusted: {rec['adjusted_score']:.3f}, Final: {rec['final_score']:.3f}")
        
        return final_recs[:top_k]
        
    # def _get_match_reason(self, movie_data, recent_activity, time_context, device):
    #     reasons = []
    #     if any(genre in recent_activity for genre in movie_data['genre']):
    #         reasons.append(f"Matches your interest in {', '.join(recent_activity)}")
    #     if time_context == 'evening' and movie_data['rating'] > 8.5:
    #         reasons.append("Perfect for evening viewing")
    #     if device == 'mobile' and movie_data['duration'] < 150:
    #         reasons.append("Great for mobile viewing")
    #     if movie_data['rating'] > 8.5:
    #         reasons.append("Highly rated")
    #     return '; '.join(reasons) if reasons else "Popular choice"

    def save_model(self, save_dir="./saved_model"):
        """Save the complete model and all necessary components"""
        print(f"Saving model to {save_dir}...")
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Save PyTorch model state dict
        if self.deep_model is not None:
            torch.save(self.deep_model.state_dict(), f"{save_dir}/deep_model.pth")
            print("✓ Deep learning model saved")
        
        # 2. Save encoders
        with open(f"{save_dir}/user_encoder.pkl", 'wb') as f:
            pickle.dump(self.user_encoder, f)
        with open(f"{save_dir}/item_encoder.pkl", 'wb') as f:
            pickle.dump(self.item_encoder, f)
        print("✓ Encoders saved")
        
        #print(self.movie_features)
        # 3. Save movie features
        with open(f"{save_dir}/movie_features.json", 'w') as f:
            json.dump(self.movie_features, f, indent=2)
        print("✓ Movie features saved")
        
        # 4. Save model metadata
        with open(f"{save_dir}/model_metadata.json", 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        print("✓ Model metadata saved")
        
        # 5. Save user profiles (if any)
        with open(f"{save_dir}/user_profiles.json", 'w') as f:
            json.dump(self.user_profiles, f, indent=2)
        print("✓ User profiles saved")
        
        print(f"Model successfully saved to {save_dir}")
        
        # Create a manifest file
        manifest = {
            "model_type": "HybridPersonalizationSystem",
            "files": [
                "deep_model.pth",
                "user_encoder.pkl", 
                "item_encoder.pkl",
                "movie_features.json",
                "model_metadata.json",
                "user_profiles.json"
            ],
            "metadata": self.model_metadata
        }
        
        with open(f"{save_dir}/manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        print("✓ Manifest created")
    
    @classmethod
    def load_model(cls, save_dir="./saved_model"):
        """Load the complete model and all necessary components"""
        print(f"Loading model from {save_dir}...")
        
        # Check if all required files exist
        required_files = [
            "deep_model.pth",
            "user_encoder.pkl", 
            "item_encoder.pkl",
            "movie_features.json",
            "model_metadata.json"
        ]
        
        for file in required_files:
            if not os.path.exists(f"{save_dir}/{file}"):
                raise FileNotFoundError(f"Required file {file} not found in {save_dir}")
        
        # Initialize new instance
        system = cls()
        
        # 1. Load model metadata
        with open(f"{save_dir}/model_metadata.json", 'r') as f:
            system.model_metadata = json.load(f)
        print("✓ Model metadata loaded")
        
        # 2. Load encoders
        with open(f"{save_dir}/user_encoder.pkl", 'rb') as f:
            system.user_encoder = pickle.load(f)
        with open(f"{save_dir}/item_encoder.pkl", 'rb') as f:
            system.item_encoder = pickle.load(f)
        print("✓ Encoders loaded")
        
        # 3. Load movie features
        with open(f"{save_dir}/movie_features.json", 'r') as f:
            # Convert string keys back to integers for movie IDs
            movie_features_str = json.load(f)
            system.movie_features = {int(k): v for k, v in movie_features_str.items()}
        print("✓ Movie features loaded")
        
        # 4. Load user profiles
        if os.path.exists(f"{save_dir}/user_profiles.json"):
            with open(f"{save_dir}/user_profiles.json", 'r') as f:
                system.user_profiles = json.load(f)
        print("✓ User profiles loaded")
        
        # 5. Load and initialize PyTorch model
        system.deep_model = DeepCF(
            num_users=system.model_metadata['num_users'],
            num_items=system.model_metadata['num_items'],
            embedding_dim=system.model_metadata['embedding_dim'],
            hidden_dims=system.model_metadata['hidden_dims']
        )
        system.deep_model.load_state_dict(torch.load(f"{save_dir}/deep_model.pth"))
        system.deep_model.eval()  # Set to evaluation mode
        print("✓ Deep learning model loaded and initialized")
        
        print(f"Model successfully loaded from {save_dir}")
        return system
        
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'num_users': self.model_metadata.get('num_users', 0),
            'num_items': self.model_metadata.get('num_items', 0),
            'num_movies': len(self.movie_features),
            'embedding_dim': self.model_metadata.get('embedding_dim', 0),
            'hidden_dims': self.model_metadata.get('hidden_dims', [])
        }

