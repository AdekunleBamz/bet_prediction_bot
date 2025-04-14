import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_initial_training_data():
    # Create synthetic training data based on extensive historical patterns
    # Football data - 200 matches with realistic patterns
    np.random.seed(42)  # For reproducibility
    
    def generate_match_data(n_matches, favorite_odds_range, underdog_odds_range, win_prob):
        home_odds = np.random.uniform(*favorite_odds_range, n_matches)
        away_odds = np.random.uniform(*underdog_odds_range, n_matches)
        implied_prob_home = 1 / home_odds
        implied_prob_away = 1 / away_odds
        odds_ratio = home_odds / away_odds
        
        # Generate results with some upsets based on win_prob
        results = np.random.choice([1, 0], n_matches, p=[win_prob, 1-win_prob])
        
        return np.column_stack([home_odds, away_odds, implied_prob_home, implied_prob_away, odds_ratio]), results

    # Generate different match scenarios
    # Strong home favorites
    f1, r1 = generate_match_data(40, (1.2, 1.5), (5.0, 8.0), 0.75)
    
    # Moderate home favorites
    f2, r2 = generate_match_data(40, (1.5, 2.0), (3.5, 5.0), 0.60)
    
    # Slight home favorites
    f3, r3 = generate_match_data(40, (2.0, 2.5), (2.5, 3.5), 0.55)
    
    # Even matches
    f4, r4 = generate_match_data(40, (2.4, 2.6), (2.4, 2.6), 0.50)
    
    # Slight away favorites
    f5, r5 = generate_match_data(40, (2.5, 3.5), (2.0, 2.5), 0.45)
    
    # Strong away favorites
    f6, r6 = generate_match_data(40, (5.0, 8.0), (1.2, 1.5), 0.25)

    # Combine all scenarios
    football_features = np.vstack([f1, f2, f3, f4, f5, f6])
    football_labels = np.concatenate([r1, r2, r3, r4, r5, r6])

    # Add some noise and variations
    noise = np.random.normal(0, 0.05, football_features.shape)
    football_features = np.abs(football_features + noise)  # Keep odds positive

    football_data = {
        'features': football_features,
        'labels': football_labels
    }

    # Basketball data - Similar approach but with different odds ranges
    # Strong home favorites
    b1, br1 = generate_match_data(40, (1.1, 1.3), (3.5, 5.0), 0.80)
    
    # Moderate home favorites
    b2, br2 = generate_match_data(40, (1.3, 1.6), (2.5, 3.5), 0.65)
    
    # Slight home favorites
    b3, br3 = generate_match_data(40, (1.6, 1.9), (2.0, 2.5), 0.55)
    
    # Even matches
    b4, br4 = generate_match_data(40, (1.9, 2.1), (1.9, 2.1), 0.50)
    
    # Slight away favorites
    b5, br5 = generate_match_data(40, (2.0, 2.5), (1.6, 1.9), 0.45)
    
    # Strong away favorites
    b6, br6 = generate_match_data(40, (3.5, 5.0), (1.1, 1.3), 0.20)

    # Combine all scenarios
    basketball_features = np.vstack([b1, b2, b3, b4, b5, b6])
    basketball_labels = np.concatenate([br1, br2, br3, br4, br5, br6])

    # Add some noise and variations
    noise = np.random.normal(0, 0.03, basketball_features.shape)
    basketball_features = np.abs(basketball_features + noise)

    basketball_data = {
        'features': basketball_features,
        'labels': basketball_labels
    }

    return football_data, basketball_data

def initialize_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Get initial training data
    football_data, basketball_data = create_initial_training_data()

    # Scale the features
    scaler = StandardScaler()
    football_features_scaled = scaler.fit_transform(football_data['features'])
    basketball_features_scaled = scaler.fit_transform(basketball_data['features'])

    # Initialize and train football model with optimized parameters
    football_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    football_model.fit(football_features_scaled, football_data['labels'])
    
    # Save both model and scaler
    joblib.dump(football_model, 'models/football_model.joblib')
    joblib.dump(scaler, 'models/football_scaler.joblib')

    # Initialize and train basketball model with optimized parameters
    basketball_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    basketball_model.fit(basketball_features_scaled, basketball_data['labels'])
    
    # Save both model and scaler
    joblib.dump(basketball_model, 'models/basketball_model.joblib')
    joblib.dump(scaler, 'models/basketball_scaler.joblib')

    print("Models initialized and trained successfully!")
    
    # Print some model statistics
    print("\nFootball Model Statistics:")
    print(f"Total matches in training: {len(football_data['labels'])}")
    print(f"Home wins: {sum(football_data['labels'])} ({sum(football_data['labels'])/len(football_data['labels'])*100:.1f}%)")
    print(f"Away wins: {len(football_data['labels'])-sum(football_data['labels'])} ({(1-sum(football_data['labels'])/len(football_data['labels']))*100:.1f}%)")
    
    print("\nBasketball Model Statistics:")
    print(f"Total matches in training: {len(basketball_data['labels'])}")
    print(f"Home wins: {sum(basketball_data['labels'])} ({sum(basketball_data['labels'])/len(basketball_data['labels'])*100:.1f}%)")
    print(f"Away wins: {len(basketball_data['labels'])-sum(basketball_data['labels'])} ({(1-sum(basketball_data['labels'])/len(basketball_data['labels']))*100:.1f}%)")

if __name__ == '__main__':
    initialize_models() 