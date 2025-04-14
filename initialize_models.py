import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_football_training_data(n_samples=1000):
    """Generate synthetic training data for football predictions"""
    np.random.seed(42)
    
    # Base features for all markets
    home_odds = np.random.uniform(1.2, 5.0, n_samples)
    away_odds = np.random.uniform(1.2, 5.0, n_samples)
    
    # Additional features
    home_goals_avg = np.random.uniform(0.5, 3.0, n_samples)
    away_goals_avg = np.random.uniform(0.5, 3.0, n_samples)
    home_corners_avg = np.random.uniform(3, 8, n_samples)
    away_corners_avg = np.random.uniform(3, 8, n_samples)
    first_half_goals_avg = np.random.uniform(0.5, 1.5, n_samples)
    
    # Generate results for different markets
    data = {
        'match_result': {
            'features': np.column_stack([
                home_odds, away_odds,
                1/home_odds, 1/away_odds,
                home_goals_avg, away_goals_avg
            ]),
            'labels': (np.random.random(n_samples) < 1/home_odds).astype(int)
        },
        'btts': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_goals_avg, away_goals_avg,
                home_goals_avg * away_goals_avg
            ]),
            'labels': ((home_goals_avg > 0.8) & (away_goals_avg > 0.8)).astype(int)
        },
        'over_under_2_5': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_goals_avg, away_goals_avg,
                home_goals_avg + away_goals_avg
            ]),
            'labels': ((home_goals_avg + away_goals_avg) > 2.5).astype(int)
        },
        'first_half_result': {
            'features': np.column_stack([
                home_odds, away_odds,
                first_half_goals_avg,
                home_goals_avg/2, away_goals_avg/2
            ]),
            'labels': (np.random.random(n_samples) < 1/home_odds).astype(int)
        },
        'first_team_score': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_goals_avg, away_goals_avg
            ]),
            'labels': (home_goals_avg > away_goals_avg).astype(int)
        },
        'total_corners_over_9_5': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_corners_avg, away_corners_avg
            ]),
            'labels': ((home_corners_avg + away_corners_avg) > 9.5).astype(int)
        }
    }
    
    return data

def generate_basketball_training_data(n_samples=1000):
    """Generate synthetic training data for basketball predictions"""
    np.random.seed(43)
    
    # Base features
    home_odds = np.random.uniform(1.2, 4.0, n_samples)
    away_odds = np.random.uniform(1.2, 4.0, n_samples)
    
    # Additional features
    home_points_avg = np.random.uniform(85, 115, n_samples)
    away_points_avg = np.random.uniform(85, 115, n_samples)
    home_first_quarter_avg = np.random.uniform(20, 30, n_samples)
    away_first_quarter_avg = np.random.uniform(20, 30, n_samples)
    
    # Generate results for different markets
    data = {
        'match_winner': {
            'features': np.column_stack([
                home_odds, away_odds,
                1/home_odds, 1/away_odds,
                home_points_avg, away_points_avg
            ]),
            'labels': (np.random.random(n_samples) < 1/home_odds).astype(int)
        },
        'total_points_over_200': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_points_avg, away_points_avg,
                home_points_avg + away_points_avg
            ]),
            'labels': ((home_points_avg + away_points_avg) > 200).astype(int)
        },
        'first_quarter_winner': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_first_quarter_avg, away_first_quarter_avg
            ]),
            'labels': (home_first_quarter_avg > away_first_quarter_avg).astype(int)
        },
        'point_spread_home': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_points_avg, away_points_avg,
                home_points_avg - away_points_avg
            ]),
            'labels': ((home_points_avg - away_points_avg) > 5).astype(int)
        },
        'first_half_winner': {
            'features': np.column_stack([
                home_odds, away_odds,
                home_points_avg/2, away_points_avg/2
            ]),
            'labels': ((home_points_avg/2) > (away_points_avg/2)).astype(int)
        }
    }
    
    return data

def train_and_save_models():
    """Train and save models for all prediction markets"""
    print("Initializing models...")
    
    # Train football models
    football_data = generate_football_training_data()
    for market, data in football_data.items():
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data['features'])
        
        # Train model
        model.fit(X_scaled, data['labels'])
        
        # Save model and scaler
        joblib.dump(model, os.path.join(MODEL_DIR, f'football_{market}_model.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'football_{market}_scaler.joblib'))
        
        print(f"Trained football {market} model")
    
    # Train basketball models
    basketball_data = generate_basketball_training_data()
    for market, data in basketball_data.items():
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data['features'])
        
        # Train model
        model.fit(X_scaled, data['labels'])
        
        # Save model and scaler
        joblib.dump(model, os.path.join(MODEL_DIR, f'basketball_{market}_model.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'basketball_{market}_scaler.joblib'))
        
        print(f"Trained basketball {market} model")
    
    print("\nAll models initialized and trained successfully!")
    print("\nFootball Markets:")
    for market in football_data.keys():
        print(f"• {market.replace('_', ' ').title()}")
    
    print("\nBasketball Markets:")
    for market in basketball_data.keys():
        print(f"• {market.replace('_', ' ').title()}")

if __name__ == '__main__':
    train_and_save_models() 