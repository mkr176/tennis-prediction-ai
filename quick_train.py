#!/usr/bin/env python3
"""
Quick tennis model training without optimization for immediate testing
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import warnings
from tennis_elo_system import TennisEloSystem
warnings.filterwarnings('ignore')

def quick_train():
    print("🎾 QUICK TENNIS MODEL TRAINING")
    print("Getting basic models for testing")
    print("=" * 50)

    # Load tennis data
    try:
        matches_df = pd.read_csv('../data/tennis_matches.csv')
        print(f"✅ Loaded {len(matches_df):,} matches")
    except:
        print("❌ No tennis data found")
        return False

    # Build ELO system
    elo_system = TennisEloSystem()
    elo_system.build_from_match_data(matches_df)

    # Create balanced features quickly
    features_list = []

    for idx, match in matches_df.iterrows():
        if idx > 5000:  # Limit for quick training
            break

        winner = match['winner']
        loser = match['loser']
        surface = match['surface']
        tournament_type = match.get('tournament_type', 'atp_250')

        # Create both perspectives
        for player_wins, player1, player2 in [(1, winner, loser), (0, loser, winner)]:
            p1_elo = elo_system.get_player_elo_features(player1, surface)
            p2_elo = elo_system.get_player_elo_features(player2, surface)

            features = {
                'target': player_wins,
                'player_elo_diff': p1_elo['overall_elo'] - p2_elo['overall_elo'],
                'surface_elo_diff': p1_elo['surface_elo'] - p2_elo['surface_elo'],
                'total_elo': p1_elo['overall_elo'] + p2_elo['overall_elo'],
                'player1_elo': p1_elo['overall_elo'],
                'player2_elo': p2_elo['overall_elo'],
                'ace_diff': match.get('winner_aces', 0) - match.get('loser_aces', 0) if player_wins else match.get('loser_aces', 0) - match.get('winner_aces', 0),
                'double_fault_diff': match.get('loser_double_faults', 0) - match.get('winner_double_faults', 0) if player_wins else match.get('winner_double_faults', 0) - match.get('loser_double_faults', 0),
                'is_clay': 1 if surface == 'clay' else 0,
                'is_grass': 1 if surface == 'grass' else 0,
                'is_hard': 1 if surface == 'hard' else 0,
            }
            features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Prepare data
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols].fillna(0)
    y = features_df['target']

    print(f"Training samples: {len(X):,}")
    print(f"Features: {len(feature_cols)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest (fast)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Train XGBoost (basic)
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    print(f"\n🎯 QUICK RESULTS:")
    print(f"Random Forest: {rf_accuracy:.3f}")
    print(f"XGBoost: {xgb_accuracy:.3f}")

    # Use best model
    if rf_accuracy > xgb_accuracy:
        best_model = rf_model
        best_accuracy = rf_accuracy
        model_name = "Random Forest"
    else:
        best_model = xgb_model
        best_accuracy = xgb_accuracy
        model_name = "XGBoost"

    print(f"Best: {model_name} ({best_accuracy:.3f})")

    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/tennis_85_percent_model.pkl')
    joblib.dump(feature_cols, 'models/tennis_features.pkl')
    joblib.dump(elo_system, 'models/tennis_elo_complete.pkl')

    print(f"✅ Models saved!")
    return True

if __name__ == "__main__":
    quick_train()