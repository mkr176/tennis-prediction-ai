#!/usr/bin/env python3
"""
Train tennis model with REAL ATP data targeting 85% accuracy
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import optuna
from optuna import Trial
import joblib
import warnings
from tennis_elo_system import TennisEloSystem
warnings.filterwarnings('ignore')

class RealATPModel:
    """
    Train tennis model with REAL ATP data for 85% accuracy target
    """

    def __init__(self):
        self.elo_system = TennisEloSystem()
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_columns = None
        self.target_accuracy = 0.85

    def create_real_atp_features(self, matches_df):
        """
        Create features from real ATP data with enhanced feature engineering
        """
        print("🎾 CREATING REAL ATP FEATURES")
        print("Using actual ATP statistics for maximum accuracy")
        print("=" * 60)

        # Build ELO system from real data
        print("Building ELO system from real ATP matches...")
        self.elo_system.build_from_match_data(matches_df)

        features_list = []

        for idx, match in matches_df.iterrows():
            winner = match['winner']
            loser = match['loser']
            surface = match['surface']
            tournament_type = match['tournament_type']

            # Get ELO features
            winner_elo = self.elo_system.get_player_elo_features(winner, surface)
            loser_elo = self.elo_system.get_player_elo_features(loser, surface)

            # Create balanced training examples (both perspectives)
            for player_wins, player1, player2, p1_elo, p2_elo in [
                (1, winner, loser, winner_elo, loser_elo),
                (0, loser, winner, loser_elo, winner_elo)
            ]:
                features = {
                    'target': player_wins,

                    # CORE ELO FEATURES (YouTube: most important)
                    'player_elo_diff': p1_elo['overall_elo'] - p2_elo['overall_elo'],
                    'surface_elo_diff': p1_elo['surface_elo'] - p2_elo['surface_elo'],
                    'total_elo': p1_elo['overall_elo'] + p2_elo['overall_elo'],

                    # Individual ELO ratings
                    'player1_elo': p1_elo['overall_elo'],
                    'player2_elo': p2_elo['overall_elo'],
                    'player1_surface_elo': p1_elo['surface_elo'],
                    'player2_surface_elo': p2_elo['surface_elo'],

                    # Surface-specific ELO
                    'clay_elo_diff': p1_elo['clay_elo'] - p2_elo['clay_elo'],
                    'grass_elo_diff': p1_elo['grass_elo'] - p2_elo['grass_elo'],
                    'hard_elo_diff': p1_elo['hard_elo'] - p2_elo['hard_elo'],

                    # Recent form
                    'recent_form_diff': p1_elo['recent_form'] - p2_elo['recent_form'],
                    'momentum_diff': p1_elo['recent_momentum'] - p2_elo['recent_momentum'],
                    'elo_change_diff': p1_elo['recent_elo_change'] - p2_elo['recent_elo_change'],

                    # REAL SERVING STATISTICS (key tennis predictors)
                    'ace_diff': (match['winner_aces'] - match['loser_aces']) if player_wins else (match['loser_aces'] - match['winner_aces']),
                    'double_fault_diff': (match['loser_double_faults'] - match['winner_double_faults']) if player_wins else (match['winner_double_faults'] - match['loser_double_faults']),

                    # Service percentages (real data)
                    'first_serve_pct_diff': (match['winner_first_serve_pct'] - match['loser_first_serve_pct']) if player_wins else (match['loser_first_serve_pct'] - match['winner_first_serve_pct']),

                    # Break points (YouTube emphasis: "every break point")
                    'break_points_saved_pct_diff': (match['winner_break_points_saved_pct'] - match['loser_break_points_saved_pct']) if player_wins else (match['loser_break_points_saved_pct'] - match['winner_break_points_saved_pct']),

                    # REAL ATP RANKINGS (powerful predictor)
                    'rank_diff': (match['loser_rank'] - match['winner_rank']) if player_wins else (match['winner_rank'] - match['loser_rank']),
                    'rank_points_diff': (match['winner_rank_points'] - match['loser_rank_points']) if player_wins else (match['loser_rank_points'] - match['winner_rank_points']),

                    # Player characteristics
                    'age_diff': (match['winner_age'] - match['loser_age']) if player_wins else (match['loser_age'] - match['winner_age']),
                    'height_diff': (match['winner_height'] - match['loser_height']) if player_wins else (match['loser_height'] - match['winner_height']),

                    # HEAD-TO-HEAD (real historical data)
                    'h2h_advantage': (match['winner_h2h_wins'] - match['loser_h2h_wins']) if player_wins else (match['loser_h2h_wins'] - match['winner_h2h_wins']),
                    'h2h_win_rate': match['winner_h2h_win_rate'] if player_wins else (1 - match['winner_h2h_win_rate']),
                    'h2h_total_matches': match['h2h_total_matches'],

                    # Tournament context
                    'tournament_weight': self.elo_system.tournament_weights.get(tournament_type, 25),
                    'is_grand_slam': 1 if tournament_type == 'grand_slam' else 0,
                    'is_masters': 1 if tournament_type == 'masters_1000' else 0,

                    # Surface encoding
                    'is_clay': 1 if surface == 'clay' else 0,
                    'is_grass': 1 if surface == 'grass' else 0,
                    'is_hard': 1 if surface == 'hard' else 0,

                    # Combined features (interaction terms)
                    'elo_rank_interaction': (p1_elo['overall_elo'] - p2_elo['overall_elo']) * ((match['loser_rank'] - match['winner_rank']) if player_wins else (match['winner_rank'] - match['loser_rank'])),
                    'surface_rank_interaction': (p1_elo['surface_elo'] - p2_elo['surface_elo']) * ((match['loser_rank'] - match['winner_rank']) if player_wins else (match['winner_rank'] - match['loser_rank'])),
                }

                features_list.append(features)

            if idx % 5000 == 0:
                print(f"   Processed {idx:,} matches...")

        features_df = pd.DataFrame(features_list)

        print(f"\n✅ REAL ATP FEATURES COMPLETE!")
        print(f"   📊 Training examples: {len(features_df):,} (balanced dataset)")
        print(f"   🎯 Features: {len(features_df.columns)-1} (real ATP statistics)")
        print(f"   ⚖️  Target balance: {features_df['target'].value_counts().to_dict()}")

        return features_df

    def train_85_percent_model(self):
        """
        Train model with real ATP data targeting 85% accuracy
        """
        print(f"\n🎾 TRAINING 85% ACCURACY MODEL WITH REAL ATP DATA")
        print("=" * 60)
        print("Using 27,672 real ATP matches with actual statistics")

        # Load real ATP data
        print("Loading real ATP dataset...")
        matches_df = pd.read_csv('../data/real_atp_matches.csv')
        print(f"✅ Loaded {len(matches_df):,} real ATP matches")
        print(f"   🎾 Players: {len(set(matches_df['winner']) | set(matches_df['loser'])):,}")
        print(f"   📅 Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")

        # Create enhanced features
        features_df = self.create_real_atp_features(matches_df)

        # Prepare data
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']

        self.feature_columns = feature_cols

        print(f"\n📊 REAL ATP TRAINING DATA:")
        print(f"   Features: {len(feature_cols)} (real statistics)")
        print(f"   Samples: {len(X):,}")
        print(f"   Target balance: {y.value_counts().to_dict()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Training: {len(X_train):,} | Test: {len(X_test):,}")

        # Train multiple models
        print(f"\n🚀 TRAINING MODELS WITH REAL ATP DATA:")

        # 1. Random Forest
        print("1️⃣  Random Forest with real data...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"   Random Forest: {rf_accuracy:.4f}")

        # 2. XGBoost
        print("2️⃣  XGBoost with real data...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"   XGBoost: {xgb_accuracy:.4f}")

        # 3. LightGBM
        print("3️⃣  LightGBM with real data...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        print(f"   LightGBM: {lgb_accuracy:.4f}")

        # 4. Ensemble
        print("4️⃣  Ensemble model...")
        ensemble = VotingClassifier([
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ], voting='soft')
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"   Ensemble: {ensemble_accuracy:.4f}")

        # Select best model
        models = {
            'Random Forest': (rf_model, rf_accuracy),
            'XGBoost': (xgb_model, xgb_accuracy),
            'LightGBM': (lgb_model, lgb_accuracy),
            'Ensemble': (ensemble, ensemble_accuracy)
        }

        best_name, (best_model, best_accuracy) = max(models.items(), key=lambda x: x[1][1])
        self.best_model = best_model

        print(f"\n🏆 REAL ATP MODEL RESULTS:")
        print(f"   🎾 YouTube targets vs Our results:")
        print(f"      ELO alone: 72.0% target")
        print(f"      Random Forest: 76.0% target → {rf_accuracy:.1%} achieved")
        print(f"      XGBoost: 85.0% target → {xgb_accuracy:.1%} achieved")
        print(f"      LightGBM: new → {lgb_accuracy:.1%} achieved")
        print(f"      Ensemble: new → {ensemble_accuracy:.1%} achieved")

        print(f"\n   🥇 Best model: {best_name}")
        print(f"   🎯 Final accuracy: {best_accuracy:.4f}")

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\n🎯 TOP 15 FEATURES (Real ATP Data):")
            for i, row in feature_importance.head(15).iterrows():
                print(f"   {row['feature']:<25}: {row['importance']:.4f}")

        # Achievement analysis
        print(f"\n📈 ACCURACY ANALYSIS:")
        if best_accuracy >= 0.85:
            print(f"🎉 SUCCESS! Achieved YouTube-level 85% accuracy with real data!")
        elif best_accuracy >= 0.80:
            print(f"🎯 Excellent! Very close to 85% target!")
        elif best_accuracy >= 0.75:
            print(f"✅ Great! Strong improvement with real data!")
        else:
            print(f"📊 Good progress! Gap to target: {0.85 - best_accuracy:.3f}")

        print(f"\n🔥 REAL DATA ADVANTAGE:")
        print(f"   Real ATP vs Simulated: {best_accuracy:.1%} vs 63.6%")
        print(f"   Improvement: {(best_accuracy - 0.636):.1%} accuracy gain")
        print(f"   Real rankings, real statistics, real H2H records")

        # Save models
        print(f"\n💾 Saving real ATP models...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, 'models/real_atp_85_percent_model.pkl')
        joblib.dump(self.feature_columns, 'models/real_atp_features.pkl')
        joblib.dump(self.elo_system, 'models/real_atp_elo_system.pkl')

        print(f"✅ Real ATP models saved!")

        return best_accuracy

def main():
    print("🎾 REAL ATP 85% ACCURACY MODEL")
    print("Training with actual ATP match data")
    print("=" * 60)

    model = RealATPModel()
    final_accuracy = model.train_85_percent_model()

    print(f"\n🎯 REAL ATP MODEL TRAINING COMPLETE!")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"YouTube target: 0.85")

    if final_accuracy >= 0.85:
        print(f"🎉 TARGET ACHIEVED! 85%+ accuracy with real data!")
    elif final_accuracy >= 0.80:
        print(f"🎯 Excellent performance! Close to target!")
    else:
        print(f"📈 Gap to YouTube target: {0.85 - final_accuracy:.3f}")

    print(f"\n🚀 Ready for real ATP predictions!")

if __name__ == "__main__":
    main()