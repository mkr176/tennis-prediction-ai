#!/usr/bin/env python3
"""
Test the exact user case that was failing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_user_case():
    """Test the exact case from the user"""
    print("🎾 TESTING USER'S EXACT CASE")
    print("=" * 50)
    print("Predicting Thomas Martin vs Damir v Etcheverry Dzumhur...")
    print("   Surface: Hard")
    print("   Tournament: Atp 500")

    predictor = TennisPredictor()

    result = predictor.predict_match(
        "Thomas Martin",
        "Damir v Etcheverry Dzumhur",
        "hard",
        "atp_500"
    )

    if result:
        print(f"\n🎯 PREDICTION SUCCESSFUL!")
        print(f"Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Player 1 ({result['player1']}): {(1-result['player2_prob']):.1%}")
        print(f"Player 2 ({result['player2']}): {result['player2_prob']:.1%}")
    else:
        print("❌ Prediction failed")

if __name__ == "__main__":
    test_user_case()