#!/usr/bin/env python3
"""
Test the enhanced player name matching system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_name_matching():
    """Test various name matching scenarios"""
    predictor = TennisPredictor()

    if not predictor.load_model():
        print("❌ Could not load model")
        return

    print("🎾 TESTING ENHANCED PLAYER NAME MATCHING")
    print("=" * 50)

    # Test cases with various name formats
    test_cases = [
        ("thomas martin", "Expected: Should find player with similar name"),
        ("THOMAS MARTIN", "Expected: Case insensitive match"),
        ("damir dzumhur", "Expected: Should find Damir Dzumhur"),
        ("DAMIR DZUMHUR", "Expected: Case insensitive for Damir Dzumhur"),
        ("novak djokovic", "Expected: Should find Novak Djokovic"),
        ("NOVAK DJOKOVIC", "Expected: Case insensitive for Novak Djokovic"),
        ("djokovic", "Expected: Should find Novak Djokovic by last name"),
        ("novak", "Expected: Should find Novak Djokovic by first name"),
        ("rafael nadal", "Expected: Should find Rafael Nadal"),
        ("rafa nadal", "Expected: Should find Rafael Nadal"),
        ("carlos alcaraz", "Expected: Should find Carlos Alcaraz"),
        ("jannik sinner", "Expected: Should find Jannik Sinner"),
        ("alexander zverev", "Expected: Should find Alexander Zverev"),
        ("zverev", "Expected: Should find Alexander Zverev by last name"),
    ]

    for test_name, expected in test_cases:
        print(f"\n🔍 Testing: '{test_name}'")
        print(f"   {expected}")

        # Test the find_player_match method
        matched_name, score = predictor.find_player_match(test_name)
        if matched_name:
            print(f"   ✅ Found: '{matched_name}' (score: {score:.2f})")
        else:
            print(f"   ❌ No match found")
            # Get suggestions
            suggestions = predictor.suggest_similar_players(test_name, 3)
            if suggestions:
                print(f"   💡 Suggestions: {', '.join(suggestions)}")

    print(f"\n" + "=" * 50)
    print("🎯 TESTING ACTUAL PREDICTION WITH FUZZY MATCHING")
    print("=" * 50)

    # Test prediction with the problematic names from the user
    test_predictions = [
        ("Thomas Martin", "Damir Dzumhur"),
        ("thomas martin", "damir v etcheverry dzumhur"),
        ("NOVAK DJOKOVIC", "rafael nadal"),
        ("djokovic", "nadal"),
    ]

    for player1, player2 in test_predictions:
        print(f"\n🎾 Testing prediction: '{player1}' vs '{player2}'")
        result = predictor.predict_match(player1, player2, "hard", "atp_500")
        if result:
            print(f"   ✅ Prediction successful!")
            print(f"   Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
        else:
            print(f"   ❌ Prediction failed")

if __name__ == "__main__":
    test_name_matching()