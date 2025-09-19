#!/usr/bin/env python3
"""
Test apostrophe and special character handling in tennis player names
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_apostrophe_matching():
    """Test apostrophe and special character matching"""
    predictor = TennisPredictor()

    if not predictor.load_model():
        print("❌ Could not load model")
        return

    print("🎾 TESTING APOSTROPHE AND SPECIAL CHARACTER MATCHING")
    print("=" * 60)

    # Test the normalization function first
    print("🔧 Testing name normalization:")
    test_normalizations = [
        "O'Connell",
        "o'connell",
        "O'Connell",  # Curly apostrophe
        "OConnell",   # No apostrophe
        "Müller",     # Umlaut
        "José",       # Accent
        "François",   # Accent and cedilla
        "Novák",      # Accent
    ]

    for name in test_normalizations:
        variations = predictor.normalize_name(name)
        print(f"   '{name}' → {variations}")

    print(f"\n" + "=" * 60)
    print("🎯 TESTING APOSTROPHE PLAYER MATCHING")
    print("=" * 60)

    # Test cases with apostrophes and special characters
    apostrophe_test_cases = [
        ("oconnell", "Should find O'Connell without apostrophe"),
        ("o'connell", "Should find O'Connell with straight apostrophe"),
        ("O'Connell", "Should find O'Connell with curly apostrophe"),
        ("O'CONNELL", "Should find O'Connell case insensitive"),
        ("christopher oconnell", "Should find Christopher O'Connell"),
        ("chris oconnell", "Should find Christopher O'Connell"),
        ("muller", "Should find Müller without umlaut"),
        ("Müller", "Should find Müller with umlaut"),
        ("jose", "Should find José without accent"),
        ("José", "Should find José with accent"),
    ]

    for test_name, expected in apostrophe_test_cases:
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

    print(f"\n" + "=" * 60)
    print("🎯 TESTING PREDICTION WITH APOSTROPHE NAMES")
    print("=" * 60)

    # Test actual predictions with apostrophe variations
    prediction_test_cases = [
        ("oconnell", "novak djokovic", "O'Connell (no apostrophe) vs Djokovic"),
        ("o'connell", "rafael nadal", "O'Connell (straight apostrophe) vs Nadal"),
        ("muller", "roger federer", "Müller (no umlaut) vs Federer"),
    ]

    for player1, player2, description in prediction_test_cases:
        print(f"\n🎾 Testing: {description}")
        print(f"   Predicting: '{player1}' vs '{player2}'")

        result = predictor.predict_match(player1, player2, "hard", "atp_500")
        if result:
            print(f"   ✅ Prediction successful!")
            print(f"   Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
            print(f"   Matched players: {result['player1']} vs {result['player2']}")
        else:
            print(f"   ❌ Prediction failed")

    print(f"\n" + "=" * 60)
    print("🔍 TESTING SPECIAL CHARACTER VARIATIONS")
    print("=" * 60)

    # Test various special character scenarios
    special_char_tests = [
        "d'avion",  # French-style apostrophe
        "o brien",  # Space instead of apostrophe
        "mcenroe",  # Scottish/Irish prefix
        "van der meer",  # Dutch prefix
        "de minaur",  # Compound surname
        "del potro",  # Spanish compound
    ]

    for name in special_char_tests:
        print(f"\n🔍 Testing special case: '{name}'")
        matched_name, score = predictor.find_player_match(name)
        if matched_name:
            print(f"   ✅ Found: '{matched_name}' (score: {score:.2f})")
        else:
            suggestions = predictor.suggest_similar_players(name, 3)
            if suggestions:
                print(f"   💡 Suggestions: {', '.join(suggestions)}")
            else:
                print(f"   ❌ No matches or suggestions found")

    print(f"\n" + "=" * 60)
    print("✅ APOSTROPHE AND SPECIAL CHARACTER TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_apostrophe_matching()