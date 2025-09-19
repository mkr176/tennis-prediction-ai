#!/usr/bin/env python3
"""
Test the enhanced surname-specific suggestions system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_surname_suggestions():
    """Test surname-specific suggestions"""
    predictor = TennisPredictor()

    if not predictor.load_model():
        print("❌ Could not load model")
        return

    print("🎾 TESTING ENHANCED SURNAME-SPECIFIC SUGGESTIONS")
    print("=" * 60)

    # Test cases where players don't exist - should show smart suggestions
    test_cases = [
        ("John Smith", "Rafael Nadal", "Non-existent player vs existing player"),
        ("Roger Federer", "Jane Doe", "Existing player vs non-existent player"),
        ("Michael Jordan", "Lebron James", "Neither player exists"),
        ("Carlos Something", "Novak Somebody", "Partial names that don't exist"),
        ("Alex Zverev", "Rafa Nadal", "Close but not exact names"),
        ("Tommy Robredo", "David Ferrer", "Players who might not be in recent data"),
    ]

    for player1, player2, description in test_cases:
        print(f"\n🔍 Testing: {description}")
        print(f"   Players: '{player1}' vs '{player2}'")

        result = predictor.predict_match(player1, player2, "hard", "atp_500")

        if not result:
            print("   ❌ Prediction failed (as expected)")
        else:
            print("   ✅ Prediction successful!")
            print(f"   Winner: {result['predicted_winner']} ({result['confidence']:.1%})")

    print(f"\n" + "=" * 60)
    print("🎯 TESTING INDIVIDUAL SURNAME SUGGESTIONS")
    print("=" * 60)

    # Test individual surname suggestion function
    test_names = [
        "John Smith",
        "Carlos Something",
        "Alex Zverev",
        "Tommy Robredo",
        "Rafael Someone"
    ]

    for name in test_names:
        print(f"\n🔍 Testing surname suggestions for: '{name}'")
        surname_info = predictor.get_surname_specific_suggestions(name)

        if surname_info['surname_suggestions']:
            print(f"   🔍 Similar surnames: {', '.join(surname_info['surname_suggestions'])}")
        else:
            print("   🔍 No similar surnames found")

        if surname_info['firstname_suggestions']:
            print(f"   👤 Same first name: {', '.join(surname_info['firstname_suggestions'])}")
        else:
            print("   👤 No players with same first name")

    print(f"\n" + "=" * 60)
    print("🎯 TESTING SUGGESTION QUALITY")
    print("=" * 60)

    # Test suggestion quality for common tennis names
    tennis_test_cases = [
        "Roger Federer",  # Should find exact match
        "roger federer",  # Case insensitive
        "Federer",        # Last name only
        "Fed",           # Partial name
        "Roger Fed",     # Partial combination
        "Novak",         # First name only
        "Djoko",         # Partial last name
        "Rafa",          # Nickname
    ]

    for name in tennis_test_cases:
        print(f"\n🔍 Suggestions for: '{name}'")
        suggestions = predictor.suggest_similar_players(name, 3)
        if suggestions:
            print(f"   💡 Top suggestions: {', '.join(suggestions)}")
        else:
            print("   ❌ No suggestions found")

if __name__ == "__main__":
    test_surname_suggestions()