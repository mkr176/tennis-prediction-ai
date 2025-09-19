#!/usr/bin/env python3
"""
Test cases that should fail to demonstrate enhanced error messages
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_failure_cases():
    """Test cases that should fail to show enhanced error messages"""
    predictor = TennisPredictor()

    if not predictor.load_model():
        print("❌ Could not load model")
        return

    # Temporarily lower the matching threshold to see failures
    print("🎾 TESTING ENHANCED ERROR MESSAGES FOR NON-MATCHES")
    print("=" * 60)

    # Test cases with completely non-tennis names that should fail
    test_cases = [
        ("Basketball Player", "Soccer Player", "Completely unrelated names"),
        ("XXXXXX YYYYYY", "ZZZZZZ AAAAAA", "Random letter combinations"),
        ("A B", "C D", "Single letters"),
        ("123 456", "789 000", "Numbers instead of names"),
        ("", "Empty Name", "Empty name test"),
        ("VeryLongNameThatDoesntExist", "AnotherVeryLongNameThatDoesntExist", "Very long non-existent names"),
    ]

    for player1, player2, description in test_cases:
        print(f"\n🔍 Testing: {description}")
        print(f"   Players: '{player1}' vs '{player2}'")

        try:
            result = predictor.predict_match(player1, player2, "hard", "atp_500")
            if result:
                print("   ✅ Unexpected success!")
                print(f"   Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
            else:
                print("   ❌ Failed as expected - good error handling")
        except Exception as e:
            print(f"   ⚠️  Exception occurred: {str(e)[:100]}...")

    print(f"\n" + "=" * 60)
    print("🎯 TESTING SURNAME-SPECIFIC SUGGESTIONS FOR REAL CASES")
    print("=" * 60)

    # Test with names that might partially match but we want to see suggestions
    realistic_cases = [
        "John Martinez",  # Common first name, tennis-like surname
        "Carlos Rodriguez",  # Tennis first name, common surname
        "Rafael Johnson",  # Tennis first name, common surname
        "Mike Williams",  # Common name combination
        "David Thompson",  # Common name combination
    ]

    for name in realistic_cases:
        print(f"\n🔍 Testing realistic partial match: '{name}'")

        # Check what the system finds
        matched_name, score = predictor.find_player_match(name)
        if matched_name:
            print(f"   ✅ Found match: '{matched_name}' (score: {score:.2f})")
        else:
            print(f"   ❌ No match found")

        # Show suggestions
        suggestions = predictor.suggest_similar_players(name, 3)
        if suggestions:
            print(f"   💡 Suggestions: {', '.join(suggestions)}")

        # Show surname-specific suggestions
        surname_info = predictor.get_surname_specific_suggestions(name)
        if surname_info['surname_suggestions']:
            print(f"   🔍 Similar surnames: {', '.join(surname_info['surname_suggestions'])}")
        if surname_info['firstname_suggestions']:
            print(f"   👤 Same first name: {', '.join(surname_info['firstname_suggestions'])}")

if __name__ == "__main__":
    test_failure_cases()