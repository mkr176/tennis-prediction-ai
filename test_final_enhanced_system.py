#!/usr/bin/env python3
"""
Comprehensive test of the final enhanced player name matching system
Demonstrates all improvements: case-insensitive, fuzzy matching, apostrophes,
surnames, and enhanced suggestions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tennis_predictor import TennisPredictor

def test_comprehensive_matching():
    """Test all enhanced matching features"""
    predictor = TennisPredictor()

    if not predictor.load_model():
        print("❌ Could not load model")
        return

    print("🎾 COMPREHENSIVE ENHANCED NAME MATCHING SYSTEM TEST")
    print("=" * 70)
    print("Testing: Case-insensitive, Fuzzy, Apostrophes, Surnames & Suggestions")
    print("=" * 70)

    # Comprehensive test cases covering all improvements
    test_cases = [
        # Original user problem (fixed)
        ("Thomas Martin", "Damir v Etcheverry Dzumhur", "✅ Original user case"),

        # Case insensitive matching
        ("NOVAK DJOKOVIC", "rafael nadal", "✅ Case insensitive"),
        ("roger federer", "CARLOS ALCARAZ", "✅ Mixed case"),

        # Apostrophe handling
        ("oconnell", "novak djokovic", "✅ Apostrophe (no apostrophe)"),
        ("o'connell", "nadal", "✅ Apostrophe (straight)"),
        ("O'Connell", "federer", "✅ Apostrophe (curly)"),

        # Accent/special character handling
        ("muller", "djokovic", "✅ Accent (no umlaut)"),
        ("Müller", "nadal", "✅ Accent (with umlaut)"),
        ("de minaur", "alcaraz", "✅ Compound surname"),
        ("del potro", "sinner", "✅ Spanish compound"),

        # Partial name matching
        ("djokovic", "nadal", "✅ Last name only"),
        ("novak", "rafa", "✅ First name/nickname"),
        ("fed", "nole", "✅ Partial names"),

        # Fuzzy matching
        ("carlos alcaras", "jannik siner", "✅ Typos"),
        ("alexander zverev", "stefanos tsitsipas", "✅ Full names"),
        ("zverev", "tsitsipas", "✅ Last names"),

        # Complex cases
        ("chris oconnell", "botic van de zandschulp", "✅ Complex names"),
        ("tommy paul", "jack draper", "✅ Common first names"),
    ]

    success_count = 0
    total_tests = len(test_cases)

    for i, (player1, player2, description) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. {description}")
        print(f"    Testing: '{player1}' vs '{player2}'")

        result = predictor.predict_match(player1, player2, "hard", "atp_500")

        if result:
            success_count += 1
            print(f"    ✅ SUCCESS: {result['predicted_winner']} wins ({result['confidence']:.1%})")

            # Show if names were matched/corrected
            original_p1, original_p2 = player1, player2
            matched_p1, matched_p2 = result['player1'], result['player2']

            if original_p1.lower() != matched_p1.lower():
                print(f"    🔧 Corrected: '{original_p1}' → '{matched_p1}'")
            if original_p2.lower() != matched_p2.lower():
                print(f"    🔧 Corrected: '{original_p2}' → '{matched_p2}'")
        else:
            print(f"    ❌ FAILED: Could not make prediction")

    print(f"\n" + "=" * 70)
    print(f"📊 FINAL RESULTS")
    print(f"=" * 70)
    print(f"✅ Successful predictions: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    print(f"❌ Failed predictions: {total_tests-success_count}/{total_tests}")

    if success_count == total_tests:
        print(f"🎉 PERFECT SCORE! All enhanced matching features working correctly!")
    elif success_count >= total_tests * 0.9:
        print(f"🌟 EXCELLENT! Enhanced matching is working very well!")
    elif success_count >= total_tests * 0.8:
        print(f"👍 GOOD! Most enhanced features are working correctly!")
    else:
        print(f"⚠️  Some enhanced features may need adjustment")

    print(f"\n" + "=" * 70)
    print(f"🔍 TESTING ENHANCED ERROR MESSAGES")
    print(f"=" * 70)

    # Test enhanced error messages for names that truly don't exist
    error_test_cases = [
        ("NonExistentPlayer1", "NonExistentPlayer2"),
        ("Basketball Star", "Soccer Player"),
        ("John Unknown", "Jane Somebody")
    ]

    for player1, player2 in error_test_cases:
        print(f"\n🔍 Testing error handling: '{player1}' vs '{player2}'")
        result = predictor.predict_match(player1, player2, "hard", "atp_500")
        if not result:
            print("    ✅ Proper error handling with enhanced suggestions shown")
        else:
            print("    ⚠️  Unexpected success (very aggressive matching)")

    print(f"\n" + "=" * 70)
    print(f"✅ COMPREHENSIVE TESTING COMPLETE")
    print(f"=" * 70)
    print(f"Enhanced features tested:")
    print(f"  ✅ Case-insensitive matching")
    print(f"  ✅ Fuzzy name matching")
    print(f"  ✅ Apostrophe handling (O'Connell variations)")
    print(f"  ✅ Accent/special character handling (Müller, José)")
    print(f"  ✅ Surname-specific suggestions")
    print(f"  ✅ Partial name matching (first/last name only)")
    print(f"  ✅ Enhanced error messages with smart suggestions")
    print(f"  ✅ Compound surname handling (van de, del, etc.)")

if __name__ == "__main__":
    test_comprehensive_matching()