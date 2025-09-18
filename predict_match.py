#!/usr/bin/env python3
"""
Interactive Tennis Match Predictor - Terminal Interface
87.4% accuracy model for real-time predictions
"""

import sys
import os
sys.path.append('src')

import argparse
from tennis_predictor import TennisPredictor

def interactive_prediction():
    """Interactive mode - ask user for input"""
    print("🎾 TENNIS MATCH PREDICTOR - 87.4% ACCURACY")
    print("Built with real ATP data • Interactive mode")
    print("=" * 60)

    # Initialize predictor
    predictor = TennisPredictor()

    while True:
        print("\n🔮 PREDICT A TENNIS MATCH")
        print("-" * 30)
        print("💡 Type 'players' to see available players, or 'quit' to exit")

        # Get player names
        player1 = input("Enter Player 1 name: ").strip()
        if not player1:
            print("❌ Player name cannot be empty")
            continue

        if player1.lower() in ['quit', 'exit']:
            print("🎾 Thanks for using Tennis Match Predictor!")
            break

        if player1.lower() == 'players':
            available_players = predictor.get_available_players(20)
            if available_players:
                print(f"\n📋 Available players ({len(available_players)} showing):")
                for i, player in enumerate(available_players[:20], 1):
                    print(f"  {i:2d}. {player}")
                if len(predictor.get_available_players()) > 20:
                    total = len(predictor.get_available_players())
                    print(f"  ... and {total-20} more players")
            else:
                print("❌ No players found. Make sure the model is trained.")
            continue

        player2 = input("Enter Player 2 name: ").strip()
        if not player2:
            print("❌ Player name cannot be empty")
            continue

        if player2.lower() == 'players':
            available_players = predictor.get_available_players(20)
            if available_players:
                print(f"\n📋 Available players ({len(available_players)} showing):")
                for i, player in enumerate(available_players[:20], 1):
                    print(f"  {i:2d}. {player}")
                if len(predictor.get_available_players()) > 20:
                    total = len(predictor.get_available_players())
                    print(f"  ... and {total-20} more players")
            else:
                print("❌ No players found. Make sure the model is trained.")
            continue

        # Get surface
        print("\nSurface options: clay, grass, hard")
        surface = input("Enter surface (default: hard): ").strip().lower()
        if not surface:
            surface = "hard"
        elif surface not in ["clay", "grass", "hard"]:
            print(f"⚠️  Unknown surface '{surface}', using 'hard'")
            surface = "hard"

        # Get tournament type
        print("\nTournament options: grand_slam, masters_1000, atp_500, atp_250")
        tournament = input("Enter tournament type (default: atp_500): ").strip().lower()
        if not tournament:
            tournament = "atp_500"
        elif tournament not in ["grand_slam", "masters_1000", "atp_500", "atp_250"]:
            print(f"⚠️  Unknown tournament '{tournament}', using 'atp_500'")
            tournament = "atp_500"

        # Make prediction
        print(f"\n🤖 Predicting {player1} vs {player2}...")
        print(f"   Surface: {surface.title()}")
        print(f"   Tournament: {tournament.replace('_', ' ').title()}")

        try:
            prediction = predictor.predict_match(player1, player2, surface, tournament)

            if prediction:
                print(f"\n🏆 PREDICTION RESULT:")
                print(f"   🎾 {player1} vs {player2}")
                print(f"   🏟️  Surface: {surface.title()} court")
                print(f"   🏆 Tournament: {tournament.replace('_', ' ').title()}")
                print(f"")
                print(f"   🥇 Predicted Winner: {prediction['predicted_winner']}")
                print(f"   📊 {player1}: {prediction['player1_win_probability']:.1%}")
                print(f"   📊 {player2}: {prediction['player2_win_probability']:.1%}")
                print(f"   🎯 Model Confidence: {prediction['confidence']:.1%}")
                print(f"   ⭐ High Confidence: {'Yes' if prediction['is_high_confidence'] else 'No'}")
                print(f"")
                print(f"   🤖 Model: {prediction['prediction_method']}")
                print(f"   🎯 Target Accuracy: {prediction['model_accuracy_target']:.0%}")

                # Show ELO comparison
                if 'elo_favorite' in prediction:
                    print(f"\n📈 ELO COMPARISON:")
                    print(f"   ELO Favorite: {prediction['elo_favorite']}")
                    print(f"   ELO Confidence: {prediction['elo_confidence']:.1%}")

            else:
                print("❌ Could not make prediction. Players might not be in the system.")
                # Show some available players as examples
                available_players = predictor.get_available_players(10)
                if available_players:
                    print(f"💡 Available players include: {', '.join(available_players[:5])}")
                    if len(available_players) > 5:
                        print(f"   ...and {len(available_players)-5} more players")

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            print("💡 Make sure the model is trained. Run: python3 train_real_atp_model.py")

        # Ask if user wants to continue
        print(f"\n" + "=" * 60)
        continue_choice = input("Predict another match? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("🎾 Thanks for using Tennis Match Predictor!")
            break

def command_line_prediction(player1, player2, surface="hard", tournament="atp_500"):
    """Command line mode - direct prediction"""
    print("🎾 TENNIS MATCH PREDICTOR - 87.4% ACCURACY")
    print("=" * 60)

    predictor = TennisPredictor()

    print(f"🤖 Predicting: {player1} vs {player2}")
    print(f"   Surface: {surface.title()}")
    print(f"   Tournament: {tournament.replace('_', ' ').title()}")

    try:
        prediction = predictor.predict_match(player1, player2, surface, tournament)

        if prediction:
            print(f"\n🏆 RESULT: {prediction['predicted_winner']} wins!")
            print(f"   📊 {player1}: {prediction['player1_win_probability']:.1%}")
            print(f"   📊 {player2}: {prediction['player2_win_probability']:.1%}")
            print(f"   🎯 Confidence: {prediction['confidence']:.1%}")

            return prediction
        else:
            print("❌ Could not make prediction")
            # Show some available players as examples
            available_players = predictor.get_available_players(10)
            if available_players:
                print(f"💡 Available players include: {', '.join(available_players[:5])}")
                if len(available_players) > 5:
                    print(f"   ...and {len(available_players)-5} more players")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the model is trained: python3 train_real_atp_model.py")
        return None

def show_famous_examples():
    """Show predictions for famous tennis rivalries"""
    print("🎾 FAMOUS TENNIS RIVALRIES - PREDICTIONS")
    print("=" * 60)

    predictor = TennisPredictor()

    famous_matches = [
        ("Novak Djokovic", "Rafael Nadal", "clay", "grand_slam"),
        ("Carlos Alcaraz", "Novak Djokovic", "grass", "grand_slam"),
        ("Daniil Medvedev", "Rafael Nadal", "hard", "masters_1000"),
        ("Stefanos Tsitsipas", "Carlos Alcaraz", "clay", "atp_500"),
        ("Alexander Zverev", "Jannik Sinner", "hard", "atp_250")
    ]

    for player1, player2, surface, tournament in famous_matches:
        try:
            prediction = predictor.predict_match(player1, player2, surface, tournament)
            if prediction:
                print(f"\n🎾 {player1} vs {player2} ({surface})")
                print(f"   Winner: {prediction['predicted_winner']} ({prediction['confidence']:.1%})")
            else:
                print(f"\n❌ Could not predict {player1} vs {player2}")
        except:
            print(f"\n❌ Error predicting {player1} vs {player2}")

def main():
    parser = argparse.ArgumentParser(
        description="🎾 Tennis Match Predictor (87.4% accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 predict_match.py                                    # Interactive mode
  python3 predict_match.py --examples                         # Famous rivalries
  python3 predict_match.py "Djokovic" "Nadal"                # Quick prediction
  python3 predict_match.py "Alcaraz" "Medvedev" --surface grass --tournament grand_slam
        """
    )

    parser.add_argument('player1', nargs='?', help='First player name')
    parser.add_argument('player2', nargs='?', help='Second player name')
    parser.add_argument('--surface', choices=['clay', 'grass', 'hard'], default='hard',
                       help='Court surface (default: hard)')
    parser.add_argument('--tournament', choices=['grand_slam', 'masters_1000', 'atp_500', 'atp_250'],
                       default='atp_500', help='Tournament type (default: atp_500)')
    parser.add_argument('--examples', action='store_true',
                       help='Show predictions for famous tennis rivalries')

    args = parser.parse_args()

    if args.examples:
        show_famous_examples()
    elif args.player1 and args.player2:
        command_line_prediction(args.player1, args.player2, args.surface, args.tournament)
    else:
        interactive_prediction()

if __name__ == "__main__":
    main()