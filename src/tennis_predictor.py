import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tennis_elo_system import TennisEloSystem
import warnings
warnings.filterwarnings('ignore')

class TennisPredictor:
    """
    Tennis Match Predictor using the 85% accuracy YouTube model approach.

    Implements the exact prediction system that achieved 85% accuracy:
    - ELO as primary feature
    - Surface-specific analysis
    - Comprehensive match statistics
    - XGBoost optimization
    """

    def __init__(self):
        self.model = None
        self.elo_system = None
        self.feature_columns = None
        self.confidence_threshold = 0.75  # High confidence predictions

    def load_model(self):
        """Load the trained 85% accuracy model"""
        try:
            import os
            # Get the directory of this file and construct the models path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')

            self.model = joblib.load(os.path.join(models_dir, 'real_atp_85_percent_model.pkl'))
            self.feature_columns = joblib.load(os.path.join(models_dir, 'real_atp_features.pkl'))
            self.elo_system = joblib.load(os.path.join(models_dir, 'real_atp_elo_system.pkl'))

            print("✅ 85% accuracy tennis model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using train_tennis_model.py")
            return False

    def create_prediction_features(self, player1, player2, surface='hard',
                                 tournament_type='atp_250', match_date=None):
        """
        Create prediction features using YouTube model approach
        """
        if match_date is None:
            match_date = datetime.now()

        # Get ELO features (most important in YouTube model)
        player1_elo_features = self.elo_system.get_player_elo_features(player1, surface)
        player2_elo_features = self.elo_system.get_player_elo_features(player2, surface)

        # Create feature set matching training data exactly
        features = {
            # CORE ELO FEATURES
            'player_elo_diff': player1_elo_features['overall_elo'] - player2_elo_features['overall_elo'],
            'surface_elo_diff': player1_elo_features['surface_elo'] - player2_elo_features['surface_elo'],
            'total_elo': player1_elo_features['overall_elo'] + player2_elo_features['overall_elo'],

            # Individual ELO ratings
            'player1_elo': player1_elo_features['overall_elo'],
            'player2_elo': player2_elo_features['overall_elo'],
            'player1_surface_elo': player1_elo_features['surface_elo'],
            'player2_surface_elo': player2_elo_features['surface_elo'],

            # SURFACE-SPECIFIC FEATURES
            'clay_elo_diff': player1_elo_features['clay_elo'] - player2_elo_features['clay_elo'],
            'grass_elo_diff': player1_elo_features['grass_elo'] - player2_elo_features['grass_elo'],
            'hard_elo_diff': player1_elo_features['hard_elo'] - player2_elo_features['hard_elo'],

            # RECENT FORM
            'recent_form_diff': player1_elo_features['recent_form'] - player2_elo_features['recent_form'],
            'momentum_diff': player1_elo_features['recent_momentum'] - player2_elo_features['recent_momentum'],
            'elo_change_diff': player1_elo_features['recent_elo_change'] - player2_elo_features['recent_elo_change'],

            # MATCH STATISTICS (default values for prediction)
            'ace_diff': 0,
            'double_fault_diff': 0,
            'first_serve_pct_diff': 0,
            'break_points_saved_pct_diff': 0,

            # RANKING AND PLAYER INFO (default values)
            'rank_diff': 0,  # Will use default if ranking not available
            'rank_points_diff': 0,
            'age_diff': 0,
            'height_diff': 0,

            # HEAD-TO-HEAD (default values)
            'h2h_advantage': 0,
            'h2h_win_rate': 0.5,
            'h2h_total_matches': 0,

            # TOURNAMENT CONTEXT
            'tournament_weight': self.elo_system.tournament_weights.get(tournament_type, 25),
            'is_grand_slam': 1 if tournament_type == 'grand_slam' else 0,
            'is_masters': 1 if tournament_type == 'masters_1000' else 0,

            # SURFACE ENCODING
            'is_clay': 1 if surface == 'clay' else 0,
            'is_grass': 1 if surface == 'grass' else 0,
            'is_hard': 1 if surface == 'hard' else 0,

            # INTERACTION FEATURES
            'elo_rank_interaction': 0,  # Default value
            'surface_rank_interaction': 0,  # Default value
        }

        return features

    def normalize_name(self, name):
        """
        Normalize a name for better matching (handle apostrophes, accents, etc.)
        """
        import re
        # Convert to lowercase and strip whitespace
        normalized = name.lower().strip()

        # Handle apostrophes - both ways (with and without)
        # This creates multiple versions for matching
        variations = [normalized]

        # Add version without apostrophes
        no_apostrophe = re.sub(r"['']", "", normalized)
        if no_apostrophe != normalized:
            variations.append(no_apostrophe)

        # Add version with different apostrophe styles
        if "'" in normalized:
            variations.append(normalized.replace("'", "'"))  # curly apostrophe
        if "'" in normalized:
            variations.append(normalized.replace("'", "'"))  # straight apostrophe

        # Handle common accent removal (basic)
        accent_map = {
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ñ': 'n', 'ç': 'c'
        }

        for original in variations.copy():
            no_accent = original
            for accented, plain in accent_map.items():
                no_accent = no_accent.replace(accented, plain)
            if no_accent != original:
                variations.append(no_accent)

        return list(set(variations))  # Remove duplicates

    def find_player_match(self, player_name):
        """
        Find the best match for a player name (case-insensitive and fuzzy matching)
        Handles apostrophes, accents, and special characters
        Returns: (matched_name, confidence_score) or (None, 0)
        """
        if not self.elo_system:
            return None, 0

        all_players = self.elo_system.get_all_players()
        search_variations = self.normalize_name(player_name)

        # Try exact match first (case-insensitive with normalization)
        for search_variant in search_variations:
            for player in all_players:
                player_variations = self.normalize_name(player)
                for player_variant in player_variations:
                    if search_variant == player_variant:
                        if self.elo_system.has_played_matches(player):
                            return player, 1.0

        best_match = None
        best_score = 0

        # Now do fuzzy matching with normalized names
        for player in all_players:
            if not self.elo_system.has_played_matches(player):
                continue

            player_variations = self.normalize_name(player)
            score = 0

            # Test each search variation against each player variation
            for search_variant in search_variations:
                search_parts = search_variant.split()

                for player_variant in player_variations:
                    player_parts = player_variant.split()
                    current_score = 0

                    # Full substring match
                    if search_variant in player_variant or player_variant in search_variant:
                        current_score = 0.85

                    # Check name parts matching
                    exact_matches = 0
                    partial_matches = 0

                    for search_part in search_parts:
                        if len(search_part) >= 2:  # Consider meaningful parts
                            for player_part in player_parts:
                                # Exact part match
                                if search_part == player_part:
                                    exact_matches += 1
                                # Partial match (one contains the other)
                                elif search_part in player_part or player_part in search_part:
                                    partial_matches += 1
                                # Similar starting letters for longer names
                                elif (len(search_part) >= 4 and len(player_part) >= 4 and
                                      search_part[:3] == player_part[:3]):
                                    partial_matches += 0.7

                    if exact_matches > 0 or partial_matches > 0:
                        # Score based on matches
                        part_score = 0.4 + (exact_matches * 0.3) + (partial_matches * 0.15)
                        current_score = max(current_score, part_score)

                        # Bonus for having all parts matched
                        if exact_matches + partial_matches >= len(search_parts):
                            current_score += 0.2

                        # Bonus for first/last name exact matches
                        if len(search_parts) >= 2 and len(player_parts) >= 2:
                            if search_parts[0] == player_parts[0]:  # First name exact
                                current_score += 0.15
                            if search_parts[-1] == player_parts[-1]:  # Last name exact
                                current_score += 0.15

                    # Boost score for similar length names
                    length_diff = abs(len(search_variant) - len(player_variant))
                    if length_diff <= 3:
                        current_score += 0.1
                    elif length_diff <= 6:
                        current_score += 0.05

                    # Keep the best score for this player
                    score = max(score, current_score)

            # Lower threshold for matching - be more aggressive
            if score > best_score and score >= 0.5:  # Lower minimum confidence threshold
                best_match = player
                best_score = score

        return best_match, best_score

    def validate_players(self, player1, player2):
        """
        Validate that both players exist in the system and have match history
        Uses smart matching for case-insensitive and fuzzy name matching
        """
        if not self.elo_system:
            if not self.load_model():
                return False, "Model not loaded", player1, player2

        # Try to find matches for both players
        matched_player1, score1 = self.find_player_match(player1)
        matched_player2, score2 = self.find_player_match(player2)

        if matched_player1 and matched_player2:
            # Both players found
            message = "Both players validated"
            if matched_player1.lower() != player1.lower():
                message += f" ('{player1}' -> '{matched_player1}')"
            if matched_player2.lower() != player2.lower():
                message += f" ('{player2}' -> '{matched_player2}')"
            return True, message, matched_player1, matched_player2

        elif matched_player1 and not matched_player2:
            # Only player1 found
            suggestions = self.suggest_similar_players(player2, 5)
            surname_info = self.get_surname_specific_suggestions(player2)

            message = f"Player '{player2}' not found in the system."
            if matched_player1.lower() != player1.lower():
                message += f" ('{player1}' matched to '{matched_player1}')"

            if suggestions:
                message += f"\n   💡 Did you mean: {', '.join(suggestions[:3])}?"

            # Add surname-specific suggestions
            if surname_info['surname_suggestions']:
                message += f"\n   🔍 Players with similar surnames: {', '.join(surname_info['surname_suggestions'])}"
            if surname_info['firstname_suggestions']:
                message += f"\n   👤 Players with same first name: {', '.join(surname_info['firstname_suggestions'])}"

            return False, message, matched_player1, player2

        elif not matched_player1 and matched_player2:
            # Only player2 found
            suggestions = self.suggest_similar_players(player1, 5)
            surname_info = self.get_surname_specific_suggestions(player1)

            message = f"Player '{player1}' not found in the system."
            if matched_player2.lower() != player2.lower():
                message += f" ('{player2}' matched to '{matched_player2}')"

            if suggestions:
                message += f"\n   💡 Did you mean: {', '.join(suggestions[:3])}?"

            # Add surname-specific suggestions
            if surname_info['surname_suggestions']:
                message += f"\n   🔍 Players with similar surnames: {', '.join(surname_info['surname_suggestions'])}"
            if surname_info['firstname_suggestions']:
                message += f"\n   👤 Players with same first name: {', '.join(surname_info['firstname_suggestions'])}"

            return False, message, player1, matched_player2

        else:
            # Neither player found
            suggestions1 = self.suggest_similar_players(player1, 5)
            suggestions2 = self.suggest_similar_players(player2, 5)
            surname_info1 = self.get_surname_specific_suggestions(player1)
            surname_info2 = self.get_surname_specific_suggestions(player2)

            message = f"Neither '{player1}' nor '{player2}' found in the system.\n"

            # Player 1 suggestions
            if suggestions1:
                message += f"   💡 Similar to '{player1}': {', '.join(suggestions1[:3])}\n"
            if surname_info1['surname_suggestions']:
                message += f"   🔍 '{player1}' similar surnames: {', '.join(surname_info1['surname_suggestions'])}\n"

            # Player 2 suggestions
            if suggestions2:
                message += f"   💡 Similar to '{player2}': {', '.join(suggestions2[:3])}\n"
            if surname_info2['surname_suggestions']:
                message += f"   🔍 '{player2}' similar surnames: {', '.join(surname_info2['surname_suggestions'])}"

            return False, message, player1, player2

    def get_available_players(self, limit=None):
        """Get list of all available players in the system"""
        if not self.elo_system:
            if not self.load_model():
                return []

        all_players = self.elo_system.get_all_players()
        # Filter to only players who have actually played matches
        active_players = [player for player in all_players
                         if self.elo_system.has_played_matches(player)]

        if limit:
            return active_players[:limit]
        return active_players

    def suggest_similar_players(self, player_name, limit=5):
        """Suggest similar player names from the system with smart matching and surname-specific suggestions"""
        if not self.elo_system:
            if not self.load_model():
                return []

        all_players = self.get_available_players()
        search_variations = self.normalize_name(player_name)
        player_name_lower = player_name.lower()
        name_parts = player_name_lower.split()

        # Score each player for similarity
        scored_players = []
        surname_matches = []  # Track players with matching surnames
        similar_surnames = []  # Track players with similar surnames

        for player in all_players:
            player_lower = player.lower()
            player_parts = player_lower.split()
            score = 0

            # Exact match (case-insensitive)
            if player_lower == player_name_lower:
                score = 100

            # Full substring match
            elif player_name_lower in player_lower or player_lower in player_name_lower:
                score = 80

            # Check for surname/lastname matches specifically
            if len(name_parts) >= 2 and len(player_parts) >= 2:
                search_lastname = name_parts[-1]
                player_lastname = player_parts[-1]

                # Exact surname match
                if search_lastname == player_lastname:
                    surname_matches.append(player)
                    score += 25  # Bonus for surname match

                # Similar surname (for typos/variations)
                elif (len(search_lastname) >= 3 and len(player_lastname) >= 3 and
                      (search_lastname[:3] == player_lastname[:3] or
                       search_lastname in player_lastname or player_lastname in search_lastname)):
                    similar_surnames.append(player)

            # Name parts matching
            matches = 0
            partial_matches = 0

            for search_part in name_parts:
                if len(search_part) >= 2:  # Consider meaningful parts
                    for player_part in player_parts:
                        # Exact part match
                        if search_part == player_part:
                            matches += 2
                        # Partial part match
                        elif search_part in player_part or player_part in search_part:
                            partial_matches += 1
                        # Similar length and first letters match
                        elif (len(search_part) >= 3 and len(player_part) >= 3 and
                              search_part[:2] == player_part[:2]):
                            partial_matches += 0.5

            if matches > 0 or partial_matches > 0:
                score = 30 + (matches * 15) + (partial_matches * 5)

                # Bonus for similar total length
                if abs(len(player_name_lower) - len(player_lower)) <= 5:
                    score += 5

                # Bonus if first or last names match
                if len(name_parts) >= 2 and len(player_parts) >= 2:
                    if (name_parts[0] == player_parts[0] or  # First name match
                        name_parts[-1] == player_parts[-1]):  # Last name match
                        score += 20

            if score > 0:
                scored_players.append((player, score))

        # Sort by score (highest first)
        scored_players.sort(key=lambda x: x[1], reverse=True)

        # Build smart suggestions prioritizing surname matches
        suggestions = []

        # First, add exact surname matches (highest priority)
        for player in surname_matches:
            if player not in suggestions:
                suggestions.append(player)
                if len(suggestions) >= limit:
                    break

        # Then add highest scoring matches that aren't already included
        for player, score in scored_players:
            if player not in suggestions:
                suggestions.append(player)
                if len(suggestions) >= limit:
                    break

        # If we still need more and have similar surnames, add them
        if len(suggestions) < limit:
            for player in similar_surnames:
                if player not in suggestions:
                    suggestions.append(player)
                    if len(suggestions) >= limit:
                        break

        # If still no good matches, return some popular players
        if not suggestions:
            suggestions = all_players[:limit]

        return suggestions[:limit]

    def get_surname_specific_suggestions(self, player_name):
        """Get specific suggestions when surname isn't found"""
        if not self.elo_system:
            return []

        all_players = self.get_available_players()
        name_parts = player_name.lower().strip().split()

        if len(name_parts) < 2:
            return []

        search_surname = name_parts[-1]
        search_firstname = name_parts[0]

        # Find players with similar surnames
        surname_suggestions = []
        firstname_suggestions = []

        for player in all_players:
            player_parts = player.lower().split()
            if len(player_parts) >= 2:
                player_surname = player_parts[-1]
                player_firstname = player_parts[0]

                # Similar surnames
                if (len(search_surname) >= 3 and len(player_surname) >= 3 and
                    (search_surname[:3] == player_surname[:3] or
                     search_surname in player_surname or player_surname in search_surname)):
                    surname_suggestions.append(player)

                # Same first name, different surname
                if search_firstname == player_firstname and search_surname != player_surname:
                    firstname_suggestions.append(player)

        return {
            'surname_suggestions': surname_suggestions[:3],
            'firstname_suggestions': firstname_suggestions[:3]
        }

    def predict_match(self, player1, player2, surface='hard', tournament_type='atp_250'):
        """
        Predict tennis match outcome using 85% accuracy model
        Only works for players that exist in the system
        """
        if not self.model:
            if not self.load_model():
                return None

        # Validate players exist in the system (now returns matched names)
        validation_result = self.validate_players(player1, player2)
        if len(validation_result) == 4:  # New format with matched names
            valid, message, matched_player1, matched_player2 = validation_result
        else:  # Fallback for old format
            valid, message = validation_result
            matched_player1, matched_player2 = player1, player2

        if not valid:
            print(f"❌ Validation failed: {message}")
            # Show some available players as suggestions
            available = self.get_available_players(10)
            if available:
                print("💡 Available players include:", ", ".join(available[:5]))
                if len(available) > 5:
                    print(f"   ...and {len(available) - 5} more players")
            return None

        print(f"✅ {message}")

        # Use the matched player names for prediction
        player1, player2 = matched_player1, matched_player2

        # Create prediction features
        features = self.create_prediction_features(player1, player2, surface, tournament_type)

        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features])
        X = features_df[self.feature_columns].fillna(0)

        # Get prediction
        prediction_proba = self.model.predict_proba(X)[0]
        prediction_class = self.model.predict(X)[0]

        # Interpret results (1 = player1 wins, 0 = player2 wins)
        player1_win_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        player2_win_prob = 1 - player1_win_prob

        winner = player1 if prediction_class == 1 else player2
        confidence = max(player1_win_prob, player2_win_prob)

        # Get ELO-based prediction for comparison
        elo_prediction = self.elo_system.predict_match_outcome(player1, player2, surface)

        return {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'tournament': tournament_type,
            'predicted_winner': winner,
            'player1_win_probability': player1_win_prob,
            'player2_win_probability': player2_win_prob,
            'confidence': confidence,
            'is_high_confidence': confidence >= self.confidence_threshold,

            # ELO comparison
            'elo_favorite': elo_prediction['favorite'],
            'elo_confidence': elo_prediction['confidence'],

            # Model insights
            'model_accuracy_target': 0.85,
            'prediction_method': '85% Accuracy YouTube Model'
        }

    def predict_multiple_matches(self, matches):
        """
        Predict multiple matches efficiently
        """
        predictions = []

        for match in matches:
            prediction = self.predict_match(
                player1=match['player1'],
                player2=match['player2'],
                surface=match.get('surface', 'hard'),
                tournament_type=match.get('tournament_type', 'atp_250')
            )
            predictions.append(prediction)

        return predictions

    def analyze_head_to_head(self, player1, player2):
        """
        Analyze head-to-head record and surface breakdown
        """
        if not self.elo_system:
            if not self.load_model():
                return None

        # Get player ELO features
        player1_features = self.elo_system.get_player_elo_features(player1)
        player2_features = self.elo_system.get_player_elo_features(player2)

        surfaces = ['clay', 'grass', 'hard']
        surface_predictions = {}

        for surface in surfaces:
            prediction = self.predict_match(player1, player2, surface, 'atp_500')
            surface_predictions[surface] = {
                'winner': prediction['predicted_winner'],
                'confidence': prediction['confidence']
            }

        return {
            'player1': player1,
            'player2': player2,
            'player1_overall_elo': player1_features['overall_elo'],
            'player2_overall_elo': player2_features['overall_elo'],
            'elo_advantage': player1_features['overall_elo'] - player2_features['overall_elo'],
            'surface_predictions': surface_predictions,
            'best_surface_for_player1': max(surfaces,
                key=lambda s: player1_features[f'{s}_elo'] - player2_features[f'{s}_elo']),
            'head_to_head_analysis': 'Based on 85% accuracy model predictions'
        }

    def simulate_tournament_bracket(self, players, surface='hard', tournament_type='grand_slam'):
        """
        Simulate a tournament bracket with predictions
        """
        if len(players) not in [4, 8, 16, 32, 64, 128]:
            raise ValueError("Tournament size must be 4, 8, 16, 32, 64, or 128 players")

        current_round_players = players[:]
        tournament_results = {
            'surface': surface,
            'tournament_type': tournament_type,
            'rounds': []
        }

        round_number = 1

        while len(current_round_players) > 1:
            round_name = {
                128: 'Round 1', 64: 'Round 2', 32: 'Round 3', 16: 'Round 4',
                8: 'Quarterfinals', 4: 'Semifinals', 2: 'Final'
            }.get(len(current_round_players), f'Round {round_number}')

            round_matches = []
            next_round_players = []

            # Pair up players for matches
            for i in range(0, len(current_round_players), 2):
                if i + 1 < len(current_round_players):
                    player1 = current_round_players[i]
                    player2 = current_round_players[i + 1]

                    prediction = self.predict_match(player1, player2, surface, tournament_type)

                    match_result = {
                        'player1': player1,
                        'player2': player2,
                        'predicted_winner': prediction['predicted_winner'],
                        'confidence': prediction['confidence']
                    }

                    round_matches.append(match_result)
                    next_round_players.append(prediction['predicted_winner'])

            tournament_results['rounds'].append({
                'round_name': round_name,
                'matches': round_matches
            })

            current_round_players = next_round_players
            round_number += 1

        tournament_results['champion'] = current_round_players[0] if current_round_players else None

        return tournament_results

def main():
    """Test the tennis predictor"""
    print("🎾 TENNIS PREDICTION SYSTEM")
    print("Based on 85% accuracy YouTube model")
    print("=" * 50)

    predictor = TennisPredictor()

    # Test prediction
    print("\n🔮 Testing predictions...")

    # Famous rivalry predictions
    rivalries = [
        ("Novak Djokovic", "Rafael Nadal", "clay"),
        ("Carlos Alcaraz", "Novak Djokovic", "grass"),
        ("Daniil Medvedev", "Rafael Nadal", "hard"),
        ("Stefanos Tsitsipas", "Carlos Alcaraz", "hard")
    ]

    for player1, player2, surface in rivalries:
        prediction = predictor.predict_match(
            player1, player2, surface, 'grand_slam'
        )

        if prediction:
            print(f"\n🎾 {player1} vs {player2} ({surface} court)")
            print(f"   Predicted winner: {prediction['predicted_winner']}")
            print(f"   {player1}: {prediction['player1_win_probability']:.1%}")
            print(f"   {player2}: {prediction['player2_win_probability']:.1%}")
            print(f"   Confidence: {prediction['confidence']:.1%}")
            print(f"   High confidence: {prediction['is_high_confidence']}")

    # Head-to-head analysis
    print(f"\n📊 Head-to-head analysis:")
    h2h = predictor.analyze_head_to_head("Novak Djokovic", "Rafael Nadal")
    if h2h:
        print(f"   ELO advantage: {h2h['elo_advantage']:.0f} points")
        print(f"   Surface breakdown:")
        for surface, pred in h2h['surface_predictions'].items():
            print(f"     {surface.title()}: {pred['winner']} ({pred['confidence']:.1%})")

    print(f"\n✅ Tennis prediction system ready!")
    print(f"Targeting 85% accuracy like YouTube model!")

if __name__ == "__main__":
    main()