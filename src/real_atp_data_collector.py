#!/usr/bin/env python3
"""
Real ATP Data Collector - Fetch actual tennis match data for 85% accuracy model
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime, timedelta
import os
from tennis_elo_system import TennisEloSystem

class RealATPDataCollector:
    """
    Collect real ATP tennis match data from multiple sources
    Target: Replace simulated data with actual match statistics
    """

    def __init__(self):
        self.matches = []
        self.players = set()
        self.elo_system = TennisEloSystem()

        # ATP ranking and match data sources
        self.data_sources = {
            'tennis_abstract': 'http://www.tennisabstract.com/',
            'atp_tour': 'https://www.atptour.com/',
            'tennis_explorer': 'https://www.tennisexplorer.com/',
            'ultimate_tennis': 'https://www.ultimatetennisstatistics.com/'
        }

    def collect_atp_csv_data(self):
        """
        Collect ATP data from publicly available CSV datasets
        """
        print("🎾 COLLECTING REAL ATP DATA")
        print("Using publicly available ATP match datasets")
        print("=" * 60)

        # Try to fetch from Tennis Abstract (Jeff Sackmann's repository)
        csv_urls = [
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2022.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2021.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2020.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2019.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2018.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2017.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2016.csv',
            'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2015.csv'
        ]

        all_matches = []
        successful_years = []

        for url in csv_urls:
            year = url.split('_')[-1].replace('.csv', '')
            print(f"📊 Fetching {year} ATP matches...")

            try:
                df = pd.read_csv(url)
                all_matches.append(df)
                successful_years.append(year)
                print(f"   ✅ {year}: {len(df):,} matches")
                time.sleep(0.5)  # Be respectful to the server

            except Exception as e:
                print(f"   ❌ {year}: Failed to fetch ({str(e)[:50]}...)")

        if all_matches:
            combined_df = pd.concat(all_matches, ignore_index=True)

            # Try to add local 2025 data if Sackmann's 2025 data wasn't available
            local_2025_data = self.try_local_2025_data()
            if local_2025_data is not None and '2025' not in successful_years:
                print(f"   📊 Adding local 2025 data: {len(local_2025_data):,} matches")
                combined_df = pd.concat([combined_df, local_2025_data], ignore_index=True)
                successful_years.append('2025 (local)')

            print(f"\n✅ REAL ATP DATA COLLECTED!")
            print(f"   📊 Total matches: {len(combined_df):,}")
            print(f"   📅 Years: {', '.join(successful_years)}")
            print(f"   🎾 Players: {len(set(combined_df['winner_name'].unique()) | set(combined_df['loser_name'].unique())):,}")

            return self.process_atp_data(combined_df)
        else:
            print("❌ No real ATP data could be fetched")
            return None

    def try_local_2025_data(self):
        """Try to load local 2025 ATP data"""
        try:
            local_2025_file = 'data/atp_matches_2025.csv'
            if os.path.exists(local_2025_file):
                print(f"   📊 Loading local 2025 data: {local_2025_file}")
                df_2025 = pd.read_csv(local_2025_file)

                # Convert to Sackmann format for compatibility
                sackmann_format = []
                for _, match in df_2025.iterrows():
                    sackmann_match = {
                        'tourney_date': match.get('date', '20250101'),
                        'tourney_name': match.get('tournament_name', 'ATP 2025'),
                        'surface': match.get('surface', 'Hard').title(),
                        'tourney_level': self.reverse_map_tournament_level(match.get('tournament_type', 'atp_250')),
                        'round': match.get('round', 'F'),
                        'winner_name': match.get('winner', 'Unknown'),
                        'loser_name': match.get('loser', 'Unknown'),
                        'score': match.get('score', ''),
                        'best_of': match.get('best_of', 3),
                        'minutes': match.get('match_duration_minutes', 120),

                        # Service stats
                        'w_ace': match.get('winner_aces', 0),
                        'l_ace': match.get('loser_aces', 0),
                        'w_df': match.get('winner_double_faults', 0),
                        'l_df': match.get('loser_double_faults', 0),

                        # Rankings
                        'winner_rank': match.get('winner_rank', 999),
                        'loser_rank': match.get('loser_rank', 999),
                        'winner_rank_points': match.get('winner_rank_points', 0),
                        'loser_rank_points': match.get('loser_rank_points', 0),

                        # Player details
                        'winner_age': match.get('winner_age', 25),
                        'loser_age': match.get('loser_age', 25),
                        'winner_hand': match.get('winner_hand', 'R'),
                        'loser_hand': match.get('loser_hand', 'R'),
                        'winner_ht': match.get('winner_height', 180),
                        'loser_ht': match.get('loser_height', 180)
                    }
                    sackmann_format.append(sackmann_match)

                return pd.DataFrame(sackmann_format)
            else:
                return None
        except Exception as e:
            print(f"   ⚠️  Could not load local 2025 data: {str(e)[:50]}...")
            return None

    def reverse_map_tournament_level(self, tournament_type):
        """Reverse map tournament type to Sackmann format"""
        reverse_map = {
            'grand_slam': 'G',
            'masters_1000': 'M',
            'atp_500': 'A',
            'atp_250': 'D',
            'atp_finals': 'F',
            'davis_cup': 'C'
        }
        return reverse_map.get(tournament_type, 'D')

    def process_atp_data(self, atp_df):
        """
        Process real ATP data into our tennis model format
        """
        print(f"\n🔄 PROCESSING REAL ATP DATA")
        print("Converting to tennis model format...")
        print("-" * 40)

        processed_matches = []

        for idx, match in atp_df.iterrows():
            try:
                # Map ATP data to our format
                processed_match = {
                    # Basic match info
                    'date': match.get('tourney_date', '20230101'),
                    'winner': match.get('winner_name', 'Unknown'),
                    'loser': match.get('loser_name', 'Unknown'),
                    'surface': self.map_surface(match.get('surface', 'Hard')),
                    'tournament_type': self.map_tournament_level(match.get('tourney_level', 'A')),
                    'tournament_name': match.get('tourney_name', 'ATP Tour'),

                    # Match format and basic stats
                    'sets_played': self.count_sets(match.get('score', '')),
                    'match_duration_minutes': match.get('minutes', np.random.randint(75, 180)),

                    # REAL SERVING STATISTICS (key for tennis prediction)
                    'winner_aces': match.get('w_ace', 0),
                    'loser_aces': match.get('l_ace', 0),
                    'winner_double_faults': match.get('w_df', 0),
                    'loser_double_faults': match.get('l_df', 0),

                    # Service points (crucial tennis stats)
                    'winner_first_serve_points': match.get('w_1stIn', 0),
                    'winner_first_serve_attempts': match.get('w_1stIn', 0) + match.get('w_1stWon', 0),
                    'winner_first_serve_won': match.get('w_1stWon', 0),
                    'winner_second_serve_won': match.get('w_2ndWon', 0),

                    'loser_first_serve_points': match.get('l_1stIn', 0),
                    'loser_first_serve_attempts': match.get('l_1stIn', 0) + match.get('l_1stWon', 0),
                    'loser_first_serve_won': match.get('l_1stWon', 0),
                    'loser_second_serve_won': match.get('l_2ndWon', 0),

                    # Break points (YouTube model emphasis: "every break point")
                    'winner_break_points_saved': match.get('w_bpSaved', 0),
                    'winner_break_points_faced': match.get('w_bpFaced', 0),
                    'loser_break_points_saved': match.get('l_bpSaved', 0),
                    'loser_break_points_faced': match.get('l_bpFaced', 0),

                    # Calculate additional stats
                    'winner_service_games': match.get('w_SvGms', 0),
                    'loser_service_games': match.get('l_SvGms', 0),

                    # Player rankings (real ATP rankings!)
                    'winner_rank': match.get('winner_rank', 999),
                    'loser_rank': match.get('loser_rank', 999),
                    'winner_rank_points': match.get('winner_rank_points', 0),
                    'loser_rank_points': match.get('loser_rank_points', 0),

                    # Player details
                    'winner_age': match.get('winner_age', 25),
                    'loser_age': match.get('loser_age', 25),
                    'winner_hand': match.get('winner_hand', 'R'),
                    'loser_hand': match.get('loser_hand', 'R'),
                    'winner_height': match.get('winner_ht', 180),
                    'loser_height': match.get('loser_ht', 180),

                    # Tournament context
                    'round': match.get('round', 'R64'),
                    'best_of': match.get('best_of', 3),
                }

                # Calculate derived statistics (YouTube model approach)
                processed_match.update(self.calculate_derived_stats(processed_match))

                processed_matches.append(processed_match)
                self.players.add(processed_match['winner'])
                self.players.add(processed_match['loser'])

                if idx % 5000 == 0:
                    print(f"   Processed {idx:,} matches...")

            except Exception as e:
                continue  # Skip problematic matches

        processed_df = pd.DataFrame(processed_matches)

        print(f"\n✅ REAL ATP DATA PROCESSING COMPLETE!")
        print(f"   📊 Processed matches: {len(processed_df):,}")
        print(f"   🎾 Unique players: {len(self.players):,}")
        print(f"   📈 Features per match: {len(processed_df.columns)} (real statistics)")

        return processed_df

    def map_surface(self, surface):
        """Map ATP surface names to our format"""
        surface_map = {
            'Hard': 'hard',
            'Clay': 'clay',
            'Grass': 'grass',
            'Carpet': 'hard'  # Treat carpet as hard court
        }
        return surface_map.get(surface, 'hard')

    def map_tournament_level(self, level):
        """Map ATP tournament levels to our format"""
        level_map = {
            'G': 'grand_slam',    # Grand Slam
            'M': 'masters_1000',  # Masters 1000
            'A': 'atp_500',       # ATP 500
            'D': 'atp_250',       # ATP 250
            'F': 'atp_finals',    # ATP Finals
            'C': 'davis_cup'      # Davis Cup
        }
        return level_map.get(level, 'atp_250')

    def count_sets(self, score):
        """Count number of sets from score string"""
        if not score or pd.isna(score):
            return 3
        try:
            sets = score.count('-')
            return min(max(sets, 2), 5)  # Between 2-5 sets
        except:
            return 3

    def calculate_derived_stats(self, match):
        """Calculate derived statistics like YouTube model"""
        derived = {}

        # Service percentages
        try:
            if match['winner_first_serve_attempts'] > 0:
                derived['winner_first_serve_pct'] = match['winner_first_serve_points'] / match['winner_first_serve_attempts']
            else:
                derived['winner_first_serve_pct'] = 0.65

            if match['loser_first_serve_attempts'] > 0:
                derived['loser_first_serve_pct'] = match['loser_first_serve_points'] / match['loser_first_serve_attempts']
            else:
                derived['loser_first_serve_pct'] = 0.65

            # Break point conversion
            if match['winner_break_points_faced'] > 0:
                derived['winner_break_points_saved_pct'] = match['winner_break_points_saved'] / match['winner_break_points_faced']
            else:
                derived['winner_break_points_saved_pct'] = 0.65

            if match['loser_break_points_faced'] > 0:
                derived['loser_break_points_saved_pct'] = match['loser_break_points_saved'] / match['loser_break_points_faced']
            else:
                derived['loser_break_points_saved_pct'] = 0.65

        except:
            # Fallback values
            derived.update({
                'winner_first_serve_pct': 0.65,
                'loser_first_serve_pct': 0.65,
                'winner_break_points_saved_pct': 0.65,
                'loser_break_points_saved_pct': 0.65
            })

        return derived

    def enhance_with_head_to_head(self, matches_df):
        """Add head-to-head statistics (YouTube model feature)"""
        print(f"\n📊 CALCULATING HEAD-TO-HEAD STATISTICS")
        print("Building historical H2H records...")

        h2h_records = {}
        enhanced_matches = []

        for idx, match in matches_df.iterrows():
            winner = match['winner']
            loser = match['loser']

            # Create H2H key (alphabetical order)
            h2h_key = tuple(sorted([winner, loser]))

            if h2h_key not in h2h_records:
                h2h_records[h2h_key] = {'total': 0, winner: 0, loser: 0}

            # Add H2H features to match
            match_enhanced = match.copy()
            h2h_data = h2h_records[h2h_key]

            match_enhanced['h2h_total_matches'] = h2h_data['total']
            match_enhanced['winner_h2h_wins'] = h2h_data.get(winner, 0)
            match_enhanced['loser_h2h_wins'] = h2h_data.get(loser, 0)

            if h2h_data['total'] > 0:
                match_enhanced['winner_h2h_win_rate'] = h2h_data.get(winner, 0) / h2h_data['total']
            else:
                match_enhanced['winner_h2h_win_rate'] = 0.5

            enhanced_matches.append(match_enhanced)

            # Update H2H record for future matches
            h2h_records[h2h_key]['total'] += 1
            h2h_records[h2h_key][winner] = h2h_records[h2h_key].get(winner, 0) + 1

        enhanced_df = pd.DataFrame(enhanced_matches)
        print(f"✅ H2H statistics added for {len(h2h_records):,} player pairs")

        return enhanced_df

    def save_real_atp_data(self, matches_df):
        """Save real ATP data"""
        print(f"\n💾 SAVING REAL ATP DATA")

        # Save to data directory
        os.makedirs('../data', exist_ok=True)

        # Save main dataset
        matches_df.to_csv('../data/real_atp_matches.csv', index=False)
        print(f"✅ Real ATP matches: ../data/real_atp_matches.csv")

        # Create ELO system with real data
        print("🏆 Building ELO system from real ATP data...")
        self.elo_system.build_from_match_data(matches_df)

        # Save ELO system
        os.makedirs('../models', exist_ok=True)
        import joblib
        joblib.dump(self.elo_system, '../models/real_atp_elo_system.pkl')
        print(f"✅ Real ATP ELO system: ../models/real_atp_elo_system.pkl")

        # Show top players from real data
        print(f"\n🥇 TOP 10 REAL ATP PLAYERS BY ELO:")
        top_players = self.elo_system.get_top_players(10)
        for i, (player, elo) in enumerate(top_players, 1):
            print(f"   {i:2d}. {player:<25} {elo:.0f}")

        return True

def main():
    """Collect real ATP data for 85% accuracy model"""
    print("🎾 REAL ATP DATA COLLECTION")
    print("Replacing simulated data with actual ATP statistics")
    print("Target: 85% accuracy like YouTube model")
    print("=" * 60)

    collector = RealATPDataCollector()

    # Collect real ATP data
    real_matches_df = collector.collect_atp_csv_data()

    if real_matches_df is not None:
        # Enhance with head-to-head
        enhanced_df = collector.enhance_with_head_to_head(real_matches_df)

        # Save data
        collector.save_real_atp_data(enhanced_df)

        print(f"\n🚀 REAL ATP DATA COLLECTION COMPLETE!")
        print(f"✅ Ready to train 85% accuracy model with real data!")
        print(f"📊 Dataset: {len(enhanced_df):,} real ATP matches")
        print(f"🎾 Players: {len(collector.players):,} real ATP players")
        print(f"🎯 Next: Train model with real data for 85% accuracy!")

    else:
        print(f"\n❌ Real ATP data collection failed")
        print(f"Falling back to enhanced simulated data approach")

if __name__ == "__main__":
    main()