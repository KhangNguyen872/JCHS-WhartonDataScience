import pandas as pd
from collections import defaultdict

def train_elo(df, k_factor=30, initial_rating=1500):
    """
    Trains an Elo rating system using the given DataFrame.
    - df: DataFrame containing game results
    - k_factor: The weight given to new match results
    - initial_rating: Starting Elo rating for all teams
    """
    # Initialize Elo ratings for all teams
    elo_ratings = defaultdict(lambda: initial_rating)

    # Group games by the same match (assuming two rows per game, one per team)
    grouped = df.groupby('game_id')  # Assuming there’s a game_id column

    for game_id, group in grouped:
        if len(group) != 2:
            continue  # Skip if data is incomplete

        # Identify home and away teams
        home_team = group[group["home_away"] == "home"]["team"].values[0]
        away_team = group[group["home_away"] == "away"]["team"].values[0]

        # Get final scores
        home_score = group[group["home_away"] == "home"]["score"].values[0]
        away_score = group[group["home_away"] == "away"]["score"].values[0]

        # Determine match outcome
        if home_score > away_score:
            winner, loser = home_team, away_team
        else:
            winner, loser = away_team, home_team

        # Calculate Elo update
        def expected_score(rating_a, rating_b):
            return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

        expected_winner = expected_score(elo_ratings[winner], elo_ratings[loser])
        elo_ratings[winner] += k_factor * (1 - expected_winner)
        elo_ratings[loser] += k_factor * (0 - (1 - expected_winner))

    return dict(sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True))

def main(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # Clean column names

    elo_ratings = train_elo(df)
    
    print("Final Team Rankings:")
    for rank, (team, rating) in enumerate(elo_ratings.items(), start=1):
        print(f"{rank}. {team}: {rating:.2f}")

csv_file = "games_2022.csv"  # Replace with your actual CSV filename
main(csv_file)
