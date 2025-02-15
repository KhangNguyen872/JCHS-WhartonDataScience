import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Elo rating system parameters
BASE_ELO = 1500  # Default starting Elo for all teams
K_FACTOR = 30  # How much the rating adjusts per game
HOME_ADVANTAGE = 100  # Additional Elo points for home team

# Neural network parameters
EPOCHS = 50
BATCH_SIZE = 32

def expected_win_probability(team_elo, opponent_elo):
    """Calculates the expected probability of a team winning based on Elo ratings."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

def update_elo(winner_elo, loser_elo, k=K_FACTOR):
    """Updates Elo ratings after a game."""
    expected_winner = expected_win_probability(winner_elo, loser_elo)
    expected_loser = expected_win_probability(loser_elo, winner_elo)

    new_winner_elo = winner_elo + k * (1 - expected_winner)
    new_loser_elo = loser_elo + k * (0 - expected_loser)

    return new_winner_elo, new_loser_elo

def train_elo(df):
    """Processes game results and applies Elo rating system to rank teams."""
    elo_ratings = {}  # Dictionary to store each team's current Elo rating

    for _, row in df.iterrows():
        team = row["team"]
        game_id = row["game_id"]
        home_away = row["home_away"]
        team_score = row["team_score"]
        opponent_score = row["opponent_team_score"]

        # Find the opponent for this game
        opponent_row = df[(df["game_id"] == game_id) & (df["team"] != team)]
        if not opponent_row.empty:
            opponent = opponent_row.iloc[0]["team"]
        else:
            continue  # Skip if no opponent is found

        # Initialize Elo ratings if new team
        if team not in elo_ratings:
            elo_ratings[team] = BASE_ELO
        if opponent not in elo_ratings:
            elo_ratings[opponent] = BASE_ELO

        # Adjust Elo for home advantage
        if home_away == 'home':
            team_elo = elo_ratings[team] + HOME_ADVANTAGE
            opponent_elo = elo_ratings[opponent]
        else:
            team_elo = elo_ratings[team]
            opponent_elo = elo_ratings[opponent] + HOME_ADVANTAGE

        # Determine winner and loser
        if team_score > opponent_score:
            winner, loser = team, opponent
            winner_elo, loser_elo = team_elo, opponent_elo
        else:
            winner, loser = opponent, team
            winner_elo, loser_elo = opponent_elo, team_elo

        # Update Elo ratings
        elo_ratings[winner], elo_ratings[loser] = update_elo(winner_elo, loser_elo)

    return elo_ratings

def prepare_data(df, elo_ratings):
    """Prepares data for neural network training."""
    X = []
    y = []

    for _, row in df.iterrows():
        team = row["team"]
        game_id = row["game_id"]
        home_away = row["home_away"]
        team_score = row["team_score"]
        opponent_score = row["opponent_team_score"]

        # Find the opponent for this game
        opponent_row = df[(df["game_id"] == game_id) & (df["team"] != team)]
        if not opponent_row.empty:
            opponent = opponent_row.iloc[0]["team"]
        else:
            continue  # Skip if no opponent is found

        # Get Elo ratings
        team_elo = elo_ratings.get(team, BASE_ELO)
        opponent_elo = elo_ratings.get(opponent, BASE_ELO)

        # Add home advantage
        if home_away == 'home':
            team_elo += HOME_ADVANTAGE
        else:
            opponent_elo += HOME_ADVANTAGE

        # Features: Team Elo, Opponent Elo, Home/Away (encoded as 1 for home, 0 for away)
        X.append([team_elo, opponent_elo, 1 if home_away == 'home' else 0])

        # Label: 1 if team wins, 0 if team loses
        y.append(1 if team_score > opponent_score else 0)

    return np.array(X), np.array(y)

def build_neural_network(input_shape):
    """Builds a simple neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),  # Input layer
        Dense(32, activation='relu'),                            # Hidden layer
        Dense(1, activation='sigmoid')                            # Output layer
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(csv_file):
    """Loads CSV file, processes the data, and ranks teams using Elo ratings and neural network."""
    df = pd.read_csv(csv_file)

    # Train Elo rankings
    elo_ratings = train_elo(df)

    # Prepare data for neural network
    X, y = prepare_data(df, elo_ratings)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train neural network
    model = build_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Evaluate neural network
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nNeural Network Test Accuracy: {accuracy:.2f}")

    # Use neural network to predict win probabilities
    predictions = model.predict(X_test)

    # Print top teams based on Elo ratings
    sorted_teams = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print("\n🏀 Final Team Rankings (Elo Scores):")
    for rank, (team, rating) in enumerate(sorted_teams, start=1):
        print(f"{rank}. {team} - Elo: {rating:.2f}")

# Run program
if __name__ == "__main__":
    csv_file = "games_2022.csv"  # Replace with your CSV file name
    main(csv_file)