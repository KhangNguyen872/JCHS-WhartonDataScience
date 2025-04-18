import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import shutil

# Clear previous tuner results
shutil.rmtree('tuner_results', ignore_errors=True)
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

def train_elo(df, all_teams):
    """Processes game results and applies Elo rating system to rank teams."""
    elo_ratings = {team: BASE_ELO for team in all_teams}  # Initialize Elo for all teams

    # Iterate over unique game IDs
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id]
        if len(game_df) != 2:
            continue  # Skip if the game doesn't have exactly two teams

        team_row = game_df.iloc[0]
        opponent_row = game_df.iloc[1]

        team = team_row["team"]
        opponent = opponent_row["team"]
        home_away = team_row["home_away"]
        team_score = team_row["team_score"]
        opponent_score = opponent_row["team_score"]

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
    teams = []  # To store team information for group-based splitting

    # Iterate over unique game IDs
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id]
        if len(game_df) != 2:
            continue  # Skip if the game doesn't have exactly two teams

        team_row = game_df.iloc[0]
        opponent_row = game_df.iloc[1]

        team = team_row["team"]
        opponent = opponent_row["team"]
        home_away = team_row["home_away"]
        team_score = team_row["team_score"]
        opponent_score = opponent_row["team_score"]

        # Get Elo ratings
        team_elo = elo_ratings.get(team, BASE_ELO)
        opponent_elo = elo_ratings.get(opponent, BASE_ELO)

        # Add home advantage
        if home_away == 'home':
            team_elo += HOME_ADVANTAGE
        else:
            opponent_elo += HOME_ADVANTAGE

        # Features: Team Elo, Opponent Elo, Home/Away, and additional statistics
        features = [
            team_elo,
            opponent_elo,
            1 if home_away == 'home' else 0,  # Home/Away
            team_row["FGA_2"],  # 2-point field goal attempts
            team_row["FGM_2"],  # 2-point field goals made
            team_row["FGA_3"],  # 3-point field goal attempts
            team_row["FGM_3"],  # 3-point field goals made
            team_row["BLK"],    # Blocks
            team_row["STL"],    # Steals
            team_row["TOV"],    # Turnovers
            team_row["AST"],    # Assists
            team_row["OREB"],   # Offensive rebounds
            team_row["DREB"],   # Defensive rebounds
            team_row["FTA"],    # Free throw attempts
            team_row["FTM"],    # Free throws made
        ]
        X.append(features)

        # Label: 1 if team wins, 0 if team loses
        y.append(1 if (team_score > opponent_score) else 0)

        # Store team information for group-based splitting
        teams.append(team)

    return np.array(X), np.array(y), np.array(teams)

def build_model(hp):
    """Builds a neural network model with hyperparameter tuning."""
    model = Sequential()

    # Tune the number of layers and units
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                      activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Tune the learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(csv_file, region_csv_file):
    """Loads CSV file, processes the data, and ranks teams using Elo ratings and neural network."""
    df = pd.read_csv(csv_file)
    region_df = pd.read_csv(region_csv_file)

    # Merge the region information with the main dataframe
    df = df.merge(region_df, on="team")

    # Get the list of all teams
    all_teams = region_df["team"].unique()

    # Split data into train and test sets using GroupShuffleSplit
    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(group_splitter.split(df, groups=df["team"]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Train Elo rankings on the training set only
    elo_ratings = train_elo(train_df, all_teams)

    # Prepare data for neural network using the trained Elo ratings
    X_train, y_train, train_teams = prepare_data(train_df, elo_ratings)
    X_test, y_test, test_teams = prepare_data(test_df, elo_ratings)

    # Verify no overlapping teams between train and test sets
    train_teams_set = set(train_teams)
    test_teams_set = set(test_teams)
    overlap = train_teams_set.intersection(test_teams_set)
    if overlap:
        print(f"Warning: {len(overlap)} teams appear in both train and test sets: {overlap}")
    else:
        print("No teams overlap between train and test sets.")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning with Keras Tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,  # Increase executions for better results
        directory='tuner_results',
        project_name='team_ranking'
    )

    # Perform hyperparameter search
    tuner.search(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the best model
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"\nNeural Network Test Accuracy: {accuracy:.2f}")

    # Print top teams based on Elo ratings within each region
    regions = region_df["region"].unique()
    for region in regions:
        region_teams = region_df[region_df["region"] == region]["team"]
        region_elo_ratings = {team: elo_ratings[team] for team in region_teams if team in elo_ratings}
        sorted_teams = sorted(region_elo_ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏀 Final Team Rankings for {region} (Elo Scores):")
        for rank, (team, rating) in enumerate(sorted_teams, start=1):
            print(f"{rank}. {team} - Elo: {rating:.2f}")

# Run program
if (__name__ == "__main__"):
    csv_file = "games_2022.csv"
    region_csv_file = "Team Region Groups.csv"  # Replace with your region CSV file name
    main(csv_file, region_csv_file)