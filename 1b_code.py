import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


def load_data():
    """Load the datasets."""
    east_games = pd.read_csv("East Regional games to predict.csv")
    games_2022 = pd.read_csv("games_2022.csv")
    return east_games, games_2022


def pair_games(games_df):
    """Pair teams and their stats for each game."""
    paired_games = games_df.groupby("game_id").agg({
        "team": list,
        "team_score": list,
        "opponent_team_score": list,
        "FTA": list,
        "FGA_2": list,
        "FGA_3": list,
        "FTM": list,
        "FGM_3": list,
        "OREB": list,
        "DREB": list,
        "TOV": list,
        "rest_days": list,
        "travel_dist": list,
    }).reset_index()

    paired_data = []
    for _, row in paired_games.iterrows():
        teams = row["team"]
        scores = row["team_score"]
        opponent_scores = row["opponent_team_score"]
        ftas = row["FTA"]
        fga_2s = row["FGA_2"]
        fga_3s = row["FGA_3"]
        ftms = row["FTM"]
        fgm_3s = row["FGM_3"]
        orebs = row["OREB"]
        drebs = row["DREB"]
        tovs = row["TOV"]
        rest_days = row["rest_days"]
        travel_dists = row["travel_dist"]

        if len(teams) == 2:  # Ensure there are exactly two teams per game
            paired_data.append({
                "game_id": row["game_id"],
                "team_1": teams[0],
                "team_2": teams[1],
                "team_1_score": scores[0],
                "team_2_score": opponent_scores[0],
                "FTA_1": ftas[0],
                "FGA_2_1": fga_2s[0],
                "FGA_3_1": fga_3s[0],
                "FTM_1": ftms[0],
                "FGM_3_1": fgm_3s[0],
                "OREB_1": orebs[0],
                "DREB_1": drebs[0],
                "TOV_1": tovs[0],
                "rest_days_1": rest_days[0],
                "travel_dist_1": travel_dists[0],
                "FTA_2": ftas[1],
                "FGA_2_2": fga_2s[1],
                "FGA_3_2": fga_3s[1],
                "FTM_2": ftms[1],
                "FGM_3_2": fgm_3s[1],
                "OREB_2": orebs[1],
                "DREB_2": drebs[1],
                "TOV_2": tovs[1],
                "rest_days_2": rest_days[1],
                "travel_dist_2": travel_dists[1],
            })

    return pd.DataFrame(paired_data)


def standardize_team_names(paired_df):
    """Standardize team names using a mapping."""
    team_name_mapping = {
        "rhode_island_rams": "rhode_island_rams",
        "nc_state_wolfpack": "nc_state_wolfpack",
        "liberty_flames": "liberty_flames",
        "drexel_dragons": "drexel_dragons",
        "massachusetts_minutewomen": "massachusetts_minutewomen",
        "buffalo_bulls": "buffalo_bulls",
        "fairfield_stags": "fairfield_stags",
        "uconn_huskies": "uconn_huskies",
        "american_university_eagles": "american_university_eagles",
    }
    paired_df["team_1"] = paired_df["team_1"].map(team_name_mapping).fillna(paired_df["team_1"])
    paired_df["team_2"] = paired_df["team_2"].map(team_name_mapping).fillna(paired_df["team_2"])
    return paired_df


def calculate_team_metrics(games_df):
    """Calculate performance metrics for each team."""
    team_metrics = {}
    for team, data in games_df.groupby("team_1"):
        total_FTA = data["FTA_1"].sum()
        total_FGA_2 = data["FGA_2_1"].sum()
        total_FGA_3 = data["FGA_3_1"].sum()

        metrics = {
            "offensive_efficiency": data["team_1_score"].sum() / total_FGA_2 if total_FGA_2 != 0 else 0,
            "defensive_efficiency": data["team_2_score"].sum() / total_FGA_2 if total_FGA_2 != 0 else 0,
            "rebounding_rate": (data["OREB_1"].sum() + data["DREB_1"].sum()) / total_FGA_2 if total_FGA_2 != 0 else 0,
            "turnover_rate": data["TOV_1"].sum() / total_FGA_2 if total_FGA_2 != 0 else 0,
            "free_throw_percentage": data["FTM_1"].sum() / total_FTA if total_FTA != 0 else 0,
            "three_point_percentage": data["FGM_3_1"].sum() / total_FGA_3 if total_FGA_3 != 0 else 0,
            "avg_rest_days": data["rest_days_1"].mean(),
            "avg_travel_dist": data["travel_dist_1"].mean(),
        }
        team_metrics[team] = metrics
    return team_metrics


def prepare_training_data(paired_df, team_metrics):
    """Prepare training data (X, y) for the model."""
    X = []
    y = []
    for _, row in paired_df.iterrows():
        home_team = row["team_1"]
        away_team = row["team_2"]
        if home_team in team_metrics and away_team in team_metrics:
            home_features = list(team_metrics[home_team].values())
            away_features = list(team_metrics[away_team].values())
            features = home_features + away_features
            X.append(features)
            y.append(1 if row["team_1_score"] > row["team_2_score"] else 0)
    return np.array(X), np.array(y)


def train_model(X, y):
    """Train a logistic regression model and return accuracy."""
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, imputer, scaler, accuracy


def predict_games(east_games, team_metrics, model, imputer, scaler):
    """Predict winning probabilities for the East Regional games."""
    predictions = []
    for _, row in east_games.iterrows():
        home_team = row["team_home"]
        away_team = row["team_away"]
        if home_team in team_metrics and away_team in team_metrics:
            home_features = list(team_metrics[home_team].values())
            away_features = list(team_metrics[away_team].values())
            features = home_features + away_features
            features = imputer.transform([features])
            features = scaler.transform(features)
            prob = model.predict_proba(features)[0][1]  # Probability of home team winning
            predictions.append(prob)
        else:
            predictions.append(0.5)  # Default probability if team data is missing
    return predictions


def main():
    """Main function to execute the workflow."""
    # Load data
    east_games, games_2022 = load_data()

    # Pair games and standardize team names
    paired_df = pair_games(games_2022)
    paired_df = standardize_team_names(paired_df)

    # Calculate team metrics
    team_metrics = calculate_team_metrics(paired_df)

    # Prepare training data
    X, y = prepare_training_data(paired_df, team_metrics)

    # Train the model and get accuracy
    model, imputer, scaler, accuracy = train_model(X, y)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Predict winning probabilities
    predictions = predict_games(east_games, team_metrics, model, imputer, scaler)

    # Add predictions to the East Regional games dataframe
    east_games["WINNING %"] = predictions

    # Save predictions to a new CSV file
    east_games.to_csv("East_Regional_Predictions.csv", index=False)
    print("Predictions saved to East_Regional_Predictions.csv")


if __name__ == "__main__":
    main()