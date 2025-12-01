import pandas as pd
import warnings
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prepare_Data import prepare_data
from Training.eval import evaluation
from models.my_models import train_model
from utils.helperfunctions import create_matchups_with_averages, createMatchups, CreateGames_df, create_one_matchup_per_team, game_avg, predict_future

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # folder containing results.py
csv_path = os.path.join(BASE_DIR, "NFLPlaybyPlay.csv")


warnings.filterwarnings("ignore")

df = pd.read_csv(csv_path)

games_df = CreateGames_df(df)
games = games_df[['GameID', 'posteam','Win', 'Season']]
games_averages = game_avg(games_df)
matchups = createMatchups(games)
half_matchups = matchups[matchups['Season_A'] <= 2011]
half_avg = games_averages[games_averages['Season'] <= 2011]
matchupsfull = create_matchups_with_averages(matchups, games_averages)
BestTeams = games_averages.head(10)
Bestmatchup = create_one_matchup_per_team(BestTeams)
X_test_scaled, X_train_scaled, y_test, y_train, scaler, train_cols = prepare_data(matchupsfull)
best_model, scaler, train_cols, tree, svm, mlp = train_model(X_train_scaled,y_train, scaler, train_cols)
results, y_pred_tree, y_pred_logreg = evaluation(best_model, tree, svm, mlp, X_test_scaled, y_test)

prediction = predict_future(Bestmatchup, best_model, scaler, train_cols)



