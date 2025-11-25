import pandas as pd
from Training.Trainer import CreateGames_df, createMatchups, create_matchups_with_averages, create_one_matchup_per_team, training, predict_future, game_avg

if __name__ == "__main__":
    df = pd.read_csv("NFLPlaybyPlay.csv")

    games_df = CreateGames_df(df)
    games = games_df[['GameID', 'posteam','Win', 'Season']]
    games_averages = game_avg(games_df)
    matchups = createMatchups(games)
    half_matchups = matchups[matchups['Season_A'] <= 2011]
    half_avg = games_averages[games_averages['Season'] <= 2011]
    matchupsHalf = create_matchups_with_averages(matchups, games_averages)
    BestTeams = games_averages.head(10)
    Bestmatchup = create_one_matchup_per_team(BestTeams)
    m1, scaler, cols = training(matchupsHalf)
    prediction = predict_future(Bestmatchup, m1, scaler, cols)


    print(prediction)
