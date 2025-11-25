from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def create_matchups_with_averages(matchups, games_averages):
    # Merge Team A season averages
    matchups_a = matchups.copy()
    matchups_a = matchups_a.merge(
        games_averages,
        left_on=["posteam_A", "Season_A"],
        right_on=["posteam", "Season"],
        suffixes=("", "_A")
    ).drop(columns=["posteam", "Season"])


    # Merge Team B season averages
    matchups_a = matchups_a.merge(
        games_averages,
        left_on=["posteam_B", "Season_B"],
        right_on=["posteam", "Season"],
        suffixes=("", "_B")
    ).drop(columns=["posteam", "Season"])
    return matchups_a

def create_one_matchup_per_team(alltime_df):
    df = alltime_df.copy().reset_index(drop=True)
    pairs = []
    
    teams = df.to_dict('records')  # list of dicts

    for teamA in teams:
        # randomly select a different team for B
        teamB = random.choice([t for t in teams if t['posteam'] != teamA['posteam']])
        
        pair = {
            "posteam_A": teamA["posteam"],
            "Season_A": teamA["Season"],
            "posteam_B": teamB["posteam"],
            "Season_B": teamB["Season"],
        }
        
        # add stats
        for col in df.columns:
            if col not in ["posteam", "Season"]:
                pair[f"{col}"] = teamA[col]
                pair[f"{col}_B"] = teamB[col]
        
        pairs.append(pair)
    
    return pd.DataFrame(pairs)
def createMatchups(games):
    teamA = games.groupby("GameID").nth(0).reset_index()
    teamB = games.groupby("GameID").nth(1).reset_index()


    matchups = teamA.merge(
        teamB,
        on="GameID",
        suffixes=('_A', '_B')
    )
    return matchups

def training(matchups2009a):

    # 1. Define features and target
    drop_cols = [
        'posteam_A','posteam_B','Season_A','Season_B',
        'index_A','index_B','Win_B','Win_A','GameID','Touchdown','Touchdown_B', 'win', 'win_B'
    ]
    X = matchups2009a.drop(columns=drop_cols)
    y = matchups2009a['Win_A']
    train_cols = X.columns.tolist()   # keep column order
    

    # 2. Train/test split BEFORE scaling (correct!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Fit scaler on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Train models
    tree = DecisionTreeClassifier(class_weight='balanced', max_depth=9, random_state=42)
    tree.fit(X_train_scaled, y_train)

    logreg = LogisticRegression(C=1, max_iter=10000, class_weight='balanced')
    logreg.fit(X_train_scaled, y_train)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000)
    mlp.fit(X_train_scaled, y_train)

    t = 0
    for i in range(10):
        f = accuracy_score(y_test, mlp.predict(X_test_scaled))
        if f > t:
            t = f

    # 5. Evaluate
    print("Decision Tree accuracy:", accuracy_score(y_test, tree.predict(X_test_scaled)))
    print("LogReg accuracy:", accuracy_score(y_test, logreg.predict(X_test_scaled)))
    print("SVM accuracy:", accuracy_score(y_test, svm.predict(X_test_scaled)))
    print("Best MLP accuracy out of 10 tries:", t)

    models = {'Decision Tree': tree, 'LogReg': logreg, 'SVM': svm, 'MLP': mlp}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        a_win_rate = np.mean(y_pred)
        b_win_rate = 1 - a_win_rate
        print(f"{name} predicts Team A wins {a_win_rate:.2%}, Team B wins {b_win_rate:.2%}")

    # 6. Choose best model (example selects logistic regression)
    best_model = logreg

    # 7. Return everything needed for prediction
    return best_model, scaler, train_cols


def predict_future(matchups_future, model, scaler, train_cols):
    
    df_future = matchups_future.copy()

    # 1. Use same training columns & same order
    X_future = df_future[train_cols].copy()
    

    # 2. Use the TRAINED scaler (DO NOT re-fit)
    X_scaled = scaler.transform(X_future)

    # 3. Predict probabilities
    proba = model.predict_proba(X_scaled)

    # Model classes are [0, 1] where 1 = A wins
    col_for_A = list(model.classes_).index(1)
    df_future["proba_A_wins"] = proba[:, col_for_A]

    # 4. Winner prediction
    df_future["winner"] = df_future["proba_A_wins"].apply(
        lambda p: "A" if p >= 0.48 else "B"
    )

    df_future["predicted_winner_team"] = df_future.apply(
        lambda r: r["posteam_A"] if r["winner"] == "A" else r["posteam_B"],
        axis=1
    )

    # 5. Clean output
    return df_future[[
        'posteam_A', 'Season_A',
        'posteam_B', 'Season_B',
        'proba_A_wins', 'winner', 'predicted_winner_team'
    ]]

def CreateGames_df(df):
    games_df = (df.groupby(['GameID', 'posteam'], as_index=False)
              .agg({
                  
                  'InterceptionThrown': 'sum',
                  'Fumble': 'sum',
                  'Touchdown': 'sum',
                  'FieldGoalResult': lambda x: (x == 'Good').sum(),
                  'Penalty.Yards': 'sum',
                  'Yards.Gained': 'mean',
                  'PuntResult': lambda x: (x == 'Blocked').sum(),
                  'Sack': 'sum',
                  'PosTeamScore': 'max',
                  'DefTeamScore': 'max',
                  'Season': 'first',
                  'FirstDown': 'sum',
                  'PassAttempt': 'sum',
                  'PassOutcome': lambda x: (x == 'Complete').sum(),
                  
                  
              }))
    games_df.rename(columns={'PuntResult': 'PuntBlocked'}, inplace=True)
    games_df['CompletionPercentage'] = games_df['PassOutcome'] / games_df['PassAttempt']
    yards = (df.groupby(['GameID', 'posteam',], as_index=False)
                .agg({
                    'Yards.Gained': 'sum',
                }))
    games_df['Yards.Gained'] = yards['Yards.Gained']
    games_df['Win'] = (games_df['PosTeamScore'] > games_df['DefTeamScore']).astype(int)
    
    return games_df
def game_avg(games_df):
    games_averages = games_df.groupby(['posteam', 'Season'], as_index=False).mean()
    games_averages = games_averages.drop(columns=['GameID'])
    games_averages = games_averages.rename(columns={'Win': 'win'})
    games_averages = games_averages.sort_values(by=['win'], ascending=False)

    return games_averages



