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

def prepare_data(matchups):
     # 1. Define features and target
    drop_cols = [
        'posteam_A','posteam_B','Season_A','Season_B',
        'index_A','index_B','Win_B','Win_A','GameID','Touchdown','Touchdown_B', 'win', 'win_B'
    ]
    X = matchups.drop(columns=drop_cols)
    y = matchups['Win_A']
    train_cols = X.columns.tolist()   # keep column order
    

    # 2. Train/test split BEFORE scaling (correct!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Fit scaler on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_test_scaled, X_train_scaled, y_test, y_train, scaler, train_cols