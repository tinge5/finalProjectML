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

def evaluation(logreg, tree, svm, mlp, X_test_scaled, y_test):
    
    t = 0
    for i in range(10):
        f = accuracy_score(y_test, mlp.predict(X_test_scaled))
        if f > t:
            t = f
    y_pred_tree = tree.predict(X_test_scaled)
    y_pred_logreg = logreg.predict(X_test_scaled)

    results = []
    results.append(("Decision Tree accuracy", accuracy_score(y_test, y_pred_tree)))
    results.append(("LogReg accuracy:", accuracy_score(y_test, y_pred_logreg)))
    results.append(("SVM accuracy:", accuracy_score(y_test, svm.predict(X_test_scaled))))
    results.append(("Best MLP accuracy out of 10 tries:", t))

    models = {'Decision Tree': tree, 'LogReg': logreg, 'SVM': svm, 'MLP': mlp}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        a_win_rate = np.mean(y_pred)
        b_win_rate = 1 - a_win_rate
        print(f"{name} predicts Team A wins {a_win_rate:.2%}, Team B wins {b_win_rate:.2%}")

    return results, y_pred_tree, y_pred_logreg