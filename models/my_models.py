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

def train_model(X_train_scaled,y_train, scaler, train_cols):
    tree = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
    tree.fit(X_train_scaled, y_train)

    logreg = LogisticRegression(C=1, max_iter=10000, class_weight='balanced')
    logreg.fit(X_train_scaled, y_train)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000)
    mlp.fit(X_train_scaled, y_train)
    

    # 5. Evaluate
        # 6. Choose best model (example selects logistic regression)
    best_model = logreg

    # 7. Return everything needed for prediction
    return best_model, scaler, train_cols, tree, svm, mlp

