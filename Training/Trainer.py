import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("data/nfl_cleaned.csv")

# Feature setup
df['target'] = (df['yards_gained'] > 5).astype(int)
df = pd.get_dummies(df, columns=['play_type', 'posteam', 'defteam'], drop_first=True)

X = df.drop(columns=['target'])
y = df['target']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc:.3f}")
