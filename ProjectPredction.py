import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('./African projects Dataset.csv')

# count total NaN values across all columns
print('empty values', df.isna().sum().sum())

# drop empty values
df.dropna(inplace=True)

# one-hot method to encode the categorical variables
df = pd.get_dummies(df, columns=["regionname", "countryname", "lendinginstr"])

X = df.drop(['project result'], axis=1)
y = df['project result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the test set results and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# List of columns
print(X_train.columns)

# Save the model
joblib.dump(clf, 'decision_tree.joblib')

# Predict a new data point
new_data_point = [[80000000, 0, 0, 0, 0, 0, 1, 0, 0, 0, #1
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #2
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #3
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0,       #4
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #5
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #6
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #7
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,       #8
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       #9
                  0, 0, 0]]                           #10
clf = joblib.load('decision_tree.joblib')
prediction = clf.predict(new_data_point)
print("Prediction:", prediction)
