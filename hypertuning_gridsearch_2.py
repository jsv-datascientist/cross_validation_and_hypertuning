from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_wine(return_X_y=True)

# Define parameter grid
param_grid = {
    'max_depth': [4, 6],
    'min_samples_split': [3, 7]
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Perform Grid Search with DecisionTreeClassifier, the parameter grid, cv=3, and scoring='accuracy'
model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)

#  Print the best parameters found in the Grid Search
print(f"Best parameters are {grid_search.best_params_}")
#  Predict on the test data and calculate final accuracy

best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy is {accuracy}")