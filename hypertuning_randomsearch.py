from sklearn.datasets import load_wine
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the "cancer cake" recipe data
X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)

# Split the data into "training batter" and "testing batter"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the "ingredient amounts" for our logistic regression cake
param_distributions = {
    'C': [0.1, 0.5, 0.75, 1, 5, 10, 25, 50, 75, 100],
    'solver': ['liblinear', 'saga']
}

# Create and train our "random recipe picker"
#  adjust number of iterations
random_search = RandomizedSearchCV(LogisticRegression(max_iter=1000), param_distributions, n_iter=5, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Display the "best recipe"
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_}")