from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Create a Decision Tree Regressor model
dt_reg = DecisionTreeRegressor()

# Set up the hyperparameter grid for tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
}

#  Set up Grid Search with cross-validation
#  Perform Grid Search to find the best hyperparametersy
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X,y)

# Get the best model
best_model = grid_search.best_estimator_

# Perform cross-validation on the best model
scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-scores}")
print(f"Mean cross-validation MSE score: {-scores.mean():.2f}")