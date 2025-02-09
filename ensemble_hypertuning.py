from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Load the wine dataset and split into training and test sets
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base estimator and AdaBoost classifier
base_estimator = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME')

# Define the parameter grid for GridSearch
param_grid = {'n_estimators': [10, 50], 
            'learning_rate': [0.01, 0.1], 
            'estimator__max_depth': [1, 2],
             'estimator__min_samples_split' : [5, 8],
            'estimator__min_samples_leaf' : [2, 4, 6]
            }

# Perform GridSearch with cross-validation
grid_search = GridSearchCV(ada_clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearch
print(f"Best parameters for AdaBoost: {grid_search.best_params_}")