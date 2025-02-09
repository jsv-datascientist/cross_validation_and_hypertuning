from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#  Load the wine dataset
X, y = load_wine(return_X_y=True)

#  Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Create two ensemble models with different configurations
model1 = RandomForestClassifier()
model2 =  GradientBoostingClassifier()

#  Perform 5-fold cross-validation for each model
scores1 = cross_val_score(model1, X_train, y_train, cv=5)
scores2 = cross_val_score(model2, X_train, y_train, cv=5)


#  Print mean cross-validation scores for each model
print(f" Model1 {scores1} and {scores1.mean()}")
print(f" Model1 {scores2} and {scores2.mean()}")