# Random forest project

In this project the Titanic dataset is analyzed and survival predictions are made using random forest classifier from sklearn.

First, a random forest with default hyperparameters is tried. Then, we try with GridSearchCV to choose the best hyperparametrs. In third place, we try with RandomizedSearchCV to choose the best hyperparameters. When we use GridSearchCV less options of hyperparameters are included because of time to run concerns.

The estimators selected with GridSearchCV and RandomizedSearchCV show similar performance in the test dataset.