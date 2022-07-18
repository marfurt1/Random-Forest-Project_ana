# Random forest project

In this project the Titanic dataset is analyzed and survival predictions are made using random forest classifier from sklearn.

First, a random forest with default hyperparameters is tried. Then, we try with GridSearchCV to choose the best hyperparametrs. In third place, we try with RandomizedSearchCV to choose the best hyperparameters. When we use GridSearchCV less options of hyperparameters are included because of time to run concerns.

The estimators selected with GridSearchCV and RandomizedSearchCV show similar performance in the test dataset.

We compare the performace of this models with XGBoost, this algorithm trains a bunch of decision trees in a sequential way. Where each individual model learns from mistakes made by the previous model. The performance of XGBoost with default hyperparamters is worst that the three random forests trained. The accuracy increases when we select the best hyperparameters by random grid search. In this case, the accuracy is similar to the random forest with hyperparameters selected by grid search, buy it is lower compared to the radonm forest with hyperparameters selected by random grid search.