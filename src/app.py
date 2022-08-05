## Random forest for titanic dataset

# Import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

## Read data
url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df_raw = pd.read_csv(url)


## Check duplicates
df_duplicates = df['PassengerId'].duplicated().sum()
print(f'It seems that there are {df_duplicates} duplicated passenger according to the PassengerId feature.')

## Drop irrelevant columns

df_transf = df_raw.copy()
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
df_transf = df_transf.drop(drop_cols, axis = 1)

df = df_transf.copy()


# Split the dataset so to avoid bias

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)


## Process data before modelling

# Fill missing values for Age and Embarked

# Nan values of Age will be replaced by the mean
# Nan values of Embarked will be replaced by the mode

imputer_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer_mean = imputer_mean.fit(X_train[['Age']])
X_train['Age'] = imputer_mean.transform(X_train[['Age']])


imputer_mode = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer_mode = imputer_mode.fit(X_train[['Embarked']])
X_train['Embarked'] = imputer_mode.transform(X_train[['Embarked']])


# Tranform the X_test with mean (of age) and mode (of embarked) from the train data

X_test['Age'] = imputer_mean.transform(X_test[['Age']])

X_test['Embarked'] = imputer_mode.transform(X_test[['Embarked']])


# Encode categorical variables: Age and Embarked

X_train[['Sex','Embarked']]=X_train[['Sex','Embarked']].astype('category')
X_test[['Sex','Embarked']]=X_test[['Sex','Embarked']].astype('category')


X_train['Sex']=X_train['Sex'].cat.codes
X_train['Embarked']=X_train['Embarked'].cat.codes

X_test['Sex']=X_test['Sex'].cat.codes
X_test['Embarked']=X_test['Embarked'].cat.codes




## Start modeling

# Train, test datasets


## 1. Random forest with default hyperparameters

rfc = RandomForestClassifier(random_state=1107)

rfc.fit(X_train, y_train)

print(f'Accuracy in train dataset: {rfc.score(X_train, y_train)}')

print(f'Accuracy in test dataset: {rfc.score(X_test, y_test)}')


## 2. Optimize model hyperparameters to select the better forest

# 2.1 Grid Search CV (tries with all the possible combinations)

param_grid = [{'max_depth': [8, 12, 16], 
         'min_samples_split': [12, 16, 20], 
         'criterion': ['gini', 'entropy']}]

# GridSearchCV 

rfc2=RandomForestClassifier(random_state=1107)

grid =  GridSearchCV(estimator=rfc2, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid.fit(X_train, y_train)

# save best estimator into model_cv
model_cv=grid.best_estimator_

print('Accuracy of random forest selected by CV in test set (grid search):',grid.score(X_test, y_test))





# 2.2 Randomized search CV
# In this case, not all parameters values are tried out. This allows us to include more hyperparameters in the grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Criterio
criterion=['gini','entropy']

# Create the random grid
random_grid = {'n_estimators': n_estimators,
#'max_features': max_features, # Son muy pocas variables por lo cual no vale la pena aplicarlo
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap,
'criterion':criterion}

rfc3=RandomForestClassifier(random_state=1107)

grid_random=RandomizedSearchCV(estimator=rfc3,n_iter=100,cv=5,random_state=1107,param_distributions=random_grid)
# n_iter: number of parameter settings that are sampled

grid_random.fit(X_train,y_train)

# Save best estimator
model_cv_2 = grid_random.best_estimator_

# Accuracy in test data 
print('Accuracy of random forest selected by CV in test set (random grid search):',grid_random.score(X_test, y_test))


# Save best estimator in models folder for future data
filename = '../models/modelo_random_forest.sav'
pickle.dump(model_cv_2, open(filename, 'wb'))


## 3. XGBoost

# 3.1 XGBoost with default hyperparameters

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_xgb_pred = xgb.predict(X_test)
print('Accuracy of xgboost in test set:',accuracy_score(y_test, y_xgb_pred))


# 3.2 XGBoost with hyperparameters selected by random grid search

xgb_2 = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid_xgb = RandomizedSearchCV(xgb_2,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

xgb_2 = grid_xgb.best_estimator_
y_pred_xgb_2 = xgb_2.predict(X_test)
print('Accuracy of xgboost with hyperparameters selected by random grid search in test set:',accuracy_score(y_test, y_pred_xgb_2))