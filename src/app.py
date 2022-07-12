## Random forest for titanic dataset

# Import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
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


## Read data
url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df_raw = pd.read_csv(url)

## Copy data
df = df_raw.copy()

## Check duplicates
df_duplicates = df['PassengerId'].duplicated().sum()
print(f'It seems that there are {df_duplicates} duplicated passenger according to the PassengerId feature.')

## Drop irrelevant columns
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
df = df.drop(drop_cols, axis = 1)


## Fill missing values

# Fill missing values of age with mean by sex
#  
df.loc[df['Sex']=='female','Age']=df[(df['Sex']=='female')].fillna(df.loc[df['Sex']=='female','Age'].mean())

df.loc[df['Sex']=='male','Age']=df[(df['Sex']=='male')].fillna(df.loc[df['Sex']=='male','Age'].mean())

# Fill missing values of Embarked with mode

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


## Encoding categorical variables

# Change type of variables 

df[['Sex','Embarked']]=df[['Sex','Embarked']].astype('category')

# Encoding

df['Sex']=df['Sex'].cat.codes

df['Embarked']=df['Embarked'].cat.codes


## Start modeling

# Train, test datasets

X = df.drop('Survived', axis=1)

y= df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1107)


## 1. Random forest with default hyperparameters

rfc1 = RandomForestClassifier(random_state=1107)

rfc1.fit(X_train, y_train)

y_pred = rfc1.predict(X_test)

print('Accuracy Random Forest with default parameters in test set:',rfc1.score(X_test, y_test))


## 2. Optimize model hyperparameters to select the better forest

# 2.1 Grid Search CV (tries with all the possible combinations)

param_grid = [{'max_depth': [8, 12, 16], 
         'min_samples_split': [12, 16, 20], 
         'criterion': ['gini', 'entropy']}]

rfc2=RandomForestClassifier(random_state=1107)

grid=GridSearchCV(estimator=rfc2,param_grid=param_grid, cv=5, n_jobs=-1,verbose=2)

grid.fit(X_train,y_train)

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
filename = '../models/final_model.sav'
pickle.dump(model_cv_2, open(filename, 'wb'))