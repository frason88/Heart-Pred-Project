import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('heart.csv')
print(df.head())


print(df.shape)

X= df.drop('target', axis= 1)
y= df.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10, stratify= y)


#++++++++++++++++++++++ Random Forest ++++++++++++++++++++++

from sklearn.ensemble import RandomForestClassifier


model_1 = RandomForestClassifier()
model_1.fit(X_train, y_train)
y_pred1 = model_1.predict(X_test)


from sklearn.metrics import accuracy_score, mean_squared_error
print(accuracy_score(y_test, y_pred1))

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 110, num = 11)]
#max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

#Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state = 10, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_

# Manually provide the best parameters to model for training
model_12 = RandomForestClassifier(**{'n_estimators': 1000,
 'min_samples_split': 2,
 'min_samples_leaf': 4,
 'max_features': 'auto',
 'max_depth': 1,
 'bootstrap': True})

result_12= model_12.fit(X_train, y_train)

pred_12 = result_12.predict(X_test)


from sklearn.metrics import accuracy_score, mean_squared_error
print(accuracy_score(y_test, pred_12))

print(np.sqrt(mean_squared_error(y_test, pred_12)))

#++++++++++++++++++++ KNN ++++++++++++++++++

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


estimators_21 = []
estimators_21.append(('standardize', StandardScaler()))
estimators_21.append(('knn', KNeighborsClassifier()))
model_21 = Pipeline(estimators_21)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=10)
results = cross_val_score(model_21, X_train, y_train, cv=kfold)
print(results.mean())
print(results.std())

search_space = [{
                 'knn__n_neighbors': range(1,20),
                 'knn__weights': ['distance']
                }]


from sklearn.model_selection import GridSearchCV

kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=10)
clf = GridSearchCV(model_21, search_space, cv=kfold, return_train_score=True ,verbose=False)
clf = clf.fit(X_train, y_train)
clf.best_estimator_


estimators_22 = []
estimators_22.append(('standardize', StandardScaler()))
estimators_22.append(('knn', KNeighborsClassifier(n_neighbors = 19)))
model_22 = Pipeline(estimators_22)
model_22= model_22.fit(X_train, y_train)


pred_22 = model_22.predict(X_test)
print(accuracy_score(y_test, pred_22))


import pickle
# open a file, where you ant to store the data
file = open('Heart_model.pkl', 'wb')

# dump information to that file
pickle.dump(model_22, file)

















