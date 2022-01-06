import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('.//training_set_features.csv', index_col='respondent_id')
labels = pd.read_csv('./training_set_labels.csv', index_col='respondent_id')['h1n1_vaccine']
test_features = pd.read_csv('./test_set_features.csv', index_col='respondent_id')

num_cols = train.select_dtypes('number').columns

cat_cols = ['race', 'sex',
            'marital_status', 'rent_or_own', 'hhs_geo_region',
            'census_msa', 'employment_industry', 'employment_occupation']

ord_cols = ['age_group', 'education', 'income_poverty',
            'employment_status']
# Impute train
for col in num_cols:
    train[col] = train[col].fillna(value=-1)
for col in (cat_cols + ord_cols):
    train[col] = train[col].fillna(value='missing')

# Impute test
for col in num_cols:
    test_features[col] = test_features[col].fillna(value=-1)
for col in (cat_cols + ord_cols):
    test_features[col] = test_features[col].fillna(value='missing')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

objList = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
           'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']
for feat in objList:
    train[feat] = le.fit_transform(train[feat].astype(str))
for feat in objList:
    test_features[feat] = le.fit_transform(test_features[feat].astype(str))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = GradientBoostingRegressor(n_estimators=200, random_state=100)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
score = metrics.roc_auc_score(y_test, y_pred)
print(i, ' Roc Auc Score:', score)
ypoints.append(score)

from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
score = metrics.roc_auc_score(y_test, y_pred)
print(' Roc Auc Score:', score)
ypoints.append(score)

# test_pred = regressor.predict(test_features)

# print(test_pred)
xpoints = np.arange(1, 101)

plt.plot(xpoints, ypoints)
plt.title("Curve plotted using the given points")
plt.xlabel("Number of Estimators")
plt.ylabel("Test Accuracies")
plt.show()