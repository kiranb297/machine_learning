# Using Gradient boosting with parameter boosting

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data_full = pd.read_csv(data_path)

# set prediction value to y before removing it from the test data
y=house_data_full.Price

# remove prediction value from the test data
house_data=house_data_full.drop('Price',axis=1)

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X=house_data[features]
# contains categorical columns having cordinality greater than 10
categorical_cols = [cname for cname in X.columns 
                    if X[cname].nunique() < 10 and 
                       X[cname].dtype == "object"]

# Get numerical columns list ( datatype of int and float)
numerical_cols = [cname for cname in X.columns 
                  if X[cname].dtype in ['int64', 'float64']]

# contains all columns expect colums having cordinality greater than 10
req_cols = categorical_cols + numerical_cols

# Copying only required colums from data
X_train_all = X[req_cols].copy()

X_train,X_valid,y_train,y_valid=train_test_split(X_train_all,y,train_size=0.8, test_size=0.2,random_state=0)


X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

my_model=XGBRegressor(n_estimators=1000,learning_rate=0.05,early_stopping_rounds=5)
my_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=False)

predictions = my_model.predict(X_valid)

mae13 = mean_absolute_error(predictions,y_valid)
print('MAE13 => ', mae13)