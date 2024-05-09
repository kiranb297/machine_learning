# Using Cross validation for RandomForestRegressor

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

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

categorical_cols = [cname for cname in X.columns 
                    if X[cname].nunique() < 10 and 
                       X[cname].dtype == "object"]

# Get numerical columns list ( datatype of int and float)
numerical_cols = [cname for cname in X.columns 
                  if X[cname].dtype in ['int64', 'float64']]

# contains all columns expect colums having cordinality greater than 10
req_cols = categorical_cols + numerical_cols

# Copying only required colums from data
X_train = X[req_cols].copy()

X_training,X_valid,y_train,y_valid=train_test_split(X_train,y,train_size=0.8, test_size=0.2,random_state=0)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
pipeline_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
pipeline_model.fit(X_training,y_train)
predics=pipeline_model.predict(X_valid)
mae=mean_absolute_error(predics,y_valid)
print(mae)

scores=-1*cross_val_score(pipeline_model,X_train,y,cv=5,scoring='neg_mean_absolute_error')

print('MAE12 => ', scores.mean())