# Using Pipeline to simplify the coding and to combine both imputation one-hot-encoding in single model

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



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

# split data into train(80%) and validation(20%) data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# get Categorical columns list that has cordinality less than 10 and having datatype object
categorical_cols = [cname for cname in X_train_full.columns 
                    if X_train_full[cname].nunique() < 10 and 
                       X_train_full[cname].dtype == "object"]

# Get numerical columns list ( datatype of int and float)
numerical_cols = [cname for cname in X_train_full.columns 
                  if X_train_full[cname].dtype in ['int64', 'float64']]

# contains all columns expect colums having cordinality greater than 10
req_cols = categorical_cols + numerical_cols

# Copying only required colums from data
X_train = X_train_full[req_cols].copy()
X_valid = X_valid_full[req_cols].copy()


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

# Preprocessing of training data, fit model 
pipeline_model.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
predicted_value = pipeline_model.predict(X_valid)

mae11=mean_absolute_error(y_valid,predicted_value)

print('MAE11 => ', mae11)