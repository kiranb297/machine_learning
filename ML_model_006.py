# Missing values (1. Dropping columns with missing values)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path) 

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Declare Y with prediction feature
y=house_data.Price

# Describe X with required features
X = house_data[features]

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)


# List all columns which has missing data
columns_with_missing_values=[col for col in train_X.columns
                             if train_X[col].isnull().any()]

# Drop the entire column which has missing data
droped_train_X= train_X.drop(columns_with_missing_values,axis=1)
droped_val_X= val_X.drop(columns_with_missing_values,axis=1)

house_model_6=RandomForestRegressor(random_state=1)
house_model_6.fit(droped_train_X,train_y)

predicted_value=house_model_6.predict(droped_val_X)

# validate the model using MAE using validation data y
mae6=mean_absolute_error(val_y,predicted_value)

print('MAE6 => ', mae6)