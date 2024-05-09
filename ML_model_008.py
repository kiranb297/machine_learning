# Using Random forest and adding categical variable concept
# 1. Removing columns containing categorical variables

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path)

# Copy the data into another variable
house_data_copy = house_data.copy()

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Declare Y with prediction feature
y=house_data_copy.Price

# Describe X with required features
X_full = house_data_copy[features]


# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X_full,y,random_state=1)

#Remove or exclude columns with categorical variables from the copied data
X_train_removed_cat = train_X.select_dtypes(exclude=['object'])
X_val_removed_cat = val_X.select_dtypes(exclude=['object'])

house_model_4=RandomForestRegressor(random_state=1)
house_model_4.fit(X_train_removed_cat,train_y)

predicted_value=house_model_4.predict(X_val_removed_cat)

# validate the model using MAE using validation data y
mae8=mean_absolute_error(val_y,predicted_value)

print('MAE8 => ', mae8)