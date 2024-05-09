# Using Random forest to generate more number of tress and to get optimized MAE model
# insted of DecisionTreeRegressor we will use RandomForestRegressor

import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path) 

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Describe X with required features
X = house_data[features]
# Declare Y with prediction feature
y=house_data.Price

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

house_model_4=RandomForestRegressor(random_state=1)
house_model_4.fit(train_X,train_y)

predicted_value=house_model_4.predict(val_X)

print(predicted_value)

# validate the model using MAE using validation data y
mae4=mean_absolute_error(val_y,predicted_value)

print('MAE4 => ', mae4)