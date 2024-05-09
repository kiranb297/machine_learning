# splitting the data into train and validation data and observing the the MAE

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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

# create a model and fit with train X and train Y
house_model_2 = DecisionTreeRegressor(random_state=1)
house_model_2.fit(train_X,train_y)

# Predict Price from created model for validation data X
predicted_value=house_model_2.predict(val_X)

print(predicted_value)

# validate the model using MAE using validation data y
mae2=mean_absolute_error(val_y,predicted_value)

print('MAE2 => ', mae2)