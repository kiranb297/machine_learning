# Sample model 1 with MAE

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

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

# create a model and fit with X and Y
house_model_1 = DecisionTreeRegressor(random_state=1)
house_model_1.fit(X,y)

# Predict Price from created model
predicted_value=house_model_1.predict(X)

print(predicted_value)

# validate the model using MAE
mae1=mean_absolute_error(y,predicted_value)
print('MAE1 => ', mae1)