# Using Random forest and adding categical variable concept
# 3. One-hot encoding (Creating columns for each categorical  variables based on cardinality)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path)

# Copy the data into another variable
house_data_copy = house_data.copy()

#categorical columns
categorical_columns=list(house_data_copy.select_dtypes(include=['object']).columns)

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# store high and low cardinality columns 
low_cardinality_cols=[col for col in categorical_columns if house_data_copy[col].nunique()<10]
high_cardinality_cols=list(set(categorical_columns)-set(low_cardinality_cols))

# we will one-hot-encode only for low cardinality columns
oh_encoder=OneHotEncoder(handle_unknown='ignore')

temp_encoded_data=pd.DataFrame(oh_encoder.fit_transform(house_data_copy[low_cardinality_cols]))

# oh encode will remove index we need to add the index
temp_encoded_data.index=house_data_copy.index

# temp_encode_data will contain only encoded new columns from low cardinality columns variables
# so need to concat with other columns
# create a variable containing all columns expect low cardinality columns
num_data=house_data_copy.drop(low_cardinality_cols,axis=1)

oh_encoded_data=pd.concat([num_data,temp_encoded_data],axis=1)

# need to assign string type to all columns
oh_encoded_data.columns=oh_encoded_data.columns.astype(str)

# Describe X with required features
X = oh_encoded_data[features]
# Declare Y with prediction feature
y=oh_encoded_data.Price

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

house_model_4=RandomForestRegressor(random_state=1)
house_model_4.fit(train_X,train_y)

predicted_value=house_model_4.predict(val_X)

print(predicted_value)

# validate the model using MAE using validation data y
mae10=mean_absolute_error(val_y,predicted_value)

print('MAE10 => ', mae10)