# Using Random forest to generate more number of tress and Passing opt max_leaf_node

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

# Describe X with required features
X = house_data[features]
# Declare Y with prediction feature
y=house_data.Price

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)


# create a function that return MAE
def my_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
    maemodel = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=1)
    maemodel.fit(train_X,train_y)
    pred_val=maemodel.predict(val_X)
    mae=mean_absolute_error(val_y,pred_val)
    return mae

# Get the optimized max_leaf_node value
maes={}
for leaf_nodes in [200,400,600,800,1000]:
    maes.update({leaf_nodes:my_mae(leaf_nodes,train_X,val_X,train_y,val_y)})

opt_max_leaf_node=min(maes,key=maes.get)
# print(opt_max_leaf_node)

house_model_5=RandomForestRegressor(max_leaf_nodes=opt_max_leaf_node,random_state=1)
house_model_5.fit(train_X,train_y)

predicted_value=house_model_5.predict(val_X)

print(predicted_value)

# validate the model using MAE using validation data y
mae5=mean_absolute_error(val_y,predicted_value)

print('MAE5 => ', mae5)