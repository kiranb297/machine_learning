# Using Random forest and adding categical variable concept
# 2. Ordinal encoding(replacing unique categorical variables with different integer values)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path)

# Copy the data into another variable
house_data_copy = house_data.copy()

# Declare Y with prediction feature
y=house_data_copy.Price

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Describe X with required features
X = house_data_copy[features]

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)


#categorical columns (All object columns)
categorical_columns=list(train_X.select_dtypes(include=['object']).columns)

# Columns that can be safely ordinal encoded
# (List of Object columns that are present in both train and validation data)
good_label_cols = [col for col in categorical_columns if 
                   set(val_X[col]).issubset(set(train_X[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(categorical_columns)-set(good_label_cols))

# drop all bad columns from data
drop_X_train = train_X.drop(bad_label_cols,axis=1)
drop_X_val = val_X.drop(bad_label_cols,axis=1)


# Ordinal encoder
oridinal_encoder=OrdinalEncoder()
drop_X_train[good_label_cols]=oridinal_encoder.fit_transform(train_X[good_label_cols])
drop_X_val[good_label_cols]=oridinal_encoder.transform(val_X[good_label_cols])


house_model_4=RandomForestRegressor(random_state=1)
house_model_4.fit(drop_X_train,train_y)

predicted_value=house_model_4.predict(drop_X_val)

# validate the model using MAE using validation data y
mae9=mean_absolute_error(val_y,predicted_value)

print('MAE9 => ', mae9)