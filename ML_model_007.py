# Missing values (2. Imputating missing values with median or average value)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path) 

# Declare critical features
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Declare Y with prediction feature
y=house_data.Price
# Describe X with required features
X_full = house_data[features]

# Exclude columns with string (object) consider only numeric columns
X_numb=X_full.select_dtypes(exclude=['object'])

# Split the data into training and validation data
train_X,val_X,train_y,val_y = train_test_split(X_numb,y,random_state=1)

# Define imputer and fit the data with imputed values
my_imputer = SimpleImputer(strategy='constant')
imputed_X_train=pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_X_val=pd.DataFrame(my_imputer.transform(val_X))

# Imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_val.columns = val_X.columns

house_model_7=RandomForestRegressor(random_state=1)
house_model_7.fit(imputed_X_train,train_y)
predicted_value=house_model_7.predict(imputed_X_val)

# validate the model using MAE using validation data y
mae7=mean_absolute_error(val_y,predicted_value)

print('MAE7 => ', mae7)