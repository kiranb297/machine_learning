# Using Gradient boosting with parameter boosting

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import seaborn as sns


# path to csv file
data_path="Data_Sets\melb_data.csv"

# Read data from csv
house_data = pd.read_csv(data_path)

# converst Date into seperate month and year columns 
house_data['Date']=pd.to_datetime(house_data['Date'],format='mixed')
house_data['year']=house_data['Date'].apply(lambda date:date.year)
house_data['month']=house_data['Date'].apply(lambda date:date.month)

# To find the columns corelation plot corelation heat map
# categorical_columns = house_data.select_dtypes('object').columns
# numerical_columns = house_data.select_dtypes(['int64', 'float64']).columns

# catData=house_data[categorical_columns].copy()
# numData=house_data[numerical_columns].copy()
# for col in categorical_columns:
#     catData[col], _ = pd.factorize(catData[col])
# data = pd.concat([catData, numData], axis=1)
# correlation = data.corr()
#print(correlation['Price'])

#heat map
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,cmap='Blues')
# plt.show()

# drop columns which is not impacting the price column (by observing heat map empty values and iterating drop columns and observing MAE)
house_data=house_data.drop(columns=['Date','Suburb','Address','SellerG','Regionname'],axis=1)
categorical_columns = house_data.select_dtypes('object').columns

# convert categorical columns into numerical (ordinal encoding)
for col in categorical_columns:
    house_data[col],_=house_data[col].factorize()

# Drop target column (Price)
X= house_data.drop(columns='Price',axis= 1 )
y= house_data['Price']

# Split the dataset into train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=103)

# Fill all empty values with Simple imputer 
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns

X_train=imputed_X_train
X_test=imputed_X_test

# Standardize the data set using minmaxscaler
scaler = MinMaxScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)

my_model=XGBRegressor(n_estimators=1000,learning_rate=0.05,early_stopping_rounds=5)
my_model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],verbose=False)

predictions = my_model.predict(X_test)

mae14 = mean_absolute_error(predictions,Y_test)
print('MAE14 => ', mae14)