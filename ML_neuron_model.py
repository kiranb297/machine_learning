# Buuilding a neuron model using tensorflow

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,callbacks

# Read datasets and feature development
house_data=pd.read_csv('Data_Sets\melb_data.csv')
# converst Date into seperate month and year columns
house_data['Date']=pd.to_datetime(house_data['Date'],format='mixed')
house_data['year']=house_data['Date'].apply(lambda date:date.year)
house_data['month']=house_data['Date'].apply(lambda date:date.month)

# drop columns which is not impacting the price column
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

# Declare early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# declare Model and add as many layers you want and add a activation function
# Number of units is equal to number of features (Number of columns)
model = Sequential()
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(units=64,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))
model.add(layers.Dense(units=256,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))
model.add(layers.Dense(units=1024,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))
model.add(layers.Dense(units=4096,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))

# compile the model with optimizer
model.compile(optimizer='adam',
              loss='mae')

# Train the model
history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=32,epochs=1000,callbacks=[early_stopping])

history_df= pd.DataFrame(model.history.history)

predection = model.predict(X_test)
error = mean_absolute_error(Y_test,predection)
print(error)

# history_df.plot()
# plt.plot(history_df)
# plt.show()