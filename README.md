# machine_learning
Here i tried to capture the evolution of Machine learning model (Simple model to Complex deep neuron network model).
This repository contains 14 different approach of machine learning + 1 deep learning approach for the same DataSet 
i.e Malbore Hosing data (Predicting the house price).
1. Sample Decision Tree Regressor model with calculation Mean Absolute Error (MAE).
2. Decision Tree Regressor model by splitting the data into train and validation data and observing the the MAE.
3. In Decision Tree Regressor model Optimizing the size of the tree (optimized max_leaf_node value) to get better MAE and to make better prediction.
4. Using Random forest to generate more number of tress and to get optimized MAE model (RandomForestRegressor MODEL).
5. In RandomForestRegressor model Optimizing the size of the tree (optimized max_leaf_node value) to get better MAE and to make better prediction.

 Applying Feature engineering for RandomForestRegressor
1. Missing values
   6. Dropping columns with missing values and observing the the MAE.
   7. Imputating missing values with median or average value and observing the the MAE.
2. categical values
   8. Removing columns containing categorical variables and observing the the MAE.
   9. Ordinal encoding(replacing unique categorical variables with different integer values) and observing the the MAE.
   10. One-hot encoding (Creating columns for each categorical  variables based on cardinality) and observing the the MAE.
11. Using Pipeline to simplify the coding and to combine both imputation one-hot-encoding in single model.
12. Using Cross validation for RandomForestRegressor and observing the the MAE.
13. Using Gradient boosting with parameter boosting (XGBRegressor model) and observing the the MAE.
14. Applying feature engineering for Gradient boosting (XGBRegressor model) and observing the the MAE.
15. Applying feature engineering for neuron networks with deep layers(hidden layers) and observing the the error.
