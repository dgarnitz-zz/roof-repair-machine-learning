import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#load data
data = pd.read_excel('./regression-training-data/PriceData.xlsx', sheet_name='SeasonalData', header=None)

#extract data 
X = data.iloc[1:,2:16]
y = data.iloc[1:,0]

#remove the training set 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 111) 

#initialize the model
        #sklearn documentation recommends this solver for small datasets
        #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
mlp = MLPRegressor(hidden_layer_sizes=(14,), max_iter=5000) 
        #I experimented with early stopping - made performance materially worse

#standardize the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#train the model
mlp.fit(x_train, y_train)

#test generalization on a new dataset
y_prediction = mlp.predict(x_test)
print("MSE: ")
print (mean_squared_error(y_test, y_prediction))
print("MAE: ")
print(mean_absolute_error(y_test, y_prediction))
print("R2 of the model: ")
print(r2_score(y_test, y_prediction))

#cross validate with 5 fold CV
cv = cross_validate(mlp, x_train, y_train, cv=5, return_train_score=False)
print(cv)

#Residual Plot 
sns.set(style="whitegrid")
sns.residplot(y_prediction, y_test, lowess=True, color="g")
plt.title("Neural Network Residuals")
plt.savefig("neuralnet-residuals.png")

#test prediction with raw data before saving
data = [1, 0, 0, 0, 8.21, 3.9, 0, 0, 1, 0, 0, 0, 1, 0] #observed value of 50
data = np.reshape(data, (1, -1))
data = scaler.transform(data)
print("The price of the contract in GBP per Square Meter is:")
print(mlp.predict(data))

# # save the model to disk
# with open('../API/price-predict/neuralnet.sav', 'wb') as path:
#     pickle.dump(mlp, path)
