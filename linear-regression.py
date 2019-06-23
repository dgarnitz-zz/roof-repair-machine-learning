import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
X = data.iloc[1:,2:14]
y = data.iloc[1:,0]

#remove the training set 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100) 

#initialize the model
linreg = LinearRegression()

#standardize the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#train the model
linreg.fit(x_train, y_train)

#test generalization on a new dataset
y_prediction = linreg.predict(x_test)
print("MSE: ")
print (mean_squared_error(y_test, y_prediction))
print("MAE: ")
print(mean_absolute_error(y_test, y_prediction))
print("R2 of the model: ")
print(r2_score(y_test, y_prediction))

#cross validate with 5 fold CV
cv = cross_validate(linreg, x_train, y_train, cv=5, return_train_score=False)
print(cv)

#Residual Plot --> check for normality, residuals should be balanced on each side, ideally normally distributed
                    # want to underestimate as often as you overestimate
                    # want to overestimate by the same amount as you underestimate
                    # does not make sense to do this for one hot encoded data

sns.set(style="whitegrid")
sns.residplot(y_prediction, y_test, lowess=True, color="g")
plt.title("Residuals")
plt.show()

# save the model to disk
# with open('../API/price-predict/lin-reg.sav', 'wb') as path:
#     pickle.dump(linreg, path)
