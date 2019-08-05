import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#load data
data = pd.read_excel('../regression-training-data/PriceData.xlsx', sheet_name='SeasonalDataWithoutDates', header=None)

#extract data 
X = data.iloc[1:,1:15]
y = data.iloc[1:,0]

#did not scatter plot because data is mostly categorical, and there is nothing to plot the Y values against

#output prices histogram - Y values only, as it does not make sense to plot categorical data
fig1 = plt.figure()
histogram = fig1.add_subplot(111)
n, bins, patches = plt.hist(y, density = 0, facecolor = 'green', alpha = .2)
histogram.set_xlabel("Bid Price in £ per square meter")
histogram.set_ylabel("Frequency")
# plt.show()
# plt.savefig('histogram.png')

#method to plot the results, called in the file where the model is trained
