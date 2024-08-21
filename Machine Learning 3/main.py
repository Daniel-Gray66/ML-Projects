import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

#Loading the data frame 
df = pd.read_csv("homeprices.csv")

# Calculate the median 

median_bedrooms =  math.floor(df.bedrooms.median())
print(median_bedrooms)
#Setting it equal to itself somehow updates it
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

#Creatinga linear Regression object

reg = linear_model.LinearRegression()
#First Argument is ur independent variables,Second one is the dependent
reg.fit(df[['area','bedrooms','age']],df.price)


print("Coefficients:", reg.coef_)


predicted_price = reg.predict([[2500, 4, 5]])
print("Predicted Price for house with area=2500, bedrooms=4, age=5:", predicted_price)