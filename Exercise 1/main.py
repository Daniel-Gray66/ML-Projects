import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model

#Point of this assignment is to familiarize myself with python and the basics of simple linear regression
#Creating the data frame

df = pd.read_csv("canada_per_capita_income.csv")
#Plotting the graph for data frame

plt.scatter(df['year'],df['Capita'], color ='red', marker = '+')
plt.xlabel('year')
plt.ylabel('Capita')
plt.title("Canada")

#Creae the line of best fit usign simple linear regression 

reg = linear_model.LinearRegression()
reg.fit(df[['year']],df['Capita'])

predictions = reg.predict(df[['year']])

plt.plot(df['year'],predictions, color ='blue')
plt.show()


