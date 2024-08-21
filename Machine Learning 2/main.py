import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 

df = pd.read_csv("homeprices.csv")
print(df)


plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.xlabel('Area')  
plt.ylabel('Price')
plt.title('Home Prices Scatter Plot')  


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df[['price']])

#Making the predictions 
predictions = reg.predict(df[['area']])
print(predictions)

#Posting the predictions back to the csv file
df['Future'] = predictions

print(df)

plt.plot(df['area'],predictions,color = 'blue')

plt.show()  

#df.to_csv("name of file you want to send the data to")



