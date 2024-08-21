import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model

#Focus of this activity is to make sure to fix and correct data

#Creating the data frame 
df = pd.read_csv("hiring.csv")

#Filling the missing data (Make sure to check the name of the series)
median_test = df.test_score.median()
df.test_score = df.test_score.fillna(median_test)

df.experience = df.experience.fillna('zero')

#Creating the linear modle instance
reg = linear_model.LinearRegression()

#Then We need to train the linear model instance using linear regression multiple

reg.fit(df[['experience', 'test_score', 'interview_score(out of 10)']], df.salary)

#giving an estomade to someone who has 2 years of exp, 9 test score and 6/10


prediction_data_1 = pd.DataFrame([[2, 9, 6]], columns=['experience', 'test_score', 'interview_score(out of 10)'])
prediction_data_2 = pd.DataFrame([[10, 11, 10]], columns=['experience', 'test_score', 'interview_score(out of 10)'])

predicted_salary_1 = reg.predict(prediction_data_1)
predicted_salary_2 = reg.predict(prediction_data_2)


print(predicted_salary_1)
print(predicted_salary_2)

print(df)