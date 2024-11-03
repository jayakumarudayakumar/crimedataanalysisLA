# Crime Data Analysis Of Los Angeles From 2020 Till Present

##  Project Description
The project is about the crime occurrences in Los Angeles from 2020 to the present.
1. ​It is a real-time dataset being updated on a bi-weekly basis and available for public use. (Source: Data.gov)​
2. Our primary goal is to analyze the crime occurrences, frequency, and severity from the variables in the dataset
3. Our secondary goal is to utilize predictive modeling methods to recognize the most vulnerable time, age, and geographical locations

## ​Questions Vs Models Chosen​
1. Is there a correlation between the time of the day and the reported crime frequency?
   
   Model: Linear Regression, Rationale - This model helps us to make comparisons and predictions between the independent variable (Time) and dependent variable (crime count)

2. How severe are the vandalism crimes in LA? Misdemeanor Vs Felony. Which one of them would be expected to occur the most?

   Model: Binary Classification through Logistic Regression, Rationale: This model comparatively performed well in classifying and predicting the vandalism crime categories: misdemeanor and felony.​

3. ​How can we analyze and predict incident outcomes (crime statuses) based on the type of crime, date/time of occurrence, and geographical locations?

   Model: Decision-Tree Model, Rationale: It is comparatively the best method that provided better output in performing multi-class classification and predictions of the "Crime Status Codes" concerning the factors such as time occurrences and locations
