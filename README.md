# Crime Data Analysis Of Los Angeles From 2020 Till Present

## INTRODUCTION
This project proposal is based on the dataset “Crime Data in the City of Los Angeles from 2020 – Present.” It is a real-time dataset that will be updated bi-weekly and available for public use. Our goal is to perform data analysis to understand the crime occurrences in Los Angeles and their severity concerning various factors and make the best predictions using predictive models to recognize the most vulnerable premises, time of the day, gender, etc., and the geographical locations that are heavily affected. 

To attain our goal, we have dissected it into the following questions: 
1. Is there a correlation between the time of the day and the reported crime frequency? 
2. How severe are vandalism crimes in Los Angeles? Misdemeanor Vs Felony. Which one of them would be expected to occur the most?​ 
3. How can we analyze and predict incident outcomes (crime statuses) based on the type of crime, date/time of occurrence, and geographical locations? 

## Models for Analysis 

Model 1: Binary Classification using Logistic Regression 
Rationale: This model performed well in classifying and predicting the vandalism crime categories: misdemeanor and felony. 

Model 2: Decision-Tree Model 
Rationale: It is the best method that provides better output in performing multi-class classification and predictions of the "Crime Status Codes" concerning factors such as time occurrences and locations.​ 

Model 3: Support Vector Machine Model 
Rationale: Since there is an extreme imbalance between “crime status,” we have used the SVM model to do binary classification on the prevalent crime status ‘IC - Investigation Continues.” 

After careful consideration and several trials, we chose these models to provide better outcomes. 
