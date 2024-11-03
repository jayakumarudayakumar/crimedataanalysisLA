# Crime Data Analysis Of Los Angeles From 2020 Till Present

##  Project Description
The project is about the crime occurrences in Los Angeles from 2020 to the present.
1. ​It is a real-time dataset being updated on a bi-weekly basis and available for public use. (Source: Data.gov)​
2. The primary goal is to analyze the crime occurrences, frequency, and severity from the variables in the dataset
3. The Secondary goal is to utilize predictive modeling methods to recognize the most vulnerable time, age, and geographical locations

## ​Questions Vs Models Chosen​
1. Is there a correlation between the time of the day and the reported crime frequency?
   
   Model: Linear Regression, Rationale - This model helps us to make comparisons and predictions between the independent variable (Time) and dependent variable (crime count)

2. How severe are the vandalism crimes in LA? Misdemeanor Vs Felony. Which one of them would be expected to occur the most?

   Model: Binary Classification through Logistic Regression, Rationale: This model comparatively performed well in classifying and predicting the vandalism crime categories: misdemeanor and felony.​

3. ​How can we analyze and predict incident outcomes (crime statuses) based on the type of crime, date/time of occurrence, and geographical locations?

   Model: Decision-Tree Model, Rationale: It is comparatively the best method that provided better output in performing multi-class classification and predictions of the "Crime Status Codes" concerning factors such as time occurrences and locations

## Question 1​ - Observation
Is there a correlation between the time of the day and the reported crime frequency?

EDA
1. Crime trend reaches a peak during the daytime noon and continues till the evening

Model - Linear Regression
Through the linear regression model, we tried to train the model with 80% of the data and test it using the remaining 20%
Positive Correlation: The blue line represents the linear regression fit, indicating a trend that as the hours of the day increase, the crime count also tends to increase
Analysis: How well the time of day can predict crime frequency using linear regression

## Question 2 - Observation
How severe are the vandalism crimes in LA? Misdemeanor Vs Felony. Which one of them would be expected to occur the most?​

EDA​
1. Young age people between 20 and 40 are more vulnerable to the crimes
2. Frequency is declining for older age groups
3. Misdemeanor crimes have broad distribution showing a wide range of age groups prone to "less severe" crimes​

Model - Binary Classification through Logistic Regression Model
1. The area under the ROC curve is AUC is 0.58​
2. It suggests that the model's ability to discriminate between the positive and negative classes is slightly better than random guessing, which would have an AUC of 0.5.

## Question 3 - Observation
How can we analyze and predict incident outcomes (crime statuses) based upon the type of crime, date/time of occurrence, and geographical locations?

EDA
1. Status Distribution: Depicts a comparison of crime outcomes (statuses) by location
2. Comparative Analysis: Highlights crime count variations, indicating area-specific crime rates and law enforcement activity
3. Pattern Insights: Reveals prevalent crime statuses and their concentration in certain geographical areas.

Model - Decision Tree
1. Diagnostic Ability: Plots true positives against false positives, reflecting the trade-off between sensitivity and error rate
2. Performance Benchmark: Proximity to the dashed diagonal line suggests performance is slightly better than random chance
3. AUC Metric: Typically provides a single performance score (not shown here), with a higher value indicating better accuracy​

## Conclusion
1. Analysis Success: Comprehensive analysis of Los Angeles crime data (2020-present) revealed key insights into crime patterns and severity, using Python for data processing and predictive modeling
2. Key Insights: Identified the vulnerability of individuals aged 20 to 40, trends in crime severity, and spatial crime patterns. These insights are crucial for understanding crime dynamics in Los Angeles
3. Modeling Impact: Demonstrated the effectiveness of Linear, Logistic, and Multinomial Regression in predicting crime occurrences, severity classification, and geographical analysis of crime statuses
4. Policy and Safety: The study informs strategic interventions and policy decisions, enhancing public safety by guiding resource allocation and targeted crime prevention strategies
5. Future Research: Lays the groundwork for further exploration into crime prediction and prevention, suggesting the potential of additional data, variables, and models to deepen understanding and impact.​

​
