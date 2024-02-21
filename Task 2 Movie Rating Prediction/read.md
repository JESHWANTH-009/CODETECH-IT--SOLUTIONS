# Introduction
Movie Rating Prediction project involves building a model that predicts the rating of a movie based on features like genre, director, and actors. We use regression techniques to tackle this exciting problem. This enables us to explore data analysis, preprocessing, feature engineering, and machine learning modeling techniques

# Goal
The main goal of this project is to analyze historical movie data and develop a model that accurately estimates the rating given to a movie by users or critics. By doing so, we aim to provide insights into the factors that influence movie ratings and create a model that can estimate the ratings of movies accurately.
## Content
### 1. Data exploration
- Identify the missing values 
- Use distribution plots to visualize the correlation between certain attributes (age, sex, Pclass, alone/not alone) and the outcome of survival
### 2. Data cleaning and feature engineering
- The Movie Rating Prediction project involves building a model to predict the rating of a movie based on various features, such as genre, director, and actors. This project utilizes regression techniques to analyze historical movie data and develop a model that accurately estimates the rating assigned by users or critics. Through this project, one can gain insights into data analysis, preprocessing, feature engineering, and machine learning modeling techniques, while exploring the factors influencing movie ratings.
### 3. Evaluation of different models
- Stochastic gradient descent
- Random forest
- Logistic regression
- K-nearest neighbours
- Gaussian Naive Bayes
- Linear SVM
- Decision tree
- Perceptron  

Random forest performed the best. 
### 4. Hyperparameter tuning
- Plot the importance of each feature contributing the to model training
- Drop the non-important features (not_alone and Parch)
- Identify the available parameters for tuning 
- Tune the hyperparameters of Random Forest using RandomizedSearchCV for an approximate region of search. Then use GridSearchCV for a finer degree of search. 
### 5. Confusion matrix, precision, recall, ROC-AUC score
- The confusion matrix shows the number of True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN)
- FP is aka Type 1 error, FN is aka Type 2 error
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)
- The ROC-AUC curve should be as far as possible away from the purely random classifier line

## Libraries
1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib
5. sklearn

