# -Costumer-Churn-Prediction-and-Retention-Strategies-for-Teleco-Company
This project aim was to identify the key indicators of customer churn for a telecommunications company and develop a model to predict which customers are likely to churn. Insights will be provided into effective retention strategies that the company can implement to reduce customer churn.

## Project Description

This project’s aim was to identify the key indicators of customer churn for a telecommunications company and develop a machine learning model to predict which customers are likely to churn. The project provided insights into effective retention strategies that the company can implement to reduce customer churn. The data was processed and analyzed using various techniques such as data cleaning, bivariate and multivariate analysis, and exploratory data analysis. The best-performing model was selected and evaluated, and suggestions for model improvement were provided. The ultimate goal of this project was to help the telecommunications company reduce customer churn and improve customer retention.

## Introduction

Telecom companies operate in a highly competitive industry, and customer churn is one of their biggest challenges. Customer churn refers to the rate at which customers stop using a company’s services. This is a significant problem for telecom companies because acquiring new customers is more expensive than retaining existing ones. Therefore, companies need to identify customers who are likely to churn and take appropriate measures to retain them. In this article, I will discuss a customer churn prediction model and retention strategies for telecom companies. I will use a classification model to predict customer churn and suggest some retention strategies that can be implemented to reduce churn rates.

## Objective
The goal of this classification was to analyze the data and utilize machine learning algorithms to predict customer churn and retention strategies for the telco company.

## Hypothesis and Questions

The analysis was guided by three(3) null hypothesis and their corresponding alternate hypothesis respectively. also, six(6) questions were asked

## Hypothesis
**ONE (1)**

*H0: There is no significant difference in churn rates between male and female customers*.

*H1: There is a significant difference in churn rates between male and female customers*.

**Two(2)**

*H0: There is no significant relationship between the customer’s internet service provider and their likelihood to churn*.

*H1: There is a significant relationship between the customer’s internet service provider and their likelihood to churn*.

**Three(3)**

*H0: There is no significant difference in churn rates between customers on different types of payment methods*.

*H1: There is a significant difference in churn rates between customers on different types of payment methods*.

## Questions
Here are five questions that guided the project:

- **What percentage of customers have churned?**

- **Is there a correlation between a customer’s length of tenure with the company and their likelihood of churning?**

- **Are there any specific groups of customers based on demographic that are more likely to churn than others?**
- **Can customer retention be improved by offering longer contract terms?**

- **How much money could the company save by reducing customer churn?**

- **What is the relationship between Internet Services and churn rate?**

## Data Understanding
The dataset used in this classification project is a Telco customer churn dataset. The data contains 7043 records of customers with 21 attributes that describe customer demographics, services used, and customer account information. The objective of the analysis is to predict customer churn and develop effective retention strategies to reduce churn rates.

*The dataset has 21 columns, which are described as follows:*

- CustomerID: A unique identifier for each customer.
Gender: The customer’s gender (Male/Female).

- SeniorCitizen: A binary variable indicating if the customer is a senior citizen or not (1, 0).

- Partner: A binary variable indicating if the customer has a partner or not (Yes, No).

- Dependents: A binary variable indicating if the customer has dependents or not (Yes, No).

- Tenure: The number of months the customer has been with the company.

- PhoneService: A binary variable indicating if the customer has a phone service or not (Yes, No).

- MultipleLines: A binary variable indicating if the customer has multiple lines or not (Yes, No, No phone service).

- InternetService: The type of internet service the customer has (DSL, Fiber optic, No).

- OnlineSecurity: A binary variable indicating if the customer has online security or not (Yes, No, No internet service).

- OnlineSecurity: A binary variable indicating if the customer has online backup or not (Yes, No, No internet service).

- DeviceProtection: A binary variable indicating if the customer has device protection or not (Yes, No, No internet service).

- TechSupport: A binary variable indicating if the customer has tech support or not (Yes, No, No internet service).

- StreamingTV: A binary variable indicating if the customer has streaming TV or not (Yes, No, No internet service).

- StreamingMovies: A binary variable indicating if the customer has streaming movies or not (Yes, No, No internet service).

- Contract: The type of contract the customer has (Month-to-month, One year, Two years).

- Paperless billing: A binary variable indicating if the customer has paperless billing or not (Yes, No).

- PaymentMethod: The payment method the customer uses (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).

- MonthlyCharges: The amount charged to the customer monthly.

- Total charges: The total amount charged to the customer over the entire tenure.

- Churn: This variable indicates whether a customer has churned or not. It was the target variable for the project (Yes, No)

The dataset contained no missing values, and all the attributes are in the correct data type. The next step is to perform exploratory data analysis and feature engineering to prepare the data for modeling. Also, the project aims to develop a predictive model that can identify customers who are at risk of churning and implement retention strategies to reduce churn rates.

## Packages Used
The following packages were used for the project

**Data handling**
- import pandas as pd # used for data manipulation and analysis, such as loading data into data frames and performing various data transformations.
- import numpy as np  # used for numerical operations and computations, such as handling missing values and performing array operations.

**Vizualisation (Matplotlib, Plotly, Seaborn, etc. )**
- import seaborn as sns #used for advanced data visualization, such as creating heatmaps and categorical plots.
- import matplotlib.pyplot as plt  # used for creating basic plots and charts.
- %matplotlib inline # used to create easier and view plots quickly and efficiently


**Feature Processing (Scikit-learn processing, etc. )**
- from sklearn.impute import SimpleImputer # used for imputing missing values in the data.
- from sklearn.model_selection import train_test_split  # used for splitting the data into training and testing sets.
- from sklearn.preprocessing import OrdinalEncoder   # used for encoding categorical features as integer values.
- from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # used for encoding categorical features as integer labels.
- from sklearn.preprocessing import StandardScaler # used for standardizing the data.
- from sklearn.preprocessing import MinMaxScaler   # used for scaling the data to a specific range
- from collections import Counter  # used for counting the number of occurrences of each element in a list.
- from imblearn.over_sampling import RandomOverSampler # used for oversampling the minority class to balance the dataset.
- import scipy.stats as stats   # used for performing statistical tests and calculations.
- from scipy.stats import chi2_contingency   # 

**Machine Learning (Scikit-learn Estimators, Catboost, LightGBM, etc.)**
- from sklearn.datasets import make_classification
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
- from sklearn.linear_model import LogisticRegression, SGDClassifier # used for building a logistic regression model
- from sklearn.neighbors import KNeighborsClassifier   # used for building a K-Nearest Neighbors model.
- from sklearn.svm import SVC # used for building a Support Vector Machines model.
- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, fbeta_score
- from sklearn.metrics import confusion_matrix #  used for evaluating the performance of the machine learning models.

**Hyperparameters Fine-tuning (Scikit-learn hp search, cross-validation, etc. )**
- from sklearn.model_selection import KFold, cross_val_score  # used for performing K-fold cross-validation
- from sklearn.model_selection import GridSearchCV # used for performing hyperparameter tuning through grid search.
- from sklearn.ensemble import GradientBoostingRegressor # sed for building a gradient boosting regression model.

 **Other packages**
- from tabulate import tabulate  # used for creating tables to display the results of the machine learning models.
- import os, pickle # used for saving and loading the trained machine learning models

- import warnings # used for filtering warning messages from the output.
warnings.filterwarnings('ignore')

## Dataframe and Datatypes Understanding
The dataset was loaded into a Pandas DataFrame using the pd.read_csv function. The dataset contained 21 columns/features and 7043 rows.

**Datatypes**

This output of the dataframe.info() revealed that the 21 columns in  with their corresponding data types had no missing values (since all columns have 7043 non-null values), but the TotalCharges column was in object instead of float64. This suggests that there may be some non-numeric values in this column that need to be cleaned.
