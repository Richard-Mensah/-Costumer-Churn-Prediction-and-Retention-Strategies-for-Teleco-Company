# -Costumer-Churn-Prediction-and-Retention-Strategies-for-Teleco-Company :computer: 💻
This project aim was to identify the key indicators of customer churn for a telecommunications company and develop a model to predict which customers are likely to churn. Insights will be provided into effective retention strategies that the company can implement to reduce customer churn.

## Project Description :book: 

This project’s aim was to identify the key indicators of customer churn for a telecommunications company and develop a machine learning model to predict which customers are likely to churn. The project provided insights into effective retention strategies that the company can implement to reduce customer churn. The data was processed and analyzed using various techniques such as data cleaning, bivariate and multivariate analysis, and exploratory data analysis. The best-performing model was selected and evaluated, and suggestions for model improvement were provided. The ultimate goal of this project was to help the telecommunications company reduce customer churn and improve customer retention.

## Hypothesis and Questions :s

The analysis was guided by three(3) null hypothesis and their corresponding alternate hypothesis respectively. also, six(6) questions were asked

## Hypothesis 💻
**ONE** :one:

*H0: There is no significant difference in churn rates between male and female customers*.

*H1: There is a significant difference in churn rates between male and female customers*.

**Two** :two:

*H0: There is no significant relationship between the customer’s internet service provider and their likelihood to churn*.

*H1: There is a significant relationship between the customer’s internet service provider and their likelihood to churn*.

**Three** :three:

*H0: There is no significant difference in churn rates between customers on different types of payment methods*.

*H1: There is a significant difference in churn rates between customers on different types of payment methods*.

## Questions :question:
Here are five questions that guided the project:

- **What percentage of customers have churned?**

- **Is there a correlation between a customer’s length of tenure with the company and their likelihood of churning?**

- **Are there any specific groups of customers based on demographic that are more likely to churn than others?**
- **Can customer retention be improved by offering longer contract terms?**

- **How much money could the company save by reducing customer churn?**

- **What is the relationship between Internet Services and churn rate?**

## Data Understanding :o:
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

There was no missing values in the dataset

## Dataframe and Datatypes Understanding :v:
The dataset was loaded into a Pandas DataFrame using the pd.read_csv function. The dataset contained **21 columns/features and 7043 rows**.

**Datatypes** :o:

This output of the dataframe.info() revealed that the 21 columns in  with their corresponding data types had no missing values (since all columns have 7043 non-null values), but the TotalCharges column was in object instead of float64. This suggests that there may be some non-numeric values in this column that need to be cleaned.

## Summary of findings from the analysis :blush: 
Below are some findings drawn from the jupyter notebook for your kind attention 
1. Data handling packages like pandas and numpy were used
2. Packages for feacture processing, machine learning and hyperparemeter tuning were also used to acheive the object of the analysis
3. Other packages such as tabulate, os and  warnings  were also used.
3. Regarding the hypothesis, 
- There is no significant difference in churn rates between male and female customers 
- There is a significant relationship between the customer’s internet service provider and their likelihood to churn
- Also, there is a significant difference in churn rates between customers on different types of payment methods guided the analysis.
5. Total customers of 1,869 representing 26.5% were churned whiles 5,174 customers representing 73.5% did not churn.
6. Males customers churned higher than females
7. Customers who undergo two-year contract terms have a higher retention rate than the one-year and month-to-month contract terms
8. The company has the potential to save more than $14 million in revenue by implementing efficient strategies and decreasing churn rates.
9. The best performing model was **Random Forest** 

## Medium Article ↩ :on:
Below is link to my full article 
https://medium.com/@richmensah1997/customer-churn-prediction-and-retention-strategies-for-a-telecommunications-company-42af00bdf1e9


## Author :book:⛺
Richard Mensah