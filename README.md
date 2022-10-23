# GetAround_EDA_ML_Dashboard_API_Project
Project on Exploratory Data Analysis, Supervised Machine Learning, Dashboard Building and API Creation


Link to dashboard: http://getaround-dashboard-threshold.herokuapp.com

Link to API/docs: http://getaround-api-xgboost.herokuapp.com/docs

Linkt to API/predict: http://getaround-api-xgboost.herokuapp.com/predict


The goal is to create an app that will recommend users where to plan their next holidays. It takes real up-to-date data on Weather and Hotels in the area. The application will recommend the best destinations and hotels in France.

Tasks:

Scraping location information, acquiring up-to-date weather information and accessing to hotel information in each destination
All the data will be stored in a data lake in cloud. ETL process will be run from datalake to a data warehouse.

Deliverables available:

1- Two maps with TOP 5 destinations and TOP 20 hotels in the area

2- A .csv file in an S3 bucket containing enriched information about weather and hotels for each French city

3- A SQL Database to get the cleaned data from S3
