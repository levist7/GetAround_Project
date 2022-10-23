# GetAround_EDA_ML_Dashboard_API_Project
Project on Exploratory Data Analysis, Supervised Machine Learning, Dashboard Building and API Creation

Link to dashboard: http://getaround-dashboard-threshold.herokuapp.com

Link to API/docs: http://getaround-api-xgboost.herokuapp.com/docs

Linkt to API/predict: http://getaround-api-xgboost.herokuapp.com/predict

Getaround isn online rental car rental service but in an Airbnb way. When renting a car, users have to complete a checkin flow at the beginning of the rental and a checkout flow at the end of the rental in order to:

*  Assess the state of the car and notify other parties of pre-existing damages or damages that occurred during the rental.
*  Compare fuel levels.
*  Measure how many kilometers were driven.
*  The checkin and checkout of our rentals can be done with three distinct flows:

Mobile rental agreement 📱 on native apps where driver and owner meet and both sign the rental agreement on the owner’s smartphone 

Connect where  the driver doesn’t meet the owner and opens the car with his smartphone

Paper contract 📝 is negligible

The goal is to resolve those issues and give insights on implementing a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental. It can solve the late checkout issue but may hurt Getaround/owners revenues.

Deliverabes availabe  📬:

1-  A dashboard in production (accessible via a web page above)

2-  An documented online API on Heroku server containing /predict endpoint that respect the technical descriptions. 

3-  Exploratory data analysis on Getaround data and test of various machine learning models
