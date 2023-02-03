"""
    this script is taken from one of my earlier projects.
    ref. https://github.com/LHB-Group/Civil-Work-Bidding-And-Investment-Helper

    description:
        This script tests different machine learning models by using cross validation
        provided a cleaned database, a list of features, a target column name and parameters of fold number and random state.
        It was inspired by a Kaggle notebook: https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard
        Parameters used in machine learning models are tuned by grid search. Some of the grid search experiments are presented
        in a notebook located on Notebooks/predictive_models.ipynb . Refer to this file for more details.

    inputs:
        User needs to input the following variables
        fname1: direction to the database file
        description_ML: explain why you perform this experiment
        features_list: name of feature columns to be used in machine learning model
        target_variable: name of target column to be used in machine learning model
        k_fold: number of folds in cross validation
        random_state: random state
"""
# Libraries
import pandas as pd
import numpy as np

# sklearn libraries
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb  # library independent of sklearn


from functions import cross_validate_score, score_ML_log
import warnings
warnings.filterwarnings('ignore')

# Start of User Input
fname1 = "../data/get_around_pricing_project_clean.csv"
dataset = pd.read_csv(fname1, low_memory=False)

# Number of folds in cross validation and random state
n_folds = 4
random_state = 0

# add why you do this experiment
description_ML = "dataset cleanerd | models tuned | all features"

"""
    conclusions obtained in eda_pricing notebook
    rental_price_per_day is correlated :
    * **very well** with engine_power, automatic_car, mileage
    * **well** with  model_key_, has_getaround_connect, has_gps, car_type, private_parking_available, has_air_conditioning, has_speed_regulator
    * **slightly** with paint_color_, fuel_, winter_tires
"""

features_list = [
    "model_key_",
    "mileage",
    "engine_power",
    "fuel_",
    "paint_color_",
    "car_type",
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires"
]

target_variable = "rental_price_per_day"
# End of User Input

# Separate target variable Y from features X

print("Separating labels from features...")
X = dataset.loc[:, features_list]
Y = dataset.loc[:, target_variable]

print("\n...Done...\n")
print()

print('Y : ')
print(Y.head())
print()
print('X :')
print(X.head())

# Automatically detect names of numeric/categorical columns
numeric_features = []
categorical_features = []
for i, t in X.dtypes.iteritems():
    if ('float' in str(t)) or ('int' in str(t)):
        numeric_features.append(i)
    else:
        categorical_features.append(i)

print('\nFound numeric features ', numeric_features)
print('\nFound categorical features ', categorical_features)

# Since we use Kfold, we don't divide data into train and test data set!
# And we use all dataset

X_train = X
Y_train = Y
print("\n...Done...\n")
print()

# Create pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    # missing values will be replaced by columns' median
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create pipeline for categorical features
categorical_transformer = Pipeline(
    steps=[
        # first column will be dropped to avoid creating correlations between
        # features
        ('encoder', OneHotEncoder(drop='first'))
    ])

# Use ColumnTransformer to make a preprocessor object that describes all
# the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])  # Separate target variable Y from features X


# Preprocessings on train set
print("\nPerforming preprocessings...\n")
# print(X.head())
X_train = preprocessor.fit_transform(X_train)
print('\n...Done...\n')


# Definition of models
print("\nDefining machine learning models...\n")
regressor0 = LinearRegression()
# lasso regression
lasso = Lasso(alpha=0.0001, random_state=random_state)
# elastic net regression
"""
Elastic Net Regression:
    Elastic net is a combination of the two regularized linear regression: ridge and lasso.
    Ridge utilizes an L2 penalty and lasso uses an L1 penalty.
    Elastic net uses both the L2 and the L1 penalties.
"""
ENet = ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=random_state)


# xgb
"""
    Extreme Gradient Boosting - XGB Model
    XG Model is one of the most used algorithm in ML. It is an implementation of gradient boosted decision trees.
    It was designed for speed and performance. It is an enhanced gradient boosting library and uses a gradient boosting framework.
    Ref. to the algorithm paper:
    XGBoost: A Scalable Tree Boosting System by Tianqi Chen, Carlos Guestrin
    link : https://arxiv.org/abs/1603.02754

    Parameters were tuned by GridSearch.
"""
model_xgb = xgb.XGBRegressor(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=250,
    colsample_bytree=0.9,
    subsample=0.8,
    random_state=random_state
)


# random forest
"""
    A random forest regressor.
    'A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.'

    Parameters were obtained by GridSearch.
"""
randomForestRegressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=4,
    random_state=random_state
)

print('\n...Done...\n')

models = [regressor0,
          lasso,
          ENet,
          model_xgb,
          randomForestRegressor
          ]

model_names = ["Linear Regressor Model",
               "Lasso Model",
               "Elastic Net Regressor Model",
               "XGBoost Model",
               "Random Forest Regressor Model"
               ]

count = 0

for model in models:
    print("\n**********" + model_names[count] + "**********")
    print("**********Scores on test set**********\n")
    cv_scores = cross_validate_score(
        model, n_folds, random_state, X_train, Y_train)
    score_1 = cv_scores['test_r2']
    print(
        "R2 score - mean : {:.4f}  |  std : {:.4f}\n".format(score_1.mean(), score_1.std()))

    score_2 = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
    print(
        "\nRoot mean squared error - mean : {:.4f}  |  std : {:.4f}\n".format(
            score_2.mean(),
            score_2.std()))

    score_3 = np.sqrt(-cv_scores['test_neg_mean_squared_log_error'])
    print(
        "\nRMSLE -logarithmic error - mean : {:.4f}  |  std : {:.4f}\n".format(
            score_3.mean(),
            score_3.std()))

    score_4 = -cv_scores["test_neg_mean_absolute_percentage_error"]
    print(
        "\nMean absolute percentage error - mean : {:.3f}  |  std : {:.3f}\n".format(
            score_4.mean(),
            score_4.std()))

    score_5 = cv_scores["test_explained_variance"]
    print(
        "\nExplained variance score - mean : {:.4f}  |  std : {:.4f}\n".format(
            score_5.mean(),
            score_5.std()))
    print("----------------END--------------------")
    count = count + 1
    score_ML_log(
        fname1,
        dataset,
        model,
        target_variable,
        categorical_features,
        numeric_features,
        description_ML,
        n_folds,
        random_state,
        score_1,
        score_2,
        score_3,
        score_4,
        score_5,
        flog="..\\tracking\\exp_logs.csv")
