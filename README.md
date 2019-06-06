# Costa Rican Household Poverty Level Prediction

## Team Members:
    - Nema Sobhani
    - David LaCharite

## Binder Notebook Path

https://mybinder.org/v2/gh/nemasobhani/Kaggle-Costa-Rican-Poverty-Prediction/master?filepath=FINAL_REPORT.ipynb

## Motivation & Data
  - Using Data to **Predict Costa Rican Household Poverty** (from Kaggle found [here](https://www.kaggle.com/c/costa-rican-household-poverty-prediction)).

## Libraries/Tools
    - Pandas
    - Numpy
    - Scikit Learn
    - AWS (maybe)
    - Google Colab

## Approach

1.  Data Exploration/Visualization
    -  Assess and visualize impact of different features on poverty level
    -  Find important features
        -  Manually from visualization
        -  Using sklearn feature selection

2.  Cleaning
    -  Decide on how to handle null values
    -  Filling missing values with regression
    
3.  Feature Selection/Engineering
    -  Finalize features to include in model
    -  Perform logical transformations (as well as from high scoring random subsets)
    
4.  Classification
    -  Multinomial Logistic Regression 
    -  Support Vector Machines
    -  Random Forest
    -  XGBoost
    
5.  Tuning of hyper-parameters
    -  RandomizedSearchCV / GridSearchCV


## Timeline

Action | Date
--- | --- 
Data Collection/Cleaning | May 20th
Model Construction/Testing | May 27th
Application Interface development | June 3th
