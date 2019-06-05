# Kaggle - Costa Rican Poverty Prediction
# Nema Sobhani and David LaCharite
# Callable Functions for Notebooks

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor



# DataFrame Generator Function
def dataframe_generator(rent=False):

    # Load dataframe
    df = pd.read_csv('train.csv')


    #_______________________________
    # RENT
    #_______________________________

    if rent == False:
        df.drop('v2a1', axis=1, inplace=True)


    #_______________________________
    # NUMBER OF TABLETS
    #_______________________________

    df.v18q1.fillna(0, inplace=True)


    #_______________________________
    # YEARS BEHIND IN SCHOOL
    #_______________________________

    df.drop("rez_esc", axis=1, inplace=True)


    #_______________________________
    # MIXED CATEGORY VARIABLES
    #_______________________________

    df.drop(columns=['edjefe', 'edjefa', 'dependency'], inplace=True)


    #_______________________________
    # EDUCATION
    #_______________________________

    # Find missing education value ids
    missing_ids = df[df['meaneduc'].isna()]['meaneduc'].keys()

    # Adult (18+) education by household
    educ_by_household = df[df.age >= 18].groupby('idhogar')['escolari'].mean()

    # Iterate over missing values and set them to correct value
    for i in missing_ids:

        household = df.loc[i, 'idhogar']

        # Update 'meaneduc'
        try:
            df.loc[i, 'meaneduc'] = educ_by_household[household]
        except:
            df.loc[i, 'meaneduc'] = 0

        # Update 'SQBmeaned'
        try:
            df.loc[i, 'SQBmeaned'] = educ_by_household[household] ** 2
        except:
            df.loc[i, 'SQBmeaned'] = 0


    #_______________________________
    # DROPPING IDENTIFIERS
    #_______________________________

    df.drop(columns=['Id', 'idhogar'], inplace=True)


    return df



# Rent Prediction Function
def dataframe_generator_rent():
    
    #_______________________________
    # DATAFRAME SETUP
    #_______________________________
    
    # Setting up new dataframe (including rent data)
    df_rent = dataframe_generator(rent=True)
    
    # Remove missing values for target (rent)
    df_rent_predict = df_rent.dropna()

    
    #_______________________________
    # CLASSIFICATION SETUP
    #_______________________________
    
    # Partition explanatory and response variables
    X = df_rent_predict.drop(columns='v2a1')
    y = df_rent_predict['v2a1']

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=12345)
    
    
    #_______________________________
    # CLASSIFICATION 
    # (using random forest because it consistently gave highest score)
    #_______________________________
    
    # XGB
    # clf = xgb.XGBClassifier(max_depth=6,n_estimators=100, n_jobs=-1, subsample=.7)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    
    # Random Forest
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    
    
    #_______________________________
    # FILL NAN USING PREDICTED VALUES FROM MODEL
    #_______________________________
    
    # Prepare data to fill in predicted values for rent
    df_rent_nan = df_rent[df_rent.v2a1.isna()]
    
    # Predict using model
    rent_pred = clf.predict(df_rent_nan.drop(columns='v2a1'))
    
    # Fill NaN
    df_rent_nan['v2a1'] = pd.DataFrame(rent_pred).values
    
    # Update full dataframe
    df_rent[df_rent.v2a1.isna()] = df_rent_nan
    
    
    return df_rent