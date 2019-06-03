# Kaggle - Costa Rican Poverty Prediction
# Nema Sobhani and David LaCharite
# Callable Functions for Notebooks

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split



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
