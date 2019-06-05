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
def dataframe_generator(data, rent=False):

    # Load dataframe
    df = pd.read_csv(data)


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

    # df.drop(columns=['Id', 'idhogar'], inplace=True)


    return df



# Rent Prediction Function
def dataframe_generator_rent(data):
    
    #_______________________________
    # DATAFRAME SETUP
    #_______________________________
    
    # Setting up new dataframe (including rent data)
    df_rent = dataframe_generator(data, rent=True)
    
    # Remove missing values for target (rent)
    df_rent_predict = df_rent.dropna()

    
    #_______________________________
    # CLASSIFICATION SETUP
    #_______________________________
    
    # Partition explanatory and response variables
    X = df_rent_predict.drop(columns=['v2a1', 'Id', 'idhogar'])
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
    rent_pred = clf.predict(df_rent_nan.drop(columns=['v2a1', 'Id', 'idhogar']))
    
    # Fill NaN
    df_rent_nan['v2a1'] = pd.DataFrame(rent_pred).values
    
    # Update full dataframe
    df_rent[df_rent.v2a1.isna()] = df_rent_nan
    
    
    return df_rent



# Transformed DataFrame Generator
def dataframe_generator_trans(data):
    
    # Top Features
    top_features = ['v2a1', 'meaneduc', 'SQBedjefe', 'overcrowding', 'SQBdependency', 'age', 'rooms', 'qmobilephone']

    # Best subset
    winner = \
            [['SQ_SQBedjefe',
              'LOG_qmobilephone',
              'SQ_v2a1',
              'SQBdependency',
              'SQBedjefe',
              'meaneduc',
              'qmobilephone',
              'rooms',
              'LOG_meaneduc',
              'SQ_qmobilephone',
              'v2a1',
              'SQ_overcrowding',
              'LOG_SQBdependency'],
             13,
             0.9257322175732218,
             0.8887133182436542]
            
    # Create rent-inclusive dataframe
    df_rent = dataframe_generator_rent(data)
    
    # Create transformed dataframe
    df_trans = df_rent.copy(deep=True)
    df_trans.drop(columns=top_features, inplace=True)

    for feature in winner[0]:
        if "SQ_" in feature:
            col = feature.split("SQ_")[1]
            df_trans[feature] = df_rent[col] ** 2

        elif "LOG_" in feature:
            col = feature.split("LOG_")[1]
            df_trans[feature] = df_rent[col].apply(lambda x: np.log(x) if x!=0 else x)

        else:
            col = feature
            df_trans[feature] = df_rent[col]
            
    return df_trans