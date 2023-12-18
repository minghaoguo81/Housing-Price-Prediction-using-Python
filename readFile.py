import pandas as pd
import dataProcess
import numpy as np


def trainingFile():
    dftrain = pd.read_csv('training_dataset-1.csv')

    # preprocessing

    #delete unreasonable data
    dftrain = dftrain[dftrain['bedrooms'] != 33]
    dftrain = dftrain[(dftrain['bedrooms'] != 0) | (dftrain['bathrooms'] <= 1)]

    #calculate renovated time
    dftrain['renovated_time'] = dataProcess.renovatedTime(dftrain['date'], dftrain['yr_built'], dftrain['yr_renovated'])

    #delete all data that show sell date is prior than built date or renoveted date
    dftrain = dftrain[dftrain['renovated_time'] >= 0]

    dftrain['date'] = dataProcess.dateConvert(dftrain['date'])
    dftrain.reset_index(inplace = True)
    dftrain['isRenovated'] = dataProcess.isRenovated(dftrain['yr_renovated'])
    dftrain.pop('yr_renovated')
    dftrain.pop('index')

    
    dftrain['sqft_lot'] = np.log(dftrain['sqft_lot'])
    dftrain['sqft_lot15'] = np.log(dftrain['sqft_lot15'])

    popList = ['id', 'zipcode', 'sqft_living']
    for id in popList:
        dftrain.pop(id)
    return dftrain


def testFile():
    dftest = pd.read_csv('test_dataset-1.csv')

    # preprocessing
    dftest['renovated_time'] = dataProcess.renovatedTime(dftest['date'], dftest['yr_built'], dftest['yr_renovated'])
    dftest['date'] = dataProcess.dateConvert(dftest['date'])
    dftest['isRenovated'] = dataProcess.isRenovated(dftest['yr_renovated'])
    dftest.pop('yr_renovated')

    dftest['sqft_lot'] = np.log(dftest['sqft_lot'])
    dftest['sqft_lot15'] = np.log(dftest['sqft_lot15'])
    popList = ['id', 'zipcode', 'sqft_living']
    for id in popList:
        dftest.pop(id)
    return dftest


predictor_list = ['date', 'bedrooms', 'bathrooms',
       'floors', 'waterfront','view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'sqft_lot',
       'yr_built', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'renovated_time', 'isRenovated']