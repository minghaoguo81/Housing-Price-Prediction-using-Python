import numpy as np
from datetime import *
import pandas as pd


def dateConvert(date_frame):
    """
    Convert the original date to time difference to 2014.05.01
    """
    date_frame = np.array(date_frame)

    date0 = datetime.strptime('20140501', "%Y%m%d")
    for i in range(0, len(date_frame)):
        # print(type(date_frame[i]))
        # print(type(date0))
        date_frame[i] = datetime.strptime(date_frame[i], "%Y%m%dT000000") - date0
        date_frame[i] = date_frame[i].days
    return date_frame.astype(float)


def renovatedTime(sale_date, yr_built, yr_renovated):
    output = np.array([0] * len(sale_date))
    sale_date = np.array(sale_date)
    yr_built = np.array(yr_built)
    yr_renovated = np.array(yr_renovated)
    for i in range(0, len(sale_date)):
        if yr_renovated[i] != 0:
            temp = datetime.strptime(str(yr_renovated[i]), "%Y")
            output[i] = (datetime.strptime(str(sale_date[i]), "%Y%m%dT000000") - temp).days
        else:
            temp = datetime.strptime(str(yr_built[i]), "%Y")
            output[i] = (datetime.strptime(str(sale_date[i]), "%Y%m%dT000000") - temp).days
    return output.astype(float)


def isRenovated(yr_renovated):
    output = np.array([0] * len(yr_renovated))
    for i in range(0, len(yr_renovated)):
        if yr_renovated[i] != 0:
            output[i] = 0
        else:
            output[i] = 1
    return output

def addViewDummies(dftrain):
    df = pd.get_dummies(dftrain, columns=['view'], prefix= 'view')
    df['view_0'] = df['view_0'].astype('int')
    df['view_1'] = df['view_1'].astype('int')
    df['view_2'] = df['view_2'].astype('int')
    df['view_3'] = df['view_3'].astype('int')
    df['view_4'] = df['view_4'].astype('int')
    return df

def addConditionDummies(dftrain):
    df = pd.get_dummies(dftrain, columns=['condition'], prefix='condition')
    df['condition_1'] = df['condition_1'].astype('int')
    df['condition_2'] = df['condition_2'].astype('int')
    df['condition_3'] = df['condition_3'].astype('int')
    df['condition_4'] = df['condition_4'].astype('int')
    df['condition_5'] = df['condition_5'].astype('int')
    return df

def addGradeDummies(dftrain):
    dftrain['grade_poor'] = 0
    dftrain['grade_average'] = 0
    dftrain['grade_high'] = 0
    dftrain.loc[(dftrain['grade'] >= 1) & (dftrain['grade'] <= 3), 'grade_poor'] = 1
    dftrain.loc[(dftrain['grade'] >= 4) & (dftrain['grade'] <= 10), 'grade_average'] = 1
    dftrain.loc[(dftrain['grade'] >= 11) & (dftrain['grade'] <= 13), 'grade_high'] = 1
    dftrain.pop('grade')
    return dftrain
