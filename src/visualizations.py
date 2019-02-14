# 1. frequency of each class for each feature
# 2. distributions of each feature
# 3. correlation of each input feature with the output feature


import pandas as pd
from sklearn.utils import shuffle


def read_data():
    df = pd.read_csv('../dataset/ENB2012_data.csv')
    df.rename(index=str, columns={
        'X1': 'Relative Compactness',
        'X2': 'Surface Area',
        'X3': 'Wall Area',
        'X4': 'Roof Area',
        'X5': 'Overall Height',
        'X6': 'Orientation',
        'X7': 'Glazing area',
        'X8': 'Glazing area distribution',
        'y1': 'Heating Load',
        'y2': 'Cooling Load',
    })
    df = shuffle(df)


def frequency_distribution():
