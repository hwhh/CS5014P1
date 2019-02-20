# 1. load the data set
# 2. clean the data set -- over fitting
# 3. shuffle
# 4. normalise the data
# 4. create test / train split
# 5. use Autoencoder to reduce dimensionality of data ?

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.utils import shuffle

#


def load_data():
    df = pd.read_csv('../data/ENB2012_data.csv').astype(float)
    df = shuffle(df)
    ss = StandardScaler()
    scaled_df = ss.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_df, columns=features)
    final_df = pd.concat([scaled_df, df[target]], axis=1)
    final_df.reset_index(inplace=True, drop=True)
    return df


def train(df):
    x, x_test, y, y_test = train_test_split(df[features].values, df[target].values, test_size=0.1,
                                            random_state=234)

    y1 = (LinearRegression().fit(x, y))
    # y2 = (LinearRegression().fit(x, y[:, 1]))
    # regr_multirf.fit(x, y)
    print("Training set score: %f" % y1.score(x_test, y_test))
    # print("Training set score: %f" % y2.score(x_test, y_test[:, 1]))
    # for i, (train_split, val_split) in enumerate(StratifiedKFold(n_splits=10).split(np.zeros(len(x)), y)):
    #     x_train, y_train, x_val, y_val = x[train_split], y[train_split], x[val_split], y[val_split]


# train(load_data())