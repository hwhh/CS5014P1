# 1. load the data set
# 2. clean the data set -- over fitting
# 3. shuffle
# 4. normalise the data
# 4. create test / train split
# 5. use Autoencoder to reduce dimensionality of data ?
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

X = df[[
    'Glazing area distribution',
    'Glazing area',
    'Orientation',
    'Heat Loss Coefficient']]
y = df[target_cols[0]]

# Convert the continous outputs (Y1, Y2) to discrete data


X_train, X_test, y_train, y_test = train_test_split(df[feature_cols_copy].values, y, test_size=0.1,
                                            random_state=1)

X_train = preprocessing.normalize(X_train, norm='l2', axis=1)
X_test = preprocessing.normalize(X_test, norm='l2', axis=1)

est = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='uniform')
y = est.fit_transform(y_train[:, np.newaxis])

mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 64), max_iter=5000, alpha=0.1,
                        solver='adam', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=.1, )

for train_split, val_split in StratifiedKFold(n_splits=10).split(np.zeros(len(X_train)), y):
    x_train, y_train, x_val, y_val = X_train[train_split], y[train_split], X_train[val_split], y[val_split]
    mlp.fit(x_train, y_train)
    print("Training set score: %f" % mlp.score(x_val, y_val))

#
# def load_data():
#     df = pd.read_csv('../data/ENB2012_data.csv').astype(float)
#     df = shuffle(df)
#     ss = StandardScaler()
#     scaled_df = ss.fit_transform(df[features])
#     scaled_df = pd.DataFrame(scaled_df, columns=features)
#     final_df = pd.concat([scaled_df, df[target]], axis=1)
#     final_df.reset_index(inplace=True, drop=True)
#     return df
#
#
# df = pd.read_csv('../data/ENB2012_data.csv').astype(float)
#
# columns = {
#     'X1': 'Relative Compactness',
#     'X2': 'Surface Area',
#     'X3': 'Wall Area',
#     'X4': 'Roof Area',
#     'X5': 'Overall Height',
#     'X6': 'Orientation',
#     'X7': 'Glazing area',
#     'X8': 'Glazing area distribution',
#     'Y1': 'Heating Load',
#     'Y2': 'Cooling Load',
# }
# df.rename(index=str, columns=columns, inplace=True)
#
# np.sum(((1.780 * df['Wall Area'].values),
#         (0.860 * (df['Surface Area'].values - (
#                     df['Wall Area'].values + df['Glazing area'].values + df['Roof Area'].values))),
#         (2.260 * df['Glazing area'].values),
#         (0.500 * df['Roof Area'].values)
#         ), axis=0)
#
# df['total_heat_loss'] = np.sum(((1.780 * df['Wall Area'].values),
#                                 (0.860 * (df['Surface Area'].values - (
#                                             df['Wall Area'].values + df['Glazing area'].values + df[
#                                         'Roof Area'].values))),
#                                 (2.260 * df['Glazing area'].values),
#                                 (0.500 * df['Roof Area'].values)
#                                 ), axis=0)
#
#
# fabric_heat_loss = np.sum(((1.780 * df['Wall Area'].values),
#                           (2.260 * (220.5 * df['Glazing area'].values)),
#                           (0.860 * (np.repeat(220.5, len(df.index))))
#                           (0.500 * df['Roof Area'].values)), axis=0, dtype=np.float)
#
# # (2.260 * (220.5 * df['Glazing area'].values)),
#
# # (0.860 * (np.repeat(220.5, len(df.index)).astype(float))),
#
#
# print()
# def train(df):
#     x, x_test, y, y_test = train_test_split(df[features].values, df[target].values, test_size=0.1,
#                                             random_state=234)
#
#     y1 = (LinearRegression().fit(x, y))
#     # y2 = (LinearRegression().fit(x, y[:, 1]))
#     # regr_multirf.fit(x, y)
#     print("Training set score: %f" % y1.score(x_test, y_test))
#     # print("Training set score: %f" % y2.score(x_test, y_test[:, 1]))
#     # for i, (train_split, val_split) in enumerate(StratifiedKFold(n_splits=10).split(np.zeros(len(x)), y)):
#     #     x_train, y_train, x_val, y_val = x[train_split], y[train_split], x[val_split], y[val_split]
#
# # train(load_data())
