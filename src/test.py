import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer


def plot_on_dataset(ax, name):
    # ax.set_title(name)
    # X = MinMaxScaler().fit_transform(X)
    mlps = []
    max_iter = 10000

    for label, param in zip(labels, params):
        print("training: %s" % label)

        mlp = MLPClassifier(verbose=0, random_state=0, max_iter=max_iter, **param)

        for train_split, val_split in StratifiedKFold(n_splits=10).split(np.zeros(len(X_train)), y_train):
            x_train_batch, y_train_batch = X_train[train_split], y_train[train_split]
            x_val, y_val = X_train[val_split], y_train[val_split]
            mlp.fit(x_train_batch, np.ravel(y_train_batch))
            print("Training set score: %f" % mlp.score(x_val, np.ravel(y_val)))
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X_test, y_test))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}
          ]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]

feature_cols = ['Surface Area',
                'Wall Area',
                'Roof Area',
                'Orientation',
                'Glazing area',
                'Glazing area distribution',
                'Heat Loss Coefficient']
target_cols = ['Heating Load', 'Cooling Load']

df = pd.read_csv('../data/ENB2012_data.csv').astype(float)

columns = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing area',
    'X8': 'Glazing area distribution',
    'Y1': 'Heating Load',
    'Y2': 'Cooling Load',
}
df.rename(index=str, columns=columns, inplace=True)

fabric_heat_loss = np.sum(((1.780 * df['Wall Area'].values),
                           (2.260 * (220.5 * df['Glazing area'].values)),
                           (0.860 * (np.repeat(220.5, len(df.index)))),
                           (0.500 * df['Roof Area'].values)), axis=0, dtype=np.float)

ventilation_heat_loss = (20 - 1.6) * np.repeat((0.33 * 0.5 * 771.75), len(df.index))

df['Heat Loss Coefficient'] = fabric_heat_loss + ventilation_heat_loss

x = df[feature_cols].values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

y = est.fit_transform(df[target_cols[0]].values[:, np.newaxis])

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.1, random_state=1)

fig, axes = plt.subplots(figsize=(15, 10))

plot_on_dataset(ax=axes, name='MLP Classifier')

fig.legend(axes.get_lines(), labels, ncol=3, loc="upper center")
plt.show()
