# 1. load the data set
# 2. clean the data set -- over fitting
# 3. shuffle
# 4. normalise the data
# 4. create test / train split
# 5. use Autoencoder to reduce dimensionality of data ?
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from keras import backend as K, Input, models, optimizers, losses


def create_model():
    # Input
    inp = Input(shape=(4,))  # TODO batch generator - 90000 samples in total

    # Layer 1
    x = Dense(1000, activation="relu")(inp)
    x = Dropout(0.3)(x)

    # Layer 2
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Output
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    out = Dense(30, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model


feature_cols = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
                'Overall Height', 'Orientation', 'Glazing area', 'Glazing area distribution']
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

if 'Heat Loss Coefficient' not in feature_cols:
    feature_cols += ['Heat Loss Coefficient']

X = df[[
    'Glazing area distribution',
    'Glazing area',
    'Orientation',
    'Heat Loss Coefficient']]
y = df[target_cols[0]]
est = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
y = est.fit_transform(y[:, np.newaxis])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train = preprocessing.normalize(X_train, norm='l2', axis=0)
X_test = preprocessing.normalize(X_test, norm='l2', axis=0)

skf = StratifiedKFold(n_splits=10, shuffle=True)
for i, (train_split, val_split) in enumerate(skf.split(X_train, y_train)):
    K.clear_session()
    X_train_batch, y_train_batch, X_val, y_val = X_train[train_split], y[train_split], X_train[val_split], y_train[val_split]
    checkpoint = ModelCheckpoint('./out/best_%d.h5' % i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    tb = TensorBoard(log_dir=os.path.join('./out', 'logs') + '/fold_%i' % i, write_graph=True)
    callbacks_list = [checkpoint, early, tb]

    print('#' * 50)
    print('Fold: ', i)

    model = create_model()

    model.fit(X_train_batch,
              to_categorical(y_train_batch, num_classes=30),
              validation_data=(X_val, to_categorical(y_val, num_classes=30)),
              callbacks=callbacks_list,
              batch_size=64, epochs=50)

    model.load_weights('out/best_%d.h5' % i)

# x = Input(shape=(4,) )
#
# for train_split, val_split in StratifiedKFold(n_splits=10).split(np.zeros(len(X_train)), len(y[:, 0])):
#     x_train, y_train, x_val, y_val = X_train[train_split], y[train_split], X_train[val_split], y[val_split]
#     mlp.fit(x_train, y_train)
#     print("Training set score: %f" % mlp.score(x_val, y_val))
