# 1. load the data set
# 2. clean the data set -- over fitting
# 3. shuffle
# 4. normalise the data
# 4. create test / train split
# 5. use Autoencoder to reduce dimensionality of data ?


#
#
#
# def load_data():
#     df = pd.read_csv('../data/ENB2012_data.csv')
#     print()
#
#
# load_data()


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/henryhargreaves/Documents/University/Year_4/CS5014/Practicals/P1/data/ENB2012_data.csv')
feature_cols = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
                'Overall Height', 'Orientation', 'Glazing area', 'Glazing area distribution']
target_cols = ['Heating Load', 'Cooling Load']

cols = ['Overall Height', 'Roof Area']
df.rename(index=str, columns={
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
}, inplace=True)
df.index.names = ['obs']
df.columns.names = ['vars']
tidy = (
    df[cols].stack()  # pull the columns into row variables
        .to_frame()  # convert the resulting Series to a DataFrame
        .reset_index()  # pull the resulting MultiIndex into the columns
        .rename(columns={0: 'val'})  # rename the unnamed column
)
sns.lmplot(x='obs', y='val', col='vars', hue='vars', data=tidy)

q1_heating, q3_heating = np.percentile(df['Heating Load'], [25, 75])
q1_cooling, q3_cooling = np.percentile(df['Cooling Load'], [25, 75])

df_copy = df.copy()
df_copy['Heating Load'] = df_copy['Heating Load'].apply(lambda value: 'low' if value <= q1_heating
else 'medium' if value <= q3_heating
else 'high')
df_copy['Cooling Load'] = df_copy['Cooling Load'].apply(lambda value: 'low' if value <= q1_cooling
else 'medium' if value <= q3_cooling
else 'high')
df_copy.reset_index(inplace=True, drop=True)
subset_df = df_copy[feature_cols]
ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=feature_cols)
final_df = pd.concat([scaled_df, df_copy['Heating Load']], axis=1)
final_df.reset_index(inplace=True, drop=True)

final_df.head()

# pc = parallel_coordinates(final_df, 'Heating Load', color=('#00cc00', '##ff9900', '#e62e00'))
