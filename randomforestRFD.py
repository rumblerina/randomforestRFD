from pyexpat import native_encoding
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

#Reading datafile. It has a header of X|Y|dose|mezo|hght|ra|mezolayers|quartlayers|rfdk|rfdd|rfd
df = pd.read_csv('raw_geodata.csv', header = 0, index_col=[0])

#Preparing the data
categoricals = ['mezolayers', 'quartlayers']
df = df.sample(frac = 0.25)
df = pd.get_dummies(df, columns = categoricals, dummy_na=False)
#Here we use RFD data obtained thru kriging, as actually measured data lacks some of the geodata, which is why 25% of it is only used
df = df.drop('rfdd', axis = 1)
df = df.drop('rfd', axis = 1)
df = df.astype({"X": 'float16', "Y": 'float16', 'dose': 'float16', 'mezo': 'float16', 'hght': 'float16', 'ra': 'float16', 'rfdk': 'float16'})
#training = the fraction on which learning will be done 
#testing - fraction on which how well the education was done will be checked
df_trn = df.sample(frac=0.8)
df_tst = df.drop(df_trn.index)
#features - predictors
ftr_trn = df_trn.copy()
ftr_tst = df_tst.copy()
#labels - real values
lbl_trn = ftr_trn.pop('rfdk')
lbl_tst = ftr_tst.pop('rfdk')

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

#the model itself
model = RandomForestClassifier(n_jobs=24, max_depth = 10)
model = model.fit(ftr_trn, lbl_trn)

#checking model accuracy
predx = model.predict(ftr_tst)
print(r2_score(lbl_tst, predx))

#preparing the predictor data
df_pred = pd.read_csv('raw_geodata.csv', header = 0, index_col=[0])
df_pred = df_pred.astype({"X": 'float16', "Y": 'float16', 'dose': 'float16', 'mezo': 'float16', 'hght': 'float16', 'ra': 'float16', 'rfdk': 'float16'})
df_pred = df_pred.drop('rfdd', axis = 1)
df_pred = df_pred.drop('rfd', axis = 1)
df_pred = pd.get_dummies(df_pred, columns = categoricals, dummy_na=False)
df_pred = df_pred[~df_pred.index.isin(df.index)]
df_pred = df_pred.drop(['rfdk'], axis = 1)

#Predicting and finishing
predix = model.predict(df_pred)
df_pred['rfdpred'] = predix
#df_pred = df_pred[(np.abs(stats.zscore(df_pred['rfd_pred'])) < 3)]
df_pred = undummify(df_pred, "_")
print(df_pred)
df_pred.to_csv('randomforest.csv')
