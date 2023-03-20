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

#Set up a filename for automatic saving
filename = ('prediction_RF')
#Reading datafile. It has a header of X|Y|dose|mezo|hght|ra|mezolayers|quartlayers|rfdk|rfdd|rfd
df = pd.read_csv('raw_geodata.csv', header = 0, index_col=[0])
df = df.drop('ra', axis = 1)
df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['X'], df['Y']))

#Following lines grab radium data depending on how it was obtained. Uncomment or comment to set up
# # Use kriging radium
# ra = pd.read_csv('ra_kriging.xyz', sep = " ", header = None, names = ['X', 'Y', 'ra'], na_values = '-99999')
# ra = gpd.GeoDataFrame(ra, geometry = gpd.points_from_xy(ra['X'], ra['Y']))

# # Use RF radium
# ra = pd.read_csv('Ra_randomforest.csv', sep = ",", header = 0, na_values = '-99999')
# ra = ra.drop(['dose', 'mezo', 'hght', 'ra', 'mezolayers', 'quartlayers'], axis = 1)
# ra = ra.drop(ra.columns[0], axis = 1)
# ra = gpd.GeoDataFrame(ra, geometry = gpd.points_from_xy(ra['X'], ra['Y']))

# # Use DNN radium
ra = pd.read_csv('Ra_NN.csv', sep = " ", header = None, names = ['X', 'Y', 'dose', 'mezo', 'hght', 'ratrash', 'mezolayers', 'quartlayers', 'ra'], na_values = '-99999')
ra = ra.drop(['dose', 'mezo', 'hght', 'ratrash', 'mezolayers', 'quartlayers'], axis = 1)
ra = gpd.GeoDataFrame(ra, geometry = gpd.points_from_xy(ra['X'], ra['Y']))

# # Add avg radium (average for geological soil type, required)
raAvg = pd.read_csv("C:\\Users\\Sakhayaan Gavrilyev\\Documents\\GIS data\\radon_msk\\Ra_avg_soiltype_2k.xyz", sep = " ", header = None, names = ['X', 'Y', 'raAvg'], na_values = '0')
raAvg = gpd.GeoDataFrame(raAvg, geometry = gpd.points_from_xy(raAvg['X'], raAvg['Y']))

# align data using geopandas
res = 150
df = df.sjoin_nearest(ra, how = "left", max_distance=res)
df = df.drop(['index_right'], axis = 1)
df = df.sjoin_nearest(raAvg, how = "left", max_distance=res)
df.columns = ['X', 'Y', 'dose', 'mezo', 'hght', 'mezolayers', 'quartlayers', 'rfdk', 'rfdd', 'rfd', 'trash', 'trash', 'trash', 'ra', 'trash', 'trash', 'trash', 'raAvg']
df = df.drop('trash', axis = 1)
df = df[df['ra'].notna()]

#separate into training and test datasets
categoricals = ['mezolayers', 'quartlayers']
#df = df.sample(frac = 0.25)
df = pd.get_dummies(df, columns = categoricals, dummy_na=False)
df = df.drop('rfd', axis = 1)
df = df.drop('rfdk', axis = 1)
gdf = df.copy()
df = df.dropna()
df232 = df.copy()
gdf = gdf[gdf.rfdd.isin(df.rfdd) == False]
df = df.astype({"X": 'float16', "Y": 'float16', 'dose': 'float16', 'mezo': 'float16', 'hght': 'float16',  'rfdd': 'float16','ra': 'float16', 'raAvg': 'float16'})
df_trn = df.sample(frac=0.8)
df_tst = df.drop(df_trn.index)
#features - predictors
ftr_trn = df_trn.copy()
ftr_tst = df_tst.copy()
#labels - real values
lbl_trn = ftr_trn.pop('rfdd')
lbl_tst = ftr_tst.pop('rfdd')

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

#Model itself
model = RandomForestClassifier(n_jobs=24, max_depth = 15)
model = model.fit(ftr_trn, lbl_trn)

predx = model.predict(ftr_tst)
print(r2_score(lbl_tst, predx))
print(mean_squared_error(lbl_tst, predx))

#Save diagnostic data

a = plt.axes(aspect='equal')
plt.scatter(lbl_tst, predx)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig((filename + "_scatter.png"))
plt.show()

error = predx - lbl_tst
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')
plt.savefig((filename + "_error.png"))
plt.show()

with open((filename + '_rsquared.txt'), 'w') as f:
    f.write('R2 ' + str(r2_score(lbl_tst, predx)))
    f.write(' MSE ' + str(mean_squared_error(lbl_tst, predx)))
    
gdf = gdf.astype({"X": 'float16', "Y": 'float16', 'dose': 'float16', 'mezo': 'float16', 'hght': 'float16', 'rfdd': 'float16', 'ra': 'float16', 'raAvg': 'float16'})
gdf = gdf[~gdf.index.isin(df.index)]
gdf = gdf.drop(['rfdd'], axis = 1)
gdf = gdf.dropna()
predix = model.predict(gdf)
gdf['rfdpred'] = predix
#df_pred = df_pred[(np.abs(stats.zscore(df_pred['rfd_pred'])) < 3)]
gdf = undummify(gdf, "_")

q = gdf['rfdpred'].quantile(0.995)
#lo_q = gdf['rfdpred'].quantile(0.01)
hi_q = gdf['rfdpred'].quantile(0.995)
gdf_filt = gdf[(gdf['rfdpred'] < hi_q)]

gdf_filt.to_csv(filename +'.csv')
