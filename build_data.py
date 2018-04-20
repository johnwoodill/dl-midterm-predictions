# Builds prediction data

import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler



# Import data

# Presidential approval rating
pres32 = pd.read_csv("data/roosevelt_approval.csv")
pres33 = pd.read_csv("data/truman_approval.csv")
pres34 = pd.read_csv("data/eisenhower_approval.csv")
pres35 = pd.read_csv("data/kennedy_approval.csv")
pres36 = pd.read_csv("data/johnson_approval.csv")
pres37 = pd.read_csv("data/nixon_approval.csv")
pres38 = pd.read_csv("data/ford_approval.csv")
pres39 = pd.read_csv("data/carter_approval.csv")
pres40 = pd.read_csv("data/reagan_approval.csv")
pres41 = pd.read_csv("data/bushh_approval.csv")
pres42 = pd.read_csv("data/clinton_approval.csv")
pres43 = pd.read_csv("data/bushw_approval.csv")
pres44 = pd.read_csv("data/obama_approval.csv")

# Consumer purchasing index
cpi = pd.read_csv("data/cpi.csv")


# Unemployment Rate
unemp_rate = pd.read_csv("data/unemp_rate.csv")


# Congression seats loss
closs = pd.read_csv("data/congress_losses.csv").sort_values('year')

#------------------------------------------------------
def build_data():
    # Combine presidential approval ratings
    pres = pres32.append([pres33, pres34, pres35, pres36, pres37, pres38, pres39, pres40, pres41, pres42, pres43, pres44])
    pres = pres.drop('start_date', 1)
    pres['month'] = pd.DatetimeIndex(pres['end_date']).month
    pres['year'] = pd.DatetimeIndex(pres['end_date']).year
    pres = pres.fillna(method='ffill')
    #
    # Keep only years of midterm elections
    # [1932, 1934, 1936, 1938, 1940, 1942, 1944, 1946, 1948, 1950, 1952, 1954, 1956, 1958, 1960, 
    # 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 
    # 1992, 1994,1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014]
    #
    lyear = list(set(closs.year))
    pres = pres[pres.year.isin(lyear)]
    pres.columns
    #
    # Average over year
    pres = pres.groupby(['year', 'president'], as_index=False).agg({'approve': 'mean', 'disapprove': 'mean', 'unsure': 'mean'})
    #
    # tidy unemployment rate
    unemp_rate['year'] = pd.DatetimeIndex(unemp_rate['DATE']).year
    unemp_rate = unemp_rate.groupby(['year'], as_index=False).agg({'UNRATE':'mean'})
    unemp_rate.columns = ['year', 'unrate']
    #
    #tidy CPI
    cpi.columns = ['year', 'cpi', 'perc_change_cpi']
    #
    #merge pres, cpi, unemp_rate
    pres = pd.merge(pres, closs, on=['year', 'president'], how='inner')
    pres = pd.merge(pres, cpi, how='left', on='year')
    pres = pd.merge(pres, unemp_rate, how='left', on='year')
    #
    # Fill in estimated historical unemployment rate
    pres.loc[0, 'unrate'] = 4.7
    pres.loc[1, 'unrate'] = 3.9
    return(pres)

pres = build_data()
pres.to_feather("data/pres.feather")



# Fit linear model
mod = smf.ols('senate_change ~ approve + disapprove + unsure + pres_party + midterm + perc_change_cpi + unrate', data=pres).fit()
print(mod.summary())

trump = [{'approve': 41.6, 'disapprove': 54.3, 'unsure': 4.1, 'pres_party': 0, 'midterm': 1, 'cpi': 0.886, 'unrate': 4.1}]
trump = pd.DataFrame(trump)

pred = mod.predict(trump)
pred.summary_frame()


print(pred)

#def bootstrap(x):
#    se = []
#    for i in range(0, 1000):
#        indat = x.sample(len(x.index), replace=True)
#        mod = smf.ols('senate_change ~ approve + disapprove + unsure + pres_party + midterm + cpi + unrate', data=indat).fit()
#        X = [{'approve': 41.6, 'disapprove': 54.3, 'unsure': 4.1, 'pres_party': 0, 'midterm': 1, 'cpi': 0.886, 'unrate': 4.1}]
#        se.append(mod.predict(X))
#    pse = np.std(se)
#    return(pse)

#se = bootstrap(pres)


# Keras model
y = pres['house_change']
X = pres[['approve', 'disapprove', 'unsure', 'pres_party', 'midterm', 'perc_change_cpi', 'unrate']]
sc = StandardScaler()
X_scale = sc.fit(X)
X = X_scale.transform(X)

y = np.array(y).reshape(-1, 1)
y_scale = sc.fit(y)
y = y_scale.transform(y)

ksmod = Sequential()
ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
ksmod.add(Dense(8, activation='relu'))
ksmod.add(Dense(4, activation='relu'))
ksmod.add(Dense(1, activation='sigmoid'))
ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

ksmod.fit(X, y, epochs=100, batch_size = 20)

scores = ksmod.evaluate(X, y)
scores

trump = [{'approve': 41.6, 'disapprove': 54.3, 'unsure': 4.1, 'pres_party': 0, 'midterm': 1, 'perc_change_cpi': 0.886, 'unrate': 4.1}]
trump = pd.DataFrame(trump)

trump = X_scale.transform(trump)
tpredict = ksmod.predict(np.array(trump))
print('Predicted House Seats: ', y_scale.inverse_transform(tpredict))



