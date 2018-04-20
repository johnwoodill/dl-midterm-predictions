import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler

# Read feather data
pres = pd.read_feather("data/pres.feather")

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

