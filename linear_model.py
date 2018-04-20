import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
pres = pd.read_feather("data/pres.feather")

# Fit linear model
mod = smf.ols('senate_change ~ approve + disapprove + unsure + pres_party + midterm + perc_change_cpi + unrate', data=pres).fit()
print(mod.summary())

trump = [{'approve': 41.6, 'disapprove': 54.3, 'unsure': 4.1, 'pres_party': 0, 'midterm': 1, 'perc_change_cpi': 0.886, 'unrate': 4.1}]
trump = pd.DataFrame(trump)

pred = mod.predict(trump)
print(pred[0])

def bootstrap(x):
    se = []
    for i in range(0, 1000):
        indat = x.sample(len(x.index), replace=True)
        mod = smf.ols('senate_change ~ approve + disapprove + unsure + pres_party + midterm + cpi + unrate', data=indat).fit()
        X = [{'approve': 41.6, 'disapprove': 54.3, 'unsure': 4.1, 'pres_party': 0, 'midterm': 1, 'cpi': 0.886, 'unrate': 4.1}]
        se.append(mod.predict(X))
    pse = np.std(se)
    return(pse)

se = bootstrap(pres)

print('Estimated Congressional Seats: {} +- {}'.format(pred[0].round(3), se.round(3)))