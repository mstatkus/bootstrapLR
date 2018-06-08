# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:11:48 2018

@author: Mike

http://www.statsmodels.org/dev/gettingstarted.html

perform OLS regression on resampled data
for SPM evaluation

"""

#%%
import statsmodels.api as sm
import pandas as pd
import numpy as np
from patsy import dmatrices

#%%

data=pd.read_excel('sample_data_SW175.xlsx')

#%%
y, X = dmatrices('log_k_exp ~ E + S + A + B + V', data=data, return_type='dataframe')

#%%
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

#%%
resampled_data = data.sample(frac=1,replace=True)

#%%
def resample_and_regress(data):
    resampled_data = data.sample(frac=1,replace=True)
    y, X = dmatrices('log_k_exp ~ E + S + A + B + V', data=resampled_data, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res.params

#%%
bootstrap_pars = []
for i in range(10):
    p = resample_and_regress(data)
    bootstrap_pars.append(p)

#%%
bsp=pd.concat(bootstrap_pars,axis=1)

#%%
def bootstrap(data,iterations):
    "approx 1 s per 100 iterations"
    bootstrap_pars = []
    for i in range(iterations):
        p = resample_and_regress(data)
        bootstrap_pars.append(p)
    bsp=pd.concat(bootstrap_pars,axis=1)
    bsp = bsp.transpose()
    return bsp

#%%
import timeit
start_time = timeit.default_timer()

bsp = bootstrap(data,1000)

elapsed = timeit.default_timer() - start_time
print (elapsed)

#%%
for c in bsp.columns:
    bsp.hist(column=c,bins=20)
    
#%%
bootstrap_CL = bsp.quantile([0.025,0.975])

#%%
