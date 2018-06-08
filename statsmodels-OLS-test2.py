# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:00:58 2018

@author: Mike

DFFITs and other regression quality tests

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
fitted_values = res.fittedvalues
influence = res.get_influence()
(dffits, dffits_threshold) = influence.dffits
dffits = pd.Series(dffits)
residuals = res.resid

#%%
