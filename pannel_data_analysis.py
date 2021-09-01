# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:56:38 2021

@author: Kered
"""

from linearmodels.datasets import wage_panel
import pandas as pd
from linearmodels.panel import PooledOLS
import statsmodels.api as sm
import seaborn as sns
from linearmodels.iv import IV2SLS
import pandas as pd

#Importing data 
df = pd.read_csv('fitbit_data.csv', index_col=0)
df.head()

#Data cleaning 
df.drop('Calories_y',axis=1, inplace=True)
#Putting the right format
df.Date = pd.to_datetime(df.Date)
# Import and preprocess data
dataset = pd.read_csv('fitbit_data.csv', usecols = ['Id', 'Date', 'VeryActiveMinutes', 'Calories_x'],\
 index_col = ['Id', 'Date'])
years = dataset.index.get_level_values('Date').to_list()
dataset['Date'] = pd.Categorical(years)

#Exploring type
dataset.dtypes
#OlS old school linear regression 
from statsmodels.formula.api import ols
model = ols('Calories_x ~ VeryActiveMinutes', data=df) # dependent ~ independent

model_fit = model.fit() #calculates everything

#printing summary
model_fit.summary()

#Deduction and exploring main feature of the columns datas
df.corr().Calories_x.sort_values()
#Visualization
sns.heatmap(df.corr())

#Using more complex model
ols('Calories_x ~ VeryActiveMinutes + TotalSteps + VeryActiveDistance + LightActiveDistance + FairlyActiveMinutes + TotalDistance + TrackerDistance ', data=df).fit().summary()

#Deduction and interpretation
#The condition number is large, 3.01e+03. This might indicate that there are
#strong multicollinearity or other numerical problems.

#plotting the results
y_pred = model_fit.predict()
resids = model_fit.resid
y_pred
#Plotting residual results
resids.plot()
#Observing the distribution of the residual
sns.distplot(resids).set(title='Distribution of residuals')

#Observation of the shape of the distribution
#presence of a shoulder shape
#Conclusion : strong multicollinearity 

#Linear regression with scikit-learn
df1 = df.copy()
df1['y_pred']=y_pred
df1 = df1.sort_values(by='Calories_x')
#Visualization of the regression with sns
sns.scatterplot(data = df, x = 'VeryActiveMinutes', y = 'Calories_x')
sns.lineplot(data = df1, x = 'VeryActiveMinutes', y = 'y_pred')


#Choosing the right columns for the model
X = df.drop(['SedentaryMinutes',      
'SedentaryActiveDistance' ,    
'LoggedActivitiesDistance' ,   
'ModeratelyActiveDistance' ,   
'LightlyActiveMinutes'   ,     
'FairlyActiveMinutes'     ,    
'LightActiveDistance'     ,    
'VeryActiveDistance'    ,      
'TotalSteps'      ,          
'VeryActiveMinutes'   ,        
'TotalDistance'  ,             
'TrackerDistance','Date','Weekday' ], axis=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, df.Calories_x)
#Coefficients of the regression
lr.coef_
lr.score(X, df.Calories_x)


#Pooled OLS regression

df = pd.get_dummies(df, columns=['Weekday'], drop_first=True)
df.Date = pd.to_datetime(df.Date)
year = pd.Categorical(df.Date)
df = df.set_index(['Id', 'Date'])
df['year'] = year
exog = sm.add_constant(df.VeryActiveMinutes)
mod = PooledOLS(df.Calories_x, exog, check_rank=False)
pooled_res = mod.fit(cov_type='clustered', cluster_entity=True)
#Print summary
print(pooled_res)

pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)

# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids

#Visual representation
# 3A. Homoskedasticity
import matplotlib.pyplot as plt
 # 3A.1 Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 10)
plt.show()


#residuals-plot represents predicted values 
#(x-axis) vs. residuals (y-axis). 
#If the plotted data points spread out,
# this is an indicator for growing variance
# and thus, for heteroskedasticity. 
#Since this seems to be the case in 
#our example, we might have the first violation

# 3A.2 White-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
pooled_OLS_dataset = pd.concat([df, residuals_pooled_OLS], axis=1)
pooled_OLS_dataset = pooled_OLS_dataset.drop(year, axis = 1).fillna(0)
exog = sm.tools.tools.add_constant(dataset['Calories_x']).fillna(0)
white_test_results = het_white(pooled_OLS_dataset['residual'], exog)
labels = ['LM_Stat', 'LM p_val', 'F_Stat', 'Fp_val'] 
print(dict(zip(labels, white_test_results)))
# 3A.3 Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(pooled_OLS_dataset['residual'], exog)
labels = ['LM_Stat', 'LM p_val', 'F_Stat', 'Fp_val'] 
print(dict(zip(labels, breusch_pagan_test_results)))

# 3.B Non-Autocorrelation
# Durbin-Watson-Test
from statsmodels.stats.stattools import durbin_watson

durbin_watson_test_results = durbin_watson(pooled_OLS_dataset['residual']) 
print(durbin_watson_test_results)

########FE-/RE-model will be more suitable
# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects
exog = sm.add_constant(df.VeryActiveMinutes)
endog = df.Calories_x
# random effects model
model_re = RandomEffects(endog, exog) 
re_res = model_re.fit() 
# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit() 
#print results
print(fe_res)
print(re_res)


#More accurate model(FE,RE)
#More columns in the model

# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects
# TotalSteps + VeryActiveDistance + LightActiveDistance + FairlyActiveMinutes + TotalDistance + TrackerDistance
exog = sm.add_constant(df[['TotalSteps','VeryActiveDistance','LightActiveDistance','FairlyActiveMinutes','TotalDistance','TrackerDistance']])
endog = df.Calories_x
# random effects model
model_re = RandomEffects(endog, exog) 
re_res = model_re.fit() 
# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit() 
#print results
print(fe_res)
print(re_res)

#Step 4: Perform Hausman-Test
import numpy.linalg as la
from scipy import stats
import numpy as np
def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.covdf = b[np.abs(b) < 1e8].size
    
    df = b[np.abs(b) < 1e8].size
    
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B)) 
    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval

hausman_results = hausman(fe_res, re_res) 

print('chi-Squared:' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value:'  + str(hausman_results[2]))
#Visualization
sns.scatterplot(x=df.VeryActiveMinutes, y=df.Calories_x)
plt.show()


# Fit the first stage regression and print summary
results_fs = sm.OLS(df.Calories_x, exog, check_rank=False).fit()   #dependant and Exogenous or right-hand-side variables
print(results_fs.summary())


mean_expr = np.mean(df['Calories_x'])
results_fs.predict(exog=[1, mean_expr])

# Plot predicted values

fix, ax = plt.subplots()
# ax.scatter(df.VeryActiveMinutes,y=results_fs.predict(), alpha=0.5,
#         label='predicted')

# Plot observed values

ax.scatter(df.VeryActiveMinutes, df['Calories_x'], alpha=0.5,
        label='observed')

ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('VeryActiveMinutes')
ax.set_ylabel('Calories')
plt.show()


# Fit the second stage regression and print summary
#model = ols('Calories_x ~ VeryActiveMinutes', data=df)
# Add a constant term
df['const'] = 1

# Estimate the first stage regression


reg1 = sm.OLS(endog=df['Calories_x'],
              exog=df[['const', 'VeryActiveMinutes']],
              missing='drop').fit()

# Retrieve the residuals
df['resid'] = reg1.resid

# Estimate the second stage residuals
reg2 = sm.OLS(endog=df['TotalSteps'],
              exog=df[['const', 'Calories_x', 'resid']],
              missing='drop').fit()

print(reg2.summary())





