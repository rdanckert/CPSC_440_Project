import pandas as pd
import numpy as np
import random as rd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
pitch_data = pd.read_csv("C:/Users/ranen/Downloads/stats.csv")
print(pitch_data.columns)

## xwoba is our response variable, lower xwoba is better for a pitcher
## xwoba is a pandas series object
xwoba = pitch_data.loc[:,"xwoba"]

## ff is fastball, sl is slider, ch is changeup,
cols_to_drop = ['last_name, first_name', 'player_id', 'year', 'pa', 'xwoba', 'n_cu_formatted',
       'cu_avg_speed', 'cu_avg_spin', 'cu_avg_break_x', 'cu_avg_break_z',
       'cu_avg_break', 'n_si_formatted', 'si_avg_speed', 'si_avg_spin',
       'si_avg_break_x', 'si_avg_break_z', 'n_fc_formatted', 'fc_avg_speed',
       'fc_avg_spin', 'fc_avg_break_x', 'fc_avg_break_z', 'n_fs_formatted',
       'fs_avg_speed', 'fs_avg_spin', 'fs_avg_break_x', 'fs_avg_break_z']
pitch_data = pitch_data.drop(columns=cols_to_drop)


## Replace missing values with median
pitch_medians = pitch_data.median(axis = 0, numeric_only=True)
pitch_data = pitch_data.fillna(value = pitch_medians)

## We will randomly assign 20% of the data as a validation set
n_total, d_total = pitch_data.shape
pitchers_seq = np.arange(0,n_total)
rd.shuffle(pitchers_seq)
train_indices = pitchers_seq[:int(0.8*n_total)]
valid_indices = pitchers_seq[int(0.8*n_total):]

## Subset pitch data into training and validation sets
pitch_train = pitch_data.loc[train_indices,]
xwoba_train = xwoba[train_indices]
n_train,d_train = pitch_train.shape

pitch_valid = pitch_data.loc[valid_indices]
xwoba_valid = xwoba[valid_indices]
n_valid, d_valid = pitch_valid.shape

## Don't need cross-validation for random forests

pitch_random_forest_model = RandomForestRegressor()
pitch_random_forest_model.fit(pitch_train, xwoba_train)

## Predict with random forest
xwoba_preds = pitch_random_forest_model.predict(pitch_valid)

## Random forest MSE
print(np.sum((xwoba_preds - xwoba_valid)**2)/n_valid)
## MSE 0.00087
## off by .91 woba
## Random forest MAE
np.sum(np.abs(xwoba_preds-xwoba_valid))/n_valid
## off by 23.6 woba, roughly one walk per three plate appearances.

## Fit individual random forests to each pitch
## Fastball Random Forest
fastball_cols = ['n_ff_formatted', 'ff_avg_speed', 'ff_avg_spin', 'ff_avg_break_x',
       'ff_avg_break_z']
fastball_train = pitch_train[fastball_cols]
fastball_rf = RandomForestRegressor()
fastball_rf.fit(fastball_train, xwoba_train)

fastball_valid = pitch_valid[fastball_cols]
fastball_rf_preds = fastball_rf.predict(fastball_valid)

## fastball random forest MSE and MAE
print(np.sum((fastball_rf_preds-xwoba_valid)**2)/n_valid)
## MSE 0.0095
np.sum(np.abs(fastball_rf_preds-xwoba_valid))/n_valid
## MAE 0.0236

## Slider Random Forest
slider_cols = ['n_sl_formatted', 'sl_avg_speed', 'sl_avg_spin', 'sl_avg_break_x',
       'sl_avg_break_z']

slider_train = pitch_train[slider_cols]
slider_valid = pitch_valid[slider_cols]

slider_rf = RandomForestRegressor()
slider_rf.fit(slider_train, xwoba_train)

slider_rf_preds = slider_rf.predict(slider_valid)

##MSE MAE
np.sum((slider_rf_preds - xwoba_valid)**2)/n_valid
##MSE 0.00108
np.sum(np.abs(slider_rf_preds - xwoba_valid))/n_valid
## MAE 0.026

## Changeup Random Forest

changeup_cols = ['n_ch_formatted', 'ch_avg_speed', 'ch_avg_spin', 'ch_avg_break_x',
       'ch_avg_break_z']
changeup_train = pitch_train[changeup_cols]
changeup_valid = pitch_valid[changeup_cols]

changeup_rf = RandomForestRegressor()
changeup_rf.fit(changeup_train, xwoba_train)

changeup_rf_preds = changeup_rf.predict(changeup_valid)

## MSE and MAE
mean_squared_error(changeup_rf_preds, xwoba_valid)
## MSE 0.000997
mean_absolute_error(changeup_rf_preds, xwoba_valid)
## MAE 0.0245

## Lasso Model
## Will need to center data if we want variable importance
pitch_train_var_means = pitch_train.mean(axis = 0)
pitch_train_var_std = pitch_train.std(axis = 0)
xwoba_train_mean = xwoba_train.mean()

pitch_train_centered = pitch_train.sub(pitch_train_var_means, axis = 1)
pitch_train_standardized = pitch_train_centered/pitch_train_var_std
xwoba_train_centered = xwoba_train - xwoba_train_mean
pitch_valid_centered = pitch_valid.sub(pitch_train_var_means, axis = 1)
pitch_valid_standardized = pitch_valid_centered/pitch_train_var_std
xwoba_valid_centered = xwoba_valid - xwoba_train_mean

pitch_train_standardized.to_csv("pitch_train")
pitch_valid_standardized.to_csv("pitch_valid")
xwoba_train_centered.to_csv("xwoba_train")
xwoba_valid_centered.to_csv("xwoba_valid")

## Now try with regularized linear regression, once on all, then try grouping by pitch type
## Preference to Lasso for pseudo variable selection
from sklearn.linear_model import LassoCV

pitch_lasso_model = LassoCV()
pitch_lasso_model.fit(pitch_train_standardized,xwoba_train_centered)

lasso_preds = pitch_lasso_model.predict(pitch_valid_standardized)

## Lasso MSE
np.sum((lasso_preds - xwoba_valid_centered)**2)/n_valid
## MSE 0.00108
## off by 1.08 xwoba

np.sum(np.abs(lasso_preds - xwoba_valid_centered))/n_valid
## MAE 0.0232
## off by 23.2

pitch_lasso_model.coef_

import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
n,d = pitch_train.shape
# make data
x = np.arange(d)
y = pitch_lasso_model.coef_

# plot
fig, ax = plt.subplots()

ax.stem(x, y)

ax.set(xlim=(-1, d), xticks=np.arange(0, d), xticklabels = ['fastball %', 'fastball velo', 'fastball spin',
 'fastball_x_break', 'fastball_z_break', 'slider %', 'slider velo', 'slider spin', 'slider_x_break', 
 'slider_z_break', 'changeup %', 'changeup velo', 'changeup spin', 'changeup_x_break', 'changeup_z_break'],
       ylim=(-8e-3, 5e-3), yticks=np.arange(-8e-3, 5e-3, 2e-3))

plt.xticks(rotation=270)
plt.savefig('Lasso_coefs', bbox_inches="tight")

plt.show()

## Fastball Lasso
fastball_lasso = LassoCV()
fastball_train_standardized = pitch_train_standardized[fastball_cols]
fastball_valid_standardized = pitch_valid_standardized[fastball_cols]
fastball_lasso.fit(fastball_train_standardized, xwoba_train_centered)

fastball_lasso_preds = fastball_lasso.predict(fastball_valid_standardized)

mean_squared_error(fastball_lasso_preds, xwoba_valid_centered)
## MSE 0.00095
mean_absolute_error(fastball_lasso_preds, xwoba_valid_centered)
## MAE 0.0233

## Slider Lasso
slider_lasso = LassoCV()
slider_train_standardized = pitch_train_standardized[slider_cols]
slider_valid_standardized = pitch_valid_standardized[slider_cols]
slider_lasso.fit(slider_train_standardized, xwoba_train_centered)

slider_lasso_preds = slider_lasso.predict(slider_valid_standardized)

mean_absolute_error(slider_lasso_preds, xwoba_valid_centered)
##0.0242

## Changeup Lasso
changeup_lasso = LassoCV()
changeup_train_standardized = pitch_train_standardized[changeup_cols]
changeup_valid_standardized = pitch_valid_standardized[changeup_cols]
changeup_lasso.fit(changeup_train_standardized, xwoba_train_centered)

changeup_lasso_preds = changeup_lasso.predict(changeup_valid_standardized)

mean_absolute_error(changeup_lasso_preds, xwoba_valid_centered)
## 0.0243

plt.style.use('_mpl-gallery')
n,d = pitch_train.shape
# make data
x = np.arange(d)
y = slopes.iloc[:,1]
yerr = slopes.iloc[:,2] - slopes.iloc[:,1]

# plot
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr, fmt = 'o', linewidth=2, capsize=6)

ax.set(xlim=(-1, d), xticks=np.arange(0, d), xticklabels = ['fastball %', 'fastball velo', 'fastball spin',
 'fastball_x_break', 'fastball_z_break', 'slider %', 'slider velo', 'slider spin', 'slider_x_break', 
 'slider_z_break', 'changeup %', 'changeup velo', 'changeup spin', 'changeup_x_break', 'changeup_z_break'],
       ylim=(-8e-3, 5e-3), yticks=np.arange(-8e-3, 5e-3, 2e-3))

plt.xticks(rotation=270)

plt.savefig('Bayes_coefs', bbox_inches="tight")
plt.show()



