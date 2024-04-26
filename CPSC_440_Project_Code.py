import pandas as pd
import numpy as np

pitch_data = pd.read_csv("C:/Users/ranen/Downloads/stats.csv")
print(pitch_data.columns)

## xwoba is our response variable, lower xwoba is better for a pitcher
## xwoba is a pandas series object
xwoba = pitch_data.loc[:,"xwoba"]

## ff is four-seam, sl is slider, ch is changeup, cu is curveball, si is sinker, fc is cutter, fs is splitter

cols_to_drop = ['last_name, first_name', 'player_id', 'year', 'pa']
pitch_data = pitch_data.drop(columns=cols_to_drop)

## Replace missing values with median
pitch_medians = pitch_data.median(axis = 0, numeric_only=True)
pitch_data = pitch_data.fillna(value = pitch_medians)


## This code works, but you need to set up a validation set or do CV or something like that.
from sklearn.ensemble import RandomForestRegressor

pitch_random_forest_model = RandomForestRegressor()
pitch_random_forest_model.fit(pitch_data, xwoba)


