import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

# Introduction to lists, arrays and Series'
x = [8.0, 1, 2.5, 4, 28.0]  # create list
x_with_nan = [8.0, 1, 2.5, np.nan, 4, 28.0]  # create same list with a nan value

print(x)  # print x
print(x_with_nan)  # print x with a nan value

y, y_with_nan = np.array(x), np.array(x_with_nan)  # create two arrays from our lists
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)  # create two 'Series' from our lists

print(y)  # print them
print(y_with_nan)
print(z)
print(z_with_nan)

# Measures of Central Tendency
mean_ = sum(x) / len(x)  # find the mean of a list by dividing the sum by the length
print(mean_)
mean_ = statistics.mean(x)  # find the mean with a function
print(mean_)
mean_ = statistics.fmean(x)  # find the mean with a function faster (fmean)
print(mean_)

mean_ = statistics.mean(x_with_nan)  # with a nan value, these all return nan instead of the actual value
print(mean_)

mean_ = np.nanmean(x_with_nan)  # nanmean ignores nan values, this does not mean it treats it as 0
print(mean_)
