#  Calculating Descriptive Statistics
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

#  ---------------------------------------------------------------------------------------------------------------------

# Measures of Central Tendency
#  Mean
mean = sum(x) / len(x)  # find the mean of a list by dividing the sum by the length
print(mean)
mean = statistics.mean(x)  # find the mean with a function
print(mean)
mean = statistics.fmean(x)  # find the mean with a function faster (fmean)
print(mean)

mean = statistics.mean(x_with_nan)  # with a nan value, these all return nan instead of the actual mean value
print(mean)

mean = np.nanmean(x_with_nan)  # nanmean ignores nan values, this does not mean it treats it as 0
print(mean)


#  Weighted means
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
print(wmean)
wmean = sum(x * w for (x, w) in zip(x, w)) / sum(w)
print(wmean)

w = np.array(w)
wmean = np.average(y, weights=w)
print(wmean)
wmean = np.average(z, weights=w)
print(wmean)


#  Harmonic Mean
hmean = len(x) / sum(1 / item for item in x)
print(hmean)
hmean = statistics.harmonic_mean(x)
print(hmean)
hmean = statistics.harmonic_mean(x_with_nan)
print(hmean)
hmean = statistics.harmonic_mean([1, 0, 2])
print(hmean)
#  hmean = statistics.harmonic_mean([1, 2, -2]) produces error (negative number)


#  Geometric Mean
gmean = 1
for item in x:
    gmean *= item

gmean **= 1 / len(x)
print(gmean)


#  Sample Median
n = len(x)
if n % 2:
    median = sorted(x)[round(0.5 * (n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)

print(median)


#  Sample Mode
u = [2, 3, 2, 8, 12]
mode = max((u.count(item), item) for item in set(u))[1]
print(mode)

#  ---------------------------------------------------------------------------------------------------------------------

#  Measures of Variability
#  Variance
n = len(x)
mean = sum(x) / n
var = sum((item - mean)**2 for item in x) / (n - 1)
print(var)
var = statistics.variance(x)
print(var)


#  Standard Deviation
std = var**0.5
print(std)


#  Skewness
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean = sum(x) / n
var = sum((item - mean)**2 for item in x) / (n - 1)
std = var**0.5
skew = (sum((item - mean)**3 for item in x) * n / ((n - 1) * (n - 2) * std**3))
print('Skew = ', skew)


#  Percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
quant_2 = statistics.quantiles(x, n=2)
print(quant_2)
quant_4_inc = statistics.quantiles(x, n=4, method='inclusive')
print(quant_4_inc)


#  Ranges
full_range = np.ptp(x)
print(full_range)

#  ---------------------------------------------------------------------------------------------------------------------

#  Summary of Descriptive Analysis
result = scipy.stats.describe(y, ddof=1, bias=False)
print(result)
print(result.mean)

result = z.describe()
print(result)
print(result['mean'])

#  ---------------------------------------------------------------------------------------------------------------------

#  Measures of Correlation Between Pairs of Data
#  Covariance
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n)) / (n - 1))
print(cov_xy)

cov_matrix = np.cov(x_, y_)
print(cov_matrix)


#  Correlation Coefficient


