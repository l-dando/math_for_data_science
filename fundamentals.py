#  Calculating Descriptive Statistics
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

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
var_x = sum((item - mean) ** 2 for item in x) / (n - 1)
var_y = sum((item - mean) ** 2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
print(r)

#  ---------------------------------------------------------------------------------------------------------------------

#  2D Data
#  Axes
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
print(a)
print(np.mean(a))
print(a.mean())
print(np.median(a))
print(a.var(ddof=1))

print(np.mean(a, axis=0))
print(a.mean(axis=0))
print(np.mean(a, axis=1))
print(a.mean(axis=1))  # same for median, var etc.

print(scipy.stats.gmean(a))  # default axis = 0
print(scipy.stats.gmean(a, axis=None))  # default axis = 0
print(scipy.stats.gmean(a, axis=None))  # default axis = 0

print(scipy.stats.describe(a, axis=None, ddof=1, bias=False))

#  DataFrames
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
print(df)
print(df['B'])
print(df['B'].mean())

print(df.to_numpy())
print(df.describe())
print(df.describe().at['mean', 'A'])

#  ---------------------------------------------------------------------------------------------------------------------
#  Visualizing Data
plt.style.use('ggplot')

#  Box Plots
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True, labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'}, meanprops={'linewidth': 2, 'color': 'red'})
plt.draw()
plt.savefig('boxplot.png')

#  Histograms
hist, bin_edges = np.histogram(x, bins=10)
print(hist)
print(bin_edges)

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.draw()
plt.savefig('histogram.png')

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.draw()
plt.savefig('histogram.png')

#  Pie Charts
x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.draw()
plt.savefig('pie.png')

#  Bar Charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.draw()
plt.savefig('bar.png')

#  X-Y (Scatter) Plots
x = np.arange(21)
y = 5 + 2 * x + x * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data Points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.draw()
plt.savefig('scatter.png')

#  Heatmaps
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')

plt.draw()
plt.savefig('heatmap.png')

#  plt.show()
