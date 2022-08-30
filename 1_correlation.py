import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

#  Examples of Correlation
#  NumPy Correlation Example

x = np.arange(10, 20)
print(x)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
print(y)
r = np.corrcoef(x, y)
print(r)
print(r[0, 1])

#  SciPy Correlation Example

print(scipy.stats.pearsonr(x, y))
print(scipy.stats.spearmanr(x, y))
print(scipy.stats.kendalltau(x, y))

#  Pandas Correlation Example

x = pd.Series(range(10, 20))
print(x)
y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
print(y)
print(x.corr(y))
print(y.corr(x))
print(x.corr(y, method='spearman'))
print(x.corr(y, method='kendall'))

#  ---------------------------------------------------------------------------------------------------------------------

#  Linear Correlation
#  Linear Regression: SciPy Implementation

result = scipy.stats.linregress(x, y)
print(result.slope)
print(result.intercept)
print(result.rvalue)
print(result.pvalue)
print(result.stderr)

xy = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]])
print(scipy.stats.linregress(xy))

print(xy)
print(xy.T)

print(scipy.stats.linregress(xy.T))

#  Pearson Correlation: NumPy and SciPy Implementation

r, p = scipy.stats.pearsonr(x, y)
print(r)
print(p)
print(np.corrcoef(x, y))
print(np.corrcoef(xy))

xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])

print(np.corrcoef(xyz))

#  Pearson Correlation: Pandas Implementation

x = pd.Series(range(10, 20))
print(x)
y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
print(y)
z = pd.Series([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])
print(z)

xy = pd.DataFrame({'x-values': x, 'y-values': y})
print(xy)
xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})
print(xyz)
print(xyz.corr())
print(xyz.corrwith(z))

#  ---------------------------------------------------------------------------------------------------------------------

#  Rank Correlation
#  Rank: SciPy Implementation

print(scipy.stats.rankdata(x))
print(scipy.stats.rankdata(y))
print(scipy.stats.rankdata(z))

#  Rank Correlation: NumPy and SciPy Implementation

result = scipy.stats.spearmanr(x, y)
print(result)
print(result.correlation)
print(result.pvalue)
rho, p = scipy.stats.spearmanr(x, y)
print(rho)
print(p)

xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])
corr_matrix, p_matrix = scipy.stats.spearmanr(xyz, axis=1)
print(corr_matrix)
print(p_matrix)

result = scipy.stats.kendalltau(x, y)
print(result)

#  Rank Correlation: Pandas Implementation

x, y, z = pd.Series(x), pd.Series(y), pd.Series(z)
xy = pd.DataFrame({'x-values': x, 'y-values': y})
xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})
print(x.corr(y, method='spearman'))
print(xy.corr(method='spearman'))
print(xyz.corr(method='spearman'))
print(xy.corrwith(z, method='spearman'))

#  ---------------------------------------------------------------------------------------------------------------------

#  Visualization of Correlation
#  X-Y Plots With a Regression Line

slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
print(line)

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.savefig('1_regression')
plt.draw()

#  Heatmaps of Correlation Matrices

xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])
corr_matrix = np.corrcoef(xyz).round(decimals=2)
print(corr_matrix)

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, corr_matrix[i, j], ha='center', va='center', color='r')

cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.savefig('1_heatmap')
plt.draw()

plt.show()
