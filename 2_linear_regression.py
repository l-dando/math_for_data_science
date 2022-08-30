import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

#  Linear Regression
#  Simple plot

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.draw()
plt.savefig('2_scatter.png')

#  Line of Linear Regression

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.draw()
plt.savefig('2_scatter_line.png')

#  R for Relationship

print(r)

#  Predict Future Values

speed = myfunc(10)
print(speed)

#  Bad fit

x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]
slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = (list(map(myfunc, x)))
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.draw()
plt.savefig('2_bad_line.png')

print(r)

#  Polynomial Regression
#  Simple Plot
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

fig, ax = plt.subplots()
plt.scatter(x, y)
plt.draw()

#  Line of Best Fit
mymodel = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(1, 22, 100)
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.draw()
plt.savefig('2_best_fit.png')

#  R-squared
mymodel = np.poly1d(np.polyfit(x, y, 3))
print(r2_score(y, mymodel(x)))

#  Predict Future Values
speed = mymodel(17)
print(speed)

#  Bad fit
x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]
mymodel = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(2, 95, 100)
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.draw()
plt.savefig('2_bad_line.png')
print(r2_score(y, mymodel(x)))

plt.show()
