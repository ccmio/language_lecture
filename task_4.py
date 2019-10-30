import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
e = float(math.e)


def fit_func(x, a, b, c, d):
    return a*np.sin(b*x+c)*(x+d)**2


with open('四世同堂.txt') as f:
    book = f.read()


distance = 1
d_list = []
for i in book:
    if i == '的':
        d_list.append(distance)
        distance = 1
    else:
        distance += 1

d_p = {}
for i in d_list:
    temp = i
    if temp in d_p:
        d_p[temp] += 1
    else:
        d_p[temp] = 2
scatters = []
for i in d_p.items():
    scatters.append(list(i))
scatters = np.array(sorted(scatters, key=(lambda x: x[0])))
scatters = np.log(scatters[:-1])

plt.figure(figsize=(9, 5))
plt.scatter(scatters[:, 0], scatters[:, 1], c='r', s=5)
popt, _ = curve_fit(fit_func, scatters[:, 0], scatters[:, 1], maxfev=200000)
print(popt)

f_x = np.arange(0, 6, 0.01)
f_y = fit_func(f_x, *popt)
plt.plot(f_x, f_y, label='拟合函数')
plt.title("'的'字距离分布的频次统计", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("频次", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

