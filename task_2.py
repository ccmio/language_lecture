import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import curve_fit


def fit_func(x, a, b):
    return a*np.exp(x) +b


with open('/Users/chenxi/Desktop/0-1.txt') as f:
    zero_one = f.readline()[:1000]

# zero_one = zero_one * 10
zero_one = [int(i) for i in zero_one]
# random.shuffle(zero_one)
statistic_0 = {}
statistic_1 = {}
flag_0 = 0
flag_1 = 0
print(zero_one)
for i in zero_one:
    if i == 1:
        flag_1 += 1
        if flag_0 != 0 and flag_0 in statistic_0:
            statistic_0[flag_0] += 1
        elif flag_0 != 0:
            statistic_0[flag_0] = 1
        flag_0 = 0
    else:
        flag_0 += 1
        if flag_1 != 0 and flag_1 in statistic_1:
            statistic_1[flag_1] += 1
        elif flag_1 != 0:
            statistic_1[flag_1] = 1
        flag_1 = 0
print(statistic_0, statistic_1)

scatters_0 = []
scatters_1 = []
for i in statistic_0.items():
    scatters_0.append(list(i))
scatters_0 = np.array(sorted(scatters_0, key=(lambda x: x[0])))
for i in statistic_1.items():
    scatters_1.append(list(i))
scatters_1 = np.array(sorted(scatters_1, key=(lambda x: x[0])))
plt.figure(figsize=(9, 5))

popt0, _ = curve_fit(fit_func, scatters_0[:, 0], scatters_0[:, 1])
plt.scatter(scatters_0[:, 0], scatters_0[:, 1], s=50, c='black', marker='^', label='0串')
for a, b in zip(scatters_0[:, 0], scatters_0[:, 1]):
    plt.text(a, b, (a, b), ha='left', va='top', fontsize=15, c='black')
plt.scatter(scatters_1[:, 0], scatters_1[:, 1], s=50, c='orange', marker='*', label='1串')
for a, b in zip(scatters_1[:, 0], scatters_1[:, 1]):
    plt.text(a, b, (a, b), ha='right', va='top', fontsize=15, c='orange')
f_x = np.arange(0.9, 3.1, 0.01)
f_y = fit_func(f_x, *popt0)
plt.plot(f_x, f_y, label='拟合指数函数')
plt.title("以0串分布拟合指数分布", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
print(popt0)
plt.show()

plt.figure(figsize=(9, 5))

popt1, _ = curve_fit(fit_func, scatters_1[:, 0], scatters_1[:, 1])
plt.scatter(scatters_0[:, 0], scatters_0[:, 1], s=50, c='black', marker='^', label='0串')
for a, b in zip(scatters_0[:, 0], scatters_0[:, 1]):
    plt.text(a, b, (a, b), ha='left', va='top', fontsize=15, c='black')
plt.scatter(scatters_1[:, 0], scatters_1[:, 1], s=50, c='orange', marker='*', label='1串')
for a, b in zip(scatters_1[:, 0], scatters_1[:, 1]):
    plt.text(a, b, (a, b), ha='right', va='top', fontsize=15, c='orange')

f_x = np.arange(0.9, 3.1, 0.01)
f_y = fit_func(f_x, *popt1)
plt.plot(f_x, f_y, label='拟合指数函数')
plt.title("以1串分布拟合指数分布", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
print(popt1)
plt.show()
