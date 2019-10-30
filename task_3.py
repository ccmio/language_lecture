import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
e = float(math.e)


def fit_func(x, a, b, c, d):
    return a/(x ** (b + c * np.log(x))) ** 7


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
scatters = scatters[:-1]

plt.figure(figsize=(9, 5))
# absolute_error = []
# relative_error = []
# arithmetic_mean = []
# geometric_mean = []

# for i in scatters:
#     # plt.scatter(i[0], i[1], s=5, c='red')
#     a_error = fit_func(i[0], *popt) - i[1]
#     r_error = (fit_func(i[0], *popt) - i[1]) * 100 / i[1]
#     a_mean = (a_error + r_error)/2
#     g_mean = -(a_error*r_error)**(1/2) if (a_error < 0) and (r_error < 0) else (a_error*r_error)**(1/2)
#
#     absolute_error.append([i[0], a_error])
#     relative_error.append([i[0], r_error])
#     arithmetic_mean.append([i[0], a_mean])
#     geometric_mean.append([i[0], g_mean])

popt, _ = curve_fit(fit_func, scatters[:, 0], scatters[:, 1])
print(popt)
scatters = np.log(scatters)
plt.scatter(scatters[:, 0], scatters[:, 1], c='r', s=5)


f_x = np.arange(0, 6, 0.1)
f_y = fit_func(f_x, *popt)
f_y = np.log(f_y)
plt.plot(f_x, f_y, label='拟合函数')
plt.title("'的'字距离分布的频次统计", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("频次", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()
#
# absolute_error = np.array(absolute_error)
# relative_error = np.array(relative_error)
# arithmetic_mean = np.array(arithmetic_mean)
# geometric_mean = np.array(geometric_mean)
#
# plt.figure(figsize=(9, 5))
# plt.plot(absolute_error[3:, 0], absolute_error[3:, 1], label='绝对误差')
# plt.title("拟合函数的误差", fontsize=14)
# plt.xlabel("距离", fontsize=14)
# plt.ylabel("误差", fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.grid(ls='--')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(9, 5))
# plt.plot(relative_error[3:, 0], relative_error[3:, 1], label='相对误差')
# plt.title("拟合函数的误差", fontsize=14)
# plt.xlabel("距离", fontsize=14)
# plt.ylabel("误差", fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.grid(ls='--')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(9, 5))
# plt.plot(arithmetic_mean[3:, 0], arithmetic_mean[3:, 1], label='算术平均')
# plt.title("算术平均", fontsize=14)
# plt.xlabel("距离", fontsize=14)
# plt.ylabel("误差平均值", fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.grid(ls='--')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(9, 5))
# plt.plot(geometric_mean[3:, 0], geometric_mean[3:, 1], label='几何平均')
# plt.title("几何平均", fontsize=14)
# plt.xlabel("距离", fontsize=14)
# plt.ylabel("误差平均值", fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.grid(ls='--')
# plt.legend()
# plt.show()




