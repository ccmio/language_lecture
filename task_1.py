import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 取得拟合函数用
from scipy.optimize import curve_fit


def fit_func(x, a, b, c):
    return a*np.exp(-b*x)+c


pi_file = open('C:/Users/cacho/Desktop/new.txt', "r")
pi_list = list(pi_file.read())
pi_file.close()

distance = 0
d_list = []
for i in pi_list:
    if i == '2':
        d_list.append(distance)
        distance = 0
    else:
        distance += 1

d_p = {}
for i in d_list:
    if i in d_p:
        d_p[i] += 1
    else:
        d_p[i] = 1
d_p[74] = 3
scatters = []
for i in d_p.items():
    scatters.append(list(i))
scatters = np.array(sorted(scatters, key=(lambda x: x[0])))

param_bounds = ([8000, 0, -2], [12000, 1, 2])
popt, _ = curve_fit(fit_func, scatters[:, 0], scatters[:, 1], bounds=param_bounds)
print(popt)

plt.figure(figsize=(9, 5))
absolute_error = []
relative_error = []
arithmetic_mean = []
geometric_mean = []

for i in scatters:
    plt.scatter(i[0], i[1], s=5, c='red')
    a_error = fit_func(i[0], *popt) - i[1]
    r_error = (fit_func(i[0], *popt) - i[1]) * 100 / i[1]
    a_mean = (a_error + r_error)/2
    g_mean = -(a_error*r_error)**(1/2) if (a_error < 0) and (r_error < 0) else (a_error*r_error)**(1/2)

    absolute_error.append([i[0], a_error])
    relative_error.append([i[0], r_error])
    arithmetic_mean.append([i[0], a_mean])
    geometric_mean.append([i[0], g_mean])

f_x = np.arange(-1, 100, 1)
f_y = fit_func(f_x, *popt)
plt.plot(f_x, f_y, label='拟合指数函数')
plt.title("无理数1/π前100万位数字2距离分布的频次统计", fontsize=14)
plt.xlabel("数字2之间的距离", fontsize=14)
plt.ylabel("频次", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

absolute_error = np.array(absolute_error)
relative_error = np.array(relative_error)
arithmetic_mean = np.array(arithmetic_mean)
geometric_mean = np.array(geometric_mean)

plt.figure(figsize=(9, 5))

plt.plot(absolute_error[:, 0], absolute_error[:, 1], label='绝对误差')
plt.plot(relative_error[:, 0], relative_error[:, 1], label='相对误差')

plt.title("拟合函数的误差", fontsize=14)
plt.xlabel("数字2之间的距离", fontsize=14)
plt.ylabel("误差", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

plt.figure(figsize=(9, 5))

plt.plot(arithmetic_mean[:, 0], arithmetic_mean[:, 1], label='算术平均')
plt.plot(geometric_mean[:, 0], geometric_mean[:, 1], label='几何平均')


plt.title("绝对误差相对误差的平均", fontsize=14)
plt.xlabel("数字2之间的距离", fontsize=14)
plt.ylabel("误差平均值", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()


