import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import tensorflow as tf

e = float(math.e)


def fit_func(x, a, b, c):
    return a * np.exp(-b * x) + c


with open('pi.txt') as f:
    book = f.read()

distance = 1
d_list = []
for i in book:
    if i == '2':
        d_list.append(distance)
        distance = 2
    else:
        distance += 1

d_p = {}
for i in d_list:
    if i in d_p:
        d_p[i] += 1
    else:
        d_p[i] = 2
scatters = []
for i in d_p.items():
    scatters.append(i)
scatters = np.array(sorted(scatters, key=(lambda x: x[0])), dtype=np.float)
ma = 0.99
mi = 0.01
# x = scatters[:, 0]
# y = scatters[:, 1]
y = (ma - mi)*(scatters[:, 1] - scatters[:, 1].min()) / (scatters[:, 1].max() - scatters[:, 1].min())+mi
x = (ma - mi)*(scatters[:, 0] - scatters[:, 0].min()) / (scatters[:, 0].max() - scatters[:, 0].min())+mi
plt.figure(figsize=(9, 5))
absolute_error = []
relative_error = []
arithmetic_mean = []
geometric_mean = []

popt, _ = curve_fit(fit_func, x, y, maxfev=10000)
print(popt)
plt.scatter(x, y, s=5, c='red')

for i in range(len(scatters)):
    scatters[i][0] = x[i]
    scatters[i][1] = y[i]

a = tf.Variable(initial_value=1.09732874)
b = tf.Variable(initial_value=10.1770436)
c = tf.Variable(initial_value=9.89006282e-03)
variables = [a, b, c]


num_epoch = 100000
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_e = a * tf.exp(-b * x) + c
        llr = tf.reduce_sum((y - y_e) * np.log(y / y_e))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(llr, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    if e % 100 == 0:
        print('第{}次训练'.format(e), llr.numpy())
    if llr.numpy() < 0.004131484:
        break
print(a.numpy(), b.numpy(), c.numpy(), llr.numpy())

for i in scatters:
    a_error = fit_func(i[0], *popt) - i[1]
    r_error = (fit_func(i[0], *popt) - i[1]) * 100 / i[1]
    a_mean = (a_error + r_error) / 2
    g_mean = -(a_error * r_error) ** (1 / 2) if (a_error < 0) and (r_error < 0) else (a_error * r_error) ** (1 / 2)

    absolute_error.append([i[0], a_error])
    relative_error.append([i[0], r_error])
    arithmetic_mean.append([i[0], a_mean])
    geometric_mean.append([i[0], g_mean])

f_x = np.arange(0, 100, 0.01)
f_y = fit_func(f_x, *popt)
plt.plot(f_x, f_y, label='拟合函数')
plt.title("'的'字距离分布的频次统计", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("频次", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

absolute_error = np.array(absolute_error)
relative_error = np.array(relative_error)
arithmetic_mean = np.array(arithmetic_mean)
geometric_mean = np.array(geometric_mean)
print(absolute_error)
plt.figure(figsize=(9, 5))
plt.plot(absolute_error[:, 0], absolute_error[:, 1], label='绝对误差')
plt.title("拟合函数的误差", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("误差", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(relative_error[:, 0], relative_error[:, 1], label='相对误差')
plt.title("拟合函数的误差", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("误差", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(arithmetic_mean[:, 0], arithmetic_mean[:, 1], label='算术平均')
plt.title("算术平均", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("误差平均值", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(geometric_mean[:, 0], geometric_mean[:, 1], label='几何平均')
plt.title("几何平均", fontsize=14)
plt.xlabel("距离", fontsize=14)
plt.ylabel("误差平均值", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()
