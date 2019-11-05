import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
'''
=================定义函数=====================
'''


def func(x, a, b):
    return a*np.power(x, -b)


# def func1(x, a, b, c):
#     return a*np.square(x+b)+c


def func2(x, a, b, c):
    return a*np.sin(b*x+c)


def X2(y, y_):
    return np.mean(np.square(y - y_) / (y + 1e-10))


'''
=================统计并获得散点图=====================
'''
with open('./平凡的世界.txt') as f:
    book = f.read()
chars = list(set(book))

# 统计每个字的频率
char_dict = {}
for char in book:
    chinese = '\u4e00' <= char <= '\u9fff'
    if chinese and char in char_dict:
        char_dict[char] += 1
    elif chinese:
        char_dict[char] = 1
# 统计频次（出现1，2，...100次的字）
statistic = np.zeros(100)
for value in char_dict.values():
    if value <= 94:
        statistic[value-1] += 1
    elif value <= 101:
        statistic[value-2] += 1
# 绘制散点图
lnx = np.log(np.arange(1, 101, 1))
lny = np.log(statistic)
plt.scatter(lnx, lny)


'''
=================拟合=====================
'''
# popt1, pcov1 = curve_fit(func1, lnx, lny)
# y1 = func1(lnx, *popt1)
# plt.plot(lnx, y1, c='orange')

popt2, pcov2 = curve_fit(func2, lnx, lny)
y2 = func2(lnx, *popt2)
plt.plot(lnx, y2, c='b')
plt.show()
print('正弦函数卡方：', X2(y2, lny))


'''
=================训练参数=====================
'''
a = tf.Variable(initial_value=-6.81887963)
b = tf.Variable(initial_value=0.16926294)
c = tf.Variable(initial_value=5.24984482)
variables = [a, b, c]
lrlist = []
num_epoch = 10000
initial_learning_rate = 0.002
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y = a * tf.sin(b * lnx + c)
        llr = tf.reduce_sum(tf.square(lny - y) / (y + 1e-10))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(llr, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    if e % 10 == 0:
        print('第{}次训练, loss：{:.6f}'.format(e, llr))
    # if llr.numpy() <= 3.4157488:
    #     break
print(a.numpy(), b.numpy(), c.numpy())
'''
a=-7.2501407
b=0.15367302
c=5.3309155
'''