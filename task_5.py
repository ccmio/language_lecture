import matplotlib.pyplot as plt
import numpy as np


def stirling(n, theta):
    return np.log(np.sqrt(2 * np.pi * n) * np.power(n / np.e, n) * np.exp(theta / 12 / n))


def stirling2(n, theta):
    return np.log(np.sqrt(2 * np.pi * n) * np.power(n / np.e, n) * np.exp(1/12/n - theta))


absolute_error_c8 = []
relative_error_c8 = []
absolute_error1 = []
relative_error1 = []
absolute_error_b16 = []
relative_error_b16 = []
plt.figure(figsize=(9, 5))
for i in range(3, 50):
    x = np.log(i)
    y = np.float(np.math.factorial(i))
    y = np.log(y)
    z1 = stirling(i, 1)
    b16 = stirling(i, 1 - 1 / 30 / i ** 2 + 1 / 105 / i ** 4)
    c8 = stirling2(i, 1/360/i**3)
    a_error_c8 = c8 - y
    a_error1 = z1 - y
    a_error_b16 = b16 - y

    r_error_c8 = a_error_c8 * 100/ y
    r_error1 = a_error1 * 100 / y
    r_error_b16 = a_error_b16 * 100 / y

    absolute_error_c8.append([x, a_error_c8])
    relative_error_c8.append([x, r_error_c8])
    absolute_error1.append([x, a_error1])
    relative_error1.append([x, r_error1])
    absolute_error_b16.append([x, a_error_b16])
    relative_error_b16.append([x, r_error_b16])
    plt.scatter(x, y, c='r', s=20)

x = np.arange(2, 50, 0.1)
y0 = stirling2(x, 1 / 360 / x ** 3)
y1 = stirling(x, 1)
y2 = stirling(x, 1 - 1 / 30 / x ** 2 + 1 / 105 / x ** 4)
plt.plot(np.log(x), y0, c='y', label='c8')
plt.plot(np.log(x), y1, c='b', label='改进1次')
plt.plot(np.log(x), y2, c='g', label='b16')
plt.title("散点拟合", fontsize=14)
plt.xlabel("log(n)", fontsize=14)
plt.ylabel("log(y)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()

absolute_error_c8 = np.array(absolute_error_c8)
relative_error_c8 = np.array(relative_error_c8)
absolute_error1 = np.array(absolute_error1)
relative_error1 = np.array(relative_error1)
absolute_error_b16 = np.array(absolute_error_b16)
relative_error_b16 = np.array(relative_error_b16)

plt.figure(figsize=(9, 5))
plt.plot(absolute_error_c8[:, 0], absolute_error_c8[:, 1], label='c8', linestyle=":", linewidth=2)
plt.plot(absolute_error1[:, 0], absolute_error1[:, 1], label='改进1次', linestyle="--", linewidth=2)
plt.plot(absolute_error_b16[:, 0], absolute_error_b16[:, 1], label='b16', linewidth=3)
plt.title("绝对误差", fontsize=14)
plt.xlabel("log(n)", fontsize=14)
plt.ylabel("误差值", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()

plt.show()

plt.figure(figsize=(9, 5))
plt.plot(relative_error_c8[:, 0], relative_error_c8[:, 1], label='c8', linestyle=":", linewidth=2)
plt.plot(relative_error1[:, 0], relative_error1[:, 1], label='改进1次', linestyle="--", linewidth=2)
plt.plot(relative_error_b16[:, 0], relative_error_b16[:, 1], label='b16', linewidth=3)
plt.title("相对误差", fontsize=14)
plt.xlabel("log(n)", fontsize=14)
plt.ylabel("误差值", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(ls='--')
plt.legend()
plt.show()
