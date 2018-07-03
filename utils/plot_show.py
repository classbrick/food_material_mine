import matplotlib.pyplot as plt
import numpy as np


COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def show_plot(x_list, y_list, y_name_list, color_list=COLOR_LIST, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    if len(y_name_list) > len(color_list):
        return
    my_x_ticks = np.arange(x_min, x_max, 0.1)
    my_y_ticks = np.arange(y_min, y_max, 0.1)
    plt.figure()
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.subplot(2, 2, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title(y_name_list[0])  # 第一幅图片标题
    plt.plot(x_list, y_list[0])  # 绘制第一幅图片

    plt.subplot(2, 2, 2)  # 第二个子图
    plt.title(y_name_list[1])  # 第二幅图片标题
    plt.plot(x_list, y_list[1])  # 绘制第一幅图片

    plt.subplot(2, 2, 3)  # 第三个子图
    plt.title(y_name_list[2])  # 第三幅图片标题
    plt.plot(x_list, y_list[2])  # 绘制第一幅图片

    plt.subplot(2, 2, 4)  # 第四个子图
    plt.title(y_name_list[3])  # 第四幅图片标题
    plt.plot(x_list, y_list[3])  # 绘制第一幅图片

    plt.show()
    plt.pause(1008611)

