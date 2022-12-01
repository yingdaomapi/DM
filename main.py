import numpy as np
import matplotlib.pyplot as plt
import math


def readdata():  # 获取data数据中坐标值
    data = np.loadtxt("kmeans_data")  # 读取dat数据
    x_data = data[:, 0]  # 横坐标
    y_data = data[:, 1]  # 纵坐标
    return data, x_data, y_data


def init(k):  # 初始化生成k个随机类别中心
    data, x_data, y_data = readdata()
    class_center = []
    for i in range(k):
        # 在数据的最大值与最小值间给出随机值
        x = np.random.randint(np.min(x_data), np.max(x_data))
        y = np.random.randint(np.min(y_data), np.max(y_data))
        class_center.append(np.array([x, y]))  # 以数组方式添加，方便后面计算距离
    return class_center


def dist(a, b):  # 计算两个坐标间的欧氏距离
    dist = math.sqrt(math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2))
    return dist


def dist_rank(center, data, k):  # 得到与类中心最小距离的类别位置索引
    tem = []
    for m in range(k):
        d = dist(data, center[m])
        tem.append(d)
    loc = tem.index(min(tem))
    return loc


def means(arr):  # 计算类的平均值当作类的新中心
    sum_x, sum_y = 0, 0
    for n in arr:
        sum_x += n[0]
        sum_y += n[1]
    mean_x = sum_x / len(arr)
    mean_y = sum_y / len(arr)
    return [mean_x, mean_y]


def divide(center, data, k):  # 将每一个二维坐标分到与之欧式距离最近的类里
    cla_arr = [[]]
    for i in range(k - 1):  # 创建与k值相同维度的空数组存取坐标
        cla_arr.append([])
    for j in range(len(data)):
        loc = dist_rank(center, data[j], k)
        cla_arr[loc].append(list(data[j]))
    return cla_arr


def new_center(cla, k):  # 计算每类平均值更新类中心
    new_cen = []
    for g in range(k):
        new = means(cla[g])
        new_cen.append(new)
    return new_cen


def index_element(arr, data, sw, k):  # 索引第sw个元素对应的类别
    index = []
    for i in range(len(data)):  # 遍历每一个数据
        for j in range(k):  # 遍历每一个类别
            tem = arr[j]
            for d in range(len(tem)):  # 遍历类别内的每一个数据
                if data[i][0] == tem[d][0] and data[i][1] == tem[d][1]:  # 如果横纵坐标数值都相等
                    index.append((j + 1))  # 归为j+1类
                else:
                    continue
    return index[sw]


def Kmeans(k, n, sw):  # 获取n次更新后类别中心以及第sw个元素对应的类别
    data, x_data, y_data = readdata()  # 读取数据
    center = init(k)  # 获取初始类别中心
    while n > 0:
        cla_arr = divide(center, data, k)  # 将数据分到随机选取的类中心的里
        center = new_center(cla_arr, k)  # 更新类别中心
        n -= 1
    sse1 = 0
    for j in range(k):
        for i in range(len(cla_arr[j])):  # 计算每个类里的误差平方
            # 计算每个类里每个元素与元素中心的误差平方
            dist1 = math.pow(dist(cla_arr[j][i], center[j]), 2)
            sse1 += dist1
    sse1 = sse1 / len(data)
    index = index_element(cla_arr, data, sw, k)
    return center, index, sse1, cla_arr


def hand():
    SSE = []
    for k in range(2, 9):
        while (1):
            try:
                center_l, index, sse1, cla_arr = Kmeans(k, 1000, 10)
                SSE.append(sse1)
                print("类别中心为:", center_l)
                print("所查元素属于类别：", index)
                print('k值为{0}时的误差平方和为{1}'.format(k, sse1))  # format格式化占位输出误差平方和
                break
            except ZeroDivisionError:
                pass
    x = np.linspace(2, 8, 7)  # 创建等间距大小为7的数组
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文不能显示的问题
    plt.xlabel("k值")  # 横坐标名称
    plt.ylabel("误差平方和")  # 纵坐标名称
    plt.title("手肘图")  # 曲线名
    plt.plot(x, SSE)  # 画出曲线
    plt.show()
    print(SSE)

hand()


def visualization(cla, k):  # 聚类可视化展现
    cla_x = [[]]
    cla_y = [[]]
    for m in range(k - 1):  # 创建与k值相同维度的空数组存取x坐标和y坐标
        cla_x.append([])
        cla_y.append([])
    for i in range(k):  # 遍历k次读取k个类别
        for j in cla[i]:  # 遍历每一类存取横纵坐标
            cla_x[i].append(j[0])
            cla_y[i].append(j[1])
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文不能显示的问题
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("聚类图")
    plt.scatter(cla_x[0], cla_y[0], c='r', marker='h')
    plt.scatter(cla_x[1], cla_y[1], c='y', marker='.')
    plt.scatter(cla_x[2], cla_y[2], c='g', marker='o')
    plt.scatter(cla_x[3], cla_y[3], c='b', marker=',')
    plt.show()


def Best_Kmeans(k, n):  # 获取n次更新后类别中心以及第sw个元素对应的类别
    data, x_data, y_data = readdata()  # 读取数据
    center = init(k)  # 获取初始类别中心
    while n > 0:
        cla_arr = divide(center, data, k)  # 将数据分到随机选取的类中心的里
        center = new_center(cla_arr, k)  # 更新类别中心
        n -= 1
        visualization(cla_arr, k)
    sse1 = 0
    for j in range(k):
        for i in range(len(cla_arr[j])):  # 计算每个类里的误差平方
            # 计算每个类里每个元素与元素中心的误差平方
            dist1 = math.pow(dist(cla_arr[j][i], center[j]), 2)
            sse1 += dist1
    sse1 = sse1 / len(data)
    return center, sse1, cla_arr


k = 4
n = 5
Best_Kmeans(k, n)