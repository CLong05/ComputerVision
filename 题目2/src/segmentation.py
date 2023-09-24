import math
import cv2
import random
import numpy as np


class area:
    '''
    定义⼀个类表示图中的区域，包含寻找根节点和合并两个节点这两个⽅法
    '''

    def __init__(self, num_node):
        '''
        初始化area类
        :param num_node: 该区域中的节点数量
        '''
        self.parent = [i for i in range(num_node)]  # 根节点初始化为自身
        self.size = [1 for _ in range(num_node)]  # 初始化的大小为1
        self.num_set = num_node  # 区域总数为节点总数

    def find_parent(self, u):
        '''
        寻找节点u的根节点
        :param u: 输入的子节点
        :return: 节点u的根节点
        '''
        if self.parent[u] == u:  # 本身为根节点
            return u
        self.parent[u] = self.find_parent(self.parent[u])  # 本身不是根节点，则回溯寻找根节点
        return self.parent[u]

    def merge2node(self, u, v):
        '''
        合并图中的两个节点
        :param u: 待合并节点
        :param v: 待合并节点
        '''
        u = self.find_parent(u)
        v = self.find_parent(v)
        # 若u、v分属不同根节点，则进行合并
        if u != v:
            # 子节点多的根节点作为父节点
            if self.size[u] > self.size[v]:
                self.parent[v] = u
                self.size[u] += self.size[v]
                # 被合并的根节点的子节点数目变为1
                self.size[v] = 1
            else:
                self.parent[u] = v
                self.size[v] += self.size[u]
                self.size[u] = 1
            self.num_set -= 1


def create_edge(img, width, x1, y1, x2, y2):
    '''
    计算RGB距离并创建图的边
    :param img: 待处理图片
    :param width: 图片宽度
    :param x1: 像素点1的x坐标
    :param y1: 像素点1的y坐标
    :param x2: 像素点2的x坐标
    :param y2: 像素点2的y坐标
    :return: 构建好的边信息
    '''
    # 分通道计算RGB距离
    r = math.pow((img[0][y1, x1] - img[0][y2, x2]), 2)
    g = math.pow((img[1][y1, x1] - img[1][y2, x2]), 2)
    b = math.pow((img[2][y1, x1] - img[2][y2, x2]), 2)
    return (y1 * width + x1, y2 * width + x2, math.sqrt(r + g + b))


def build_graph(img, width, height):
    '''
    建立图结构
    :param img: 输入的待处理图片
    :param width: 图片宽度
    :param height: 图片高度
    :return: 构建好的图
    '''
    graph = []
    for y in range(height):
        for x in range(width):
            # 计算八邻域内的相似度，实现每个方向取4个方向即可
            if x < width - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y))
            if y < height - 1:
                graph.append(create_edge(img, width, x, y, x, y + 1))
            if x < width - 1 and y < height - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y + 1))
            if x < width - 1 and y > 0:
                graph.append(create_edge(img, width, x, y, x + 1, y - 1))
    return graph


def segment_graph(sorted_graph, num_node, k):
    '''
    对图结构进行分割
    :param sorted_graph: 根据权重对所有的边排序后的图结构
    :param num_node: 节点总数
    :param k: 控制聚类程度的阈值
    :return: 分割后的结果
    '''
    res = area(num_node)
    # 记录各个区域的类内不相似度
    threshold = [k] * num_node
    for edge in sorted_graph:
        u = res.find_parent(edge[0])  # 边的端点1
        v = res.find_parent(edge[1])  # 边的端点2
        w = edge[2]  # 边的权重
        # 如果边接连的两点不属于同一区域
        if u != v:
            # 如果边的权重小于阈值
            if w <= threshold[u] and w <= threshold[v]:
                # 合并两个节点
                res.merge2node(u, v)
                parent = res.find_parent(u)
                # 更新最大类内间距
                threshold[parent] = np.max([w, threshold[u], threshold[v]]) + k / res.size[parent]
    return res


def remove_small_area(res, sorted_graph, min_size):
    '''
    移除像素个数小于阈值的区域
    :param res: 分割后的图结构
    :param sorted_graph: 根据权重对所有的边排序后的图结构
    :param min_size: 每个分割区域的最少像素个数
    :return: 处理后的图结构
    '''
    for edge in sorted_graph:
        u = res.find_parent(edge[0])
        v = res.find_parent(edge[1])
        if u != v:
            if res.size[u] < min_size or res.size[v] < min_size:
                res.merge2node(u, v)
    print('  区域个数: ', res.num_set)
    return res


def generate_image(res, width, height, areaset):
    '''
    生成图像分割后的结果图
    :param res:分割后的图结构
    :param width: 图像宽度
    :param height: 图像高度
    :param areaset: 记录结果图的颜色标签
    :return: 分割后的结果图
    '''
    color = [(int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))
             for i in range(width * height)]  # 随机生成颜色
    save_img = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            color_idx = res.find_parent(y * width + x)
            areaset.append(color_idx)
            save_img[y, x] = color[color_idx]
    return save_img


def cal_IOU(img_gt, res, width, height, areaset):
    '''
    生成标记前景后的图片并计算IOU
    :param img_gt:数据集提供的mask
    :param res:分割后的图结构
    :param width:图像宽度
    :param height:图像高度
    :param areaset:记录结果图的颜色标签
    :return:标记前景后的图片和IOU
    '''
    save_img = np.zeros((height, width, 3), np.uint8)
    areaset = list(set(areaset))  # 划分出的所有区域
    for i in range(len(areaset)):  # 每个区域遍历
        total_area = 0  # 这个区域的面积
        gt_area = 0  # 在前景图中是前景的面积
        # 寻找这个区域中是前景的点
        for y in range(height):
            for x in range(width):
                if (res.find_parent(y * width + x) == areaset[i]):
                    if (img_gt[y, x, 0] > 128):
                        gt_area += 1
                    total_area += 1
        # 如果前景占比大于一半则在标记图中标为全白
        if (float(gt_area / total_area) > 0.5):
            for y in range(height):
                for x in range(width):
                    if (res.find_parent(y * width + x) == areaset[i]):
                        save_img[y, x] = (255, 255, 255)
        # 否则不是前景则全黑
        else:
            for y in range(height):
                for x in range(width):
                    if (res.find_parent(y * width + x) == areaset[i]):
                        save_img[y, x] = (0, 0, 0)
    # 计算IOU
    IOU = 0.0
    R1_and_R2 = 0
    R1_or_R2 = 0
    for y in range(height):
        for x in range(width):
            if (save_img[y, x, 0] > 128 and img_gt[y, x, 0] > 128):
                R1_and_R2 += 1
            if (save_img[y, x, 0] > 128 or img_gt[y, x, 0] > 128):
                R1_or_R2 += 1
    IOU = (float)(R1_and_R2 / R1_or_R2)
    return IOU, save_img


def weight(edge):
    '''
    按照权重从小到大排序
    '''
    return edge[2]


if __name__ == "__main__":
    index = 0
    # 超参数
    min_size = 50  # 每个区域最小面积
    # 控制聚类程度的阈值，不同图片选取不同阈值，使其区域总数在50-70之间
    k = [6.6, 3, 5.8, 2, 1.8, 6, 6.8, 6, 5, 12]
    while index < 10:
        id = 19 + index * 100
        print(f"Processing image {id}.png ...")
        img_number = str(id)
        # 读取相应编号原图和前景图
        img = cv2.imread("imgs/" + img_number + ".png")
        img_gt = cv2.imread("gt/" + img_number + ".png")
        # 先在当前目录生成原图和前景图
        cv2.imwrite("result2/" + img_number + "_origin.png", img)
        cv2.imwrite("result2/" + img_number + "_gt.png", img_gt)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        img = np.asarray(img, dtype=float)
        # 分开组成grb
        img = cv2.split(img)
        # 建立图结构
        graph = build_graph(img, width, height)
        sorted_graph = sorted(graph, key=weight)  # 根据权重对所有的边进行排序
        # 分割
        res = segment_graph(sorted_graph, width * height, k[index])
        res = remove_small_area(res, sorted_graph, min_size)
        # 生成结果图
        areaset = []
        img = generate_image(res, width, height, areaset)
        IOU, img2 = cal_IOU(img_gt, res, width, height, areaset)
        print(f"  IOU: {IOU}")
        # # 将IOU的值写到区域标记图的左上角
        # cv2.putText(img2, "IOU=%.4f" % IOU, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # # 显示图片
        # cv2.imshow('result', img)
        # cv2.imshow('区域标记', img2)
        # 按序号保存图片
        cv2.imwrite("result2/" + img_number + '_result.png', img)
        cv2.imwrite("result2/" + img_number + '_mask.png', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 读取下一张图片
        index += 1
