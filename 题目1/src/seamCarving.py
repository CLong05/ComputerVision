from PIL import Image
import numpy as np
from tqdm import tqdm
import imageio


class SeamCarving():
    def getDynamicEnergyMap(self, img, mask):
        '''
        获取动态规划能量图和路径图
        :param img: ndarray (height, width)灰度图
        :param mask: ndarray (height, width)蒙版
        :return:
            dynamicEnergyMap : ndarray (height, width) 动态规划能量图
            routeMap : ndarray (height, width) seam路径图
        '''
        height, width = img.shape
        img = img * mask  # 用Mask处理图像
        paddingImg = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)  # 图像外围填充一圈0
        dynamicEnergyMap = np.zeros((height + 2, width + 2))  # 记录动态规划的计算结果
        routeMap = np.ones((height, width), dtype=int) * -1  # 记录动态规划算得的路径
        for hIndex in range(height):  # 从上到下逐行处理
            M = np.zeros((width, 3))
            # 采用 Forward Seam Removing 机制进行计算（以矩阵为计算单元）
            M[:, 0] = dynamicEnergyMap[hIndex, :-2] + np.abs(paddingImg[hIndex][1:-1] - paddingImg[hIndex + 1][:-2])
            M[:, 1] = dynamicEnergyMap[hIndex, 1:-1]
            M[:, 2] = dynamicEnergyMap[hIndex, 2:] + np.abs(paddingImg[hIndex][1:-1] - paddingImg[hIndex + 1][2:])
            colEnergy = np.abs(paddingImg[hIndex + 1][2:] - paddingImg[hIndex + 1][:-2])
            dynamicEnergyMap[hIndex + 1, 1:-1] = np.min(M, axis=1)
            routeMap[hIndex] = np.argmin(M, axis=1) - 1
            # 处理边界点
            # 像素(i,0)只能选择(i-1,0)和(i-1,1)
            if M[0, 1] <= M[0, 2]:
                routeMap[hIndex, 0] = 0
                dynamicEnergyMap[hIndex + 1, 1] = M[0, 1]
            else:
                routeMap[hIndex, 0] = 1
                dynamicEnergyMap[hIndex + 1, 1] = M[0, 2]
            # 像素(i,-1)只能选择(i-1,-1)和(i-1,-2)
            if M[-1, 0] <= M[-1, 1]:
                routeMap[hIndex, -1] = -1
                dynamicEnergyMap[hIndex + 1, -1] = M[-1, 0]
            else:
                routeMap[hIndex, -1] = 0
                dynamicEnergyMap[hIndex + 1, -1] = M[-1, 1]
            dynamicEnergyMap[hIndex + 1, 1:-1] += colEnergy  # 计算最终的energy
        return dynamicEnergyMap[1:-1, 1:-1], routeMap

    def getSeam(self, routeMap, index):
        '''
        通过index获取该像素对应的seam
        :param routeMap: ndarray (height, width) getDynamicEnergyMap得到的seam路径图
        :param index: int 最后一行像素的索引
        :return: list(int) seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        '''
        seamIndex = np.zeros(routeMap.shape[0], dtype=int)
        seamIndex[seamIndex.size - 1] = index
        for i in range(routeMap.shape[0] - 2, -1, -1):
            seamIndex[i] = seamIndex[i + 1] + routeMap[i + 1, seamIndex[i + 1]]
        return seamIndex

    def removeSeam(self, img, seamIndex):
        '''
        去除img中的seam
        :param img: ndarray (height, width, _) 像素矩阵
        :param seamIndex: list(int) seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        :return: ndarray (height-1, width, _)新的像素矩阵
        '''
        isRetain = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        isRetain[seamIndex[:, None] == np.arange(img.shape[1])] = False
        if len(img.shape) == 3:
            newImg = img[isRetain].reshape((img.shape[0], img.shape[1] - 1, img.shape[2]))
        else:
            newImg = img[isRetain].reshape((img.shape[0], img.shape[1] - 1))
        return newImg

    def markSeam(self, rgbImg, seamIndex):
        '''
        标记rgbImg中的seam为黑色
        :param img : ndarray (height, width, 3) RGB像素矩阵
        :param seamIndex : list(int) seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        :return newImg : ndarray (height, width, 3) 新的RGB像素矩阵
        '''
        isMark = np.zeros((rgbImg.shape[0], rgbImg.shape[1]), dtype=bool)
        isMark[seamIndex[:, None] == np.arange(rgbImg.shape[1])] = True
        newImg = rgbImg.copy()
        newImg[isMark] = 0
        return newImg

    def changeShape(self, img, new_shape, mask=None, file_path="process", id=0):
        '''
        改变img的尺寸
        :param img : ndarray (height, width, 3) 原始图像
        :param new_shape: tuple(new_height, new_width) 新尺寸
        :param mask : ndarray (height, width) Mask
        :param file_path : str 保存文件夹路径
        :param id : int 图像id
        '''
        # 确保新尺寸小于原尺寸
        assert len(new_shape) == 2
        assert img.shape[0] >= new_shape[0] and img.shape[1] >= new_shape[1]
        # 修改蒙版
        if mask is None:
            newMask = np.ones((img.shape[0], img.shape[1]))
        else:
            newMask = mask.copy()
            newMask[newMask < 0.5] = 0.2
        # 计算缩小的高度和宽度
        reducedHeight = img.shape[0] - new_shape[0]
        reducedWidth = img.shape[1] - new_shape[1]
        # 获取灰度图
        grayImg = np.asarray(Image.fromarray(np.uint8(img)).convert('L')).copy().astype(int)
        # 获取像素索引图，便于后续标记seam
        indexMap = np.zeros((img.shape[0], img.shape[1], 2), dtype=int)
        for i in range(img.shape[0]):
            indexMap[i, :, 0] = i
            indexMap[i, :, 1] = range(img.shape[1])
        grayImgLists = list()  # 灰度图状态
        maskLists = list()  # mask状态
        imgLists = list()  # 图像状态
        indexLists = list()  # 图像索引矩阵状态
        seamLists = list()  # 每个状态操作的seam
        costLists = list()  # 每个状态的最小cost
        chosenLists = list()  # 当前状态的上一个状态(对于[i,j]，0:[i,j-1]; 1:[i-1,j])
        # 动态规划
        for hIndex in tqdm(range(reducedHeight + 1)):
            grayImgLists.append(list())
            maskLists.append(list())
            imgLists.append(list())
            indexLists.append(list())
            seamLists.append(list())
            costLists.append(list())
            chosenLists.append(list())
            for wIndex in range(reducedWidth + 1):
                if wIndex != 0:
                    # 获取垂直seam
                    dynamicEnergyMap, routeMap = self.getDynamicEnergyMap(grayImgLists[-1][wIndex - 1],
                                                                          maskLists[-1][wIndex - 1])
                    colSeamIndex = self.getSeam(routeMap, np.argmin(dynamicEnergyMap[-1, :]))
                    colCost = costLists[-1][wIndex - 1] + np.min(dynamicEnergyMap[-1, :])
                if hIndex != 0:
                    # 获取水平seam时，将相关矩阵转置后、通过获取垂直seam的方式获取
                    tranGrayImg = np.transpose(grayImgLists[-2][wIndex], (1, 0))
                    tranImg = np.transpose(imgLists[-2][wIndex], (1, 0, 2))
                    tranMask = np.transpose(maskLists[-2][wIndex], (1, 0))
                    tranIndexMap = np.transpose(indexLists[-2][wIndex], (1, 0, 2))
                    dynamicEnergyMap, routeMap = self.getDynamicEnergyMap(tranGrayImg, tranMask)
                    rowSeamIndex = self.getSeam(routeMap, np.argmin(dynamicEnergyMap[-1, :]))
                    rowCost = costLists[-2][wIndex] + np.min(dynamicEnergyMap[-1, :])
                if wIndex == 0 and hIndex == 0:
                    # 起始写入原始数据
                    grayImgLists[-1].append(grayImg)
                    maskLists[-1].append(newMask)
                    imgLists[-1].append(img)
                    indexLists[-1].append(indexMap)
                    seamLists[-1].append(None)
                    costLists[-1].append(0.0)
                    chosenLists[-1].append(-1)
                elif hIndex == 0:
                    # 只有垂直seam
                    grayImgLists[-1].append(self.removeSeam(grayImgLists[-1][-1], colSeamIndex))
                    maskLists[-1].append(self.removeSeam(maskLists[-1][-1], colSeamIndex))
                    imgLists[-1].append(self.removeSeam(imgLists[-1][-1], colSeamIndex))
                    indexLists[-1].append(self.removeSeam(indexLists[-1][-1], colSeamIndex))
                    seamLists[-1].append(colSeamIndex)
                    costLists[-1].append(colCost)
                    chosenLists[-1].append(1)
                elif wIndex == 0:
                    # 只有水平seam
                    grayImgLists[-1].append(np.transpose(self.removeSeam(tranGrayImg, rowSeamIndex), (1, 0)))
                    maskLists[-1].append(np.transpose(self.removeSeam(tranMask, rowSeamIndex), (1, 0)))
                    imgLists[-1].append(np.transpose(self.removeSeam(tranImg, rowSeamIndex), (1, 0, 2)))
                    indexLists[-1].append(np.transpose(self.removeSeam(tranIndexMap, rowSeamIndex), (1, 0, 2)))
                    seamLists[-1].append(rowSeamIndex)
                    costLists[-1].append(rowCost)
                    chosenLists[-1].append(0)
                else:
                    # 比较cost
                    if rowCost > colCost:
                        grayImgLists[-1].append(self.removeSeam(grayImgLists[-1][-1], colSeamIndex))
                        maskLists[-1].append(self.removeSeam(maskLists[-1][-1], colSeamIndex))
                        imgLists[-1].append(self.removeSeam(imgLists[-1][-1], colSeamIndex))
                        indexLists[-1].append(self.removeSeam(indexLists[-1][-1], colSeamIndex))
                        seamLists[-1].append(colSeamIndex)
                        costLists[-1].append(colCost)
                        chosenLists[-1].append(1)
                    else:
                        grayImgLists[-1].append(np.transpose(self.removeSeam(tranGrayImg, rowSeamIndex), (1, 0)))
                        maskLists[-1].append(np.transpose(self.removeSeam(tranMask, rowSeamIndex), (1, 0)))
                        imgLists[-1].append(np.transpose(self.removeSeam(tranImg, rowSeamIndex), (1, 0, 2)))
                        indexLists[-1].append(np.transpose(self.removeSeam(tranIndexMap, rowSeamIndex), (1, 0, 2)))
                        seamLists[-1].append(rowSeamIndex)
                        costLists[-1].append(rowCost)
                        chosenLists[-1].append(0)

        # 输出最终图像
        Image.fromarray(np.uint8(imgLists[-1][-1])).save(file_path + "/" + str(id) + "_final.png")
        # 标记所有seam，输出对应图像
        isMark = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        for i in range(indexLists[-1][-1].shape[0]):
            for j in range(indexLists[-1][-1].shape[1]):
                isMark[indexLists[-1][-1][i, j, 0], indexLists[-1][-1][i, j, 1]] = False
        markImg = img.copy()
        markImg[isMark] = 0
        Image.fromarray(np.uint8(markImg)).save(file_path + "/" + str(id) + "_final2.png")

        # 生成gif图
        hIndex = len(imgLists) - 1
        wIndex = len(imgLists[0]) - 1
        gifList = list()
        while chosenLists[hIndex][wIndex] != -1:
            # 反向索引得到顺序删除的seam
            gifList.append(np.uint8(imgLists[hIndex][wIndex]))
            if chosenLists[hIndex][wIndex] == 0:
                markImg = np.transpose(
                    self.markSeam(np.transpose(imgLists[hIndex - 1][wIndex], (1, 0, 2)), seamLists[hIndex][wIndex]),
                    (1, 0, 2))
                hIndex -= 1
            else:
                markImg = self.markSeam(imgLists[hIndex][wIndex - 1], seamLists[hIndex][wIndex])
                wIndex -= 1
            gifList.append(np.uint8(markImg))
        gifList.append(np.uint8(imgLists[0][0]))
        with imageio.get_writer(file_path + "/" + str(id) + "_output.gif", mode='I') as writer:
            for i in range(len(gifList) - 1, -1, -1):
                writer.append_data(gifList[i])  # 将图片写入writer，生成gif


if __name__ == '__main__':
    imgDict = dict()  # 存储待处理图片的字典
    id = 19  # 学号的后两位
    while id < 101:
        # 读取学号对应的图片
        img = Image.open('../imgs/{}.png'.format(id))
        imgDict[id] = dict()
        imgDict[id]['img'] = np.asarray(img).copy().astype(int)
        # 读取学号对应Mask
        img = Image.open('../gt/{}.png'.format(id))
        imgArray = np.asarray(img).copy().astype(float)
        # 对Mask进行二值化
        imgArray[imgArray > 0] = 1
        imgDict[id]['mask'] = np.asarray(imgArray[:, :, 0])
        # 获取下一张图片的编号
        id += 100

    model = SeamCarving()
    for id in imgDict.keys():
        height, width, _ = imgDict[id]['img'].shape
        # 根据公式计算缩小的尺寸: 1-背景区域面积/（3*图像面积）
        backgroundProportion = 1 - np.sum(imgDict[id]['mask']) / imgDict[id]['mask'].size
        areaProportion = 1 - backgroundProportion / 3
        height *= areaProportion
        width *= areaProportion
        print(f"Processing #{id}...")
        # 执行Seam Carving
        model.changeShape(imgDict[id]['img'], (int(height), int(width)), imgDict[id]['mask'], 'result1', id)
