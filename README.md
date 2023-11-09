本仓库内为2022年春季学期中山大学计算机视觉课程的期末大作业。

文件说明：

题目1                            :  题目1代码与实验结果

- src文件夹                 :  源代码
- result1文件夹           :  实验结果
  - gif文件夹             :  119.png与319.png对应的gif动图
  - config文件夹       :  其他图片的执行结果

题目2                          	        :  题目2代码与实验结果

- src文件夹              	        :  源代码
- result2文件夹       	        :  实验结果

题目3                    		 :  题目3代码与实验结果

- dataset文件夹           		 :  测试集与训练集数据
- result3文件夹    		 :  实验结果
- src文件夹       		 :  源代码
  - segmentation.py    		 :  复用题目2代码，生成训练集与测试集
  - lassification.py          	 :  用于完成题目3的题目要求

计算机视觉期末大作业.pdf      :  实验要求

实验报告.pdf          :  实验报告

说明：
题目2.3共同组成一个图象区域判别器，具体过程如下
1. 实现基于 Graphbased image segmentation 方法，将每张图分割为 50~70 个区域
2. 提取每一个区域的归一化 RGB 颜色直方图特征  和全图的归一化 RGB 颜色直方图  
3. 将每一个区域的颜色对比度特征定义为区域颜色直方图和全图颜色直方图的拼接  
4. 采用PCA 算法对颜色对比度特征进行降维取前 20 维
5. 构建 visual bag of words dictionary：
  - 使用Kmeans聚类方法获得50个聚类中心点作为visual word；
  - 计算颜色对比度特征和50个visual word特征的点积结果，得到50维数据；
  - 再与之前PCA算法得到的20维数据拼接，得到70维的数据集。  
6. 将该数据集作为训练集训练非线性SVM  ，使其能够判别图中区域属于前景还是背景    

---

This is the final assignment of the Computer Vision course, Sun Yat-Sen University in the spring semester of 2022.

Document description:

题目1 	:   Code and experimental results of Question 1

- src folder 			:  Source code
- result1 folder	 :  Results
  - gif folder			 :  119.png and 319.png corresponding gif
  -  config folder	 :  Other image execution results

题目2	 :   Code and experimental results of Question 2

- src folder 				:  Source code
- result2 folder 		:  Results

题目3 	:   Code and experimental results of Question 3

- dataset folder 		:  Test set and training set data
- result3 folder 		:  Results of experiments
- src folder				 :  Source code
  - segmentation.py 	:  Reuse problem 2 code to generate training sets and test sets
  - classification.py 	:  Used to complete the requirements of problem 3

计算机视觉期末大作业.pdf   :  Experimental requirements

实验报告.pdf    	:  Experiment report

Explanation:
Task 2.3 together forms an image region discriminator, and the specific process is as follows:
1. Implement the Graph-based image segmentation method to segment each image into 50-70 regions.
2. Extract the normalized RGB color histogram features for each region and the entire image.
3. Define the color contrast feature for each region as the concatenation of the region's color histogram and the global color histogram.
4. Use the PCA algorithm to reduce the dimensionality of the color contrast features to the top 20 dimensions.
5. Build a visual bag of words dictionary:
  - Utilize the K-means clustering method to obtain 50 cluster centroids as visual words.
  - Calculate the dot product between the color contrast features and the 50 visual word features, resulting in a 50-dimensional dataset.
  - Concatenate the 20-dimensional data obtained from the PCA with the 50-dimensional data, resulting in a 70-dimensional dataset.
6. Use the constructed dataset as the training set to train a non-linear SVM to classify image regions as foreground or background.
