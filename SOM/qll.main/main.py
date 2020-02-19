# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
sys.path.append('../qll.tool')
import SOMNet


#创建som网络对象
som = SOMNet.SOMnet()

#生成坐标在3以内的二维数据点作为训练集
som.loadDataMat(200,3)

#训练
som.train()

#显示聚类效果
som.showCluster(plt)
plt.show()