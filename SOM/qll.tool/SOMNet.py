# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:36:20 2020

@author: Administrator
"""


import numpy as np
import sys
sys.path.append('qll.tool')
import Function as fun

class SOMnet(object):
    def __init__(self):    #构造函数，用于初始化网络参数
        self.lratemax = 0.8   #最大学习率
        self.lratemin = 0.05  #最小学习率
        self.rmax = 5.0   #最大聚类半径
        self.rmin = 0.5    #最小聚类半径
        self.steps = 20000   #迭代次数，不小于训练集行数的5倍
        self.lratelist = []  #记录每次学习率
        self.rlist = []     #记录每次聚类半径
        self.w = []       #竞争层权重向量
        '''
        竞争层（即输出层）的神经元个数为聚类种类数，对于
        二维平面阵的竞争层来说，每个种类也用一个二维向量来
        表示，M为第一维取值个数，N为第二维取值个数,
        M*N即为种类数，也就是竞争层神经元个数
        
        M、N的值越接近，聚类效果会越好
        '''
        self.M = 2      #第一维取值个数
        self.N = 2      #第二维取值个数
        self.dataMat = []   #训练集
        self.classLabel = []  #聚类后类别标签
        
    def ratecacl(self,i):
        #计算第i次迭代的学习率和聚类半径
        rate = self.lratemax - ((i+1.0)*(self.lratemax-self.lratemin)/self.steps)
        r = self.rmax - ((i+1.0)*(self.rmax-self.rmin)/self.steps)
        return rate,r
    
    def init_grid(self):     #初始化竞争层网络的值，即对应的各种类向量
        k = 0
        grid = np.zeros((self.M*self.N,2))
        for i in range(self.M):
            for j in range(self.N):
                grid[k,:] = [i,j]
                k += 1      
        return grid
    
    
    def loadDataMat(self,n,top):
        self.dataMat = fun.createSet(n,top)
    
    def train(self):     #训练函数：训练数据集
        dm,dn = np.shape(self.dataMat)
        grid = self.init_grid()
        self.w = np.random.random(size=(self.M*self.N,dn))  #初始化权重向量
        if self.steps < 5*dm:
            self.steps = 5*dm
        
        for i in range(self.steps):    #主循环：训练权重向量w
            lrate,r = self.ratecacl(i)  #求出本次迭代的学习率和聚类半径
            self.lratelist.append(lrate)
            self.rlist.append(r)
            #1  随机取出一个样本
            k = np.random.randint(0,dm)
            mySample = self.dataMat[k,:]
            
            #2  计算出竞争层的最优节点
            minIndx = (fun.distM(np.mat(mySample),np.mat(self.w))).argmin()
            
            #3  计算邻域
            d1 = int(np.floor(minIndx/self.M))
            d2 = np.mod(minIndx,self.N)  #求出最优节点对应的类别向量
            disMat = fun.distM(np.mat([d1,d2]),grid)
            nodelindx = (disMat<r).nonzero()[1]
            for j in range(self.M*self.N):
                if nodelindx.__contains__(j):
                    self.w[j,:] += lrate*(mySample[0] - self.w[j,:])
        #主循环结束
            
            
        #为dataMat的每条数据判定类别
        self.classLabel = []
        for i in range(dm):
            kind_i = (fun.distM(np.mat(self.dataMat[i,:]),self.w)).argmin()
            self.classLabel.append(kind_i)
            
        self.classLabel = np.mat(self.classLabel)
        
    def showCluster(self,plt):   #绘图函数：根据不同种类绘制不同样式的散点图
        colors = ['red','blue','black','pink','brown',
                  'green']
        lst = list(np.unique(self.classLabel.tolist()[0]))
        for i in range(len(self.classLabel.T)):
            k = lst.index(self.classLabel[0,i])
            plt.scatter(self.dataMat[i,0],self.dataMat[i,1],c=colors[k])
            