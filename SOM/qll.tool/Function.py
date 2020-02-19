# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:57:13 2020

@author: Administrator
"""
'''
工具函数模块，提供网络类中要用到的工具函数
'''
import numpy as np

def createSet(n,top):  #生成n个随机二维点
    data = top * np.random.random(size=(n,2))
    return data


def ecluddistance(A,B):  #欧氏距离函数
    result = np.linalg.norm(A - B)
    return result

def distM(dataMat1,dataMat2):    #求两个矩阵各行向量之间的距离
    m1,n1 = np.shape(dataMat1)
    m2,n2 = np.shape(dataMat2)
    
    dist = np.zeros((m1,m2))
    
    for i in range(m1):
        for j in range(m2):
            dist[i,j] = ecluddistance(dataMat1[i,:],dataMat2[j,:])
    
    return np.mat(dist)