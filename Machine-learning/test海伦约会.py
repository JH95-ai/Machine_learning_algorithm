from numpy import *
import numpy as np
import 海伦约会 as a
import matplotlib.pyplot as plt
datingDataMat,datingLabels=a.file2matrix('/home/jethro/文档/Machine-Learning-master/kNN/2.海伦约会/datingTestSet.txt')
print(datingDataMat)
print(datingLabels[0:20])
normDataSet,ranges,minVals=a.autoNorm(datingDataMat)
print('normDataSet是:')
print(normDataSet)
print('ranges是：')
print(ranges)
print('minVals是：')
print(minVals)
b=a.datingClassTest()
c=a.classifyperson()
print(c)
