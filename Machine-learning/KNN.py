from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#使用K近邻算法将每组数据划分到某个类中
#inX分类的输入向量
#dataSet输入的训练样本集
#标签向量为labels
#k表示用于选择最近邻居的数目
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #距离计算，运用欧拉公式
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistances=distances.argsort()
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        votelabel=labels[sortedDistances[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
