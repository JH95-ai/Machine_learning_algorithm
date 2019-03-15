import numpy as np
from os import listdir
#将32*32的二进制图像转换为1*1024向量
"""
Parameters:
    filename-文件名
Returns:
    returnVect -返回二进制图像的1*1024向量
Modify：
    2018-12-20
"""
def imgchangevector(filename):
    #创建1*1024零向量
    returnVect=np.zeros((1,1024))
    #打开文件
    fr=open(filename)
    #按行读取
    #lineStr = fr.readlines()
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
#filename='/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/testDigits/0_13.txt'
#a=imgchangevector(filename)
#print(a[0,32:63])
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
"""
    函数说明:手写数字分类测试
Parameters:
    无
Returns:
    无
Modify:
    2018-12-20 
"""
def handwritingClassTest():
    #测试集的Labels
    haLabels=[]
    #返回trainDigit目录下的文件名
    filename='/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/trainingDigits'
    trainingFileList=listdir(filename)
    #返回文件夹下文件的个数
    m=len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat=np.zeros((m,1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        filenameStr=trainingFileList[i]
        #获得分类的数字
        classNumber=int(filenameStr.split('_')[0])
        #将获得的类别添加到haLabels中
        haLabels.append(classNumber)
        #将每一个文件中的1*1024数据存储到trainMat矩阵中
        trainingMat[i,:]=imgchangevector('/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/trainingDigits%s'%(filenameStr))
    #返回testDigits目录下的文件名
    testfileList=listdir('/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/testDigits')
    #错误检测计数
    errorcount=0.0
    #测试数据的数量
    mTest=len(testfileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        filenameStr=testfileList[i]
        #获得分类的数字
        classNumber=int(filenameStr.split('_')[0])
        #获得测试集的1*1024向量,用于训练
        vectorUnderTest=imgchangevector('/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/testDigits%s'%(filenameStr))
        #获得预测结果
        classifierResult=classify0(vectorUnderTest,trainingMat,haLabels,3)
        print("分类返回结果为%d\t真实结果为%d"%(classifierResult,classNumber))
        if(classifierResult!=classNumber):
            errorcount+=1.0
    print("总共错了%d个数据\n错误率为%f%%"%(errorcount,errorcount/mTest))

if __name__=='__main__':
    handwritingClassTest()