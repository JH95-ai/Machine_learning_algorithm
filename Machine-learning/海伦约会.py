from numpy import *
import operator
import numpy as np
def file2matrix(filename):
    fr=open(filename,'r',encoding='utf-8')
    arrayOLines=fr.readlines()
    #arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
    #得到文件行数
    numberOfLines=len(arrayOLines)
    #创建返回的Numpy矩阵
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    #解析文件数据到列表
    for line in arrayOLines:
        line=line.strip()#截取掉所有的回车字符,然后使用tab字符 \t
        # 将上一步得到的整行数据分割成一个元素列表
        listFromLine=line.split('\t')
        #选取前3个元素,将它们存储到特征矩阵中
        returnMat[index,:]=listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
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
#归一化特征值
def autoNorm(dataSet):
    #将每列的最小值放在minVals中，
    # dataSet.min(0)中的参数0让函数可以从列中选取最小值
    #而不是选取当前行的最小值
    minVals=dataSet.min(0)
    #将最大值放在maxVals中
    maxVals=dataSet.max(0)
    #为了归一化特征值，我们必须使用当前值减去最小值，然后除以取值范围
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    #特征值相除
    normDataSet=normDataSet /tile(ranges,(m,1))

    return normDataSet,ranges,minVals

#分类机针对约会网站的测试代码
def datingClassTest():
    #得到所有数据的百分之十
    hoRatio=0.10
    #首先使用了file2matrix和autoNorm()函数从文件中读取数据并将其转换为归一化特征值
    #接着计算测试向量的数量,此部决定了normMat向量中哪些数据用于测试
    #哪些数据用于分类器的训练样本
    datingDataMat,datingLabels=file2matrix('/home/jethro/文档/Machine-Learning-master/kNN/2.海伦约会/datingTestSet.txt')
    #数据归一化，返回归一化的数据,归一化数据的范围,最小的数据值
    normat,ranges,minVals=autoNorm(datingDataMat)
    #获取normat的行数
    m=normat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs=int(m*hoRatio)
    #分类器错误计数
    errorCount=0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        #
        classifierResult=classify0(normat[i,:],normat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("分类结果:%s,真实类别:%d"%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1
        print('错误率为%f:'%(errorCount/float(numTestVecs)*100))

def classifyperson():
    #输出结果
    resultList=['讨厌','有点喜欢','非常喜欢']
    #三维特征用户
    precentTats=float(input("玩视频游戏所耗时间百分比:"))
    ffMiles=float(input("每年获得的飞行常客里程数:"))
    iceCream=float(input("每周消费的冰淇淋:"))
    #打开的文件名
    filename='/home/jethro/文档/Machine-Learning-master/kNN/2.海伦约会/datingTestSet.txt'
    #打开并处理数据
    datingDataMat,datingLabels=file2matrix(filename)
    #训练集归一化
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #生成Numpy数组,测试集
    inArr=np.array([ffMiles,precentTats,iceCream])
    #测试集归一化
    norminArr=(inArr-minVals)/ranges
    #返回分类结果
    classifierResult=classify0(norminArr,normMat,datingLabels,3)
    #打印结果
    print("你可能%s这个人"%(resultList[classifierResult-1]))
if __name__ == '__main__':
    datingClassTest()