import numpy as np
from functools import reduce
from math import log
'''
创建实验样本
Parameters:
    无
Returns:
    postingList -实验样本切分的词条
    classVec -类别标签向量
Time:
    2018-1-3
'''
def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec#返回实验样本切分的词条和类别标签向量
'''
将切分的实验样本词条整理成不重复的词条,也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet -返回不重复的词条列表,也就是词汇表
Time:
    2018-1-3
'''
def createVocabList(dataSet):
    vocabSet=set([]) #创建一个空的不重复列表
    for document in dataSet:
        vocabSet=vocabSet |set(document)#取并集
    return list(vocabSet)
'''
根据vocabList词汇表,将inputSet向量化,向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet -切分的词条列表
Returns:
    returnVec -文档向量，词集模型
Time:
    2018-1-3
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:   #遍历整个词条
        if word in inputSet:   #如果词条存在于词汇表中,则置1
            returnVec[vocabList.index(word)]=1
        else:
            print("the wordL %s is not in my Vocabulary!"%word)
    return returnVec     #返回文档向量
'''
朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix -训练文档矩阵,即set0fWords2Vec返回的returnVec构成的矩阵
    trainCategory -训练类别标签向量,即loadDataSet返回的classVec
Returns:
    P0Vect - 非的条件概率数组
    p1Vect -侮辱类的条件概率数组
    pAbsive -文档属于侮辱类的概率
Time:
    2019-1-3
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #计算训练的文档数目
    numWords=len(trainMatrix[0]) #计算每篇文档的词条数
    pAbusive=sum(trainCategory)/float(numTrainDocs)#文档属于侮辱类的概率
    p0Num=np.zeros(numWords)
    p1Num=np.zeros(numWords) #创建numpy.zeros数组
    p0Denom=0.0
    p1Denom=0.0    #分母初始化为0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num +=trainMatrix[i]
            p1Denom +=sum(trainMatrix[i])
        #统计属于侮辱类的条件概率所需的数据,即P(w0|1),P(w1|1),P(w2|1)...
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
        #统计属于非侮辱类的条件概率所需的数据,即P(w0|0),P(w1|0),P(w2|0)...
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom  #相除
    return p0Vect,p1Vect,pAbusive
    #返回属于侮辱类的条件概率数组,属于非侮辱类的条件概率数组,文档属于类的概率
'''
朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify -待分类的词条数组
    p0Vec -非侮辱性的条件概率数组
    p1Vec -侮辱类的条件概率数组
    pClass1 -文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 -属于侮辱类
Time:
    2019-1-3
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #p1=reduce(lambda x,y:x*y,vec2Classify*p1Vec)*pClass1 #对应元素相乘
    #p0=reduce(lambda x,y:x*y,vec2Classify*p0Vec)*(1.0-pClass1)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1>p0:
        return 1
    else:
        return 0
'''
测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
Time:
    2019-1-3
'''
def testingNB():
    testlists,testclasses=loadDataSet()  #创建实验样本
    vocablists=createVocabList(testlists) #创建词汇表
    trainMat=[]    #创建空表
    for i in testlists:
        trainMat.append(setOfWords2Vec(vocablists,i))
        #将实验样本向量化
    p0v,p1v,pAb=trainNB0(np.array(trainMat),np.array(testclasses))
    #训练朴素贝叶斯分类器
    testEntry=['love','my','dalmation'] #测试样本1
    thisDoc=np.array(setOfWords2Vec(vocablists,testEntry))
    #测试样本向量化
    if classifyNB(thisDoc,p0v,p1v,pAb):#执行分类并打印分类结果
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')
    testEntry=['stupid','garbage'] #测试样本2
    thisDoc=np.array(setOfWords2Vec(vocablists,testEntry))
    #测试样本向量化
    if classifyNB(thisDoc,p0v,p1v,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')		#执行分类并打印分类结果
if __name__=='__main__':
    testingNB()