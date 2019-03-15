# -*- coding: UTF-8 -*-
import numpy as np
import random
import re
'''
将切分的实验样本词条值整理成不重复的词条列表,也就是词汇表
Parameters:
    dataSet -整理的样本数据集
Returns:
    vocabSet -返回不重复的词条列表,也就是词汇表
Time:
    2018-1-3
'''
def createVocabList(dataSet):
    vocabSet=set([])   #创建一个空的不重复列表
    for document in dataSet:
        vocabSet=vocabSet |set(document)  #取并集
    return list(vocabSet)
'''
根据vocabList词汇表,将inputSet向量化,向量的每个元素为1或0
Parameters:
    vocabList -createVocabList返回的列表
    inputSet -切分的词条列表
Returns:
    returnvec -文本向量,词集模型
Time:
    2019-1-3
'''
def setofwordsvec(vocabList,inputSet):
    returnvec=[0]*len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet:  #遍历每个词条
        if word in vocabList: #如果词条存在于词汇表中,则置1
            returnvec[vocabList.index(word)]=1
        else:
            print("the word:%s not in my vocablist",word)
    return returnvec   #返回文档向量
'''
根据vocabList词汇表,构建词袋模型
Parameters:
    vocabList - createvocabList返回的列表
    inputSet -切分的词条列表
Returns:
    returnvec -文档向量，词袋模型
Time:
    2019-1-3
'''
def bagofwordvec(vocabList,inputSet):
    returnvec=[0]*len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet:    #遍历每个词表
        if word in vocabList:
            returnvec[vocabList.index(word)]+=1 #如果词条存在于词汇表中,则计数加一
    return returnvec        #返回词袋模型
'''
朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文档矩阵，即setofwordvec返回的returnvec构成的矩阵
    trainCategory -训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0vect -非侮辱类的条件概率数组
    P1vect -侮辱类的条件概率数组
    pAbusive -文档属于侮辱类的概率
Time:
    2019-1-3
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #计算训练的文档数目
    numwords=len(trainMatrix[0]) #计算每篇文档的词条数
    pAbusive=sum(trainCategory)/float(numTrainDocs)#文档属于侮辱类的概率
    p0Num=np.ones(numwords)
    p1Num=np.ones(numwords) #创建numpy.ones数组,词条出现初始化为1
    p0Denom=2.0;p1Denon=2.0 #分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:  #统计属于侮辱类的条件概率
            p1Num+=trainMatrix[i]
            p1Denon+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1vec=np.log(p1Num/p1Denon)
    p0vec=np.log(p0Num/p0Denom)
    return p1vec,p0vec,pAbusive
'''
朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify -待分类的词条数组
    p0vec -非侮辱类的条件概率数组
    p1vec -侮辱类的条件概率数组
Returns:
    0 -属于非侮辱类
    1 -属于侮辱类
Time 
    2019-1-7
'''
def classifyNB(vecClassify,p0vec,p1vec,pClass1):
    p1=sum(vecClassify*p1vec)+np.log(pClass1)
    #对应元素相乘.logA*B=logA+logB，所以这里加上log(pClass1)
    p0=sum(vecClassify*p0vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
'''
接收一个大字符串并将其解析为字符串列表
Parameters:
    无
Returns:
    无
Time:
    2019-1-7
'''
def textParse(bigString):
    listOfToken=re.split('\W+',bigString) #将字符串转换为字符列表
    return [tok.lower() for tok in listOfToken if len(tok)>2]
    #除了单个字母,例如大写的I，其他单词变成小写
'''
测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
Time:
    2019-1-7
'''
def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):  #遍历25个txt文件
        wordList=textParse(open('/home/jethro/文档/email/spam/%d.txt'%i,'r',encoding='ISO-8859-1').read())
        #读取每个垃圾邮件,并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)  #标记垃圾邮件,1表示垃圾文件
        wordList=textParse(open('/home/jethro/文档/email/ham/%d.txt' %i,'r',encoding='ISO-8859-1').read())
        #读取每个非垃圾邮件,并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0) #标记非垃圾邮件,0表示非垃圾邮件
    vocabList=createVocabList(docList) #创建词汇表,不重复
    traingSet=list(range(50));testSet=[]
    #从50个邮件中,随机挑选出40个作为训练集,10个作为测试集
    for i in range(10):   #随机选取索引值
        randIndex=int(random.uniform(0,len(traingSet)))
        #随机选取索引值
        testSet.append(traingSet[randIndex])
        #添加测试集的索引值
        del(traingSet[randIndex]) #在训练集列表中删除已添加的索引值
    trainMat=[];trainClasses=[] #创建训练集矩阵和训练集类别标签系向量
    for docIndex in traingSet: #遍历训练集
        trainMat.append(setofwordsvec(vocabList,docList[docIndex]))
        #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])
        #将类别添加到训练集类别标签系向量中
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    #训练朴素贝叶斯模型
    errorcount=0 #错误分类计数
    for docIndex in testSet: #遍历测试集
        wordVector=setofwordsvec(vocabList,docList[docIndex])
        #测试集的词集模型
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            #如果分类错误
            errorcount+=1
            print('分类错误的测试集:',docList[docIndex])
    print('errorcount', errorcount)
    print('错误率:%.2f%%'%(float(errorcount)/len(testSet)))

if __name__=='__main__':
    spamTest()