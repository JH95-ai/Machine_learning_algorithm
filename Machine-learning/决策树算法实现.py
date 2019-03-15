#1.创建数据集和标签
from math import log #从math包中导入log函数
import operator #导入operator包
def creatDataSet():
    dataSet=[[0, 0, 0, 0, 'no'],						#数据集
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
    labels=['年龄', '有工作', '有自己的房子', '信贷情况'] #特征标签
    return dataSet,labels  #函数返回数据集和特征标签.
#2.计算香农熵,根据label分类,从而计算香农熵
'''
Parameters:
    dataSet -- 数据集
Returns:
    shannonEnt --经验熵（香农熵）
Time:
    2018-12-26
'''
def calcShannonEnt(dataSet):
    numEntries=len(dataSet) #求出几行数据
    labelCounts={} #创建一个空字典
    for featVec in dataSet:#计算数据集中的数据
        currentLabel=featVec[-1]#取出最后一列
        if currentLabel not in labelCounts:
            labelCounts[currentLabel]=0
        #如果labelCounts中没有该标签,则加入该标签且统计为0
        labelCounts[currentLabel]+=1 #该标签出现次数+1
    shannonEnt=0.0 #初始化香农熵为0
    for key in labelCounts:
        prob=float(labelCounts[key]/numEntries) #计算该label出现的频率
        shannonEnt -=prob *log(prob,2)#计算香农熵,以2为底
    print('香农熵为:',shannonEnt)
    return shannonEnt #返回香农熵
'''
Parameters:
    dataSet -- 待划分的数据集
    axis --划分数据集的特征
    value --需要返回的特征的值
Returns:
    无
Time:
    2018-12-26
'''
#3划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[] #创建一个空列表
    for featVec in dataSet:
        if featVec[axis] ==value:
            reduceFeatVec=featVec[:axis]
            #按照value值选定特征之外的特征
            reduceFeatVec.extend(featVec[axis+1:]) #同上
            retDataSet.append(reduceFeatVec) #同上
    return retDataSet
'''
Parameters:
    dataSet --数据集
Returns:
    bestFeature --信息增益最大的特征的索引值
Time:
    2018-12-26
'''
#4选择最佳数据集划分方式
def choosebestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1 #最后一列用作labels
    baseEntropy=calcShannonEnt(dataSet) #计算未分类时的香农熵
    bestInfoGain=0.0
    bestFeature=-1 #初始化信息增益与最优特征
    for i in range(numFeatures): #迭代所有特征
        featList= [example[i] for example in dataSet]
        #featList表示dataSet中每个列表第i个数组合起来,构成的新列表
        uniqueVals=set(featList) #得到唯一值
        newEntropy=0.0  #熵初始化为0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value) #调用函数
            prob=len(subDataSet)/float(len(dataSet))#计算特征出现的概率
            newEntropy+=prob*calcShannonEnt(subDataSet) #调用函数
        infoGain=baseEntropy-newEntropy#计算信息增益,或者说熵的减少量
        if (infoGain>bestInfoGain):  #与目前所得最优熵就行比较
            bestInfoGain =infoGain
            bestFeature = i #用于分类的最好的特征index
    #print('bestFeature是：',bestFeature)
    return bestFeature   #返回信息增益最大的特征的索引值
'''
Parameters:
    classList --类标签列表
Returns:
    sortedClassCount[0][0] --出现此处最多的元素(类标签)
Time:
    2018-12-26
'''
def majorityCnt(classList):
    classCount={} #建立一个空字典
    for vote in classList: #对于classList中的所有项作计算
        if vote not in classCount.keys(): #当前class不在classCount的keys中
            classCount[vote] =0 #添加当前class并设置出现次数为0
        classCount[vote]+=1 #当前class出现次数为1
    sortedClassCount =sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #对classCount中的内容排序,倒序排列
    return sortedClassCount[0][0] #返回出现次数最多的类的名字
'''
Parameter:
    dataSet -训练数据集
    labels - 分类属性标签
    featLabels -存储选择的最优特征标签
Returns:
    myTree -决策树
Time:
    2018-12-26
'''
def createTree(dataSet,labels,featLabels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0]==len(classList)):
        return classList[0] #当所有的类都相等时停止划分
    if len(dataSet[0])==1 or len(labels)==0:#数据集中没有更多特征用于分类时停止划分
        return majorityCnt(classList)
    bestFeat=choosebestFeatureToSplit(dataSet)#取出最好的特征赋值给bestFeat
    bestFeatLabel=labels[bestFeat]#最优特征的标签
    myTree={bestFeatLabel:{}} #根据最优特征的标签生成树
    del(labels[bestFeat]) #删除已经使用的特征标签
    featValues=[example[bestFeat] for example in dataSet]
    #得到训练集中所有最优特征的属性值
    uniqueVals=set(featValues) #去掉重复的属性值
    for value in uniqueVals:#遍历特征,创建决策树.
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree
'''
Parameters:
    myTree --决策树
Returns:
    numLeafs --决策数的叶子结点的数目
Time:
    2018-12-26
'''
def getNumLeafs(myTree):
    numLeafs=0          #初始化叶子
    firstStr=next(iter(myTree)) #获取结点属性
    secondDict=myTree[firstStr]  #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #测试该结点是否为字典,如果不是，代表该结点为叶子结点
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs
'''
获取决策树的层数
Parameters:
    myTree -决策树
Returns:
    maxDepth -决策树的层数
Time:
    2019-1-2
'''
def getTreeDepth(myTree):
    maxDepth=0     #初始化决策树深度
    firstStr=next(iter(myTree))
    #python3中myTree.keys()返回的是dict_keys,不在是list,
    # 所以不能使用myTree.keys()[0]的方法获取结点属性,
    # 可以使用list(myTree.key())[0]
    secondDict=myTree[firstStr]   #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
        #测试该结点是否为字典,如果不是字典,代表此节点为叶子节点
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth =1
        if thisDepth >maxDepth:
            maxDepth=thisDepth  #更新层数
    return maxDepth
'''
绘制节点
Parameters:
    nodeTxt -节点名
    centerpt -文本位置
    parentPt -标注的箭头位置
    nodeTYpe -节点格式
Returns:
    无
Time:
    2019-1-2
'''
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args=dict(arrowstyle="<-")   #定义箭头格式
    font=FontProperties(fname)