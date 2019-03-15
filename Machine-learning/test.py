import 决策树算法实现 as decision_tree
myDats,labels=decision_tree.creatDataSet()
a=decision_tree.splitDataSet(myDats,0,1)
#print(a)
b=decision_tree.splitDataSet(myDats,0,0)
#print(b)
c=decision_tree.choosebestFeatureToSplit(myDats)
#print(c)
featLabels=[]
myTree=decision_tree.createTree(myDats,labels,featLabels)
print(myTree)


