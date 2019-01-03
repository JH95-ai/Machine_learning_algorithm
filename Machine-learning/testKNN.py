import KNN
group,labels=KNN.createDataSet()
print(group)
print(labels)
a=KNN.classify0([0, 0], group, labels, 3)
print(a)