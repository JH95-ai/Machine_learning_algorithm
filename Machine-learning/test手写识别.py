import numpy as np
import 手写识别
filename='/home/jethro/文档/Machine-Learning-master/kNN/3.数字识别/testDigits/0_13.txt'
testreturnVect=手写识别.imgchangevector(filename)
#print(testreturnVect[0,32:63])
filenumber=手写识别.handwritingClassTest()
