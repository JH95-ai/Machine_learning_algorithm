from functools import reduce
def testreduce(a,b,c,d):
    a1=reduce(lambda x,y:x*y,a*b)
    a2=reduce(lambda x,y:x*y,c*d)
    return a1,a2
testreduce=testreduce(1,2,3,4)
print(testreduce)
