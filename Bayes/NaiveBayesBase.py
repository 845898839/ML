import numpy as np
from functools import  reduce

def LoadDataSet():
    dataList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    label = [0,1,0,1,0,1]
    return  dataList,label

def CreateVocabList(dataSet):
    vocabSet = set()
    for line in dataSet:
        vocabSet = vocabSet.union(set(line))
    return list(vocabSet)

def WordsToVec(vocabList, inputSet):
    result = [0]* len(vocabList)
    for word in inputSet:
        if word in vocabList:
            result[vocabList.index(word)] = 1
        else:
            print("error",word)
    return  result

def Fit(trainX, trainY):
    p1Class = sum(trainY) / len(trainY)
    p0Num = np.ones(len(trainX[0]))
    p1Num = np.ones(len(trainX[0]))
    p0Denom = 2.0 #拉普拉斯变换
    p1Denom = 2.0
    for i in range(len(trainX)):
        if trainY[i] == 1:
            p1Num += trainX[i]
            p1Denom += sum(trainX[i])
        else:
            #print("p0num:",p0Num, p0Num.shape)
            #print("trainX:",trainX[i], trainX[i].shape)
            p0Num += trainX[i]
            p0Denom += sum(trainX[i])
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p0Vec, p1Vec, p1Class

def Predict(p0V, p1V, p1Class, testV):
    temp1 = p1V * testV

    p1 = reduce(lambda x,y : x*y, [x for x in p1V * testV if x != 0]) * p1Class
    p0 = reduce(lambda x,y : x*y, [x for x in p0V * testV if x != 0]) * (1 - p1Class)
    print("p1",p1,[x for x in p1V * testV if x != 0])
    print("p0",p0,[x for x in p0V * testV if x != 0])
    if p1 > p0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    data, label = LoadDataSet()
    # 预处理
    myVocabList = CreateVocabList(data)
    trainData = [];
    for line in data:
        trainData.append(WordsToVec(myVocabList, line))
    #拟合训练
    p0V, p1V, p1 = Fit(np.array(trainData), np.array(label))
    #预测
    test1 = ['stupid','garbage']
    result = Predict(p0V, p1V, p1, WordsToVec(myVocabList, np.array(test1)))
    print(result)

    test1 = ['love','my','dalmation']
    result = Predict(p0V, p1V, p1, WordsToVec(myVocabList, np.array(test1)))
    print(result)