import  numpy as np

class NaiveBayesBase(object):
    def __init__(self):
        pass

    def fit(self, trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])