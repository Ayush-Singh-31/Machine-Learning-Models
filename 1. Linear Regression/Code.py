import numpy as np
class LinearRegression:
    def __init__(self):
        self.path = None
        self.data = []
        self.target = None
        self.features = None
        self.parameters = []
        self.costFunction = None

    def importCSV(self, path):
        self.path = path
        file = open(self.path)
        self.data = [line.strip().split(',') for line in file]
        file.close()
        self.features = self.data[0]
        self.data = self.data[1:]
        self.data = [[float(x) if x != "NA" else 0.0 for x in row]for row in self.data]
        self.data = [list(map(float, col)) for col in self.data]
        return
    
    def setFeatures(self, features):
        self.features = features
        return
    
    def setTarget(self, target):
        self.target = target
        self.features.remove(target)
        return
    
    def getFeatures(self):
        return self.features

    def getTarget(self):
        return self.target
    
    def makeParameters(self):
        self.parameters = [0] * len(self.data)
        self.parameters[0] = 1
        return
    
    def makeCostFunction(self):
        self.costFunction = 0.5*[]

lr = LinearRegression()
lr.importCSV("Dataset/HousingData.csv")
print(lr.features)
