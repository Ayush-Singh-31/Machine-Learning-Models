import random

class LinearRegression:
    def __init__(self):
        self.path = None
        self.data = []           
        self.features = []      
        self.target = None     
        self.X = []             
        self.y = []            
        self.parameters = []    
        self.interceptTerm = 0.0
        self.costFunction = 0.0
        self.learningRate = 0.01

    def importCSV(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            raw = [line.strip().split(',') for line in f]
        self.features = raw[0]
        self.data = [
            [float(val) if val != "NA" else 0.0 for val in row]
            for row in raw[1:]
        ]
        return

    def setFeatures(self, features):
        self.features = features
        return

    def setTarget(self, target):
        self.target = target
        idx_map = {name: i for i, name in enumerate(self.features)}
        target_idx = idx_map[self.target]
        self.X = [
            [row[i] for i, name in enumerate(self.features) if name != self.target]
            for row in self.data
        ]
        self.y = [row[target_idx] for row in self.data]
        return

    def getFeatures(self):
        return self.features

    def getTarget(self):
        return self.target

    def makeParameters(self):
        n_preds = len(self.features) - 1
        self.parameters = [0.0] * n_preds
        if n_preds > 0:
            self.parameters[0] = 1.0
        return

    def makeInterceptTerm(self):
        n = len(self.X)
        if n == 0:
            self.interceptTerm = 0.0
        else:
            residuals = []
            for xi, yi in zip(self.X, self.y):
                y_pred = sum(p * x for p, x in zip(self.parameters, xi))
                residuals.append(yi - y_pred)
            self.interceptTerm = sum(residuals) / n
        return

    def getInterceptTerm(self):
        return self.interceptTerm

    def makeCostFunction(self):
        cost = 0.0
        for xi, yi in zip(self.X, self.y):
            y_pred = sum(p * x for p, x in zip(self.parameters, xi)) + self.interceptTerm
            cost += (y_pred - yi) ** 2
        self.costFunction = 0.5 * cost
        return
    
    def setLearningRate(self, rate):
        self.learningRate = rate
        return
    
    def getLearningRate(self):
        return self.learningRate
    
    def batchGradientDescent(self, iterations=1000):
        if not self.parameters:
            self.makeParameters()
        self.makeInterceptTerm()
        for _ in range(iterations):
            grad_w = [0.0] * len(self.parameters)
            grad_b = 0.0
            for xi, yi in zip(self.X, self.y):
                y_pred = sum(p * x for p, x in zip(self.parameters, xi)) + self.interceptTerm
                error = yi - y_pred
                grad_b += error
                for j in range(len(self.parameters)):
                    grad_w[j] += error * xi[j]
            for j in range(len(self.parameters)):
                self.parameters[j] += self.learningRate * grad_w[j]
            self.interceptTerm += self.learningRate * grad_b
            self.makeCostFunction()
        return
    
    def stochasticGradientDescent(self, epochs=1000):
        if not self.parameters:
            self.makeParameters()
        self.makeInterceptTerm()
        n = len(self.X)
        for _ in range(epochs):
            combined = list(zip(self.X, self.y))
            random.shuffle(combined)
            for xi, yi in combined:
                y_pred = sum(p * x for p, x in zip(self.parameters, xi)) + self.interceptTerm
                error = yi - y_pred

                for j in range(len(self.parameters)):
                    self.parameters[j] += self.learningRate * error * xi[j]
                self.interceptTerm += self.learningRate * error
            self.makeCostFunction()

if __name__ == "__main__":
    lr = LinearRegression()
    lr.importCSV("Dataset/Dummy-LR.csv")
    lr.setTarget("MEDV")
    lr.makeParameters()
    lr.setLearningRate(0.001)
    lr.batchGradientDescent(iterations=5000)
    print("Weights:", lr.parameters)
    print("Bias:", lr.getInterceptTerm())
    print("Final cost:", lr.costFunction)

    lr = LinearRegression()
    lr.importCSV("Dataset/Dummy-LR.csv")
    lr.setTarget("MEDV")
    lr.makeParameters()
    lr.setLearningRate(0.001)
    lr.stochasticGradientDescent(epochs=5000)
    print("Weights:", lr.parameters)
    print("Bias:", lr.getInterceptTerm())
    print("Final cost:", lr.costFunction)