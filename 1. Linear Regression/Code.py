import numpy as np
class LinearRegression:
    def __init__(self):
        self.path = None
        self.data = []

    def importCSV(self, path):
        self.path = path
        file = open(self.path)
        self.data = [line.strip().split(',') for line in file]
        file.close()
        return
    
