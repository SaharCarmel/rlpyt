import numpy as np

class Coach():
    def __init__(self, envOptions, vectorSize):
        self.envOptions = envOptions
        self.vectorSize = vectorSize
        self.envList = []
    
    def generateVector(self):
        self.envList = []
        for i in range(self.vectorSize):
            _envNumber = np.random.uniform(0,len(self.envOptions)-1,1)
            _envNumber = np.around(_envNumber,0)
            self.envList.append(self.envOptions[int(_envNumber[0])])
        return self.envList
    
    def generateStatistics(self):
        uniqueVals = self.envOptions
        _statisticsList = dict.fromkeys(uniqueVals, 0)
        for env in _statisticsList:
            _statisticsList[env] = self.envList.count(env)
        return _statisticsList

