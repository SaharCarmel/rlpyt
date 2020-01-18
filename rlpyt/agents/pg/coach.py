import numpy as np

class Coach():
    def __init__(self, envOptions, vectorSize):
        self.envOptions = envOptions
        self.vectorSize = vectorSize
    
    def generateVector(self):
        _envList = []
        for i in range(self.vectorSize):
            _envNumber = np.random.uniform(0,len(self.envOptions)-1,1)
            _envNumber = np.around(_envNumber,0)
            _envList.append(self.envOptions[int(_envNumber[0])])
        return _envList
