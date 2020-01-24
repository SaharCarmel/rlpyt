import numpy as np
from rlpyt.utils.logging import logger


class Bandit():
    def __init__(self, env, beta, initialQ):
        self.env = env
        self.Q = initialQ
        self.i = 0
        self.beta= beta

    def updateQ(self, reward, iteration):
        self.Q = self.beta**(iteration - self.i)*self.Q + (1-self.beta**(iteration - self.i))*reward
        self.i = iteration
    
    def __repr__(self):
        return "Env:{} with current Q:{} and last time used in iteration:{}".format(self.env, self.Q, self.i)

class Coach():
    def __init__(self, envOptions, vectorSize, algo='Random', beta=0.1, initialQ=1):
        self.envOptions = envOptions
        self.vectorSize = vectorSize
        self.envList = []
        self.algo = algo
        self.generateFunctions = {'Random': self.generateRandomVector,
                                  'Bandit': self.generateBanditVector}
        self.banditList = [Bandit(env, beta, initialQ) for env in envOptions]
        self.currentEnv = []

    def generateVector(self, reward, iteration):
        reward = self._calculateAvgReturn(reward)
        return self.generateFunctions[self.algo](reward, iteration)

    def generateStatistics(self):
        uniqueVals = self.envOptions
        _statisticsList = dict.fromkeys(uniqueVals, 0)
        for env in _statisticsList:
            _statisticsList[env] = self.envList.count(env)
        return _statisticsList

    def generateRandomVector(self,reward, iteration):
        self.envList = []
        for i in range(self.vectorSize):
            _envNumber = np.random.uniform(0, len(self.envOptions)-1, 1)
            _envNumber = np.around(_envNumber, 0)
            self.envList.append(self.envOptions[int(_envNumber[0])])
        return self.envList

    def chooseEnv(self):
        _maxEnv = ''
        _tempMax = 0
        for bandit in self.banditList:
            if bandit.Q > _tempMax:
                _maxEnv = bandit
                _tempMax = bandit.Q
                self.currentEnv = bandit
        for item in self.banditList:
            logger.record_tabular('Q-{}'.format(item.env), item.Q)
        

    def generateBanditVector(self,reward, iteration):
        self.currentEnv.updateQ(reward, iteration)
        self.chooseEnv()
        self.envList = []
        for i in range(self.vectorSize):
            self.envList.append(self.currentEnv.env)
        return self.envList
    
    def _calculateAvgReturn(self,reward):
        _sum = 0
        _returnAvg = 0
        _discountedReturnAvg = 0
        for item in reward:
            _returnAvg += item['Return']
            _discountedReturnAvg += item['DiscountedReturn']
        _returnAvg = _returnAvg / len(reward)
        _discountedReturnAvg = _discountedReturnAvg / len(reward)
        return _returnAvg
    
    def generateInitialVector(self):
        self.currentEnv = self.banditList[0]
        return [self.envOptions[0] for i in range(self.vectorSize)]