import numpy as np
from rlpyt.utils.logging import logger
import numpy as np
import math


class Bandit():
    def __init__(self, env, beta, initialQ):
        self.env = env
        self.Q = initialQ
        self.counts = 1
        self.beta= beta
        self.score = 0

    def updateQ(self, reward, iteration, totalCounts):
        self.Q = self.beta*self.Q + reward
        self.counts = self.counts*self.beta + iteration
        self.score = self.Q +math.sqrt((2*math.log(totalCounts))/(self.counts))
    
    def __repr__(self):
        return "Env:{} with current Q:{} and last time used in iteration:{}".format(self.env, self.Q, self.i)

class Coach():
    def __init__(self, envOptions, vectorSize, algo='Random', beta=0.1, initialQ=1):
        self.envOptions = envOptions
        self.vectorSize = vectorSize
        self.envList = []
        self.algo = algo
        self.beta = beta
        self.generateFunctions = {'Random': self.generateRandomVector,
                                  'Bandit': self.generateBanditVector}
        self.banditList = [Bandit(env, beta, initialQ) for env in envOptions]
        self.currentEnv = []
        self.totalCounts = 0

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

    def sortQ(self, val):
        return val.score

    def chooseEnv(self):
        self.banditList.sort(key=self.sortQ, reverse=True)
        self.currentEnv = self.banditList[0]
        for item in self.banditList:
            logger.record_tabular('Q-{}'.format(item.env), item.Q)
            logger.record_tabular('Score-{}'.format(item.env), item.score)

        

    def generateBanditVector(self,reward, iteration):
        self.totalCounts = self.totalCounts*self.beta + 1
        for bandit in self.banditList:
            if bandit.env == self.currentEnv.env:
                bandit.updateQ(reward, 1, totalCounts=self.totalCounts)
            else:
                bandit.updateQ(0,0,totalCounts=self.totalCounts)
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