import numpy as np

'''
Author: Hendrik Pieres

'''

class Experience:
    def __init__(self, inputSize, outputSize, maxMemory = 10000, discount = .9):
        self.maxMemory = maxMemory
        self.stateMemory = np.zeros((self.maxMemory, inputSize))
        self.actionMemory = np.zeros((self.maxMemory), dtype=int)
        self.rewardMemory = np.zeros((self.maxMemory))
        self.nextstateMemory = np.zeros((self.maxMemory, inputSize))
        self.gameOverMemory = np.zeros(self.maxMemory, dtype=bool)
        self.discount = discount
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.currentIndex = 0

    def remember(self, state, action, reward, nextState, game_over):
        indexMod = self.currentIndex % self.maxMemory
        self.stateMemory[indexMod] = state
        self.actionMemory[indexMod] = action
        self.rewardMemory[indexMod] = reward
        self.nextstateMemory[indexMod] = nextState
        self.gameOverMemory[indexMod] = game_over
        self.currentIndex += 1

    def getTrainInstance(self, modelLearn, modelDecide, batchSize):
        minLength = min(self.currentIndex, batchSize)
        
        elements = np.random.randint(0, min(self.currentIndex, self.maxMemory), size=minLength)
        outputs = np.zeros((self.inputSize, self.outputSize))
        actions = self.actionMemory[elements]
        rewards = self.rewardMemory[elements]
        gameOvers = self.gameOverMemory[elements]
        states = self.stateMemory[elements]
        nextStates = self.nextstateMemory[elements]

        outputs = modelLearn.predict_on_batch(states)
        newOutputs = np.max(modelDecide.predict_on_batch(nextStates), axis = 1)

        for index, act in enumerate(actions):
            if gameOvers[index]:
                outputs[index, act] = rewards[index]
            else:
                test = rewards[index]*0.1 + self.discount * newOutputs[index]
                outputs[index, act] = test
        return states, outputs