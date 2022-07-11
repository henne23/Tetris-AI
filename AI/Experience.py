import numpy as np

'''
Author: Hendrik Pieres

'''

class Experience:
    def __init__(self, inputSize, outputSize, maxMemory = 20000, discount = .95):
        self.maxMemory = maxMemory
        self.stateMemory = np.zeros((self.maxMemory, inputSize), dtype=int)
        self.rewardMemory = np.zeros((self.maxMemory))
        self.nextstateMemory = np.zeros((self.maxMemory, inputSize))
        self.gameOverMemory = np.zeros(self.maxMemory, dtype=bool)
        self.discount = discount
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.currentIndex = 0

    def remember(self, state, reward, nextState, game_over):
        indexMod = self.currentIndex % self.maxMemory
        self.stateMemory[indexMod] = state
        self.rewardMemory[indexMod] = reward
        self.nextstateMemory[indexMod] = nextState
        self.gameOverMemory[indexMod] = game_over
        self.currentIndex += 1

    def getTrainInstance(self, modelLearn, modelDecide, batchSize):
        minLength = min(self.currentIndex, batchSize)
        
        elements = np.random.randint(0, min(self.currentIndex, self.maxMemory), size=minLength)
        outputs = np.zeros((self.inputSize, self.outputSize))
        rewards = self.rewardMemory[elements]
        gameOvers = self.gameOverMemory[elements]
        states = self.stateMemory[elements]
        nextStates = self.nextstateMemory[elements]

        outputs = modelLearn.predict_on_batch(states)
        newOutputs = np.max(modelDecide.predict_on_batch(nextStates), axis = 1)

        for index, _ in enumerate(rewards):
            if gameOvers[index]:
                outputs[index] = rewards[index]
            else:
                outputs[index] = rewards[index] + self.discount * newOutputs[index]
        return states, outputs