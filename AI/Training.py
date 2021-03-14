import numpy as np

from AI.Experience import Experience
from AI.actions import actions
from constants.GameStates import START, GAME_OVER

'''
Author: Hendrik Pieres

'''

class Training:
    def __init__(self, game, modelLearn, modelDecide, batchSize):
        self.game = game
        self.modelLearn = modelLearn
        self.modelDecide = modelDecide
        self.epsilon = .1
        self.loss = .0
        self.targetLine = game.height
        self.batchSize = batchSize
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1])

    def randmax(self, values):
        max_values = []
        current_max = values[0]
        index = 0
        for v in values:
            if v > current_max:
                max_values = [index]
                current_max = v
            elif v == current_max:
                max_values.append(index)
            index += 1
        if len(max_values) == 0:
            return np.random.randint(0,len(values)-1)
        else:
            return np.random.choice(max_values)

    def getReward(self):
        for i in range(self.game.height, 0, -1):
            for j in range(self.game.width):
                if self.game.field.values[i][j] == 0 and self.game.field.values[i-1][j] > 0:
                    self.targetLine -= 1
        

    def train(self):
        if self.game.changeFigure == None:
            cTyp = 10
        else:
            cTyp = self.game.changeFigure.typ
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, len(actions))
        else:
            inputLayer = self.game.field.values.reshape((1,-1))
            figures = np.asarray([self.game.Figure.typ, self.game.nextFigure.typ, cTyp]).reshape((1,-1))
            inputLayer = np.append(inputLayer, figures, axis=1)
            q = self.modelDecide.predict(inputLayer)
            action = self.randmax(q[0])
        
        oldState = self.game.field.values.reshape((1,-1))
        figures = np.asarray([self.game.Figure.typ, self.game.nextFigure.typ, cTyp]).reshape((1,-1))
        oldState = np.append(oldState, figures, axis=1)

        if action == 0:
            self.game.left()
        elif action == 1:
            self.game.right()
        elif action == 2:
            self.game.down()
        elif action == 3:
            self.game.rotate()
        elif action == 4:
            self.game.change()

        reward = self.getReward()   
        if self.game.changeFigure == None:
            cTyp = 10
        else:
            cTyp = self.game.changeFigure.typ
        state = self.game.field.values.reshape((1,-1))
        figures = np.asarray([self.game.Figure.typ, self.game.nextFigure.typ, cTyp]).reshape((1,-1))
        state = np.append(state, figures, axis=1)

        if self.game.state == GAME_OVER:
            self.exp.remember(oldState, action, reward, state, True)
        else:
            self.exp.remember(oldState, action, reward, state, False)

        inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
        self.loss += self.modelLearn.train_on_batch(inputs, outputs)