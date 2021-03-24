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
        self.updateModel = 500
        self.targetLine = game.height - 1
        self.batchSize = batchSize
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1])

    def randmax(self, values):
        # kann ggf. gelöscht werden, damit auch keine Code-Fragmente kopiert werden müssen
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

    def getReward(self, oldKilledLines):
        val = np.copy(self.game.field.values)
        newTargetLine = self.targetLine
        dropX, dropY = self.game.wouldDown()
        fig = self.game.Figure.image()
        if self.game.state == GAME_OVER:
            return -1
        if self.game.killedLines > oldKilledLines:
            return self.game.killedLines - oldKilledLines
        # i = Zeile, j = Spalte
        for i in range(4):
                for j in range(4):
                    # Die Prüfung kann hier stattfinden, da bereits in der Spielumgebung geprüft wird, das Werte nicht out of bounds laufen
                    if i + dropY < 20 and j + dropX < 10:
                        val[i+dropY][j+dropX] += fig[i][j]
        for i in range(self.targetLine, 0, -1):
            if np.sum(val[i]) == 0:
                break
            diff = self.game.field.values[i] - self.game.field.values[i-1]
            if min(diff) < 0:
                self.targetLine -= 1
            diff = val[i] - val[i-1]
            if min(diff) < 0:
                newTargetLine -= 1
        if newTargetLine < self.targetLine:
            return -0.5
        
        # Namensgebung überarbeiten -> Schauen, ob direkte Höhenbetrachtung oder Höhenveränderung besser ist
        factor = .7
        breite = np.sum(val[self.targetLine])/self.game.width
        hoehe = np.sum(val, axis=1).argmax()/(self.game.height-1)
        reward = hoehe * factor + breite * (1-factor)
        return reward

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
        oldKilledLines = self.game.killedLines

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

        reward = self.getReward(oldKilledLines)   
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
        
        if self.game.moves % self.batchSize == 0:
            inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
            self.loss += self.modelLearn.train_on_batch(inputs, outputs)

        if self.game.moves % self.updateModel == 0:
            self.game.save_model(self.modelLearn)
            self.modelDecide.load_weights("model_Tetris.h5")