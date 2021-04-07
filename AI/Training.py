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
        self.state = np.zeros(4, dtype=int)
        self.updateModel = 1
        self.targetLine = game.height - 1
        self.batchSize = batchSize
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1])

    def getReward(self, nextState):

        # Aktuelle Problematik: Es wäre wünschenswert alle vier Kriterien in den Reward des States einfließen zu lassen. Jedoch
        # ist es schwierig, die Werte sauber zu normieren, damit bspw. ein weiteres Loch nicht weniger gewichtet wird, als eine
        # zunehmende Höhe.
        '''
        killedLines = nextState[0]
        holes = nextState[1]/np.sum(self.game.field.values)
        height = nextState[2]/self.game.height
        bumpiness = nextState[3]
        return 1+killedLines**2
        
        # Check for killed Lines
        if nextState[0]:
            return nextState[0]
        # Check for additional holes
        if nextState[1] > self.state[1]:
            return -0.5
        # Check for new height
        if nextState[2] > self.state[2]:
            return -0.3
        # Check for new bumpiness
        if nextState[3] > self.state[3]:
            return -0.1
        # return a positive feedback, if the next state does not lead to a higher height, new holes or a bigger bumpiness
        return 0.1
        
        killedLines = nextState[0]
        holes = 0.5*nextState[1]/np.sum(self.game.field.values)
        height = 0.3*nextState[2]/self.game.height
        bumpiness = 0.2*nextState[3]/171 # 171 is the maximum bumpiness
        if killedLines:
            return killedLines
        else:
            return -(holes+height+bumpiness)
        '''
        return 1 + nextState[0]**2 * self.game.width

    def getRewardOld(self, oldKilledLines):
    
        # Nächster Umstrukturierungsansatz: Nicht jede Aktion bewerten, sondern alle möglichen Endstates (wo kann der nächste Block landen)
        # Frage: Ist das dann noch Q-Learning?
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

    def getStateValue(self, field):
        fieldCopy = np.copy(field)
        lines = np.sum(fieldCopy, axis=1)
        killedLines = np.sum([x >= self.game.width for x in lines])
        if killedLines > 0:
            for index, val in enumerate(lines):
                if val > 9:
                    for i in range(index, 1, -1):
                        for j in range(self.game.width):
                            fieldCopy[i][j] = fieldCopy[i - 1][j]
        height = self.game.height - np.sum(fieldCopy, axis=1).argmax()
        for i in range(len(fieldCopy)-1):
            fieldCopy[i+1] = fieldCopy[i+1] - abs(fieldCopy[i])
        holes = np.sum([fieldCopy[x] < 0 for x in range(len(fieldCopy))])
        b = (field!=0).argmax(axis=0)
        b = np.where(b>0,self.game.height-b,0)
        bumpiness = np.sum(np.abs(b[:-1]-b[1:]))
        return np.array([killedLines, holes, height, bumpiness])

    def getNextPosSteps(self):
        states = {}
        fig = self.game.Figure
        numRot = len(fig.Figures[fig.typ])
        for r in range(numRot):
            length, start = fig.length(fig.typ, r)
            maxX = self.game.width - length + 1
            img = fig.image(fig.typ, r)
            for x in range(start, maxX+start):
                dropX, dropY = self.game.wouldDown(x=x, img=img)
                field = np.copy(self.game.field.values)
                for i in range(4):
                    for j in range(4):
                        if i + dropY < 20 and j + dropX < 10:
                            field[i+dropY][j+dropX] += img[i][j]
                states[(x, r)] = self.getStateValue(field)
        return states

    def train(self):
        nextPosSteps = self.getNextPosSteps()
        nextActions, nextSteps = zip(*nextPosSteps.items())
        nextSteps = np.asarray(nextSteps)
        if (np.random.rand() <= self.epsilon or self.game.totalMoves < self.exp.maxMemory/10) and self.game.train:
            index = np.random.randint(0,len(nextPosSteps)-1)
        else:
            q = self.modelLearn.predict(nextSteps)
            index = np.argmax(q)
        
        x, r = nextActions[index]
        nextState = nextSteps[index]
        self.game.Figure.x = x
        self.game.Figure.rotation = r
        self.game.down()
        if np.sum(self.game.field.values[2]) > 0:
            pass
            #print("Break")

        if self.game.train:
            reward = self.getReward(nextState)
            if self.game.state == GAME_OVER:
                self.exp.remember(self.state, np.array([x,r]), reward, nextState, True)
            else:
                self.exp.remember(self.state, np.array([x,r]), reward, nextState, False)
            
            self.state = nextState

            # Jetzt wird nach 2.000 Zügen jedes Mal mit batch_size Erfahrungswerten gelernt -> so gewollt?
            if self.game.totalMoves > self.exp.maxMemory/10:
                inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
                self.loss += self.modelLearn.train_on_batch(inputs, outputs)

                if self.game.totalMoves % self.updateModel == 0:
                    self.game.save_model(self.modelLearn)
                    self.modelDecide.load_weights("model_Tetris.h5")

    def trainOld(self):
        if self.game.changeFigure == None:
            cTyp = 10
        else:
            cTyp = self.game.changeFigure.typ
        if np.random.rand() <= self.epsilon or self.game.moves < self.exp.maxMemory/10:
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
        
        if self.game.totalMoves % self.batchSize == 0 and self.game.totalMoves > self.exp.maxMemory/10:
            inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
            self.loss += self.modelLearn.train_on_batch(inputs, outputs)

        if self.game.totalMoves % self.updateModel == 0:
            self.game.save_model(self.modelLearn)
            self.modelDecide.load_weights("model_Tetris.h5")