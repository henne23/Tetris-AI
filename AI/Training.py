from tabnanny import verbose
import numpy as np
import pygame
import time

from AI.Experience import Experience
from constants.GameStates import GAME_OVER, START

'''
Author: Hendrik Pieres

'''

class Training:
    def __init__(self, game, modelLearn, modelDecide, batchSize):
        self.game = game
        self.modelLearn = modelLearn
        self.modelDecide = modelDecide
        self.final_epsilon = .001
        self.initial_epsilon = 1
        self.num_epochs = 5000
        self.loss = .0
        self.state = np.zeros(4, dtype=int)
        self.updateModel = 50
        self.batchSize = batchSize
        maxMemory = batchSize * int(10000/batchSize)
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1], maxMemory=maxMemory)

    def getReward(self, nextState):
        if self.game.done:
            return -1
        else:
            return 1 + nextState[0]**2 * self.game.width
        
    def getStateValue(self, field):
        fieldCopy, killedLines = self.game.break_lines(np.copy(field))
        height = np.sum(np.sum(fieldCopy, axis=1))
        # To calculate the holes and bumpiness, you need to determine the first position with a figure for each column
        b = (fieldCopy!=0).argmax(axis=0)
        # For each column, starting from the first figure, it is checked whether zero values occur. All columns add up to the resulting holes.
        holes = np.sum([fieldCopy[x][i] < 1 for i in range(0, self.game.width) for x in range(b[i], self.game.height) if b[i] > 0])
        b = np.where(b>0,self.game.height-b,0)
        bumpiness = np.sum(np.abs(b[:-1]-b[1:]))
        return np.array([killedLines, holes, height, bumpiness])

    def getNextPosSteps(self, fig=None):
        states = {}
        if fig is None:
            fig = self.game.currentFigure
        numRot = len(fig.Figures[fig.typ])
        for r in range(numRot):
            length, start = fig.length(fig.typ, r)
            maxX = self.game.width - length + 1
            img = fig.image(fig.typ, r)
            height, emptyCols = fig.height(img)
            for x in range(start, maxX+start):
                dropX, dropY = self.game.wouldDown(x=x, img=img, height=height, emptyCols=emptyCols, start=start)
                # otherwise no valid move
                if dropY >= 0:
                    field = np.copy(self.game.field.values)
                    # Diese For-Schleife konnte noch nicht aufgelöst werden
                    for i in range(4):
                        for j in range(4):
                            if i + dropY < self.game.height and j + dropX < self.game.width:
                                field[i+dropY][j+dropX] += img[i][j]
                    #field[dropY:dropY+4, dropX:dropX+4] += imgBottom
                    states[(x, r)] = self.getStateValue(field)
        return states

    def train(self):
        # Enable to exit the game during test session
        if self.game.graphics:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.done = True
                    self.game.early = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game.done = True
                        self.game.early = True
                        return          
        hold_fig = self.game.nextFigure if self.game.changeFigure is None else self.game.changeFigure
        nextPosSteps = self.getNextPosSteps()
        nextActions, nextSteps = zip(*nextPosSteps.items())
        nextSteps = np.asarray(nextSteps)
        nextPosStepsHold = self.getNextPosSteps(hold_fig)
        if nextPosStepsHold:
            nextActionsHold, nextStepsHold = zip(*nextPosStepsHold.items())
            nextStepsHold = np.asarray(nextStepsHold)
        # essential -> high probability for random actions at the beginning -> decreasing during the learning period
        epsilon = self.final_epsilon + (max(self.num_epochs-self.game.totalMoves,0)*(self.initial_epsilon-self.final_epsilon)/self.num_epochs) if not self.game.loadModel else 0.001
        if np.random.rand() <= epsilon and self.game.train:
            index = np.random.randint(0,len(nextPosSteps)-1)
        else:
            q = self.modelLearn.predict(nextSteps, verbose=False)
            if nextPosStepsHold:
                q_hold = self.modelLearn.predict(nextStepsHold, verbose=False)
                if max(q) > max(q_hold):
                    index = np.argmax(q)
                else:
                    index = np.argmax(q_hold)
                    self.game.change()
                    nextActions = nextActionsHold
                    nextSteps = nextStepsHold
            else:
                index = np.argmax(q)
            
        
        x, r = nextActions[index]
        nextState = nextSteps[index]
        self.game.currentFigure.x = x
        self.game.currentFigure.rotation = r
        self.game.down()

        if self.game.train:
            reward = self.getReward(nextState)
            if self.game.state == GAME_OVER:
                self.exp.remember(self.state, reward, nextState, True)
            else:
                self.exp.remember(self.state, reward, nextState, False)
            self.state = nextState

            if self.game.loadModel or self.game.totalMoves > self.num_epochs:
                start = time.time()
                inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
                self.loss += self.modelLearn.train_on_batch(inputs, outputs)
                self.game.trainTime += (time.time() - start)
                
                if self.game.totalMoves % self.updateModel == 0:
                    self.game.save_model(self.modelLearn)
                    self.modelDecide.load_weights("model_Tetris.h5")
                