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
        self.final_epsilon = 0
        self.initial_epsilon = 1
        self.num_epochs = 5000
        self.loss = .0
        self.state = np.zeros(4, dtype=int)
        self.updateModel = 50
        self.batchSize = batchSize
        maxMemory = batchSize * int(20000/batchSize)
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1], maxMemory=maxMemory)

    def getReward(self, nextState):
        if self.game.done:
            return -1
        else:
            return 1 + nextState[0]**2 * self.game.width

    def getHoles(self, field):
        b = (field!=0).argmax(axis=0)
        return np.sum([field[x][i] < 1 for i in range(0, self.game.width) for x in range(b[i], self.game.height) if b[i] > 0])

    def getStateValue(self, field):
        fieldCopy, killedLines = self.game.break_lines(np.copy(field))
        b = (fieldCopy!=0).argmax(axis=0)
        column_height = np.where(b>0,self.game.height-b,0)
        height = np.sum(column_height)
        # To calculate the holes and bumpiness, you need to determine the first position with a figure for each column
        # For each column, starting from the first figure, it is checked whether zero values occur. All columns add up to the resulting holes.
        holes = self.getHoles(field=fieldCopy)
        bumpiness = np.sum(np.abs(column_height[:-1]-column_height[1:]))
        return np.array([killedLines, holes, height, bumpiness])

    def getNextPosSteps(self, fig=None):
        # get all possible next steps/states
        states = {}
        if fig is None:
            fig = self.game.currentFigure
        numRot = len(fig.Figures[fig.typ])
        # for every possible rotation (depends on the tetromino)
        for r in range(numRot):
            length, start = fig.length(fig.typ, r)
            maxX = self.game.width - length + 1
            img = fig.image(fig.typ, r)
            _, emptyRows = fig.height(img)
            # for all possible x-positions
            for x in range(start, maxX+start):
                dropY = self.game.wouldDown(x=x, img=img)
                # otherwise no valid move -> emptyRows because not every tetromino starts in the first row
                if dropY >= -emptyRows:
                    field = np.copy(self.game.field.values)
                    for i in range(4):
                        for j in range(4):
                            if i + dropY < self.game.height and j + x < self.game.width:
                                field[i+dropY][j+x] += img[i][j]
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
        # always checks the hold figure too. If this is empty, the next figure is used    
        hold_fig = self.game.nextFigure if self.game.changeFigure is None else self.game.changeFigure
        nextPosSteps = self.getNextPosSteps()
        # it is possible that no next steps are possible with the current or hold figure
        if nextPosSteps:
            nextActions, nextSteps = zip(*nextPosSteps.items())
            nextSteps = np.asarray(nextSteps)
        nextPosStepsHold = self.getNextPosSteps(hold_fig)
        if nextPosStepsHold:
            nextActionsHold, nextStepsHold = zip(*nextPosStepsHold.items())
            nextStepsHold = np.asarray(nextStepsHold)
        # essential -> high probability for random actions at the beginning -> decreasing during the learning period
        # decision for random or best action based on the neural network
        epsilon = self.final_epsilon + (max(self.num_epochs-self.game.totalMoves,0)*(self.initial_epsilon-self.final_epsilon)/self.num_epochs) if not self.game.loadModel else self.final_epsilon
        if np.random.rand() <= epsilon and self.game.train:
            index = np.random.randint(0,len(nextPosSteps)-1) if len(nextPosSteps) > 1 else 0
        else:
            q = self.modelLearn.predict(nextSteps, verbose=False)
            if nextPosStepsHold:
                q_hold = self.modelLearn.predict(nextStepsHold, verbose=False)
                if max(q) >= max(q_hold):
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
        # perform the chosen action
        self.game.currentFigure.x = x
        self.game.currentFigure.rotation = r
        self.game.down()

        if self.game.train:
            # get the reward and save it in memory to learn afterwards
            reward = self.getReward(nextState)
            if self.game.state == GAME_OVER:
                self.exp.remember(self.state, reward, nextState, True)
            else:
                self.exp.remember(self.state, reward, nextState, False)
            self.state = nextState

            # the training starts after 5000 epochs to fill the memory or if the training is based on existing weights
            #if self.game.loadModel or self.game.totalMoves > self.num_epochs:
            start = time.time()
            inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
            self.loss += self.modelLearn.train_on_batch(inputs, outputs)
            self.game.trainTime += (time.time() - start)
            
            if self.game.totalMoves % self.updateModel == 0:
                self.game.save_model(self.modelLearn)
                self.modelDecide.load_weights("model_Tetris.h5")
                