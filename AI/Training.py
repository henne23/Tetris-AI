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
        self.epsilon = .05
        self.loss = .0
        self.state = np.zeros(4, dtype=int)
        self.updateModel = 50
        self.batchSize = batchSize
        maxMemory = batchSize * int(20000/batchSize)
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1], maxMemory=maxMemory)

    def getReward(self, nextState):

        # Aktuelle Problematik: Es wäre wünschenswert alle vier Kriterien in den Reward des States einfließen zu lassen. Jedoch
        # ist es schwierig, die Werte sauber zu normieren, damit bspw. ein weiteres Loch nicht weniger gewichtet wird, als eine
        # zunehmende Höhe.
        '''
        # Check for killed Lines
        if nextState[0]:
            return nextState[0]
        # Check for game ending
        if self.game.done:
            return -1
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
        if self.game.done:
            return -1
        else:
            return 1 + nextState[0]**2 * self.game.width
        '''
        killedLines = nextState[0]
        height = nextState[2]
        holes = nextState[1]
        bumpiness = nextState[3]
        return -0.51*height + 0.76*killedLines - 0.36*holes - 0.18*bumpiness
        
        if self.game.done:
            return -1
        elif killedLines:
            return killedLines
        else:
            # Formula from paper "Playing Tetris with Deep Reinforcement Learning"
            return -0.51*height + 0.76*killedLines - 0.36*holes - 0.18*bumpiness
        '''
        
    def getStateValue(self, field):
        fieldCopy, killedLines = self.game.break_lines(np.copy(field))
        #height = self.game.height - np.sum(fieldCopy, axis=1).argmax()
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
            fig = self.game.Figure
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
        nextPosSteps = self.getNextPosSteps()
        hold_fig = self.game.nextFigure if self.game.changeFigure is None else self.game.changeFigure
        nextPosStepsHold = self.getNextPosSteps(hold_fig)
        nextActionsHold, nextStepsHold = zip(*nextPosStepsHold.items())
        nextStepsHold = np.asarray(nextStepsHold)
        nextActions, nextSteps = zip(*nextPosSteps.items())
        nextSteps = np.asarray(nextSteps)
        if (np.random.rand() <= self.epsilon or (self.game.totalMoves < self.exp.maxMemory/4 and not self.game.loadModel)) and self.game.train:
            try:
                # Sometimes an error occured that could not be explained by debugging
                index = np.random.randint(0,len(nextPosSteps)-1)
            except:
                print("Something went wrong")
        else:
            q = self.modelLearn.predict(nextSteps, verbose=False)
            q_hold = self.modelLearn.predict(nextStepsHold)
            if max(q) > max(q_hold):
                index = np.argmax(q)
            else:
                index = np.argmax(q_hold)
                self.game.change()
                nextActions = nextActionsHold
                nextSteps = nextStepsHold
            
        
        x, r = nextActions[index]
        nextState = nextSteps[index]
        self.game.Figure.x = x
        self.game.Figure.rotation = r
        self.game.down()

        if self.game.train:
            reward = self.getReward(nextState)
            if self.game.state == GAME_OVER:
                #self.exp.remember(self.state, np.array([x,r]), reward, nextState, True)
                self.exp.remember(self.state, reward, nextState, True)
            else:
                #self.exp.remember(self.state, np.array([x,r]), reward, nextState, False)
                self.exp.remember(self.state, reward, nextState, False)
            self.state = nextState

            if self.game.loadModel or self.game.totalMoves > self.exp.maxMemory/4:
                start = time.time()
                inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
                self.loss += self.modelLearn.train_on_batch(inputs, outputs)
                self.game.trainTime += (time.time() - start)
                
                if self.game.totalMoves % self.updateModel == 0:
                    self.game.save_model(self.modelLearn)
                    self.modelDecide.load_weights("model_Tetris.h5")
                