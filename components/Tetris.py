from components.Figure import Figure
from GUI.Field import Field
from constants.GameStates import START, GAME_OVER
import numpy as np
import pandas as pd
import os 
import shutil
import re
import random
import time
import pygame
import matplotlib.pyplot as plt
from GUI.Settings import Settings

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Tetris:
    def __init__(self, height, width, batchSize):
        # Initialize variables for all games
        self.height = height
        self.width = width
        self.graphics = bool
        self.manual = bool
        self.train = bool
        self.darkmode = bool
        # tkinter GUI for the four settings above
        Settings(self)
        self.max_epochs = 3000
        self.max_points = 500000
        self.top = 100
        self.all_scores = []
        self.all_holes = []
        self.totalTime = 0.0
        if not self.manual:
            from AI.Training import Training
            self.loadModel = False
            if self.train:
                self.modelLearn = self.createModel()
            else:
                self.modelLearn = self.loadModelFunc()
            # Implementation of Double Q-Learning
            self.modelDecide = self.loadModelFunc(compil=False)
            self.totalMoves = 0
            try:
                path = os.getcwd()
                self.highscore = int(re.sub('model_Tetris.h5', "", os.listdir(path + "\\Save")[0]))
            except:
                self.highscore = 0
            self.training = Training(self, self.modelLearn, self.modelDecide, batchSize)
            #if self.train and not self.loadModel:
             #   print("Place %d pieces first" % (self.training.num_epochs))
            try:
                path = "C:/Users/hpieres/Documents/Git/Tetris-AI/EpochResults/"
                self.epochs = max([int(re.sub(r'\D', "", x)) for x in os.listdir(path) if len(re.sub(r'\D',"",x))>0]) + 1
            except:
                self.epochs = 1

    def init(self):
        # Initialize variables for one game
        self.startTime = time.time()
        self.trainTime = 0.0
        self.field = Field(self.height, self.width, self.graphics, self.darkmode, self.manual)
        self.done, self.early, self.pressing_down, self.pressing_left, self.pressing_right, self.switch = False, False, False, False, False, False
        self.level = 1
        self.points = [40, 100, 300, 1200]
        self.killedLines, self.score, self.figureCounter, self.pieces = 0, 0, 0, 0
        self.state = START
        self.figureSet = random.sample(range(7),7)
        self.changeFigure = None
        self.currentFigure, self.nextFigure = self.create_new_figure()
        self.start()

    def plot_results(self):
        np.savetxt("Scores.txt", np.array(self.all_scores))
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(18,8))
        #top = [max(self.all_scores[i:i+self.top]) for i in range(len(self.all_scores)-self.top)]
        window_size = 200
        windows = pd.Series(self.all_scores).rolling(window_size)
        res = windows.mean().dropna().tolist()
        plt.plot(res)
        plt.title("Scores (MA)")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.xticks(range(0,len(self.all_scores)-window_size,self.top))
        plt.savefig("Score by epoch.png")
        plt.show()

    def control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                self.early = True
                return
            if self.state == GAME_OVER:
                self.done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.rotate()
                if event.key == pygame.K_DOWN:
                    self.down()
                if event.key == pygame.K_LEFT:
                    self.pressing_left = True
                if event.key == pygame.K_RIGHT:
                    self.pressing_right = True
                if event.key == pygame.K_q:
                    self.change()
                if event.key == pygame.K_n:
                    self.init()
                if event.key == pygame.K_p:
                    self.stop()
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                    self.early = True
                    return

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.pressing_left = False
                if event.key == pygame.K_RIGHT:
                    self.pressing_right = False

        if self.pressing_down:
            self.down()
        if self.pressing_left:
            self.left()
        if self.pressing_right:
            self.right()
    
    def create_new_figure(self):
        # Creates a new figure and checks for direct intersection
        typ = self.figureSet[self.figureCounter]
        new_figure = Figure(3, 0, typ, self.width)
        y = self.wouldDown(fig=new_figure, x=new_figure.x)
        border = 0 if new_figure.typ == 0 else -1
        if y < border:
            self.state = GAME_OVER
            self.done = True
            return new_figure, None
        currentFigure = new_figure
        # Creates a new figure set with seven figures (random order)
        if self.figureCounter == len(self.figureSet)-1:
            self.figureSet = random.sample(range(len(self.figureSet)),len(self.figureSet))
            self.figureCounter = 0
        else:
            self.figureCounter += 1
        typ = self.figureSet[self.figureCounter]
        nextFigure = Figure(3,0,typ,self.width)
        self.switch = False
        return currentFigure, nextFigure

    def go_down(self):
        self.currentFigure.y += 1
        if self.intersects():
            self.currentFigure.y -= 1
            self.freeze()
            
    def side(self, dx):
        old_x = self.currentFigure.x
        edge = False
        fig = self.currentFigure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    if (
                        j + self.currentFigure.x + dx > self.width - 1  # beyond right border
                        or j + self.currentFigure.x + dx < 0  # beyond left border
                    ):
                        edge = True
        if not edge:
            self.currentFigure.x += dx
        if self.intersects():
            self.currentFigure.x = old_x

    def change(self):
        # Function checks whether the figure was changed within this move
        # Otherwise do the switch
        if not self.switch:
            if self.changeFigure == None:
                self.changeFigure = self.currentFigure
                self.currentFigure, self.nextFigure = self.create_new_figure()
            else:
                figureSwitch = self.currentFigure
                self.currentFigure = self.changeFigure
                self.currentFigure.y = 0
                self.currentFigure.x = 3
                self.changeFigure = figureSwitch
            self.switch = True
                
    def left(self):
        self.side(-1)
        if self.manual:
            time.sleep(0.15)

    def right(self):
        self.side(1)
        if self.manual:
            time.sleep(0.15)

    def down(self):
        while not self.intersects():
            self.currentFigure.y += 1
        self.currentFigure.y -= 1
        self.freeze()

    def wouldDown(self, fig=None, x=None, img=None, y=None):
        # Function is used several times (GUI update, AI learning, etc.)
        if fig is None:
            fig = self.currentFigure
        if x is None:
            x = self.currentFigure.x
        if img is None:
            img = fig.image()
        end = min(x+4,10)
        start = max(x,0)
        b = (self.field.values!=0).argmax(axis=0)
        start_check = min(b[start:end])-4 if y is None else y
        for i in range(start_check, self.height):
            for j in range(3,-1,-1):
                for k in range(3,-1,-1):
                    if img[j][k]:
                        if (j + i > self.height - 1 or self.field.values[j + i][k + x] > 0) and (j+i) >= 0:
                            return i - 1
        
    def rotate(self):
        old_rotation = self.currentFigure.rotation
        self.currentFigure.rotate()
        if self.intersects():
            self.currentFigure.rotation = old_rotation

    def intersects(self, fig=None):
        # Checks for intersection after every move (downwards or sidewards)
        fig = self.currentFigure if (fig is None) else fig
        img = fig.image()
        intersection = False
        for i in range(4):
            for j in range(4):
                if img[i][j]:
                    if (
                        i + fig.y > self.height - 1  # bottom intersection
                        or self.field.values[i + fig.y][j + fig.x] > 0  # figure intersection
                    ):
                        return True
        return intersection

    def freeze(self):
        # Updates the field values after intersection
        fig = self.currentFigure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    self.field.values[i + self.currentFigure.y][j + self.currentFigure.x] = 1
                    self.field.colors[i + self.currentFigure.y][j + self.currentFigure.x] = (self.currentFigure.typ + 1)
        self.break_lines()
        self.currentFigure, self.nextFigure = self.create_new_figure()

    def break_lines(self, field=None):
        scoreUpdate = False
        # function is called by training.getStateValue() with a field copy to evaluate every possible move
        if field is None:
            # Call by Reference
            field = self.field.values
            fieldCol = self.field.colors
            scoreUpdate = True
        lines = np.sum(field, axis=1)
        killedLines = np.sum([x >= self.width for x in lines])
        if killedLines > 0:
            for index, val in enumerate(lines):
                if val > 9:
                    for i in range(index, 0, -1):
                        for j in range(self.width):
                            field[i][j] = field[i - 1][j]
                            if scoreUpdate:
                                fieldCol[i][j] = fieldCol[i - 1][j]
            # scoreUpdate checks whether this function is used during a real game or AI learning
            if scoreUpdate:
                self.score += self.points[killedLines - 1] * self.level
                self.killedLines += killedLines
                self.level = int(self.killedLines/10) + 1
        if not scoreUpdate:
            return field, killedLines

    def stop(self):
        q = False
        while not q:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        q = True

    def start(self):
        update = 0.0
        while not self.done:
            if self.manual:
                lastFrame = time.time()
                # Speed increase during manual play
                acc = 0.9**self.level
                if self.state == START and update > acc:
                    self.go_down()
                    update = 0.0
                self.control()
            elif self.training is not None:
                if self.score > self.max_points:
                    self.done = True
                else:
                    self.training.train()
                    self.pieces += 1
                    self.totalMoves += 1

            if self.graphics:
                self.field.update(self)
            if self.manual:
                duration = time.time() - lastFrame
                lastFrame = time.time()
                update += duration
        if not self.manual:
            if self.train:# and (self.loadModel or self.totalMoves > self.training.num_epochs):
                # Statistics
                gameTime = time.time() - self.startTime
                self.totalTime += gameTime
                holes = self.training.getHoles(self.field.values)
                self.all_holes.append(holes)
                print("Epoch: %5d\tLevel: %2d\tScore: %7d\tPieces: %5d\tLines: %d\tTotal Moves: %d\tTime/Total Time: %.2f / %.2f" % (self.epochs, self.level, self.score, self.pieces, self.killedLines, self.totalMoves, gameTime, self.totalTime))
                self.epochs += 1
                self.all_scores.append(self.score)
                self.training.loss = 0.0
                if self.score > self.highscore:
                    self.save_model(self.modelLearn)
                    if self.highscore > 0:
                        os.remove("Save/%dmodel_Tetris.h5" % self.highscore)
                    self.highscore = self.score
                    shutil.copy("model_Tetris.h5", "Save/%dmodel_Tetris.h5" % self.highscore)
                if self.epochs > self.max_epochs:
                    self.plot_results()
                    self.save_model(self.modelLearn)
                    self.early = True
                    
    def save_model(self, model, first=False):
        # serialize model to JSON
        model_json = model.to_json()
        if first:
            with open("model_Tetris.json", "w") as json_file:
                json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_Tetris.h5")

    def loadModelFunc(self, compil=True):
        from keras.models import model_from_json
        try:   
            json_file = open('model_Tetris.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            if compil:
                model.compile(optimizer="adam", loss="mse")
        except:
            print("No model found")
            model = None
        try:
            model.load_weights("model_Tetris.h5")
        except:
            print("No weights found")
        return model

    def createModel(self, hidden_size=64, compil = True):
        from keras.models import Sequential
        from keras.layers import Dense
        model = None
        if self.loadModel:
            model = self.loadModelFunc(compil)
        if model == None:
            model = Sequential()
            model.add(Dense(hidden_size, input_shape=(4,), activation="relu"))
            model.add(Dense(hidden_size, activation="relu"))
            model.add(Dense(1))
            model.compile(optimizer = "adam", loss="mse")
            self.save_model(model, first=True)
        return model