from components.Figure import Figure
from GUI.Field import Field
from constants.GameStates import START, GAME_OVER
import numpy as np
import os 
import shutil
import re
import random
import time
import pygame

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Tetris:
    def __init__(self, height, width, graphics, manual, train, batchSize, darkmode):
        self.height = height
        self.width = width
        self.graphics = graphics
        self.manual = manual
        self.train = train
        self.darkmode = darkmode
        if not manual:
            from AI.Training import Training
            if train:
                self.loadModel = True
                self.modelLearn = self.createModel(height, width)
            else:
                self.modelLearn = self.loadModelFunc()
            self.modelDecide = self.loadModelFunc(compil=False)
            self.totalMoves = 0
            try:
                path = os.getcwd()
                self.highscore = int(re.sub('model_Tetris.h5', "", os.listdir(path + "\\Save")[0]))
            except:
                self.highscore = 0
            self.training = Training(self, self.modelLearn, self.modelDecide, batchSize)
            if train and not self.loadModel:
                print("Place %d pieces first" % (self.training.exp.maxMemory/10))
            try:
                path = "C:/Users/hpieres/Documents/Git/Tetris-AI/EpochResults/"
                self.epochs = max([int(re.sub(r'\D', "", x)) for x in os.listdir(path) if len(re.sub(r'\D',"",x))>0]) + 1
            except:
                self.epochs = 1

    def init(self):
        self.startTime = time.time()
        self.pieces = 0
        self.field = Field(self.height, self.width, self.graphics, self.darkmode, self.manual)
        self.done, self.early, self.pressing_down, self.pressing_left, self.pressing_right, self.switch = False, False, False, False, False, False
        self.level = 1
        self.points = [40, 100, 300, 1200]
        self.killedLines, self.score, self.figureCounter = 0, 0, 0
        self.figureAnz = 7
        self.state = START
        self.figureSet = random.sample(range(self.figureAnz),self.figureAnz)
        #self.figureSet = [0,1,2,3,4,5,6]
        #self.figureSet = [3,4,6,1,2,5,0]
        self.changeFigure = None
        self.nextFigure = self.figureSet[self.figureCounter+1]
        self.new_figure()
        self.start()

    def steuerung(self):
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
    
    def new_figure(self):
        typ = self.figureSet[self.figureCounter]
        #typ = 5
        next_figure = Figure(3, 0, typ, self.width)
        if self.intersects(next_figure):
            self.state = GAME_OVER
            self.done = True
            return
        self.Figure = next_figure
        if self.figureCounter == self.figureAnz-1:
            self.figureSet = random.sample(range(self.figureAnz),self.figureAnz)
            self.figureCounter = 0
            typ = self.figureSet[self.figureCounter]
            self.nextFigure = Figure(3,0,typ,self.width)
        else:
            self.figureCounter += 1
            typ = self.figureSet[self.figureCounter]
            self.nextFigure = Figure(3,0,typ,self.width)
        self.switch = False

    def go_down(self):
        self.Figure.y += 1
        if self.intersects():
            self.Figure.y -= 1
            self.freeze()
            
    def side(self, dx):
        old_x = self.Figure.x
        edge = False
        fig = self.Figure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    if (
                        j + self.Figure.x + dx > self.width - 1  # beyond right border
                        or j + self.Figure.x + dx < 0  # beyond left border
                    ):
                        edge = True
        if not edge:
            self.Figure.x += dx
        if self.intersects():
            self.Figure.x = old_x

    def change(self):
        if not self.switch:
            if self.changeFigure == None:
                self.changeFigure = self.Figure
                self.new_figure()
            else:
                figureSwitch = self.Figure
                self.Figure = self.changeFigure
                self.Figure.y = 0
                self.Figure.x = 3
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
            self.Figure.y += 1
        self.Figure.y -= 1
        self.freeze()

    def wouldDown(self, fig=None, x=None, img=None, height = None, emptyCols = None):
        if fig is None:
            fig = self.Figure
        if x is None:
            x = self.Figure.x
        if img is None:
            img = fig.image()
            height, emptyCols = fig.height(img)
        # Problem: Es folgt eine leere Liste, wenn x = -1
        field = np.copy(self.field.values)[:, x:x+4]
        b = (field!=0).argmax(axis=0)
        b = [x if x > 0 else 20 for x in b]
        row = min(b) - height - emptyCols
        #return x, row
        
        
        for i in range(fig.y,self.height):
            for j in range(4):
                for k in range(4):
                    if img[j][k]:
                        if(
                            j + i > self.height - 1 or
                            self.field.values[j + i][k + x] > 0
                        ):
                            return x, i - 1
        
    def rotate(self):
        old_rotation = self.Figure.rotation
        self.Figure.rotate()
        if self.intersects():
            self.Figure.rotation = old_rotation

    def intersects(self, fig=None):
        fig = self.Figure if (fig is None) else fig
        img = fig.image()
        intersection = False
        for i in range(4):
            for j in range(4):
                if img[i][j]:
                    if (
                        i + fig.y > self.height - 1  # bottom intersection
                        # or i + fig.y < 0  #
                        or self.field.values[i + fig.y][j + fig.x] > 0  # figure intersection
                    ):
                        return True
        return intersection

    def freeze(self):
        fig = self.Figure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    self.field.values[i + self.Figure.y][j + self.Figure.x] = 1
                    self.field.colors[i + self.Figure.y][j + self.Figure.x] = (self.Figure.typ + 1)
        self.break_lines()
        self.new_figure()

    def break_lines(self, field=None):

        # Quelle für Punktevergabe: https://www.onlinespiele-sammlung.de/tetris/about-tetris.php#gewinn
        
        scoreUpdate = False
        if field is None:
            field = self.field.values
            fieldCol = self.field.colors
            scoreUpdate = True
        lines = np.sum(field, axis=1)
        killedLines = np.sum([x >= self.width for x in lines])
        if killedLines > 0:
            for index, val in enumerate(lines):
                if val > 9:
                    for i in range(index, 1, -1):
                        for j in range(self.width):
                            field[i][j] = field[i - 1][j]
                            if scoreUpdate:
                                fieldCol[i][j] = fieldCol[i - 1][j]
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
                acc = 0.9**self.level
                if self.state == START and update > acc:
                    self.go_down()
                    update = 0.0
                self.steuerung()
            elif self.training is not None:
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
            if self.loadModel or self.totalMoves > self.training.exp.maxMemory / 10:
                gameTime = time.time() - self.startTime
                print("Epoch: %5d\t\tLevel: %2d\tScore: %7d\tPieces: %5d\tTime: %.2f" % (self.epochs, self.level, self.score, self.pieces, gameTime))
                self.epochs += 1
                self.training.loss = 0.0
                if self.score > self.highscore:
                    self.save_model(self.modelLearn)
                    os.remove("Save/%dmodel_Tetris.h5" % self.highscore)
                    self.highscore = self.score
                    shutil.copy("model_Tetris.h5", "Save/%dmodel_Tetris.h5" % self.highscore)
                    
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

    def createModel(self, height, width, hidden_size=32, compil = True):
        from keras.models import Sequential
        from keras.layers import Dense
        model = None
        if self.loadModel:
            model = self.loadModelFunc(compil)
        if model == None:
            model = Sequential()
            #Input-Layer -> Alle vorhandenen, unabhängigen Informationen
            #model.add(Dense(hidden_size, input_shape=(height*width + len(inputs),), activation="relu"))
            model.add(Dense(hidden_size, input_shape=(4,), activation="relu"))
            #Hidden-Layer
            model.add(Dense(hidden_size, activation="relu"))
            #Output-Layer -> Beinhaltet die Anzahl der möglichen Aktionen als Anzahl von Neuronen -> Default-Activation: linear
            #model.add(Dense(len(actions)))
            # Ansatz: Größe der Output-Schicht richtet sich nach maximal möglichen States (Rotation*xPos, Bsp. L)
            model.add(Dense(1))
            model.compile(optimizer = "adam", loss="mse")
            self.save_model(model, first=True)
        return model