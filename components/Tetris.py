from components.Figure import Figure
from GUI.Field import Field
from AI.actions import actions
from constants.GameStates import START, GAME_OVER
import numpy as np
import random
import time
import pygame

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Tetris:
    def __init__(self, height, width, graphics, manuell, train, batchSize):
        self.height = height
        self.width = width
        self.graphics = graphics
        self.manuell = manuell
        self.train = train
        if not manuell:
            from AI.Experience import Experience
            from AI.Training import Training
            self.modelLearn = self.createModel(height, width, loadModel = False)
            self.modelDecide = self.loadModel(compil=False)
            self.training = Training(self, self.modelLearn, self.modelDecide, batchSize)
        self.init()

    def init(self):
        self.field = Field(self.height, self.width, self.graphics)
        self.done = False
        self.early = False
        self.pressing_down = False
        self.pressing_left = False
        self.pressing_right = False
        self.level = 1
        self.punkte = [40, 100, 300, 1200]
        self.killedLines = 0
        self.figureAnz = 7
        self.score = 0
        self.state = START
        self.figureCounter = 0
        self.figureSet = random.sample(range(self.figureAnz),self.figureAnz)
        self.switch = False
        self.changeFigure = None
        self.nextFigure = self.figureSet[self.figureCounter+1]
        self.new_figure()
        self.start()

    def steuerung(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                self.early = True
            if self.state == GAME_OVER:
                self.done = True

            # Hier muss noch ein weiterer Counter eingebaut werden

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                #if event.key == pygame.K_w:
                    self.rotate()
                if event.key == pygame.K_DOWN:
                #if event.key == pygame.K_s:
                    self.down()
                if event.key == pygame.K_LEFT:
                #if event.key == pygame.K_a:
                    self.pressing_left = True
                if event.key == pygame.K_RIGHT:
                #if event.key == pygame.K_d:
                    self.pressing_right = True
                if event.key == pygame.K_q:
                    self.change()
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                    self.early = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                #if event.key == pygame.K_a:
                    self.pressing_left = False
                if event.key == pygame.K_RIGHT:
                #if event.key == pygame.K_d:
                    self.pressing_right = False

        if self.pressing_down:
            self.down()
        if self.pressing_left:
            self.left()
        if self.pressing_right:
            self.right()
    
    def new_figure(self):
        typ = self.figureSet[self.figureCounter]
        #typ = 0
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
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in self.Figure.image():
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
        time.sleep(0.15)

    def right(self):
        self.side(1)
        time.sleep(0.15)

    def down(self):
        while not self.intersects():
            self.Figure.y += 1
        self.Figure.y -= 1
        self.freeze()

    def rotate(self):
        old_rotation = self.Figure.rotation
        self.Figure.rotate()
        if self.intersects():
            self.Figure.rotation = old_rotation

    def intersects(self, fig=None):
        fig = self.Figure if (fig is None) else fig
        img = fig.image()
        x = fig.x
        y = fig.y
        intersection = False
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in fig.image():
                    if (
                        i + fig.y > self.height - 1  # bottom intersection
                        # or i + fig.y < 0  #
                        or self.field.values[i + fig.y][j + fig.x] > 0  # figure intersection
                    ):
                        intersection = True
        return intersection

    def freeze(self):
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in self.Figure.image():
                    self.field.values[i + self.Figure.y][j + self.Figure.x] = (
                        self.Figure.typ + 1
                    )
        self.break_lines()
        self.new_figure()

    def break_lines(self):

        # Quelle für Punktevergabe: https://www.onlinespiele-sammlung.de/tetris/about-tetris.php#gewinn

        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field.values[i][j] == 0:
                    zeros += 1
                    break
            if zeros == 0:
                lines += 1
                for i2 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field.values[i2][j] = self.field.values[i2 - 1][j]
        if lines > 0:
            self.score += self.punkte[lines-1] * self.level
            self.killedLines += lines
            self.level = int(self.killedLines/10)+1 

    def start(self):
        update = 0.0
        while not self.done:
            lastFrame = time.time()
            acc = 0.9**self.level
            if self.state == START and update > acc:
                self.go_down()
                update = 0.0
            if self.manuell:
                self.steuerung()
            elif self.training is not None:
                self.training.train()
            else:
                self.test()

            if self.graphics:
                self.field.update(self)
            duration = time.time() - lastFrame
            lastFrame = time.time()
            update += duration
        self.restart(self.early)

    def restart(self, early):
        if early:
            return
        if not self.manuell:
            if not self.train:
                print("Level: %d\tScore: %d" % (self.level, self.score))
            else:
                self.init()
        else:
            q = False
            newGame = False
            while not q:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            q = True
                            self.field.gameOver()
                        if event.key == pygame.K_RETURN:
                            q = True
                            newGame = True
            if newGame:
                self.init()
    
    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("model_Tetris.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_Tetris.h5")

    def loadModel(self, compil=True):
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

    def createModel(self, height, width, hidden_size=100, loadModel=False, compil = True):
        from keras.models import Sequential
        from keras.layers import Dense
        inputs = ["currentFigure", "nextFigure", "changeFigure"]
        model = None
        if loadModel:
            model = self.loadModel(compil)
        if model == None:
            model = Sequential()
            #Input-Layer -> Alle vorhandenen, unabhängigen Informationen
            model.add(Dense(hidden_size, input_shape=(height*width + len(inputs),), activation="relu"))
            #Hidden-Layer
            model.add(Dense(hidden_size, activation="relu"))
            #Output-Layer -> Beinhaltet die Anzahl der möglichen Aktionen als Anzahl von Neuronen -> Default-Activation: linear
            model.add(Dense(len(actions)))
            model.compile(optimizer = "adam", loss="mse")
            self.save_model(model)
        return model