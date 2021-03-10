from components.Figure import Figure
from GUI.Field import Field
from constants.GameStates import START, GAME_OVER
import numpy as np
import random
import time
import pygame


class Tetris:
    def __init__(self, height, width, graphics, manuell, train, batchSize):
        self.height = height
        self.width = width
        self.graphics = graphics
        self.manuell = manuell
        self.train = train
        if not manuell:
            from AI.Model import createModel, loadModel
            from AI.Experience import Experience
            from AI.Training import Training
            self.modelLearn = createModel(height, width, loadModel = False)
            self.modelDecide = loadModel(compil=False)
            self.training = Training(game, modelLearn, modelDecide, batchSize)
        self.init()

    def init(self):
        self.field = Field(self.height, self.width, self.manuell)
        self.done = False
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
        self.spielen()

    def steuerung(self):
        '''
        keys = pygame.key.get_pressed()
        print(np.sum(keys))

        if keys[pygame.K_LEFT]:
            game.left()
        if keys[pygame.K_RIGHT]:
            game.right()
        '''
        pressing_down = False
        pressing_left = False
        pressing_right = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if self.state == GAME_OVER:
                self.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                #if event.key == pygame.K_w:
                    self.rotate()
                if event.key == pygame.K_DOWN:
                #if event.key == pygame.K_s:
                    pressing_down = True
                if event.key == pygame.K_LEFT:
                #if event.key == pygame.K_a:
                    pressing_left = True
                if event.key == pygame.K_RIGHT:
                #if event.key == pygame.K_d:
                    pressing_right = True
                if event.key == pygame.K_q:
                    self.change()
                if event.key == pygame.K_ESCAPE:
                    self.done = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                #if event.key == pygame.K_s:
                    pressing_down = False
                if event.key == pygame.K_LEFT:
                #if event.key == pygame.K_a:
                    pressing_left = False
                if event.key == pygame.K_RIGHT:
                #if event.key == pygame.K_d:
                    pressing_right = False

            if pressing_down:
                self.down()
            if pressing_left:
                self.left()
            if pressing_right:
                self.right()

    def new_figure(self):
        typ = self.figureSet[self.figureCounter]
        next_figure = Figure(3, 0, typ, self.width)
        if self.intersects(next_figure):
            self.state = GAME_OVER
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

    def right(self):
        self.side(1)

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
            if zeros == 0:
                lines += 1
                for i2 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field.values[i2][j] = self.field.values[i2 - 1][j]
        if lines > 0:
            self.score += self.punkte[lines-1] * self.level
            self.killedLines += lines
            self.level = int(self.killedLines/10)+1 

    def spielen(self):
        update = 0.0
        while not self.done:
            lastFrame = time.time()
            acc = self.level * 0.025
            if self.state == START and update > 1.025 - acc:
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
        self.restart()

    def restart(self):
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