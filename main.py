import time
import pygame
import numpy as np

from components.Tetris import Tetris
from GUI.Field import Field
from AI.actions import actions
from constants.GameStates import START, GAME_OVER

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''


done = False
graphics = True
manuell = True
training = False
batchSize = 10

height = 20
width = 10

game = Tetris(height, width)
if graphics:
    field = Field()
if not manuell:
    from AI.Model import createModel, loadModel
    from AI.Experience import Experience
    from AI.Training import Training
    modelLearn = createModel(height, width, loadModel = False)
    modelDecide = loadModel(compil=False)
    training = Training(game, modelLearn, modelDecide, batchSize)

def test():
    pass

def updateScreen():
    field.update(game)

def steuerung():
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
            done = True
        if event.type == pygame.KEYDOWN and not game.state == GAME_OVER:
            if event.key == pygame.K_UP:
            #if event.key == pygame.K_w:
                game.rotate()
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
                game.change()
            if event.key == pygame.K_ESCAPE:
                done = True

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
            game.down()
        if pressing_left:
            game.left()
        if pressing_right:
            game.right()
 

update = 0.0
while not done:
    lastFrame = time.time()
    acc = game.level * 0.025
    if game.state == START and update > 1.025 - acc:
        game.go_down()
        update = 0.0
    if manuell:
        steuerung()
    elif training:
        training.train()
    else:
        test()

    if graphics:
        updateScreen()
    duration = time.time() - lastFrame
    lastFrame = time.time()
    update += duration

if graphics:
    field.gameOver()