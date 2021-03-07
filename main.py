import time
import pygame

from components.Tetris import Tetris
from GUI.Field import Field
from constants.GameStates import START, GAME_OVER

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''


done = False

game = Tetris(20, 10)
field = Field()
pressing_down = False
pressing_left = False
pressing_right = False

update = 0.0
while not done:
    lastFrame = time.time()
    acc = game.level * 0.025
    if game.state == START and update > 1.025 - acc:
        game.go_down()
        update = 0.0

    # Bei Verwendung der Tasten wasd wird die Texteingabe ebenfalls als Event
    # gewertet und deswegen werden die Figuren doppelt verschoben.
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

    field.update(game)
    duration = time.time() - lastFrame
    lastFrame = time.time()
    update += duration

field.gameOver()