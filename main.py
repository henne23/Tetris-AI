from components.Tetris import Tetris
import pygame
import time

'''
Author: Hendrik Pieres

'''
done = False
graphics = False
manual = False
train = True
darkmode = True
verbose = True
batchSize = 50

height = 20
width = 10
game = Tetris(height, width, graphics, manual, train, batchSize, darkmode)

while not done:
    q = False
    game.init()
    if manual and not game.early:
        while not q:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        q = True
                    if event.key == pygame.K_RETURN:
                        q = True
    if game.early:
        done = True
game.field.gameOver()