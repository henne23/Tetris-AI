from components.Tetris import Tetris
import pygame

'''
Author: Hendrik Pieres

'''
done = False
darkmode = True

graphics = False
manual = False
train = True
batchSize = 256

if train:
    height = 20
else:    
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