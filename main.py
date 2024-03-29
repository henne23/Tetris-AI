from components.Tetris import Tetris
import pygame

'''
Author: Hendrik Pieres

'''
done = False
batch_size = 256

height = 20
width = 10
game = Tetris(height, width, batch_size)

while not done:
    q = False
    game.init()
    if game.manual and not game.early:
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
game.field.game_over()