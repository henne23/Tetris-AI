"""
Created on Sun Feb 28 15:44:58 2021

@author: hendr
"""

from GameEnvironment import Tetris
from Figure import Figure
import pygame
import time


pygame.init()
screen = pygame.display.set_mode((400,600))
pygame.display.set_caption("Tetris-AI")

done = False
fps = 2
clock = pygame.time.Clock()
counter = 0
zoom = 25

# Es git kein Event, welches prüft, ob die Taste gedrückt gehalten wird
pressing_down = False
pressing_left = False
pressing_right = False

BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (128,128,128)

game = Tetris(20, 10)

while not done:
    if game.state == "start":
        game.go_down()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                game.rotate()
            if event.key == pygame.K_s:
                pressing_down = True
            if event.key == pygame.K_a:
                pressing_left = True
            if event.key == pygame.K_d:
                pressing_right = True       
        
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_s:
                pressing_down = False
            if event.key == pygame.K_a:
                pressing_left = False
            if event.key == pygame.K_d:
                pressing_right = False
                
        if pressing_down:
            game.down()
        if pressing_left:
            game.left()
        if pressing_right:
            game.right() 

#GUI                
    screen.fill(color=WHITE)
    for i in range(game.height):
        for j in range(game.width):
            if game.field[i][j] == 0:
                color = GRAY
                just_border = 1
            else:
                color = game.figure.color
                just_border = 0
            pygame.draw.rect(screen, color, [j*zoom, i*zoom, zoom, zoom], just_border)
    
    if game.figure is not None:
        f = game.figure
        for i in range(4):
            for j in range(4):
                p = i*4 + j
                if p in f.image():
                    pygame.draw.rect(screen, f.color, 
                                     [(j+f.x)*zoom, (i+f.y)*zoom, zoom, zoom])
    
    gameover_font = pygame.font.SysFont("Calibri", 65, True, False)
    text_gameover = gameover_font.render("Game Over!\n Press Esc", True, (0,0,0))
    
    if game.state == "gameover":
        screen.blit(text_gameover, [30,250])
    
    score_font = pygame.font.SysFont("Calibri", 25, True, False)
    text_score = gameover_font.render("Score: %d" % game.score, True, (0,0,0))
    
    screen.blit(text_score, [game.width-50,game.height-550])
    
    pygame.display.flip()
    clock.tick(fps)

time.sleep(5)
pygame.quit()