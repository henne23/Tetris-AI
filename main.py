import pygame
import time

from components.Tetris import Tetris
from constants.Colors import brick_colors, primary_colors
from constants.GameStates import START, GAME_OVER

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''


pygame.init()

screen = pygame.display.set_mode((580, 670))
pygame.display.set_caption("Tetris-AI")

done = False
clock = pygame.time.Clock()
counter = 0
zoom = 30
xPosGame = 130
xNextFigure = 30
xChangeFigure = 490

game = Tetris(20, 10)
#fps = 1.9 + game.level/10
fps = 30
pressing_down = False
pressing_left = False
pressing_right = False

def drawFigure(fig, x, y, zoom, gameX = 0, gameY = 0):
    if fig is not None:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in fig.image():
                    pygame.draw.rect(
                        screen,
                        fig.color,
                        [
                            x + (j + gameX) * zoom,
                            y + (i + gameY) * zoom,
                            zoom,
                            zoom,
                        ],
                    )

lastFrame = time.time()
update = 0.0
while not done:
    acc = game.level * 0.025
    if game.state == START and update > 1.025 - acc:
        game.go_down()
        update = 0

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

    screen.fill(color=primary_colors["WHITE"])
    for i in range(game.height):
        for j in range(game.width):
            if game.field[i][j] == 0:
                color = primary_colors["GRAY"]
                just_border = 1
            else:
                color = brick_colors[game.field[i][j]]
                just_border = 0
            pygame.draw.rect(
                screen,
                color,
                [xPosGame + j * zoom, 30 + i * zoom, zoom, zoom],
                just_border,
            )
    pygame.draw.rect(screen, primary_colors["GRAY"], [xNextFigure, 60, int(zoom/2)*4, int(zoom/2)*4], 1)
    pygame.draw.rect(screen, primary_colors["GRAY"], [xChangeFigure, 60, int(zoom/2)*4, int(zoom/2)*4], 1)

    drawFigure(game.Figure, xPosGame, 30, zoom, game.Figure.x, game.Figure.y)
    drawFigure(game.changeFigure, xChangeFigure, 60, int(zoom/2))
    drawFigure(game.nextFigure, xNextFigure, 60, int(zoom/2))
           
    gameover_font = pygame.font.SysFont("Calibri", 65, True, False)
    text_gameover = gameover_font.render("Game Over!\n Press Esc", True, (255, 215, 0))

    if game.state == GAME_OVER:
        screen.blit(text_gameover, [30, 250])

    score_font = pygame.font.SysFont("Calibri", 20, True, False)
    text_score = score_font.render("Score: %d" % (game.score) , True, primary_colors["BLACK"])
    screen.blit(text_score, [460, 200])
    text_score = score_font.render("Level: %d" % (game.level) , True, primary_colors["BLACK"])
    screen.blit(text_score, [460, 240])

    next_font = pygame.font.SysFont("Calibri", 12, True, False)
    text_next = next_font.render("Next Figure:", True, primary_colors["BLACK"])
    screen.blit(text_next, [30,30])

    pygame.display.flip()
    clock.tick(fps)
    duration = time.time() - lastFrame
    lastFrame = time.time()
    update += duration

pygame.quit()