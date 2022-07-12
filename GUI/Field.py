import pygame
import numpy as np

from constants.Colors import primary_colors, brick_colors
from constants.GameStates import GAME_OVER, START

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Field:
    def __init__(self, height, width, graphics, darkmode, manual):
        if graphics:
            pygame.init()
            screen_width = 600
            screen_height = 670
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Tetris-AI")
            self.clock = pygame.time.Clock()
            self.zoom = 30
            self.xPosGame = (screen_width-(self.zoom*width)) / 2
            self.xNextFigure = (int(screen_width*.25) - self.zoom*2) / 2
            self.xChangeFigure = int(screen_width*.75) + (int(screen_width*.25) - self.zoom*2) / 2
            self.xScore = 460
            if manual:
                self.fps = 30
            else:
                self.fps = 0
            if not darkmode:
                self.fieldColors = ["WHITE", "GRAY", "BLACK"]
            else:
                self.fieldColors = ["BLACK", "WHITE", "WHITE"]
        self.colors = np.zeros((height, width), dtype=int)
        self.values = np.zeros((height, width), dtype=int)

    def gameOver(self):
        pygame.quit()

    def update(self, game):
        self.screen.fill(color=primary_colors[self.fieldColors[0]])
        for i in range(game.height):
            for j in range(game.width):
                if self.values[i][j] == 0:
                    color = primary_colors[self.fieldColors[1]]
                    just_border = 1
                else:
                    color = brick_colors[self.colors[i][j]]
                    just_border = 0
                pygame.draw.rect(
                    self.screen,
                    color,
                    [self.xPosGame + j * self.zoom, 30 + i * self.zoom, self.zoom, self.zoom],
                    just_border,
                )
        dropY = game.wouldDown(y=game.currentFigure.y)
        dropY = max(dropY, 0)
        borderThickness = 4
        color = game.currentFigure.color
        img = game.currentFigure.image()
        for i in range(4):
            for j in range(4):
                if img[i][j]:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        [self.xPosGame + (j+game.currentFigure.x) * self.zoom, 30 + (i+dropY) * self.zoom, self.zoom, self.zoom],
                        borderThickness,
                    )
                
        pygame.draw.rect(self.screen, primary_colors[self.fieldColors[1]], [self.xNextFigure, 60, int(self.zoom/2)*4, int(self.zoom/2)*4], 1)
        pygame.draw.rect(self.screen, primary_colors[self.fieldColors[1]], [self.xChangeFigure, 60, int(self.zoom/2)*4, int(self.zoom/2)*4], 1)

        self.drawFigure(game.currentFigure, self.xPosGame, 30, self.zoom, game.currentFigure.x, game.currentFigure.y)
        self.drawFigure(game.changeFigure, self.xChangeFigure, 60, int(self.zoom/2))
        self.drawFigure(game.nextFigure, self.xNextFigure, 60, int(self.zoom/2))

        gameover_font = pygame.font.SysFont("Calibri", 65, True, False)
        text_gameover = gameover_font.render("Game Over!", True, primary_colors[self.fieldColors[2]])
        text_2 = gameover_font.render("Press ESC or Enter.", True, primary_colors[self.fieldColors[2]])

        if game.state == GAME_OVER and game.manual:
            self.screen.blit(text_gameover, [30, 250])
            self.screen.blit(text_2, [30, 350])

        score_font = pygame.font.SysFont("Calibri", 20, True, False)
        text_score = score_font.render("Score: %d" % (game.score) , True, primary_colors[self.fieldColors[2]])
        self.screen.blit(text_score, [self.xScore, 200])
        text_score = score_font.render("Level: %d" % (game.level) , True, primary_colors[self.fieldColors[2]])
        self.screen.blit(text_score, [self.xScore, 240])
        text_killed_lines = score_font.render("Lines: %d" % game.killedLines, True, primary_colors[self.fieldColors[2]])
        self.screen.blit(text_killed_lines, [self.xScore, 280])

        next_font = pygame.font.SysFont("Calibri", 12, True, False)
        text_next = next_font.render("Next Figure:", True, primary_colors[self.fieldColors[2]])
        self.screen.blit(text_next, [self.xNextFigure,30])

        change_font = pygame.font.SysFont("Calibri", 12, True, False)
        text_change = change_font.render("Change Figure:", True, primary_colors[self.fieldColors[2]])
        self.screen.blit(text_change, [self.xChangeFigure,30])

        pygame.display.flip()
        self.clock.tick(self.fps)

    def drawFigure(self, fig, x, y, zoom, gameX = 0, gameY = 0):
        if fig is not None:
            img = fig.image()
            for i in range(4):
                for j in range(4):
                    if img[i][j]:
                        pygame.draw.rect(
                            self.screen,
                            fig.color,
                            [
                                x + (j + gameX) * zoom,
                                y + (i + gameY) * zoom,
                                zoom,
                                zoom,
                            ],
                        )