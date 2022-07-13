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
            self.x_pos_game = (screen_width-(self.zoom*width)) / 2
            self.x_next_figure = (int(screen_width*.25) - self.zoom*2) / 2
            self.x_change_figure = int(screen_width*.75) + (int(screen_width*.25) - self.zoom*2) / 2
            self.x_score = 460
            if manual:
                self.fps = 30
            else:
                self.fps = 0
            if not darkmode:
                self.field_colors = ["WHITE", "GRAY", "BLACK"]
            else:
                self.field_colors = ["BLACK", "WHITE", "WHITE"]
        self.colors = np.zeros((height, width), dtype=int)
        self.values = np.zeros((height, width), dtype=int)

    def game_over(self):
        pygame.quit()

    def update(self, game):
        self.screen.fill(color=primary_colors[self.field_colors[0]])
        for i in range(game.height):
            for j in range(game.width):
                if self.values[i][j] == 0:
                    color = primary_colors[self.field_colors[1]]
                    just_border = 1
                else:
                    color = brick_colors[self.colors[i][j]]
                    just_border = 0
                pygame.draw.rect(
                    self.screen,
                    color,
                    [self.x_pos_game + j * self.zoom, 30 + i * self.zoom, self.zoom, self.zoom],
                    just_border,
                )
        drop_y = game.would_down(y=game.current_figure.y)
        drop_y = max(drop_y, 0)
        border_thickness = 4
        color = game.current_figure.color
        img = game.current_figure.image()
        for i in range(4):
            for j in range(4):
                if img[i][j]:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        [self.x_pos_game + (j+game.current_figure.x) * self.zoom, 30 + (i+drop_y) * self.zoom, self.zoom, self.zoom],
                        border_thickness,
                    )
                
        pygame.draw.rect(self.screen, primary_colors[self.field_colors[1]], [self.x_next_figure, 60, int(self.zoom/2)*4, int(self.zoom/2)*4], 1)
        pygame.draw.rect(self.screen, primary_colors[self.field_colors[1]], [self.x_change_figure, 60, int(self.zoom/2)*4, int(self.zoom/2)*4], 1)

        self.draw_figure(game.current_figure, self.x_pos_game, 30, self.zoom, game.current_figure.x, game.current_figure.y)
        self.draw_figure(game.change_figure, self.x_change_figure, 60, int(self.zoom/2))
        self.draw_figure(game.next_figure, self.x_next_figure, 60, int(self.zoom/2))

        gameover_font = pygame.font.SysFont("Calibri", 65, True, False)
        text_gameover = gameover_font.render("Game Over!", True, primary_colors[self.field_colors[2]])
        text_2 = gameover_font.render("Press ESC or Enter.", True, primary_colors[self.field_colors[2]])

        if game.state == GAME_OVER and game.manual:
            self.screen.blit(text_gameover, [30, 250])
            self.screen.blit(text_2, [30, 350])

        score_font = pygame.font.SysFont("Calibri", 20, True, False)
        text_score = score_font.render("Score: %d" % (game.score) , True, primary_colors[self.field_colors[2]])
        self.screen.blit(text_score, [self.x_score, 200])
        text_score = score_font.render("Level: %d" % (game.level) , True, primary_colors[self.field_colors[2]])
        self.screen.blit(text_score, [self.x_score, 240])
        text_killed_lines = score_font.render("Lines: %d" % game.killed_lines, True, primary_colors[self.field_colors[2]])
        self.screen.blit(text_killed_lines, [self.x_score, 280])

        next_font = pygame.font.SysFont("Calibri", 12, True, False)
        text_next = next_font.render("Next Figure:", True, primary_colors[self.field_colors[2]])
        self.screen.blit(text_next, [self.x_next_figure,30])

        change_font = pygame.font.SysFont("Calibri", 12, True, False)
        text_change = change_font.render("Change Figure:", True, primary_colors[self.field_colors[2]])
        self.screen.blit(text_change, [self.x_change_figure,30])

        pygame.display.flip()
        self.clock.tick(self.fps)

    def draw_figure(self, fig, x, y, zoom, game_x = 0, game_y = 0):
        if fig is not None:
            img = fig.image()
            for i in range(4):
                for j in range(4):
                    if img[i][j]:
                        pygame.draw.rect(
                            self.screen,
                            fig.color,
                            [
                                x + (j + game_x) * zoom,
                                y + (i + game_y) * zoom,
                                zoom,
                                zoom,
                            ],
                        )