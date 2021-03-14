import random
import numpy as np
from constants.Colors import brick_colors

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Figure:
    x = 0
    y = 0

    """
    Figure-matrix:

    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
    """

    Figures = [
    # Auskommentierte Positionen waren die Ursprünglichen, wurden korrigiert, damit bei Rotation der Rand nicht überschritten wird
    
        [[1, 5, 9, 13], [4, 5, 6, 7]],                                  # I
        #[[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],     # J
        [[0, 4, 5, 6], [1, 2, 5, 9], [4, 5, 6, 10], [2, 6, 9, 10]],     # J
        #[[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],   # L
        [[4, 5, 6, 8],[1, 2, 6, 10], [2, 4, 5, 6], [1, 5, 9, 10]],      # L
        [[1, 2, 5, 6]],                                                 # O
        #[[6, 7, 9, 10], [1, 5, 6, 10]],                                # S
        [[5, 6, 8, 9], [1, 5, 6, 10]],                                  # S
        #[[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],      # T
        [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [2, 5, 6, 10]],      # T
        [[4, 5, 9, 10], [2, 6, 5, 9]],                                  # Z
    ]

    def __init__(self, x_coord, y_coord, typ, width):
        self.x = x_coord
        self.y = y_coord
        self.typ = typ
        self.binar = np.zeros(16)
        self.width = width  
        self.color = brick_colors[self.typ + 1]
        self.rotation = 0

    def image(self):
        return self.Figures[self.typ][self.rotation]

    def rotate(self):
        # Die Bedingungen sollen verhindern, dass die Figur bei Rotation das Spielfeld verlässt
        
        # Hier vielleicht nochmal prüfen, ob man anhand der Rotation zweite und vierte Bedingung
        if self.typ == 0 and self.x > self.width-4:
            self.x -= (self.x % (self.width-4))
        elif self.typ < 3 and self.x < 0:
            self.x += 1
        elif not self.typ == 4 and self.x == self.width-2 and self.rotation % 2 == 1:
            self.x -= 1   
        elif self.typ > 3 and self.x < 0 and self.rotation % 2 == 1:
            self.x += 1
        self.rotation = (self.rotation + 1) % len(self.Figures[self.typ])