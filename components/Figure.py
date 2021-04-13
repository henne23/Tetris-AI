import random
import numpy as np
from constants.Colors import brick_colors

'''
Author: Hendrik Pieres

'''

class Figure:

    """
    Figure-matrix:

    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
    """

    Figures = [
        [[0, 4, 8, 12], [4, 5, 6, 7]],                                  # I
        [[0, 4, 5, 6], [1, 2, 5, 9], [4, 5, 6, 10], [2, 6, 9, 10]],     # J
        [[4, 5, 6, 8],[1, 2, 6, 10], [2, 4, 5, 6], [1, 5, 9, 10]],      # L
        [[0, 1, 4, 5]],                                                 # O
        [[5, 6, 8, 9], [1, 5, 6, 10]],                                  # S
        [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [2, 5, 6, 10]],      # T
        [[4, 5, 9, 10], [2, 6, 5, 9]],                                  # Z
    ]

    def __init__(self, x_coord, y_coord, typ, width):
        self.x = x_coord
        self.y = y_coord
        self.typ = typ
        self.width = width  
        self.color = brick_colors[self.typ + 1]
        self.rotation = 0
        self.binar = self.image()

    def image(self, typ=None, rotation=None):
        if typ is None:
            typ = self.typ
        if rotation is None:
            rotation = self.rotation
        rects = self.Figures[typ][rotation]
        binar = np.zeros(16, dtype=int)
        for p in rects:
            binar[p] = 1
        return binar.reshape((4,4))

    def length(self, typ, rotation):
        fig = np.sum(self.image(typ, rotation), axis=0)
        # Da die Tetrominos nicht immer am linken Rand starten, wird in der folgenden Zeile die Spalte ermittelt, in der 
        # der Tetromino beginnt, um die For-Schleife zu verschieben
        start = (fig!=0).argmax(axis=0)
        return np.sum([x > 0 for x in fig]), -start

    def rotate(self):
        # Die Bedingungen sollen verhindern, dass die Figur bei Rotation das Spielfeld verlässt
        if self.typ == 0 and self.x > self.width-4:
            self.x -= (self.x % (self.width-4))
        elif self.typ < 3 and self.x < 0:
            self.x += 1
        elif not (self.typ == 4 or self.typ == 0) and self.x == self.width-2 and self.rotation % 2 == 1:
            self.x -= 1 
        elif self.typ == 0 and self.x < 2:
            self.x += (self.x % (self.width-8))  
        elif self.typ > 3 and self.x < 0 and self.rotation % 2 == 1:
            self.x += 1
        self.rotation = (self.rotation + 1) % len(self.Figures[self.typ])