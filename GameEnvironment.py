"""
Created on Sun Feb 28 16:20:57 2021

@author: hendr
"""
import numpy as np
from Figure import Figure



class Tetris:
    height = 0
    width = 0
    field = np.zeros((height,width))
    score = 0
    state = "start"
    
    def __init__(self, _height, _width):
        self.height = _height
        self.width = _width
        self.field = np.zeros((_height,_width))
        self.score = 0
        self.state = "start"
        self.new_figure()
    
    def new_figure(self):
        self.figure = Figure(3,0)
        
    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()
     
    def side(self, dx):
        old_x = self.figure.x
        edge = False
        for i in range(4):
            for j in range(4):
                p = i*4+j
                if p in self.figure.image():
                    if j+self.figure.x + dx > self.width -1 or \
                        j + self.figure.x + dx < 0:
                            edge = True
        if not edge:
            self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x
     
    def left(self):
        self.side(-1)
    
    def right(self):
        self.side(1)
    
    def down(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()
    
    def rotate(self):
        # Macht z.T. an der Kante noch Probleme
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation
        
    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                p = i*4+j
                if p in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                        i + self.figure.y < 0 or \
                        self.field[i + self.figure.y][j + self.figure.x] > 0:
                            intersection = True
        return intersection
    
    def freeze(self):
        for i in range(4):
            for j in range(4):
                p = i*4+j
                if p in self.figure.image():
                    self.field[i+self.figure.y][j+self.figure.x] = self.figure.type + 1
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"
    
    def break_lines(self):
        lines = 0
        for i in range(1,self.height):
            zeros = 0
            for j in range(1, self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines +=1
                for i2 in range(i,1,-1):
                    for j in range(self.width):
                        self.field[i2][j] = self.field[i2-1][j]
            self.score += lines**2