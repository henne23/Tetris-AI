"""
Created on Sun Feb 28 16:48:49 2021

@author: hendr
"""
import random

colors = [
        (0,0,0),
        (0,240,240),    # I 
        (0,0,240),      # J
        (240,160,0),    # L
        (240,240,0),    # O 
        (0,240,0),      # S
        (160,0,240),    # T
        (240,0,0)       # Z
    ]

class Figure:
    x = 0
    y = 0
    
    f = [
        [[1,5,9,13], [4,5,6,7]],                            # I
        [[1,2,5,9], [0,4,5,6], [1,5,9,8], [4,5,6,10]],      # J
        [[1,2,6,10], [5,6,7,9], [2,6,10,11], [3,5,6,7]],    # L
        [[1,2,5,6]],                                        # O
        [[6,7,9,10], [1,5,6,10]],                           # S
        [[1,4,5,6], [1,4,5,9], [4,5,6,9], [1,5,6,9]],       # T
        [[4,5,9,10], [2,6,5,9]]                             # Z
    ]
    
    
    def __init__(self, xC, yC):
        self.x = xC
        self.y = yC
        # Type nicht komplett zufällig, sondern sieben Figuren auf einmal
        self.type = random.randint(0,len(self.f)-1)
        self.color = colors[self.type+1]
        self.rotation = 0
    
    def image(self):
        return self.f[self.type][self.rotation]
    
    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.f[self.type])