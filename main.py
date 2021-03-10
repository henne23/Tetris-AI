import time
import numpy as np

from components.Tetris import Tetris
from GUI.Field import Field
from AI.actions import actions
from constants.GameStates import START, GAME_OVER

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''


done = False
graphics = True
manuell = False
training = True
batchSize = 10

height = 20
width = 10

game = Tetris(height, width, graphics, manuell, training, batchSize)