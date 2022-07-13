from components.Figure import Figure
from GUI.Field import Field
from constants.GameStates import START, GAME_OVER
import numpy as np
import pandas as pd
import os 
import shutil
import re
import random
import time
import pygame
import matplotlib.pyplot as plt
from GUI.Settings import Settings

'''
Author: Hendrik Pieres

Basic game engine by TheMorpheus407

'''

class Tetris:
    def __init__(self, height, width, batch_size):
        # Initialize variables for all games
        self.height = height
        self.width = width
        self.graphics = bool
        self.manual = bool
        self.train = bool
        self.darkmode = bool
        # tkinter GUI for the four settings above
        Settings(self)
        self.max_epochs = 3000
        self.max_points = 500000
        self.top = 50
        self.all_scores = []
        self.all_holes = []
        self.total_time = 0.0
        if not self.manual:
            from AI.Training import Training
            self.load_model = False
            if self.train:
                self.model_learn = self.create_model()
            else:
                self.model_learn = self.load_model_func()
            # Implementation of Double Q-Learning
            self.model_decide = self.load_model_func(compil=False)
            self.total_moves = 0
            try:
                path = os.getcwd()
                self.highscore = int(re.sub('model_Tetris.h5', "", os.listdir(path + "\\Save")[0]))
            except:
                self.highscore = 0
            self.training = Training(self, self.model_learn, self.model_decide, batch_size)
            if self.train:
                print("Place %d tetrominos first" % (self.training.max_memory / 10))
            try:
                path = "C:/Users/hpieres/Documents/Git/Tetris-AI/EpochResults/"
                self.epochs = max([int(re.sub(r'\D', "", x)) for x in os.listdir(path) if len(re.sub(r'\D',"",x))>0]) + 1
            except:
                self.epochs = 1

    def init(self):
        # Initialize variables for one game
        self.start_time = time.time()
        self.train_time = 0.0
        self.field = Field(self.height, self.width, self.graphics, self.darkmode, self.manual)
        self.done, self.early, self.pressing_down, self.pressing_left, self.pressing_right, self.switch = False, False, False, False, False, False
        self.level = 1
        self.points = [40, 100, 300, 1200]
        self.killed_lines, self.score, self.figure_counter, self.pieces = 0, 0, 0, 0
        self.state = START
        self.figure_set = random.sample(range(7),7)
        self.change_figure = None
        self.current_figure, self.next_figure = self.create_new_figure()
        self.start()

    def plot_results(self):
        np.savetxt("Scores.txt", np.array(self.all_scores))
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(18,8))
        res = [max(self.all_scores[i:i+self.top]) for i in range(len(self.all_scores)-self.top)]
        #window_size = 200
        #windows = pd.Series(self.all_scores).rolling(window_size)
        #res = windows.mean().dropna().tolist()
        plt.plot(res)
        plt.title("Scores (MA)")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        #plt.xticks(range(0,len(self.all_scores)-window_size,self.top))
        plt.xticks(range(0, len(self.all_scores)-self.top, self.top))
        plt.savefig("Score by epoch.png")
        plt.show()

    def control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                self.early = True
                return
            if self.state == GAME_OVER:
                self.done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.rotate()
                if event.key == pygame.K_DOWN:
                    self.down()
                if event.key == pygame.K_LEFT:
                    self.pressing_left = True
                if event.key == pygame.K_RIGHT:
                    self.pressing_right = True
                if event.key == pygame.K_q:
                    self.change()
                if event.key == pygame.K_n:
                    self.init()
                if event.key == pygame.K_p:
                    self.stop()
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                    self.early = True
                    return

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.pressing_left = False
                if event.key == pygame.K_RIGHT:
                    self.pressing_right = False

        if self.pressing_down:
            self.down()
        if self.pressing_left:
            self.left()
        if self.pressing_right:
            self.right()
    
    def create_new_figure(self):
        # Creates a new figure and checks for direct intersection
        typ = self.figure_set[self.figure_counter]
        new_figure = Figure(3, 0, typ, self.width)
        y = self.would_down(fig=new_figure, x=new_figure.x)
        border = 0 if new_figure.typ == 0 else -1
        if y < border:
            self.state = GAME_OVER
            self.done = True
            return new_figure, None
        current_figure = new_figure
        # Creates a new figure set with seven figures (random order)
        if self.figure_counter == len(self.figure_set)-1:
            self.figure_set = random.sample(range(len(self.figure_set)),len(self.figure_set))
            self.figure_counter = 0
        else:
            self.figure_counter += 1
        typ = self.figure_set[self.figure_counter]
        next_figure = Figure(3,0,typ,self.width)
        self.switch = False
        return current_figure, next_figure

    def go_down(self):
        self.current_figure.y += 1
        if self.intersects():
            self.current_figure.y -= 1
            self.freeze()
            
    def side(self, dx):
        old_x = self.current_figure.x
        edge = False
        fig = self.current_figure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    if (
                        j + self.current_figure.x + dx > self.width - 1  # beyond right border
                        or j + self.current_figure.x + dx < 0  # beyond left border
                    ):
                        edge = True
        if not edge:
            self.current_figure.x += dx
        if self.intersects():
            self.current_figure.x = old_x

    def change(self):
        # Function checks whether the figure was changed within this move
        # Otherwise do the switch
        if not self.switch:
            if self.change_figure == None:
                self.change_figure = self.current_figure
                self.current_figure, self.next_figure = self.create_new_figure()
            else:
                figure_switch = self.current_figure
                self.current_figure = self.change_figure
                self.current_figure.y = 0
                self.current_figure.x = 3
                self.change_figure = figure_switch
            self.switch = True
                
    def left(self):
        self.side(-1)
        if self.manual:
            time.sleep(0.15)

    def right(self):
        self.side(1)
        if self.manual:
            time.sleep(0.15)

    def down(self):
        while not self.intersects():
            self.current_figure.y += 1
        self.current_figure.y -= 1
        self.freeze()

    def would_down(self, fig=None, x=None, img=None, y=None):
        # Function is used several times (GUI update, AI learning, etc.)
        if fig is None:
            fig = self.current_figure
        if x is None:
            x = self.current_figure.x
        if img is None:
            img = fig.image()
        end = min(x+4,10)
        start = max(x,0)
        b = (self.field.values!=0).argmax(axis=0)
        start_check = min(b[start:end])-4 if y is None else y
        for i in range(start_check, self.height):
            for j in range(3,-1,-1):
                for k in range(3,-1,-1):
                    if img[j][k]:
                        if (j + i > self.height - 1 or self.field.values[j + i][k + x] > 0) and (j+i) >= 0:
                            return i - 1
        
    def rotate(self):
        old_rotation = self.current_figure.rotation
        self.current_figure.rotate()
        if self.intersects():
            self.current_figure.rotation = old_rotation

    def intersects(self, fig=None):
        # Checks for intersection after every move (downwards or sidewards)
        fig = self.current_figure if (fig is None) else fig
        img = fig.image()
        intersection = False
        for i in range(4):
            for j in range(4):
                if img[i][j]:
                    if (
                        i + fig.y > self.height - 1  # bottom intersection
                        or self.field.values[i + fig.y][j + fig.x] > 0  # figure intersection
                    ):
                        return True
        return intersection

    def freeze(self):
        # Updates the field values after intersection
        fig = self.current_figure.image()
        for i in range(4):
            for j in range(4):
                if fig[i][j]:
                    self.field.values[i + self.current_figure.y][j + self.current_figure.x] = 1
                    self.field.colors[i + self.current_figure.y][j + self.current_figure.x] = (self.current_figure.typ + 1)
        self.break_lines()
        self.current_figure, self.next_figure = self.create_new_figure()

    def break_lines(self, field=None):
        score_update = False
        # function is called by training.getStateValue() with a field copy to evaluate every possible move
        if field is None:
            # Call by Reference
            field = self.field.values
            field_col = self.field.colors
            score_update = True
        lines = np.sum(field, axis=1)
        killed_lines = np.sum([x >= self.width for x in lines])
        if killed_lines > 0:
            for index, val in enumerate(lines):
                if val > 9:
                    for i in range(index, 0, -1):
                        for j in range(self.width):
                            field[i][j] = field[i - 1][j]
                            if score_update:
                                field_col[i][j] = field_col[i - 1][j]
            # score_update checks whether this function is used during a real game or AI learning
            if score_update:
                self.score += self.points[killed_lines - 1] * self.level
                self.killed_lines += killed_lines
                self.level = int(self.killed_lines/10) + 1
        if not score_update:
            return field, killed_lines

    def stop(self):
        q = False
        while not q:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        q = True

    def start(self):
        update = 0.0
        while not self.done:
            if self.manual:
                last_frame = time.time()
                # Speed increase during manual play
                acc = 0.9**self.level
                if self.state == START and update > acc:
                    self.go_down()
                    update = 0.0
                self.control()
            elif self.training is not None:
                '''
                if self.score > self.max_points:
                    self.done = True
                else:
                '''
                self.training.train()
                self.pieces += 1
                self.total_moves += 1

            if self.graphics:
                self.field.update(self)
            if self.manual:
                duration = time.time() - last_frame
                last_frame = time.time()
                update += duration
        if not self.manual:
            if self.train and (self.load_model or self.training.exp.current_index > self.training.max_memory / 10):
                # Statistics
                game_time = time.time() - self.start_time
                self.total_time += game_time
                holes = self.training.get_holes(self.field.values)
                self.all_holes.append(holes)
                print("Epoch: %5d\tLevel: %2d\tScore: %7d\tPieces: %5d\tLines: %d\tTotal Moves: %d\tTime/Total Time: %.2f / %.2f" % (self.epochs, self.level, self.score, self.pieces, self.killed_lines, self.total_moves, game_time, self.total_time))
                self.epochs += 1
                self.all_scores.append(self.score)
                self.training.loss = 0.0
                if self.score > self.highscore:
                    self.save_model(self.modelLearn)
                    if self.highscore > 0:
                        os.remove("Save/%dmodel_Tetris.h5" % self.highscore)
                    self.highscore = self.score
                    shutil.copy("model_Tetris.h5", "Save/%dmodel_Tetris.h5" % self.highscore)
                if self.epochs > self.max_epochs:
                    self.plot_results()
                    self.save_model(self.model_learn)
                    self.early = True
            else:
                print("Level: %3d\tScore: %7d\tPieces: %5d\tLines: %d" % (self.level, self.score, self.pieces, self.killed_lines))
                    
    def save_model(self, model, first=False):
        # serialize model to JSON
        model_json = model.to_json()
        if first:
            with open("model_Tetris.json", "w") as json_file:
                json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_Tetris.h5")

    def load_model_func(self, compil=True):
        from keras.models import model_from_json
        try:   
            json_file = open('model_Tetris.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            if compil:
                model.compile(optimizer="adam", loss="mse")
        except:
            print("No model found")
            model = None
        try:
            model.load_weights("model_Tetris.h5")
        except:
            print("No weights found")
        return model

    def create_model(self, hidden_size=64, compil = True):
        from keras.models import Sequential
        from keras.layers import Dense
        model = None
        if self.load_model:
            model = self.load_model_func(compil)
        if model == None:
            model = Sequential()
            model.add(Dense(hidden_size, input_shape=(4,), activation="relu"))
            model.add(Dense(hidden_size, activation="relu"))
            model.add(Dense(1))
            model.compile(optimizer = "adam", loss="mse")
            self.save_model(model, first=True)
        return model