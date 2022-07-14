from tabnanny import verbose
import numpy as np
import pygame
import time

from AI.Experience import Experience
from constants.GameStates import GAME_OVER, START

'''
Author: Hendrik Pieres

'''

class Training:
    def __init__(self, game, model_learn, model_decide, batch_size):
        self.game = game
        self.model_learn = model_learn
        self.model_decide = model_decide
        self.final_epsilon = 0
        self.initial_epsilon = 1
        self.decay_epochs = int(game.max_epochs*.7)
        self.loss = .0
        self.state = np.zeros(4, dtype=int)
        self.update_model = 10
        self.batch_size = batch_size
        self.max_memory = batch_size * int(30000/batch_size)
        self.exp = Experience(self.model_decide.input_shape[-1], self.model_decide.output_shape[-1], max_memory=self.max_memory)

    def get_reward(self, current_state, next_state):
        return 1 + next_state[0]**2 * self.game.width
        '''
        if self.game.done:
            return -1
        else:
            return 1 + next_state[0]**2 * self.game.width
        

        # no real difference compared to the reward function above (pretty fluctuating results)
        reward = 0.0
        # penalize if the game is over
        if self.game.done:
            return -1
        # reward killed lines
        elif next_state[0]:
            return 1 + next_state[0]**2 * self.game.width
        # different penalties if the amount of holes or height/bumpiness is increased compared to the previous state
        if next_state[1] > current_state[1]:
            reward -= 0.4
        if next_state[2] > current_state[2]:
            reward -= 0.2
        if next_state[3] > current_state[3]:
            reward -= 0.1
        return reward
        '''
        
    def get_holes(self, field):
        b = (field!=0).argmax(axis=0)
        return np.sum([field[x][i] < 1 for i in range(0, self.game.width) for x in range(b[i], self.game.height) if b[i] > 0])

    def get_state_value(self, field):
        field_copy, killed_lines = self.game.break_lines(np.copy(field))
        b = (field_copy!=0).argmax(axis=0)
        column_height = np.where(b>0,self.game.height-b,0)
        height = np.sum(column_height)
        # To calculate the holes and bumpiness, you need to determine the first position with a figure for each column
        # For each column, starting from the first figure, it is checked whether zero values occur. All columns add up to the resulting holes.
        holes = self.get_holes(field=field_copy)
        bumpiness = np.sum(np.abs(column_height[:-1]-column_height[1:]))
        return np.array([killed_lines, holes, height, bumpiness])

    def get_next_pos_steps(self, fig=None):
        # get all possible next steps/states
        states = {}
        if fig is None:
            fig = self.game.current_figure
        num_rot = len(fig.Figures[fig.typ])
        # for every possible rotation (depends on the tetromino)
        for r in range(num_rot):
            length, start = fig.length(fig.typ, r)
            max_x = self.game.width - length + 1
            img = fig.image(fig.typ, r)
            _, empty_rows = fig.height(img)
            # for all possible x-positions
            for x in range(start, max_x+start):
                drop_y = self.game.would_down(x=x, img=img)
                # otherwise no valid move -> emptyRows because not every tetromino starts in the first row
                if drop_y >= -empty_rows:
                    field = np.copy(self.game.field.values)
                    for i in range(4):
                        for j in range(4):
                            if i + drop_y < self.game.height and j + x < self.game.width:
                                field[i+drop_y][j+x] += img[i][j]
                    states[(x, r)] = self.get_state_value(field)
        return states

    def train(self):
        # Enable to exit the game during test session
        if self.game.graphics:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.done = True
                    self.game.early = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game.done = True
                        self.game.early = True
                        return      
        # always checks the hold figure too. If this is empty, the next figure is used    
        hold_fig = self.game.next_figure if self.game.change_figure is None else self.game.change_figure
        next_pos_steps = self.get_next_pos_steps()
        # it is possible that no next steps are possible with the current or hold figure
        if next_pos_steps:
            next_actions, next_steps = zip(*next_pos_steps.items())
            next_steps = np.asarray(next_steps)
        next_pos_steps_hold = self.get_next_pos_steps(hold_fig)
        if next_pos_steps_hold:
            next_actions_hold, next_steps_hold = zip(*next_pos_steps_hold.items())
            next_steps_hold = np.asarray(next_steps_hold)
        # essential -> high probability for random actions at the beginning -> decreasing during the learning period
        # decision for random or best action based on the neural network
        epsilon = self.final_epsilon + (max(self.decay_epochs-self.game.epochs,0)*(self.initial_epsilon-self.final_epsilon)/self.decay_epochs) if not self.game.load_model else self.final_epsilon
        if np.random.rand() <= epsilon and self.game.train:
            index = np.random.randint(0,len(next_pos_steps)) if len(next_pos_steps) > 1 else 0
        else:
            q = self.model_learn.predict(next_steps, verbose=False)
            if next_pos_steps_hold:
                q_hold = self.model_learn.predict(next_steps_hold, verbose=False)
                if max(q) >= max(q_hold):
                    index = np.argmax(q)
                else:
                    index = np.argmax(q_hold)
                    self.game.change()
                    next_actions = next_actions_hold
                    next_steps = next_steps_hold
            else:
                index = np.argmax(q)
        
        x, r = next_actions[index]
        next_state = next_steps[index]
        # perform the chosen action
        self.game.current_figure.x = x
        self.game.current_figure.rotation = r
        self.game.down()

        if self.game.train:
            # get the reward and save it in memory to learn afterwards
            reward = self.get_reward(current_state = self.state, next_state=next_state)
            if self.game.state == GAME_OVER:
                self.exp.remember(self.state, reward, next_state, True)
            else:
                self.exp.remember(self.state, reward, next_state, False)
            self.state = next_state

            # place many tetrominos without learning
            if self.exp.current_index > self.max_memory / 10 or self.game.load_model:
                start = time.time()
                inputs, outputs = self.exp.get_train_instance(self.model_learn, self.model_decide, self.batch_size)
                self.loss += self.model_learn.train_on_batch(inputs, outputs)
                self.game.train_time += (time.time() - start)
                
                if self.game.epochs % self.update_model == 0:
                    self.game.save_model(self.model_learn)
                    self.model_decide.load_weights("model_Tetris.h5")
                