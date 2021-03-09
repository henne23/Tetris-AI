import numpy as np

from AI.Experience import Experience


class Training:
    def __init__(self, game, modelLearn, modelDecide, batchSize):
        self.game = game
        self.modelLearn = modelLearn
        self.modelDecide = modelDecide
        self.epsilon = .1
        self.batchSize = batchSize
        self.exp = Experience(self.modelDecide.input_shape[-1], self.modelDecide.output_shape[-1])

    def randmax(self, values):
        max_values = []
        current_max = values[0]
        index = 0
        for v in values:
            if v > current_max:
                max_values = [index]
                current_max = v
            elif v == current_max:
                max_values.append(index)
            index += 1
        if len(max_values) == 0:
            return np.random.randint(0,len(values)-1)
        else:
            return np.random.choice(max_values)

    def getReward():
        if self.game.killedLines > 0:
            return self.game.killedLines
        

    def train(modelLearn, modelDecide, exp):
        loss = 0.0
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, len(actions))
        else:
            q = modelDecide.predict(game.field, game.Figure.typ, game.nextFigure.typ, game.changeFigure.typ)
            action = randmax(q[0])
        
        oldState = self.game.field
        oldState.append(self.game.currentFigure.typ, self.game.nextFigure.typ, self.game.changeFigure.typ)

        if action == 0:
            self.game.left()
        elif action == 1:
            self.game.right()
        elif action == 2:
            self.game.down()
        elif action == 3:
            self.game.rotate()
        elif action == 4:
            self.game.change()

        reward = self.getReward()   
        state = self.game.field
        state.append(self.game.currentFigure.typ, self.game.nextFigure.typ, self.game.changeFigure.typ)

        if game.state == GAME_OVER:
            self.exp.remember(oldState, action, reward, state, True)
        else
            self.exp.remember(oldState, action, reward, state, False)

        inputs, outputs = self.exp.getTrainInstance(self.modelLearn, self.modelDecide, self.batchSize)
        self.loss += modelLearn.train_on_batch(inputs, outputs)