import numpy as np

'''
Author: Hendrik Pieres

'''

class Experience:
    def __init__(self, input_size, output_size, max_memory = 30000, discount = .99):
        self.max_memory = max_memory
        self.state_memory = np.zeros((self.max_memory, input_size), dtype=int)
        self.reward_memory = np.zeros((self.max_memory))
        self.next_state_memory = np.zeros((self.max_memory, input_size))
        self.gameover_memory = np.zeros(self.max_memory, dtype=bool)
        self.discount = discount
        self.input_size = input_size
        self.output_size = output_size
        self.current_index = 0

    def remember(self, state, reward, next_state, gameover):
        index_mod = self.current_index % self.max_memory
        self.state_memory[index_mod] = state
        self.reward_memory[index_mod] = reward
        self.next_state_memory[index_mod] = next_state
        self.gameover_memory[index_mod] = gameover
        self.current_index += 1

    def get_train_instance(self, model_learn, model_decide, batch_size):
        # maybe include prioritized replay in the next update
        min_length = min(self.current_index, batch_size)
        elements = np.random.randint(0, min(self.current_index, self.max_memory), size=min_length)
        outputs = np.zeros((self.input_size, self.output_size))
        rewards = self.reward_memory[elements]
        gameovers = self.gameover_memory[elements]
        states = self.state_memory[elements]
        next_states = self.next_state_memory[elements]

        outputs = model_learn.predict(states)
        new_outputs = np.max(model_decide.predict(next_states), axis = 1)

        for index, _ in enumerate(rewards):
            if gameovers[index]:
                outputs[index] = rewards[index]
            else:
                outputs[index] = rewards[index] + self.discount * new_outputs[index]
        return states, outputs