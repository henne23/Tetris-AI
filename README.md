
# AI learns to play Tetris

Can you beat Artificial Intelligence?

<p align="center">
    <img src ="demo.gif" width = 600><br/>
    <i> Demo </i>
</p>

## Authors

- [@Hendrik](https://www.github.com/henne23)

## Update 6/15/25

The reward function was updated. Instead of rewarding only one thing in 
a specific order (cleared_lines, holes, height, bumpiness), I provided one 
reward based on the evaluation of all four characteristics.
Furthermore, the AI will be penalized for creating stacks in the middle
of the field and rewarded for creating clean lines on the bottom.
This solved the issue of unstable results and let the agent clear
lines on the bottom rather than in the middle of the field.


## Introduction

This repository contains the basic Tetris environment where you
can play the famous game manually. Furthermore it is extended with
a Reinforcement Learning agent that learns to play Tetris with a
Double Q-Learning algorithm.

## AI Procedure

### Training

The agent starts to place one tenth of the replay memory size of
tetrominos, evaluates the states and saves them in the Experience
class.

Afterwards the training process starts. The ratio between exploration
and selection of the best action is controlled by an epsilon
parameter that continuously decreases during training.

Given one tetromino every possible action with every possible 
rotation is checked and evaluated. The model selects the action
with the best state reward.

### State

- cleared lines
- holes
- total height
- total bumpiness (sum of differences between the current and the next column)

### The network

The neural network uses two hidden layers with 64 neurons each. ReLU is used as the
activation function for the input and hidden layers. The output layer uses a linear
function. 

- loss = MSE
- optimizer = Adam

### Points

The more lines are cleared at once the more points are achieved. Afterwards the achieved points
are multiplied with the level (this only makes sense in the manual mode as the speed
increases with the level). The level itself increases after every 10 cleared lines.

These are the points:

[1: 40, 2: 100, 3: 300, 4: 1200]

## Results

The following diagram shows the best score over a window of 250 games (this training does not include the best results what can be found in the Save-folder).

<p align="center">
    <img src ="Score over epochs.png"><br/>
    <i> Score overview </i>
</p>

## Requirements

Python (3.7)\
pygame (2.0.1)\
tensorflow (2.1.0)\
keras (2.3.1)\
matplotlib (3.5.2)\
numpy (1.21.6)\
h5py (2.10.0)\
protobuf (3.20)\
pandas (1.3.5)