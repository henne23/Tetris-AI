
# AI learns to play Tetris

Can you beat Artificial Intelligence?

<p align="center">
    <img src ="demo.gif" width = 600><br/>
    <i> Demo </i>
</p>

## Authors

- [@Hendrik](https://www.github.com/henne23)


## Introduction

This repository contents the basic Tetris environment where you
can play the famous game manually. Furthermore it is extended with
a Reinforcement Learning agent that learns to play Tetris with a
Double Q-Learning algorithm.

## AI Procedure

# Training

The agent starts to place one tenth of the replay memory size of
tetrominos, evaluates the states and saves them in the Experience
class.

Afterwards the training process starts. The ratio between exploration
and selection of the best action is controlled by an epsilon
parameter that continuously decreases during training.

Given one tetromino every possible action with every possible 
rotation is checked and evaluated. The model selects the action
with the best state.

# State

- cleared lines
- holes
- total height
- total bumpiness (sum of differences between the current and the next column)

# The network

The neural network uses two hidden layers with 64 neurons each. ReLU is used as the
activation function for the input and hidden layers. The output layer uses a linear
function. 

- loss = MSE
- optimizer = Adam

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

## Misc

It was noticeable that the results varied greatly. 
Even still very successful games with several hundred cleared lines, 
it happened that subsequent games ended after only a few tetrominos.

The following diagram shows the best score over a window of 50 games (this training does not include the best results what can be found in the Save-folder).

<p align="center">
    <img src ="Score by epoch.png" width = 600><br/>
    <i> Score overview </i>
</p>

Furthermore, it seemed as if the AI doest not strive to clear
lines on the bottom of the game, but tends to clear rows in the
middle of the field.