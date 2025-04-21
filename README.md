# CartPole-tinygrad

This project implements a Q-Learning reinforcement learning algorithm using the tinygrad framework. The agent is trained to solve the CartPole-v1 environment from Gymnasium (formerly OpenAI Gym).

## Project Overview

1- A Q-Learning agent with a neural network (Q-network) using tinygrad
2- Training on the CartPole environment from Gymnasium (formerly OpenAI Gym)
3- Experience replay to stabilize learning
4- Visualization of training progress

## Q-Learning 
Q-learning is a reinforcement learning algorithm that trains an agent to assign values to its possible actions based on its current state, without requiring a model of the environment (model-free).(1)

!["A Q-learning table mapping states to actions, initially filled with zeros and updated iteratively through training.](assets/Q-Learning_matrix_init_and_after_training.png)

## Gymnasium
 A classic benchmark used in reinforcement learning (RL) to illustrate and test RL algorithms. It simulates a simple physics-based control problem.(2)

## Requirements
- Python 3.7+
- tinygrad
- gymnasium
- numpy
- matplotlib

## Sources 
1- [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
2- [Gymnasium](https://www.bomberbot.com/reinforcement-learning/reinforcement-learning-made-easy-with-gymnasium/)


