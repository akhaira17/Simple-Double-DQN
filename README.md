# Simple Double DQN for CartPole-v0

This repository contains a clean implementation of the **Double Deep Q-Network (Double DQN)** algorithm using PyTorch. It solves the classic **CartPole-v0** environment in approximately **159 episodes**.

## Features

- Implements **Double Q-learning** to reduce overestimation bias
- Uses **experience replay** and **target networks** for stable learning
- Includes **epsilon-greedy** exploration with decay

## Algorithm Summary

Double DQN modifies the Q-learning target as:

`Target = r + γ * Q_target(s', argmax_a Q_online(s', a))`

This decouples action selection and evaluation, reducing overoptimism in action-value estimates.

## Files

- `double_dqn.py`: Main training loop and neural network
- `exploration_strategy.py`: Epsilon decay handling
- `replay_buffer.py`: Experience replay buffer implementation
- `LICENSE`: MIT license for open use and modification

## Getting Started

### Installation

Install the required dependencies:

```bash
pip install torch gym numpy
