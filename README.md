# Assignment 2 - Q-Learning: Tabular & Deep

This repository contains my implementation for Assignment 2 on Deep Q-Learning (DQN) in CartPole.

## Contents
The project includes:
- Naive neural Q-learning baseline
- Target Network (TN)
- Experience Replay (ER)
- Comparison of four configurations:
  - Naive
  - Only TN
  - Only ER
  - TN + ER
- Hyperparameter ablation study for:
  - learning rate
  - exploration factor
  - network size
  - update-to-data ratio

## File Structure
- `dqn_cartpole.py`: core implementation of the neural Q-learning agent
- `experiments.py`: runs the comparison between Naive / Only TN / Only ER / TN+ER
- `ablation.py`: runs the hyperparameter ablation studies
- `results/`: stores generated plots
- `requirements.txt`: Python dependencies

## Installation
Create and activate a Python environment if desired, then install the required packages:

pip install -r requirements.txt

## How to Run
1. Run the naive baseline
python dqn_cartpole.py

This generates: results/naive.png

2. Run the configuration comparison
python experiments.py

This generates: results/compare_configs.png

3. Run the ablation study
python ablation.py

This generates the ablation plots, such as:

results/ablation_learning_rate.png
results/ablation_exploration.png
results/ablation_network_size.png
results/ablation_update_ratio.png


# reinforcement-learning-2
>>>>>>> 47cb233cfb94fd607b919ce4f914a882de3f8bae
