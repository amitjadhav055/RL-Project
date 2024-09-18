# Reinforcement Learning with Q-Learning

This project demonstrates a basic implementation of Q-Learning, a reinforcement learning algorithm, applied to the **FrozenLake** environment using OpenAI's Gym.

The goal of this project is to train an agent to navigate a slippery 4x4 grid and reach the goal without falling into the lake. 

## Project Structure

- **src/**: Contains the source code, including the `q_learning.py` file, which is the main script implementing the Q-Learning algorithm.
- **models/**: Folder to save models (if you want to extend the project).
- **data/**: Folder to store any data if required (though this project does not use external data).
- **venv/**: Virtual environment to keep dependencies isolated.
- **requirements.txt**: Lists all required Python packages to run the project.

## Q-Learning Algorithm

Q-Learning is a model-free, off-policy reinforcement learning algorithm. It aims to find the best action to take given the current state, following a policy that maximizes the reward in the long run.

### Key Components
- **Learning Rate (α)**: Determines how much new information overrides old information.
- **Discount Factor (γ)**: How much importance we give to future rewards.
- **Exploration Rate (ε)**: Probability of choosing a random action instead of the current best action.
- **Q-Table**: Stores the expected future rewards for each state-action pair.

## Dependencies

To run this project, install the necessary dependencies using:

```bash
pip install -r requirements.txt
