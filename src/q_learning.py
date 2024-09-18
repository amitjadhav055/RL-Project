import numpy as np
import gym

# Initialize the FrozenLake environment with rendering mode
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

# Q-table initialization
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters (same as before)
learning_rate = 0.8
discount_rate = 0.95
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
num_episodes = 10000
max_steps = 100

# Q-learning algorithm (same as before)
for episode in range(num_episodes):
    state = env.reset()[0]  # Unpack reset tuple
    done = False
    for step in range(max_steps):
        exploration_threshold = np.random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, truncated, info = env.step(action)

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

        if done or truncated:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
        -exploration_decay_rate * episode
    )

# Testing the agent after training
state = env.reset()[0]  # Unpack reset tuple
env.render()
done = False
for step in range(max_steps):
    action = np.argmax(q_table[state, :])
    new_state, reward, done, truncated, info = env.step(action)
    env.render()
    state = new_state
    if done or truncated:
        break

env.close()
