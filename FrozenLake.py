import gym
import numpy as np
import matplotlib.pyplot as plt


######################
# Create environment #
######################

env = gym.make("FrozenLake-v0")


############################
# Initialize quality table #
############################

print("This environment has ", env.observation_space.n, " states and ", env.action_space.n, " possible actions from each state.")
print("Initializing quality table.")

Q = np.zeros((env.observation_space.n, env.action_space.n))


#########################
# Epsilon-greedy policy #
#########################

epsilon = 1


def policy(state):
    global epsilon
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
        epsilon *= 0.9999
    else:
        action = np.argmax(Q[state, :])
    return action


####################
# Bellman Equation #
####################

gamma = 0.99
learning_rate = 0.1


def learn(old_state, new_state, action, reward):
    old_quality = Q[old_state, action]
    if (game_finished):
        new_quality = reward
    else:
        new_quality = reward + gamma*max(Q[new_state])
    Q[old_state, action] = old_quality + learning_rate * (new_quality - old_quality)


#############
# Play game #
#############

n_games = 100000
training_history = []

for n in range(n_games):
    if ((n+1) % 1000 == 0):
        print("Finished playing ", n+1, " games.")
    game_finished = False
    state = env.reset()
    total_game_reward = 0
    n_actions = 0
    while not game_finished:
        n_actions += 1
        action = policy(state)
        new_state, reward, game_finished, _ = env.step(action)
        learn(state, new_state, action, reward)
        state = new_state
        total_game_reward += reward
    # print("Game ", n, " finished with a total reward of ", total_game_reward, " after ", n_actions, " actions.")
    training_history.append(total_game_reward)


#########################
# Plot training history #
#########################

n_in_average = 1000
average_history = [np.average(training_history[i*n_in_average: (i+1)*n_in_average]) for i in range(n_games//n_in_average)]
plt.plot(average_history)
plt.show()
