import numpy as np
import copy
import pickle
import sys

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv  # https://github.com/openai/baselines

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim
import pycuda.driver as cuda

import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


class MemoryBank():
    '''Stores and retrieves memories. Also ranks memories by usefulness, and forgets unimportant ones.'''
    def __init__(self, brain):
        self.memories = []
        self.memory_probs = []

        self.memory_batch_n = 100
        self.memory_threshold = self.memory_batch_n*20
        self.memory_size = self.memory_batch_n*1000

        self.brain = brain

    def remember(self, experiences):
        '''Remembers a list of experiences of form [[states], [actions], [rewards], [new_states], [games_over]].'''
        experiences = map(list, zip(*experiences))
        for experience in experiences:
            self.memories.append(experience)

    def rank_memories(self):
        '''Rank all memories in bank based on usefulness, or the difference in results between Q and target_Q.'''
        qualities, target_qualities = self.brain.get_qualities(list(map(list, zip(*self.memories))))
        usefulness = abs(qualities - target_qualities)
        usefulness += max(usefulness)/100  # allow a non-zero chance for each item to be selected
        usefulness = usefulness.detach().numpy()
        self.memory_probs = usefulness/sum(usefulness)

    def forget(self):
        '''Forget least useful memories.'''
        best_memories = np.random.choice(len(self.memories), self.memory_size, replace=False, p=self.memory_probs)
        self.memories = np.array(self.memories)[best_memories]
        self.memories = list(self.memories)

    def recall_batch(self):
        '''Recall a batch of memories, and forget some. Q and target_Q are used to determine how useful each memory
        is. Memories are retrieved or forgotten based on usefulness.'''
        if len(self.memories) < self.memory_threshold:
            return None
        self.rank_memories()
        best_memories = np.random.choice(len(self.memories), self.memory_batch_n, replace=False, p=self.memory_probs)
        memory_batch = np.array(self.memories)[best_memories]
        if len(self.memories) > self.memory_size:
            self.forget()
        return list(map(list, zip(*memory_batch)))


class QualityNet(nn.Module):
    '''Takes a state and returns qualities for each action.'''
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(128, 256)
        self.layer_2 = nn.Linear(256, 6)

    def forward(self, state):
        x = torch.Tensor(state)
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class DDQN():
    '''Determines the quality of an action-state pair. Can learn from experience.'''
    def __init__(self):
        self.Q = QualityNet()
        self.target_Q = copy.deepcopy(self.Q)
        self.memory = MemoryBank(self)

        self.gamma = .95
        self.learning_rate = 0.01
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.learning_rate)

        self.target_count = 0
        self.target_refresh_frequency = 5

    def update_target(self):
        self.target_Q = copy.deepcopy(self.Q)

    def get_qualities(self, memory_batch):
        state_batch, action_batch, reward_batch, new_state_batch, game_over_batch = memory_batch
        quality_batch = self.Q(state_batch)[list(range(len(action_batch))), action_batch]

        game_not_over_batch = torch.Tensor([1-i for i in game_over_batch])
        DDQN_action_batch = torch.max(self.Q(new_state_batch), dim=1)[1]
        target_next_Q_batch = self.target_Q(new_state_batch)[list(range(len(DDQN_action_batch))), DDQN_action_batch]
        target_Q_batch = torch.Tensor(reward_batch) + game_not_over_batch*self.gamma*target_next_Q_batch

        return quality_batch, target_Q_batch

    def learn(self, memory_batch):
        if memory_batch is None:
            return

        quality_batch, target_Q_batch = self.get_qualities(memory_batch)
        loss = self.loss_function(quality_batch, target_Q_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_count += 1
        if (self.target_count >= self.target_refresh_frequency):
            self.target_count = 0
            self.update_target()


class Agent():
    '''This agent undergoes the act->remember->recall->learn loop for a certain number of games.
    At the same time, it stores the results of these loops for plotting.'''
    def __init__(self):
        '''Set up an agent which runs multiple environments in parallel.'''
        def make_env(rand_seed):
            def _make():
                env = gym.make("Pong-ram-v0")
                env.seed(rand_seed)
                return env
            return _make

        # def check_cuda():
            # cuda.init()
            # print(f"Using GPU {cuda.Device(torch.cuda.current_device()).name()}")
            # return torch.cuda.is_available()

        # check_cuda()

        self.n_processes = 8
        self.n_frames_between_training = 1000

        self.envs = SubprocVecEnv([make_env(np.random.randint(sys.maxsize)) for _ in range(self.n_processes)])
        self.n_actions = self.envs.action_space.n
        self.states = self.envs.reset()

        self.brain = DDQN()

        self.training_history = []
        self.game_frames = []

        self.epsilon = 1
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.05

    def act(self, states):
        '''Choose an action using the policy, act on the environment, and return all the outputs.'''
        def policy(states):
            def _get_action(state):
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(self.n_actions)
                else:
                    action_probabilities = F.softmax(self.brain.Q(state), dim=0)
                    action = Categorical(action_probabilities).sample().item()
                return action
            return np.stack([_get_action(state) for state in states])

        actions = policy(states)
        new_states, rewards, games_over, _ = self.envs.step(actions)
        self.states = new_states
        experiences = (states, actions, rewards, new_states, games_over)
        return experiences

    def reduce_policy_randomness(self):
        '''Reduce the epsilon used in the policy for less randomness in picking actions.'''
        self.epsilon *= self.epsilon_decay
        if (self.epsilon < self.min_epsilon):
            self.epsilon = self.min_epsilon

    def play(self):
        self.states = self.envs.reset()
        print(f"Playing {self.n_processes} games simultaneously.")

        while True:
            avg_game_reward = 0
            # new_game_frames = []

            for _ in range(self.n_frames_between_training):
                experiences = self.act(self.states)
                avg_game_reward += np.mean(experiences[2])
                self.brain.memory.remember(experiences)

                # new_game_frames.append(self.envs.render(mode='rgb_array'))

            for _ in range(5):
                memory_batch = self.brain.memory.recall_batch()
                self.brain.learn(memory_batch)
            self.reduce_policy_randomness()
            self.training_history.append([avg_game_reward, self.epsilon, self.brain.learning_rate])

            # self.game_frames.append(new_game_frames)
            print(f"Average score of last {self.n_frames_between_training} frames for {self.n_processes} games was {avg_game_reward:.2f}.")  # NOQA
            if (avg_game_reward > 20):
                break
            # torch.save(agent.brain.Q.state_dict(), "saved_pong_weights.data")

    def display_game(self, game_n):
        patch = plt.imshow(self.game_frames[game_n][0])
        plt.axis('off')
        def animate(i): patch.set_data(self.game_frames[game_n][i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(self.game_frames[game_n]), interval=50)
        display(display_animation(anim, default_mode='once'))


if __name__ == "__main__":
    '''Either train a new agent or load a pre-trained agent.'''
    pretrained_weights = None
    # pretrained_weights = "saved_pong_weights.data"

    agent = Agent()
    print("Setting up agent.")

    if pretrained_weights is None:
        print("Training agent.")
        agent.play()
        training_history = np.array(agent.training_history)
        print("Game reward")
        plt.plot(training_history[:, 0])
        plt.show()
        print("Policy epsilon")
        plt.plot(training_history[:, 1])
        plt.show()
        print("Learning rate")
        plt.plot(training_history[:, 2])
        plt.show()

    else:
        print("Loading trained agent.")
        agent.brain.Q.load_state_dict(torch.load("saved_pong_weights.data"))
        with open('saved_pong_history.pickle', 'rb') as handle:
            saved_info = pickle.load(handle)
            agent.game_frames = saved_info[0]
