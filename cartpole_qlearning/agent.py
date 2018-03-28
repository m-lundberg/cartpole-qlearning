from collections import defaultdict
import random


class QLearningAgent:
    """
    A QLearning agent
    """
    def __init__(self, actions, alpha=0.5, gamma=0.9):
        # self.epsilon = 0.9    # exploration probability
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        self.Q = defaultdict(lambda: [random.random() for _ in range(actions)])

    def learn(self, state, action, reward, next_state):
        # use QLearning update rule to learn
        self.Q[state][action] = \
            (1 - self.alpha) * self.Q[state][action] \
            + self.alpha * (reward + self.gamma * max(self.Q[next_state]))

    def choose_action(self, state, action_space, epsilon=0.0):
        if random.random() < epsilon:
            # explore
            return action_space.sample()
        else:
            # pick best action
            return self.Q[state].index(max(self.Q[state]))
