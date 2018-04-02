import math
import gym
from cartpole_qlearning.agent import QLearningAgent


def discretize(value, num_bins, lower, upper):
    if value < lower or num_bins == 1:
        return 0
    if value > upper:
        return num_bins - 1

    return int(round((value - lower) / (upper - lower) * num_bins))


def to_state(observation):
    state_limits = list(zip(env.observation_space.low, env.observation_space.high))
    state_limits[1] = [-0.5, 0.5]
    state_limits[2] = [-math.radians(12), math.radians(12)]
    state_limits[3] = [-math.radians(50), math.radians(50)]

    bins = [1, 1, 6, 5]

    return tuple(discretize(s, bins[j], state_limits[j][0], state_limits[j][1]) for j, s in enumerate(observation))


def get_epsilon(episode):
    # get the probability of picking a random action (exploration)
    start_epsilon = 1.0
    min_epsilon = 0.01
    return max(min_epsilon, min(start_epsilon, 1.0 - math.log10((episode+1) / 25)))


def get_alpha(episode):
    # get the learning rate
    start_alpha = 0.5
    min_alpha = 0.1
    return max(min_alpha, min(start_alpha, 1.0 - math.log10((episode+1) / 25)))


def train(num_episodes, step_limit):
    solves = 0

    for episode in range(num_episodes):
        epsilon = get_epsilon(episode)
        alpha = get_alpha(episode)

        print(f'Starting episode {episode} using epsilon {epsilon} and alpha {alpha}')
        if solves:
            print(f'Consecutive solves: {solves}')

            if solves > 100:
                # Problem considered solved
                print('Problem solved')
                break

        observation = env.reset()

        for t in range(step_limit):
            env.render()

            state = to_state(observation)

            action = agent.choose_action(state, env.action_space, epsilon)
            next_observation, reward, done, info = env.step(action)

            if done:
                if t > 195:
                    solves += 1
                else:
                    solves = 0
                break

            next_state = to_state(next_observation)
            agent.learn(state, action, reward, alpha, next_state)
            observation = next_observation


def balance():
    while True:
        state = env.reset()

        while True:
            env.render()

            action = agent.choose_action(str(state), env.action_space)
            next_state, reward, done, info = env.step(action)

            if done:
                break

            state = next_state


if __name__ == '__main__':
    episodes = 1000
    max_steps = 250

    env = gym.make('CartPole-v0')
    agent = QLearningAgent(env.action_space.n, 0.99)

    train(episodes, max_steps)

    from pprint import pprint
    pprint(agent.Q)

    balance()

    env.close()
