import gym
from cartpole_qlearning.agent import QLearningAgent


def discretize(value, num_bins, lower, upper):
    if value < lower:
        return 0
    if value > upper:
        return num_bins - 1

    return int(round((value - lower) / (upper - lower) * num_bins))


def to_state(observation):
    # print(observation)
    return str([discretize(s, 20, env.observation_space.low[i], env.observation_space.high[i]) for i, s in enumerate(observation)])


def train(episodes, step_limit, epsilon):
    for episode in range(episodes):
        if not episode % 100:
            pass
            # print('Starting episode {}'.format(episode))

        observation = env.reset()

        for t in range(step_limit):
            # env.render()

            state = to_state(observation)

            action = agent.choose_action(state, env.action_space, epsilon)
            next_observation, reward, done, info = env.step(action)

            if done:
                break

            next_state = to_state(next_observation)
            agent.learn(state, action, reward, next_state)
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
    episodes = 10000
    step_limit = 100

    env = gym.make('CartPole-v0')
    agent = QLearningAgent(env.action_space.n, 0.4, 0.99)

    epsilon = 0.9  # probability of picking a random action (exploration)

    for i in range(10):
        print('Starting episode {e} with epsilon {eps}'.format(e=i*episodes/10, eps=epsilon))
        train(int(episodes / 10), step_limit, epsilon)
        epsilon -= 0.1

        if epsilon < 0.1:
            epsilon = 0.1
    print(agent.Q)

    balance()

    env.close()
