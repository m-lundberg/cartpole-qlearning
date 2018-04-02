# cartpole-qlearning
The result of some playing around with [Gym](https://github.com/openai/gym  "OpenAI Gym") to train an agent to balance an inverted pendulum using [Q-Learning](https://en.wikipedia.org/wiki/Q-learning "Wikipedia").

The current code manages to solve the problem by only using the angle and the angular velocity of the pole, completely ignoring the linear position and velocity of the cart (to reduce dimensionality for faster convergence). I'm sure it could be tweaked even further to improve the results.
