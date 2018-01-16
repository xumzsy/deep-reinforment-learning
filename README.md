# test-simple-reinforment-learning
Implement some basic reinforcement learning on simple "gym" environment (provided by openAI)

dql_agent.py

Implement an agent for deep Q-Learning which suits for discrete input (possible input from {a0,...,an} using pytorch.

dpg_agent.py

Implement an agent for deep policy gradient (actor and critic) which suits for continues input. It should work for other case but the parameters are tuned for Pendulum-v0

test_Cartpole-v0.py

Main function to train an agent in "Cartpole-v0" environment and print result. The environment is provided by OpenAI's gym.

test_Pendulum-v0.py

Main function to train an agent in "Pendulum-v0" environment and print result.
