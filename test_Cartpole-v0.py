import gym
import numpy as np
import dql_agent

env = gym.make('CartPole-v0')
agent = dql_agent.dql_agent(2,4)
for e in range(1000):

    state = env.reset()
    state = np.reshape(state,[1,4])

    for t in range(1000):
        action = agent.act(state)
        next_state, reward, done, debug = env.step(action)
        next_state = np.reshape(next_state,[1,4])
        if done:
            reward = -10
        agent.memorize(state, action, reward, done, next_state)

        state = next_state

        if done:
            print(e,t)
            break

    if (len(agent.memory)>agent.batch_size):
        agent.replay()
