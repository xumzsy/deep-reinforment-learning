import numpy as np

import gym
import dpg_agent

env = gym.make('Pendulum-v0')
agent = dpg_agent.dpg_agent(3,1)

for e in range(10000):

    state = env.reset()
    #env.render()
    state = np.reshape(state,[1,3])
    totol_reward = 0

    for t in range(1000):
        action = agent.act(state)
        next_state, reward, done, debug = env.step(action)
        next_state = np.reshape(next_state,[1,3])
        agent.memorize(state, action, reward, done, next_state)
        totol_reward += reward
        state = next_state

        agent.epsilon *= agent.epsilon_decay
        if (len(agent.memory)>agent.batch_size and e>=100):
            agent.replay()
        if done:
            break
        #if t%10==0:
        #    env.render()
    print(e,t,reward,totol_reward/t)    
