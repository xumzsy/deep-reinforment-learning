import numpy as np

from collections import deque
import torch
from torch.autograd import Variable

class actor_nn(torch.nn.Module):
        """nn for actor"""
        def __init__(self, state_dim,action_dim):
            super(actor_nn,self).__init__()
            self.linear1 = torch.nn.Linear(state_dim,32)
            self.linear2 = torch.nn.Linear(32,32)
            self.linear3 = torch.nn.Linear(32,32)
            self.linear4 = torch.nn.Linear(32,action_dim)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self,state):
            action = self.relu(self.linear1(state))
            action = self.relu(self.linear2(action))
            action = self.relu(self.linear3(action))
            action = self.linear4(action)
            action = -2.0 + self.sigmoid(action)*4.0
            return action

class dpg_agent():
            
    def __init__(self,state_dim,action_dim):
        self.actor_lr = 1e-3 # learning rate
        self.critic_lr = 1e-3
        # noise model
        self.noise = 0.0
        self.noise_mu = 0.0
        self.noise_theta = 0.15
        self.noise_sigma = 0.2
        self.epsilon = 4 # random noise strengh
        self.epsilon_decay = 0.99 # noise decay rate
        self.gamma = 0.99 # future reward discount
        self.batch_size = 256
        self.tau = 0.01
        self.dtype = torch.FloatTensor
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # memory
        self.memory = deque(maxlen=100000)
        
        # actor 
        self.actor = actor_nn(state_dim,action_dim)

        # slow actor
        self.actor_freezed = actor_nn(state_dim,action_dim)

        # critic
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(action_dim+state_dim,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1)
        )

        # slow critic
        self.critic_freezed = torch.nn.Sequential(
            torch.nn.Linear(action_dim+state_dim,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1)
        )
        
        # init same weights for actor/critic and slow actor/critic
        for param,param_freezed in zip(self.actor.parameters(),self.actor_freezed.parameters()):
            param_freezed.data = param.data.clone()
            
        for param,param_freezed in zip(self.critic.parameters(),self.critic_freezed.parameters()):
            param_freezed.data = param.data.clone()

        # loss_fn and optimizer
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr,weight_decay=1e-6)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr,weight_decay=1e-6)

    def memorize(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))
        

    def act(self,state):
        self.noise = self.noise_theta*(self.noise_mu-self.noise)+self.noise_sigma*np.random.randn(self.action_dim)
        noise = self.noise*self.epsilon
        State = Variable(torch.from_numpy(state).type(self.dtype),volatile=True)
        action = self.actor(State)
        return action.data.numpy() + noise

    def replay(self):
        sample = np.random.randint(len(self.memory),size=self.batch_size)
        state = np.reshape(np.asarray([self.memory[idx][0] for idx in sample]),[self.batch_size,self.state_dim])
        action = np.reshape(np.asarray([self.memory[idx][1] for idx in sample]),[self.batch_size,self.action_dim])
        reward = np.reshape(np.asarray([self.memory[idx][2] for idx in sample]),[self.batch_size,1])
        done = np.reshape(np.asarray([self.memory[idx][3] for idx in sample]),[self.batch_size,1])
        next_state = np.reshape(np.asarray([self.memory[idx][4] for idx in sample]),[self.batch_size,self.state_dim])

        action = np.reshape(action,[self.batch_size,self.action_dim])
        
        # init Variables
        State = Variable(torch.from_numpy(state).type(self.dtype))
        Action = Variable(torch.from_numpy(action).type(self.dtype))
        Reward = Variable(torch.from_numpy(reward).type(self.dtype),volatile=True)
        Next_State = Variable(torch.from_numpy(next_state).type(self.dtype),volatile=True)

        ''' update critic '''
        # calculate "real" q value from freezed critic
        Next_Action = self.actor_freezed(Next_State)
        Next_Scene = torch.cat((Next_State,Next_Action),dim=1)
        q_real = self.critic_freezed(Next_Scene)*self.gamma + Reward

        # calculate expected q value (critic)
        Scene = torch.cat((State,Action),dim=1)
        q_pred = self.critic(Scene)
        
        # backward
        critic_loss = self.critic_loss_fn(q_pred,Variable(q_real.data))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ''' update actor '''
        # calculate current best action and correspond q_value
        
        Best_Action = self.actor(State)
        Best_Scene = torch.cat((State,Best_Action),dim=1)
        actor_loss = - torch.mean(self.critic(Best_Scene))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # update freezed actor/critic
        for param,param_freezed in zip(self.actor.parameters(),self.actor_freezed.parameters()):
            param_freezed.data = param.data * self.tau + param_freezed.data * (1-self.tau)

        for param,param_freezed in zip(self.critic.parameters(),self.critic_freezed.parameters()):
            param_freezed.data = param.data * self.tau + param_freezed.data * (1-self.tau)
        
        




        
