import numpy as np
import torch
from torch.autograd import Variable

class dql_agent():

	def __init__(self,action_size,state_dim):
		# parameters
		self.lr = 1e-3 # learning rate
		self.epsilon = 1 # chance of random choice
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.gamma = 0.95 # discount rate
		self.batch_size = 128
		self.action_size = action_size
		self.state_dim = state_dim
		self.dtype = torch.FloatTensor

        # Q value nn: q[action] = f([state])
        # measure the value of each action given state in the whole future
		self.Q_model = torch.nn.Sequential(
        	torch.nn.Linear(state_dim,24),
        	torch.nn.ReLU(),
        	torch.nn.Linear(24,24),
        	torch.nn.ReLU(),
        	torch.nn.Linear(24,action_size)
        )

		self.loss_fn = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.Q_model.parameters(), lr=self.lr)

        # memory
		self.memory = []

    # add history into memory
	def memorize(self,state,action,reward,done,next_state):
		self.memory.append((state, action, reward,  done,next_state))
		if (len(self.memory)>2000):
			self.memory.pop(0);

	# generate action from state
	def act(self,state):
		# random generate action
		if np.random.rand() < self.epsilon:
			return np.random.randint(self.action_size)
		# generate best action 
		State = Variable(torch.from_numpy(state).type(self.dtype),volatile=True)
		q = self.Q_model(State)
		return np.argmax(q.data.numpy())

	def replay(self):
		for i in range(128):
			sample = np.random.randint(len(self.memory))
			state, action, reward, done, next_state = self.memory[sample]

			target = reward

			Next_State = Variable(torch.from_numpy(next_state).type(self.dtype),requires_grad=False)
			State = Variable(torch.from_numpy(state).type(self.dtype),requires_grad=False)
			if not done:
				q = self.Q_model(Next_State)
				target += self.gamma*torch.max(q.data)
			
			target_pred = self.Q_model(State)
			target_real = Variable(target_pred.data.clone())
			target_real.data[0][action] = target

			loss = self.loss_fn(target_pred,target_real)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay






