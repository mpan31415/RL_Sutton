import torch.nn.functional as F
import torch
import numpy as np
import copy
from os import getcwd

from utils import nets
from utils.replay_buffer import ReplayBuffer


class TD3_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.policy_noise = 0.2*self.max_action
		self.noise_clip = 0.5*self.max_action
		self.tau = 0.005
		self.delay_counter = 0

		self.actor = nets.Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = nets.Double_Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)
  
		self.models_dir = getcwd() + "\\chp13-policy-gradient-methods\\rl\\models\\"
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)

			# Compute the target Q
			target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			'''↓↓↓ Target Policy Smoothing Regularization ↓↓↓'''
			smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
			target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
			'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~done) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss, and Optimize the q_critic
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
		if self.delay_counter > self.delay_freq:
			# Update the Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			with torch.no_grad():
				for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = 0

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), self.models_dir+"{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), self.models_dir+"{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load(self.models_dir+"{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load(self.models_dir+"{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))



