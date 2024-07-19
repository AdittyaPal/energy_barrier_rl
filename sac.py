import argparse
from collections import namedtuple
from itertools import count

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from env import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

tau = 5e-3
policy_update = 32
gradient_steps = 1
learning_rate = 1e-4
gamma = (1-1e-2)
capacity = 10000
iteration = 1000
batch_size = 128

num_hidden_layers = 2
num_hidden_units_per_layer = 256
sample_frequency = 256
activation = 'Relu'
log_interval = 1000
load = False

env = MolecularPath(np.array([0.623, 0.028]), np.array([-0.558, 1.442]), max_steps = 500, scale = 0.010) #gym.make(args.env_name))

# Set seeds
#torch.manual_seed(7623)
#np.random.seed(7623)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
min_Val = torch.tensor(1e-8).double().to(device)

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size = capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.num_transition = 0

    def push(self, data):
        self.num_transition += 1
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head

class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self):
        super(SAC, self).__init__()

        self.actor_net = Actor(state_dim, action_dim).to(device)
        self.alpha = 1.0
        self.policy_noise = 0.4
        self.noise_clip = 1.0
        self.gamma = gamma
        
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)
        self.Q_target1 = Q(state_dim, action_dim).to(device)
        self.Q_target2 = Q(state_dim, action_dim).to(device)
        self.Q_target1.load_state_dict(self.Q_net1.state_dict())
        self.Q_target2.load_state_dict(self.Q_net2.state_dict())
        self.log_alpha = torch.tensor([np.log(self.alpha)]).to(device)
        self.log_alpha.requires_grad = True

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=learning_rate)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        self.e_target = -action_dim
        self.replay_buffer = Replay_buffer()
        self.num_training = 1
        self.writer = SummaryWriter('./exp-SAC_dual_Q_network')
        
        # hard update of the target Q nets
        for target_param, param in zip(self.Q_target1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.Q_target2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        
        # calculate the mean and std of the distribution to sample the action from
        mu, log_sigma = self.actor_net(state)
        sigma = torch.exp(log_sigma)
        
        # construct the distribution for sampling (reparameterization)
        dist = Normal(mu, sigma)
        # sample an action from the constructed distribution
        z = dist.sample()
        
        perturb_dist = Normal(torch.tensor([0.0, 0.0]), 
                          torch.tensor([self.policy_noise, self.policy_noise]))
        perturb = perturb_dist.sample().clip(-self.noise_clip, self.noise_clip)        
        # noise = (torch.randn_like(z) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # squash it to be in the range (-1, 1)
        action = torch.tanh(z + perturb).detach().cpu().data.numpy()
        return action.flatten()

    def evaluate(self, state):
        # calculate the mean and std of the distribution to sample the action from
        batch_mu, batch_log_sigma = self.actor_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        # construct the distribution for sampling (reparameterization)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))       
        z = noise.sample()
        perturb = (torch.randn_like(z) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # sample actions from the distribution
        action = torch.tanh(batch_mu + batch_sigma * z.to(device) + perturb)
        # calculate the entropy of the sampled actions
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(device) + perturb) - torch.log(1 - action.pow(2) + min_Val)
        
        return action, log_prob.sum(dim=-1, keepdim=True)

    def update(self):    
        # loop for the desired number of gradient descent steps    
        for k in range(gradient_steps):        
            # sample a batch of transitions from the replay buffer    
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            s = torch.FloatTensor(x).to(device)
            a = torch.FloatTensor(u).to(device)
            r = torch.FloatTensor(r).to(device)
            sp = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            
            #compute the Q-targets    
            with torch.no_grad():
                up_pred, ep_pred = self.evaluate(sp)
                qp1_target = self.Q_target1(sp, up_pred)
                qp2_target = self.Q_target2(sp, up_pred)
                #as the minimum of the two estimates of the values of the next state
                qp_target = torch.min(qp1_target, qp2_target) - self.alpha * ep_pred
                q_target = r + (1 - done) * self.gamma * qp_target
            
            # mini batch gradient descent
            # update the first critic
            self.Q1_optimizer.zero_grad()
            Q1 = self.Q_net1(s, a)
            # penalise the loss between the estimate from this critic and the calculated target
            Q1_loss = nn.MSELoss()(Q1.squeeze(), q_target.squeeze())
            Q1_loss.backward()
            # gradient clipping to provide more stable learning
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            
            # update the second critic
            self.Q2_optimizer.zero_grad()
            Q2 = self.Q_net2(s, a)
            #penalise the loss between the estimate from this critic and the calculated target
            Q2_loss = nn.MSELoss()(Q2.squeeze(), q_target.squeeze())
            Q2_loss.backward()
            # gradient clipping to provide more stable learning
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            
            if(self.num_training % policy_update == 0):  #update the actor and alpha is it is time to update them
                for i in range(1):#policy_update):
                    #update the actor
                    self.actor_optimizer.zero_grad()
                    a_pred, e_pred = self.evaluate(s)
                    q_pi = torch.min(self.Q_net1(s, a_pred), self.Q_net2(s, a_pred))
                    # do gradient descent for the actor nets with the entropy term
                    pi_loss = (-q_pi + self.alpha * e_pred).mean()            
                    pi_loss.backward()
                    # gradient clipping to provide more stable learning
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    self.actor_optimizer.step()
                    self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)
                    
                    alpha_loss = (-self.log_alpha.exp() * (e_pred.detach() + self.e_target)).mean()
                    # update the log alpha based on the gradient of alpha loss
                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()
                    # update alpha used for weighing the entropy term
                    self.alpha = self.log_alpha.exp().item()
                    self.writer.add_scalar('Loss/alpha_loss', alpha_loss, global_step=self.num_training)
                    
                    # soft update target Q-nets by Polyak averaging 
                    for target_param, param in zip(self.Q_target1.parameters(), self.Q_net1.parameters()):
                        target_param.data.copy_(target_param * (1 - tau) + param * tau)
                    for target_param, param in zip(self.Q_target2.parameters(), self.Q_net2.parameters()):
                        target_param.data.copy_(target_param * (1 - tau) + param * tau)

            self.num_training += 1

    def save(self):
        '''
        Method to save the actor and critic neural nets at regular intervals during training
        '''
        torch.save(self.actor_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        print("Model is saved...")
    
    
    def load(self):
        '''
        Method to load the actor and critic neural nets for testing
        '''
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("Model is loaded...")


def main():

    agent = SAC()
    state, _ = env.reset()          #start interacting with the environment
    for i in range(iteration):      #do the required number of iterations
        returns = 0                 #reset the returns counter to zero at the start of each iteration
        states = []                 #reset the list to store the states at the start of each iteration
        
        for t in range(5000):       #loop to interact with the environemnt
            #the actor selects an action given the observed state
            action = agent.select_action(state)
            #observe the next state, reward and done signal after execting the chosen action in the environment
            next_state, reward, terminate, truncate, info = env.step(action, states)
            done = terminate or truncate
            
            returns += reward       #add the reward obtained to the returns from the iteration
            states.append(state)    #store the current state in the list of visited states
            
            #store the sarsd tuple in the replay buffer
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            
            #if the replay buffer is filled, the RL agent training is started
            if agent.replay_buffer.num_transition > capacity: # and t % (gradient_steps) == 0:
                agent.update()

            #update the current state of the agent to the next state
            state = next_state
            #if the state was terminal break out of the interaction loop
            if done:
                np.savetxt(f'./sac_data/states{i}.csv', states)    #save the states visited by the agent
                state, _ = env.reset()                             #get a new start state from another interaction with the environment
                print(f"Episode {i}, net returns: {returns:.3f}, steps: {t}")
                break
        if i % log_interval == (log_interval-1):
            agent.save()


if __name__ == '__main__':
    main()
