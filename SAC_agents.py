
from enum import auto
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class RandomAgent:
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        # super(RandomAgent, self).__init__(n_actions)
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class TrainedAgent:
    ''' Agent that loads and follows a learned policy/critic
     '''
    def __init__(self, file_name = 'network-critics.pth', n_agents = "auto", discount = 0.99):

        file_name = os.path.join("NN saved", file_name)
        # Load critic
        try:
            criticsNN = torch.load(file_name)
            print(f'Loaded Critic, n_agents = {len(criticsNN)}, discount = {discount}. Network model[0]: {criticsNN[0]}')
        except:
            print(f'File {file_name} not found!')
            exit(-1)

        self.criticsNN = criticsNN
        if n_agents == "auto":
            self.n_agents = len(criticsNN)
        else:
            self.n_agents = n_agents

        self.discount = discount # to benchmark the critic

    def forward(self, state: np.ndarray):
        """ Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        """
        # mu, var = self.actorNN(torch.tensor([state]))
        # mu = mu.detach().numpy()
        # std = torch.sqrt(var).detach().numpy()
        # actions = np.clip(np.random.normal(mu, std), -1, 1).flatten()

        # return actions
        pass

    def benchmark_cirtic(self, buffers: deque, only_one_NN = False):

        Gts = deque() # for debug, delete after
        Q_approxs = deque() # for debug, delete after
        criticNN= self.criticsNN[0]

        for i in range(self.n_agents):
            # NN for this agent:
            if not only_one_NN:
                criticNN = self.criticsNN[i]
            
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a)
            inputs = np.column_stack((states,actions))
            inputs = torch.tensor(inputs, dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()

            # value function: # calculate the approximated Q(s,a) = NN(input)
            Q_approx = criticNN(inputs).squeeze()

            Q_approxs.append(Q_approx.detach().numpy())
            Gts.append(Gt_array) # for debug

        # Gts is the simulated Q(st,at) values for each agent
        return Gts, Q_approxs

class SACAgents:

    def __init__(self, n_agents, dim_local_state, dim_local_action, discount, epochs, learning_rate_critic = 10**(-3), discrete_action = False) -> None:
        '''* dim_local_state is the total size of the localized vector that the input of the Q and pi approximations use, i.e (k+1)*dim'''

        self.n_agents = n_agents
        self.dim_local_state = dim_local_state
        self.dim_local_action = dim_local_action
        self.discrete_action = discrete_action # If we use a simplified discretized actions space
        self.discount = discount
        self.epochs = epochs

        # List of NN that estimate Q
        self.criticsNN = [CriticNN(dim_local_state + dim_local_action, output_size=1) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.criticsNN[i].parameters(),lr = learning_rate_critic) for i in range(n_agents)]

    def forward(self, state) -> np.ndarray:
        ''' Function that calculates the actions to take from the state (control law) '''
        pass

    def train_cirtic(self, buffers: deque):
        epochs = self.epochs

        for i in range(self.n_agents):
            # NN for this agent:
            criticNN = self.criticsNN[i]
            critic_optimizer = self.critic_optimizers[i]

            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a)
            inputs = np.column_stack((states,actions))
            inputs = torch.tensor(inputs, dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()

            for epoch in range(epochs):
                ### Perfrom omega (critic) update:
                # Set gradient to zero
                critic_optimizer.zero_grad()
                # value function: # calculate the approximated Q(s,a) = NN(input)
                Q_approx = criticNN(inputs).squeeze()
                # Compute MSE loss
                loss = nn.functional.mse_loss(Q_approx, Gt)
                # Compute gradient
                loss.backward()
                # Clip gradient norm to avoid infinite gradient
                nn.utils.clip_grad_norm_(criticNN.parameters(), max_norm=10) 
                # Update
                critic_optimizer.step()

    def benchmark_cirtic(self, buffers: deque, only_one_NN = False):

        Gts = deque() # for debug, delete after
        Q_approxs = deque() # for debug, delete after
        criticNN= self.criticsNN[0]

        for i in range(self.n_agents):
            # NN for this agent:
            if not only_one_NN:
                criticNN = self.criticsNN[i]
            
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a)
            inputs = np.column_stack((states,actions))
            inputs = torch.tensor(inputs, dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()

            # value function: # calculate the approximated Q(s,a) = NN(input)
            Q_approx = criticNN(inputs).squeeze()

            Q_approxs.append(Q_approx.detach().numpy())
            Gts.append(Gt_array) # for debug

        # Gts is the simulated Q(st,at) values for each agent
        return Gts, Q_approxs

    def save(self,filename = "network"):
        folder ="NN saved"
        cirtic_name = filename + "-critics.pth"
        # actor_name = filename + "-actors_list.pth"
        torch.save(self.criticsNN, os.path.join(folder,cirtic_name))
        print(f'Saved Critic NN as {cirtic_name}')
        # torch.save(self.actorNN, actor_name)
        # print(f'Saved Actor NN as {actor_name}')


class CriticNN(nn.Module):
    """ Create local critic network
    """
    def __init__(self, input_size, output_size = 1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        # NN sizes: define size of hidden layer
        L1 = 400
        L2 = 200

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, L1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers. 1
        self.hidden_layer1 = nn.Linear(L1, L2)
        self.hidden_layer1_activation = nn.ReLU()

        # Create output layer. NO ACTIVATION
        self.output_layer = nn.Linear(L2, output_size)

    def forward(self, z):
        '''z must be a properly formated vector of z (torch tensor)'''
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(z)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden_layer1(l1)
        l2 = self.hidden_layer1_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out
        

class ExperienceBuffers:
    """ List of buffers for each agent.
        each agent has its own buffer: i: ['z_state', 'action', 'local_reward', 'next_z', 'is_finished']
        to get data, example: buffers.buffers[0][5].action
    """
    def __init__(self, n_agents):
        # Create buffer for each agent
        self.buffers = [deque() for i in range(n_agents)]
        self.n_agents = n_agents
        self.experience = namedtuple('experience',
                            ['z_state', 'action', 'reward', 'next_z', 'finished'])

    def append(self,z_states, actions, rewards, new_z, finished):
        # Append experience to the buffer
        for i in range(self.n_agents):
            # Create localized expereince touple. Also, flatten state and action vectors
            exp = self.experience(z_states[i].flatten(), actions[i].flatten(), rewards[i], new_z[i].flatten(), finished)
            self.buffers[i].append(exp)

    def __len__(self):
        # overload len operator
        return len(self.buffers[0])