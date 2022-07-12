
from enum import auto
import os
import numpy as np
from autograd import numpy as anp
from autograd import grad
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
    def __init__(self, file_name:str, n_agents = "auto", discount = 0.99):

        file_name = os.path.join("models", file_name)
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
            states, actions, rewards, new_states, Ni, inished = zip(*buffer)

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

    def __init__(self, n_agents, dim_local_state, dim_local_action, discount, epochs, learning_rate_critic = 10**(-3), learning_rate_actor = 10**(-3), policy_type = "normal") -> None:
        '''* dim_local_state is the total size of the localized vector that the input of the Q and pi approximations use, i.e (k+1)*dim'''

        self.n_agents = n_agents
        self.dim_local_state = dim_local_state
        self.dim_local_action = dim_local_action
        self.policy_type = policy_type # What kind of policy (NN, stochastic normal dist, etc...)
        self.discount = discount
        self.epochs = epochs

        # Define policy (actor)
        self.actors = [NormalPolicy(dim_local_state,dim_local_action) for i in range(n_agents)]
        self.learning_rate_actor = learning_rate_actor

        # List of NN that estimate Q
        self.criticsNN = [CriticNN(dim_local_state + dim_local_action, output_size=1) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.criticsNN[i].parameters(),lr = learning_rate_critic) for i in range(n_agents)]

    def forward(self, z_states, Ni) -> list:
        ''' Function that calculates the actions to take from the z_states list (control law) 
            actions: list of row vectors [u1^T, u2^T,...]'''

        actions = deque()
        for i in range(self.n_agents):
            z_state = z_states[i].flatten()
            actions.append(self.actors[i].sample_action(z_state, Ni))

        return actions

    def train(self, buffers: deque, actor_lr = None, return_grads = False):
        epochs = self.epochs

        if actor_lr is not None:
            self.learning_rate_actor = actor_lr

        # CRITIC LOOP
        for i in range(self.n_agents):
            # NN for this agent:
            criticNN = self.criticsNN[i]
            critic_optimizer = self.critic_optimizers[i]

            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a) [200x8]
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
        
        # ACTOR LOOP
        grad_norms = []
        gi_norms = []
        for i in range(self.n_agents):
            # to access buffer data: buffers.buffers[i][t].action, namedtuple('experience', ['z_state', 'action', 'reward', 'next_z', 'Ni', 'finished'])
            
            actor = self.actors[i]
            gi = 0 #initialize to 0

            for t in range(T):
                zit = buffers.buffers[i][t].z_state
                ait = buffers.buffers[i][t].action
                Nit = buffers.buffers[i][t].Ni
                
                grad_actor = actor.compute_grad(zit,ait, [1,2,3])
                # PUT Ni HERE INSTEAD of [1,2,3]
                # grad_actor = actor.clip_grad_norm(grad_actor,clip_norm=100)

                Qj_sum = 0
                for j in Nit: # REMOVE THE [0]
                    zjt = buffers.buffers[j][t].z_state
                    ajt = buffers.buffers[j][t].action
                    Q_input_tensor =  torch.tensor(np.hstack((zjt,ajt)), dtype=torch.float32)
                    Qj = self.criticsNN[j](Q_input_tensor).detach().numpy()
                    Qj_sum += Qj[0]

                gi += self.discount**t * 1/self.n_agents* grad_actor * Qj_sum

            # Update policy parameters with approx gradient gi (clipped to avoid infinity gradients)
            gi = actor.clip_grad_norm(gi, clip_norm=50)
            # MAKE SURE TO CLIP THE PARAMS from 0 to 2*Pi
            actor.parameters =  actor.parameters + self.learning_rate_actor*gi
            # actor.parameters =  np.clip(actor.parameters + self.learning_rate_actor*gi, -2*np.pi, 2*np.pi)

            # print(f"grad norms gi={np.linalg.norm(gi.flatten())}")
            if return_grads:
                grad_norms.append(np.linalg.norm(grad_actor.flatten()))
                gi_norms.append(np.linalg.norm(gi.flatten()))

        if return_grads:        
            return grad_norms, gi_norms
    
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
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a)
            inputs = np.column_stack((states,actions))
            inputs = torch.tensor(inputs, dtype=torch.float32)
            ## Actor update

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
        folder ="models"
        cirtic_name = filename + "-critics.pth"
        actors_name = filename + "-actors.pth"

        torch.save(self.criticsNN, os.path.join(folder,cirtic_name))
        print(f'Saved Critic NNs as {cirtic_name}')
        torch.save(self.actors, os.path.join(folder,actors_name))
        print(f'Saved Actors List as {actors_name}')


class CriticNN(nn.Module):
    """ Create local critic network
    """
    # NN sizes: define size of hidden layer
    L1 = 200
    L2 = 200

    def __init__(self, input_size, output_size = 1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, self.L1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers. 1
        self.hidden_layer1 = nn.Linear(self.L1, self.L2)
        self.hidden_layer1_activation = nn.ReLU()

        # Create output layer. NO ACTIVATION
        self.output_layer = nn.Linear(self.L2, output_size)

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
        
class NormalPolicy:
    """Policy that uses a multivatriable normal distribution.
        The parameters are theta, which the angles of the Rot mat for each vector:
        parameters: mu = theta * z
        gradient: w.r.t theta
        covariance matrix: Constant for now, not a parameter

        CAREFUL: Changes the shape of z,a inputs to columns

        NOTICE: individual values of p(a|z) can be greater than 1, as this is a density funct.  (pdf, continous)
        the pdf of a singular point makes no sense, neeeds to be over a differential of a (i.e pdf is per unit lenght)
    """
    def __init__(self, input_size, output_size = 2, Sigma = None) -> None:
        self.dim = output_size
        self.z_dim = input_size

        param =anp.zeros(int(self.z_dim/self.dim))

        self.parameters = param
        if Sigma is None:
            self.Sigma = anp.eye(self.dim)*0.2
        else:
            self.Sigma = Sigma

    def p_of_a(self, z:np.ndarray, a:np.ndarray) -> np.ndarray:
        ''' a needs to be a row vector (1D flat)
            z needs to be a row vector (1D flat)
        '''
        pass

    def compute_grad(self, z:np.ndarray, a:np.ndarray, Ni):
        ''' a needs to be a row vector (1D flat)
            z needs to be a row vector (1D flat)
            Ni indicates the states that are neighbors
        '''
        # Make vectors proper shape (column, for math)
        z.shape = (np.size(z),1)
        a.shape = (np.size(a),1)

        # Used to only calculate the gradient of the states that actually count
        idx = np.arange(1,int(self.z_dim/self.dim+1))<=len(Ni)

        # Define scalar function to which apply numerical gradient: https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
        def my_fun(variable):
            R0 = anp.array([[anp.cos(variable[0]), -anp.sin(variable[0])],[anp.sin(variable[0]),anp.cos(variable[0])]])*idx[0]
            R1 = anp.array([[anp.cos(variable[1]), -anp.sin(variable[1])],[anp.sin(variable[1]),anp.cos(variable[1])]])*idx[1]
            R2 = anp.array([[anp.cos(variable[2]), -anp.sin(variable[2])],[anp.sin(variable[2]),anp.cos(variable[2])]])*idx[2]
            R = anp.concatenate((R0,R1,R2),1)

            return (-1/2*(a- R @ z).T @ np.linalg.inv(self.Sigma) @ (a- R @ z))[0,0]
        
        grad_fun = grad(my_fun)
        self.grad = grad_fun(self.parameters)
        z.shape = (np.size(z),)
        a.shape = (np.size(a),)

        return self.grad
    
    def clip_grad_norm(self, grad:np.ndarray, clip_norm:float):
        # If the gradient norm is to be clipped to a value:
        grad_norm = np.linalg.norm(grad.flatten())
        # If the current norm is less than the clipping, do nothing. If more, make the norm=cliped_norm
        if grad_norm <= clip_norm:
            return grad
        else:
            return grad * clip_norm/grad_norm


    def sample_action(self, z:np.ndarray, Ni):
        # Maybe add a mask so that null states are not accounted 
        z.shape = (np.size(z),1)

        # Used to only calculate the gradient of the states that actually count
        idx = np.arange(1,int(self.z_dim/self.dim+1))<=len(Ni)

        variable = self.parameters
        R0 = anp.array([[anp.cos(variable[0]), -anp.sin(variable[0])],[anp.sin(variable[0]),anp.cos(variable[0])]])*idx[0]
        R1 = anp.array([[anp.cos(variable[1]), -anp.sin(variable[1])],[anp.sin(variable[1]),anp.cos(variable[1])]])*idx[1]
        R2 = anp.array([[anp.cos(variable[2]), -anp.sin(variable[2])],[anp.sin(variable[2]),anp.cos(variable[2])]])*idx[2]
        R = anp.concatenate((R0,R1,R2),1)

        mu = (R @ z).flatten()
        
        z.shape = (np.size(z),)
        a = np.random.multivariate_normal(mu, self.Sigma)
        
        # Clip the action to not have infinite action
        return np.clip(a,-1,+1)
        

class ExperienceBuffers:
    """ List of buffers for each agent.
        each agent has its own buffer: i: ['z_state', 'action', 'local_reward', 'next_z', 'is_finished']
        to get data, example: buffers.buffers[i][t].action
    """
    def __init__(self, n_agents):
        # Create buffer for each agent
        self.buffers = [deque() for i in range(n_agents)]
        self.n_agents = n_agents
        self.experience = namedtuple('experience',
                            ['z_state', 'action', 'reward', 'next_z', 'Ni', 'finished'])

    def append(self,z_states, actions, rewards, new_z, Ni, finished):
        # Append experience to the buffer
        for i in range(self.n_agents):
            # Create localized expereince touple. Also, flatten state and action vectors
            exp = self.experience(z_states[i].flatten(), actions[i].flatten(), rewards[i], new_z[i].flatten(), Ni[i], finished)
            self.buffers[i].append(exp)

    def __len__(self):
        # overload len operator
        return len(self.buffers[0])