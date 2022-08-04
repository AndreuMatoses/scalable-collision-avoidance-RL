import os
from turtle import forward
import numpy as np
from autograd import numpy as anp
from autograd import grad
import torch
# import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

"""  Classes used for agents and other useful functions  """

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

class NormalActorNN(nn.Module):
    """ NN for a policy that as input takes the z state and outputs 2D means and sigma of a independent normal distributions
        In this case: z[1x6] -> mu[1x2], sigma^2[1x2]
    """
    def __init__(self, input_size, lr ,dim_action):
        super().__init__()

        self.dim_action = dim_action
        # NN sizes: define size of layers
        Ls = 400
        hidden_1= 200
        hidden_2= 200

        # Ls, Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, Ls)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layer, Ls- > head1
        self.hidden_layer1 = nn.Linear(Ls, hidden_1)
        self.hidden_layer1_activation = nn.ReLU()
        # Create hidden layer, Ls- > head2
        self.hidden_layer2 = nn.Linear(Ls, hidden_2)
        self.hidden_layer2_activation = nn.ReLU()

        # ctreate output layers for each head: out_1: means (tanh in [-1,1]). out_2: sigma^2 (sigmoid in [0,1])
        self.out_1 = nn.Linear(hidden_1,dim_action)
        self.out_1_activation = nn.Tanh()
        self.out_2 = nn.Linear(hidden_2,dim_action)
        self.out_2_activation = nn.Sigmoid()

        # Use adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = lr)

    def forward(self, z):
        # Compute first layer
        l1 = self.input_layer(z)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2_head1 = self.hidden_layer1(l1)
        l2_head1 = self.hidden_layer1_activation(l2_head1)

        l2_head2 = self.hidden_layer2(l1)
        l2_head2 = self.hidden_layer2_activation(l2_head2)

        # Compute output layers
        out_1 = self.out_1(l2_head1)
        out_1 = self.out_1_activation(out_1)

        out_2 = self.out_2(l2_head2)
        out_2 = self.out_2_activation(out_2)

        # out_1 = mu, out_2 = sigma^2
        return out_1,out_2

    def sample_action(self,z:np.ndarray, N=None):
        state_tensor = torch.tensor(z, dtype=torch.float32)
        mu_tensor,sigma_tensor = self.forward(state_tensor)
        # Normally distributed value with the mu and sigma (std^2) from ActorNN
        std = np.sqrt(sigma_tensor.detach().numpy())
        action = np.random.normal(mu_tensor.detach().numpy(),std)
        # action = np.clip(action,-1,1)
        return action

    def log_p_of_a(self,z:np.ndarray, a:np.ndarray):
        state_tensor = torch.tensor(z, dtype=torch.float32)
        mu , sigma = self.forward(state_tensor)
        if mu.dim() == 1:
            p1 = torch.pow(2 * np.pi * sigma[0], -0.5) * torch.exp(-(a[0] - mu[0])**2 / (2 * sigma[0]))
            p2 = torch.pow(2 * np.pi * sigma[1], -0.5) * torch.exp(-(a[1] - mu[1])**2 / (2 * sigma[1]))
        else:
            p1 = torch.pow(2 * np.pi * sigma[:,0], -0.5) * torch.exp(-(a[:,0] - mu[:,0])**2 / (2 * sigma[:,0]))
            p2 = torch.pow(2 * np.pi * sigma[:,1], -0.5) * torch.exp(-(a[:,1] - mu[:,1])**2 / (2 * sigma[:,1]))
        
        p = p1*p2
        return torch.log(p)
        
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

        self.take_all_states = False
        self.dim = output_size
        self.z_dim = input_size

        # param =anp.array([-1.6,-1.6,-1.6])
        param =-anp.ones(int(self.z_dim/self.dim))*0

        self.parameters = param
        if Sigma is None:
            self.Sigma = anp.eye(self.dim)*0.3
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
        if self.take_all_states:
            idx = anp.ones(int(self.z_dim/self.dim))
        else:
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
        if self.take_all_states:
            idx = anp.ones(int(self.z_dim/self.dim))
        else:
            idx = np.arange(1,int(self.z_dim/self.dim+1))<=len(Ni)

        variable = self.parameters
        R0 = anp.array([[anp.cos(variable[0]), -anp.sin(variable[0])],[anp.sin(variable[0]),anp.cos(variable[0])]])*idx[0]
        R1 = anp.array([[anp.cos(variable[1]), -anp.sin(variable[1])],[anp.sin(variable[1]),anp.cos(variable[1])]])*idx[1]
        R2 = anp.array([[anp.cos(variable[2]), -anp.sin(variable[2])],[anp.sin(variable[2]),anp.cos(variable[2])]])*idx[2]
        # R3 = anp.array([[anp.cos(variable[3]), -anp.sin(variable[3])],[anp.sin(variable[3]),anp.cos(variable[3])]])*idx[3]
        R = anp.concatenate((R0,R1,R2),1)

        mu = (R @ z).flatten()
        
        z.shape = (np.size(z),)
        a = np.random.multivariate_normal(mu, self.Sigma)
        
        # Clip the action to not have infinite action
        return np.clip(a,-2,+2)
        

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

class DiscreteSoftmaxNN(nn.Module):
    """ NN for a policy that as input takes the z state and outputs 2D means and sigma of a independent normal distributions
        In this case: z[1x6] -> mu[1x2], sigma^2[1x2]
        NN for the policy with finite actions, in which the input is the state and the output are softmax probabilities for each of the actions
    """
    def __init__(self, input_size, lr , n_actions = 8):
        super().__init__()

        # Create action dictionary action_arg -> 2D continuos action array
        action_norm = 1
        action_list = []
        for n_action in range(n_actions):
            ax = np.cos(n_action/n_actions * 2*np.pi)*action_norm
            ay = np.sin(n_action/n_actions * 2*np.pi)*action_norm
            action_list.append(np.array([ax,ay]))
        self.action_list = np.array(action_list)

        # Structure z -> Ls[ReLu] -> hidden1[ReLu] -> out[softmax]
        Ls = 300
        hidden_1 = 300
        self.n_actions = n_actions

        # Ls, Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, Ls)
        self.input_layer_activation = nn.ReLU()
        # Create hidden layer, Ls- > head1
        self.hidden_layer1 = nn.Linear(Ls, hidden_1)
        self.hidden_layer1_activation = nn.ReLU()
        # ctreate output layers with softmax -> probabilities
        self.out_1 = nn.Linear(hidden_1,n_actions)
        self.out_1_activation = nn.Softmax(dim=0)

        # Use adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = lr)

    def forward(self,z):
        # Compute first layer
        l1 = self.input_layer(z)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden_layer1(l1)
        l2 = self.hidden_layer1_activation(l2)

        # Compute output layers
        out_1 = self.out_1(l2)
        out_1 = self.out_1_activation(out_1)

        return out_1

    def sample_action(self,z:np.ndarray, N=None):
        state_tensor = torch.tensor(z, dtype=torch.float32)
        probs = self.forward(state_tensor)
        chosen_action_arg = np.random.choice(self.n_actions, p = probs.squeeze().detach().numpy())

        return self.action_list[chosen_action_arg]

    def log_p_of_a(self,z:np.ndarray, a:np.ndarray):
        # Find argument corresponding to a array
        action_arg = np.where((self.action_list[:,0] == a[0]) & (self.action_list[:,1]==a[1]))[0][0]

        state_tensor = torch.tensor(z, dtype=torch.float32)
        probs = self.forward(state_tensor)
        log_prob = torch.log(probs.squeeze()[action_arg])
        return log_prob
