import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from utils import *

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
    def __init__(self, critics_name:str, actors_name:str, n_agents = "auto", discount = 0.99):

        file_name_critics = os.path.join("models", critics_name)
        # Load critic
        try:
            criticsNN = torch.load(file_name_critics)
            print(f'Loaded Critic, n_agents = {len(criticsNN)}, discount = {discount}. Network model[0]: {criticsNN[0]}')
        except:
            print(f'File {file_name_critics} not found!')
            exit(-1)

        self.criticsNN = criticsNN
        self.critics_name = critics_name

        if n_agents == "auto":
            self.n_agents = len(criticsNN)
        else:
            self.n_agents = n_agents

        # load actor
        file_name_actors = os.path.join("models", actors_name)

        try:
            actors = torch.load(file_name_actors)
            print(f'Loaded actors, n_agents = {len(criticsNN)}, discount = {discount}. Type: {type(actors[0])}')
        except:
            print(f'File {file_name_actors} not found!')
            exit(-1)
        self.actors = actors
        self.actors_name = actors_name

        self.discount = discount # to benchmark the critic

    def forward(self, z_states: list, N:list):
        """ Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        """
        actions = []
        if type(self.actors[0]) is NormalPolicy or type(self.actors[0]) is NormalActorNN or type(self.actors[0]) is DiscreteSoftmaxNN:
            # z_state in this case
            for i in range(self.n_agents):
                z_state = z_states[i].flatten()
                Ni = N[i]
                if i < len(self.actors):
                    actor = self.actors[i]
                else:
                    actor = self.actors[0]

                action = actor.sample_action(z_state, Ni)
                actions.append(action)
        else:
            print(f"Error type of policy {type(self.actors[0])}")

        return actions

    def benchmark_cirtic(self, buffers: deque, only_one_NN = False):

        Gts = deque() # for debug, delete after
        V_approxs = deque() # for debug, delete after
        criticNN= self.criticsNN[0]

        for i in range(self.n_agents):
            # NN for this agent:
            if not only_one_NN:
                if i < len(self.criticsNN):
                    criticNN = self.criticsNN[i]
                else:
                    criticNN = self.criticsNN[0]
            
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a) [200x8]. If instead V(s), input = [s]
            # inputs = np.column_stack((states,actions))
            # inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = torch.tensor(np.array(states), dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()

            # value function: # calculate the approximated Q(s,a) = NN(input)
            V_approx = criticNN(inputs).squeeze()

            V_approxs.append(V_approx.detach().numpy())
            Gts.append(Gt_array) # for debug

        # Gts is the simulated Q(st,at) values for each agent. Q(s,a)-V(s) ~= Gt-V(st) = A(s,a)
        return Gts, V_approxs

class SA2CAgents:

    def __init__(self, n_agents, dim_local_state, dim_local_action, discount, epochs, learning_rate_critic = 10**(-3), learning_rate_actor = 10**(-3)) -> None:
        '''* dim_local_state is the total size of the localized vector that the input of the Q and pi approximations use, i.e (k+1)*dim'''

        self.n_agents = n_agents
        self.dim_local_state = dim_local_state
        self.dim_local_action = dim_local_action
        # self.policy_type = policy_type # What kind of policy (NN, stochastic normal dist, etc...)
        self.discount = discount
        self.epochs = epochs

        # preload_NN = "models\\final\\cont_n5"
        preload_NN = None
        # Define policy (actor)
        if preload_NN is None:
             # self.actors = [NormalPolicy(dim_local_state,dim_local_action) for i in range(n_agents)]
            self.actors = [DiscreteSoftmaxNN(dim_local_state, lr = learning_rate_actor, n_actions=16) for i in range(n_agents)]
            # self.actors = [NormalActorNN(dim_local_state, lr = learning_rate_actor, dim_action=dim_local_action) for i in range(n_agents)]
            self.learning_rate_actor = learning_rate_actor

            # List of NN that estimate Q (or V if we use advantage)
            # self.criticsNN = [CriticNN(dim_local_state + dim_local_action, output_size=1) for i in range(n_agents)]
            self.criticsNN = [CriticNN(dim_local_state, output_size=1) for i in range(n_agents)]
            self.critic_optimizers = [optim.Adam(self.criticsNN[i].parameters(),lr = learning_rate_critic) for i in range(n_agents)]
        else:
            try:
                actors = torch.load(preload_NN + "-A2Cactors.pth")
                print(f'Loaded actors, n_agents = {len(actors)}, discount = {discount}. Type: {type(actors[0])}')
            except:
                print(f'File {preload_NN + "-A2Cactors.pth"} not found!')
                exit(-1)
            self.actors = actors
            self.learning_rate_actor = learning_rate_actor
            try:
                criticsNN = torch.load(preload_NN + "-A2Ccritics.pth")
                print(f'Loaded Critic, n_agents = {len(criticsNN)}, discount = {discount}. Network model[0]: {criticsNN[0]}')
            except:
                print(f'File {preload_NN + "-A2Ccritics.pth"} not found!')
                exit(-1)
            self.criticsNN = criticsNN
            self.critic_optimizers = [optim.Adam(self.criticsNN[i].parameters(),lr = learning_rate_critic) for i in range(n_agents)]


    def forward(self, z_states, N) -> list:
        ''' Function that calculates the actions to take from the z_states list (control law) 
            actions: list of row vectors [u1^T, u2^T,...]'''

        actions = deque()
        for i in range(self.n_agents):
            z_state = z_states[i].flatten()
            actions.append(self.actors[i].sample_action(z_state, N[i]))
            # actions.append(self.actors[i].sample_action(z_state, [1]))

        return actions

    def train_designed_policy(self, buffers: deque, actor_lr = None, return_grads = False):
        epochs = self.epochs

        if actor_lr is not None:
            self.learning_rate_actor = actor_lr

        Gts = deque() # for debug, delete after -> acces data Gts[i][t]

        # CRITIC LOOP
        for i in range(self.n_agents):
            # NN for this agent:
            criticNN = self.criticsNN[i]
            critic_optimizer = self.critic_optimizers[i]

            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a) [200x8]. If instead V(s), input = [s]
            # inputs = np.column_stack((states,actions))
            # inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = torch.tensor(np.array(states), dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()
            Gts.append(Gt_array) # for debug

            ### Perfrom omega (critic) update:
            # Set gradient to zero
            critic_optimizer.zero_grad()
            # value function: # calculate the approximated V(s) = NN(input)
            V_approx = criticNN(inputs).squeeze()
            # Compute MSE loss, as E[Gt-V(s) = A(s,a)] = 0
            loss = nn.functional.mse_loss(V_approx, Gt)
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
            # Gt: Gts[i][t]
            actor = self.actors[i]
            gi = 0 #initialize to 0

            for t in range(T):
                zit = buffers.buffers[i][t].z_state
                ait = buffers.buffers[i][t].action
                Nit = buffers.buffers[i][t].Ni
                
                grad_actor = actor.compute_grad(zit,ait, Nit)
                # PUT Nit HERE INSTEAD of [1,2,3]
                # grad_actor = actor.clip_grad_norm(grad_actor,clip_norm=100)

                # Qj_sum = 0
                Advantage_j_sum = 0
                input_tensor =  torch.tensor(zit, dtype=torch.float32)
                # Baseline is the Vi(s) for current agent. reduce variance and complexity
                Vi_baseline = self.criticsNN[i](input_tensor).detach().numpy()[0]
                for j in Nit: # i included here
                    zjt = buffers.buffers[j][t].z_state
                    ajt = buffers.buffers[j][t].action
                    # Q_input_tensor =  torch.tensor(np.hstack((zjt,ajt)), dtype=torch.float32)
                    # Qj = self.criticsNN[j](Q_input_tensor).detach().numpy()[0]
                    # input_tensor =  torch.tensor(zjt, dtype=torch.float32)
                    # Vj = self.criticsNN[j](input_tensor).detach().numpy()[0]
                    Advantage_j_sum += (Gts[j][t] - Vi_baseline)
                    # Qj_sum += Gts[j][t]

                # gi += self.discount**t * 1/self.n_agents* grad_actor * Qj_sum
                gi += self.discount**t * 1/self.n_agents* grad_actor * Advantage_j_sum

            # Update policy parameters with approx gradient gi (clipped to avoid infinity gradients)
            gi = actor.clip_grad_norm(gi, clip_norm=100)
            # MAKE SURE TO CLIP THE PARAMS from 0 to 2*Pi
            actor.parameters =  actor.parameters + self.learning_rate_actor*gi
            # actor.parameters =  np.clip(actor.parameters + self.learning_rate_actor*gi, -2*np.pi, 2*np.pi)

            # print(f"grad norms gi={np.linalg.norm(gi.flatten())}")
            if return_grads:
                grad_norms.append(np.linalg.norm(grad_actor.flatten()))
                gi_norms.append(np.linalg.norm(gi.flatten()))

        if return_grads:        
            return grad_norms, gi_norms

    def train_NN(self, buffers: deque, actor_lr = None):
        epochs = self.epochs

        if actor_lr is not None:
            self.learning_rate_actor = actor_lr

        Gts = deque() # for debug, delete after -> acces data Gts[i][t]
        T = len(buffers)

        # CRITIC LOOP
        for i in range(self.n_agents):
            # NN for this agent:
            criticNN = self.criticsNN[i]
            critic_optimizer = self.critic_optimizers[i]

            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a) [200x8]. If instead V(s), input = [s]
            inputs = torch.tensor(np.array(states), dtype=torch.float32)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()
            Gts.append(Gt_array) # for debug

            ## Perfrom omega (critic) update:
            # Set gradient to zero
            critic_optimizer.zero_grad()
            # value function: # calculate the approximated V(s) = NN(input)
            V_approx = criticNN(inputs).squeeze()
            # Compute MSE loss, as E[Gt-V(s) = A(s,a)] = 0
            loss = nn.functional.mse_loss(V_approx, Gt)
            # Compute gradient
            loss.backward()
            # Clip gradient norm to avoid infinite gradient
            nn.utils.clip_grad_norm_(criticNN.parameters(), max_norm=10) 
            # Update
            critic_optimizer.step()
        
        # ACTOR LOOP
        for i in range(self.n_agents):
            # to access buffer data: buffers.buffers[i][t].action, namedtuple('experience', ['z_state', 'action', 'reward', 'next_z', 'Ni', 'finished'])
            # Gt: Gts[i][t]
            actor = self.actors[i]
            actor_loss = torch.tensor(0, dtype=torch.float32)

            for t in range(T):
                zit = buffers.buffers[i][t].z_state
                ait = buffers.buffers[i][t].action
                Nit = buffers.buffers[i][t].Ni
                
                log_prob_tensor = actor.log_p_of_a(zit,ait)

                Advantage_j_sum = 0
                input_tensor =  torch.tensor(zit, dtype=torch.float32)
                # Baseline is the Vi(s) for current agent. reduce variance and complexity
                Vi_baseline = self.criticsNN[i](input_tensor).detach().numpy()[0]
                # Advantage_j_sum += (Gts[i][t] - Vi_baseline)
                for j in Nit: # i included here
                    Advantage_j_sum += (Gts[j][t] - Vi_baseline)
                    # Advantage_j_sum += (Gts[j][t])

                # gi += self.discount**t * 1/self.n_agents* grad_actor * Qj_sum
                # actor_loss = actor_loss -  self.discount**t * 1/self.n_agents* log_prob_tensor * Advantage_j_sum
                actor_loss = actor_loss -log_prob_tensor * 1/self.n_agents * self.discount**t * Advantage_j_sum

            # Update policy parameters
            actor.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10)
            actor.optimizer.step()

    
    def benchmark_cirtic(self, buffers: deque, only_one_NN = False):

        Gts = deque() # for debug, delete after
        V_approxs = deque() # for debug, delete after
        criticNN= self.criticsNN[0]

        for i in range(self.n_agents):
            # NN for this agent:
            if not only_one_NN:
                criticNN = self.criticsNN[i]
            
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a)
            # inputs = np.column_stack((states,actions))
            inputs = torch.tensor(np.array(states), dtype=torch.float32)
            ## Actor update

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Gt = torch.tensor(Gt_array, dtype=torch.float32).squeeze()

            # value function: # calculate the approximated Q(s,a) ~= Gt = V(s) + A(s,a) => Gt-V(s) = A(s,a)
            V_approx = criticNN(inputs).squeeze()
            V_approxs.append(V_approx.detach().numpy())
            # Q_approxs.append(Gt_array)
            Gts.append(Gt_array) # for debug

        # Gts is the simulated Q(st,at) values for each agent. Q(s,a)-V(s) ~= Gt-V(st) = A(s,a)
        return Gts, V_approxs

    def save(self,filename = "network"):
        folder ="models"
        cirtic_name = filename + "-A2Ccritics.pth"
        actors_name = filename + "-A2Cactors.pth"

        torch.save(self.criticsNN, os.path.join(folder,cirtic_name))
        print(f'Saved Critic NNs as {cirtic_name}')
        torch.save(self.actors, os.path.join(folder,actors_name))
        print(f'Saved Actors List as {actors_name}')


class SPPOAgents:
    def __init__(self, n_agents, dim_local_state, dim_local_action, discount, epochs, learning_rate_critic = 10**(-3), learning_rate_actor = 10**(-3), epsilon = 0.2) -> None:
        '''* dim_local_state is the total size of the localized vector that the input of the Q and pi approximations use, i.e (k+1)*dim'''

        self.n_agents = n_agents
        self.dim_local_state = dim_local_state
        self.dim_local_action = dim_local_action
        # self.policy_type = policy_type # What kind of policy (NN, stochastic normal dist, etc...)
        self.discount = discount
        self.epochs = epochs
        self.epsilon = epsilon

        # Define actor networks
        self.actorsNN = [NormalActorNN(dim_local_state, dim_action=dim_local_action) for i in range(n_agents)]
        self.learning_rate_actor = learning_rate_actor
        self.actor_optimizers = [optim.Adam(self.actorsNN[i].parameters(),lr = learning_rate_actor) for i in range(n_agents)]


        # List of NN that estimate Q (or V if we use advantage)
        # self.criticsNN = [CriticNN(dim_local_state + dim_local_action, output_size=1) for i in range(n_agents)]
        self.criticsNN = [CriticNN(dim_local_state, output_size=1) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.criticsNN[i].parameters(),lr = learning_rate_critic) for i in range(n_agents)]

    def forward(self, z_states, N) -> list:
        ''' Function that calculates the actions to take from the z_states list (control law) 
            actions: list of row vectors [u1^T, u2^T,...]'''

        actions = deque()
        for i in range(self.n_agents):
            z_state = z_states[i].flatten()
            actorNN = self.actorsNN[i]

            state_tensor = torch.tensor(z_state, dtype=torch.float32)
            mu_tensor,sigma_tensor = actorNN(state_tensor)

            # Normally distributed value with the mu and sigma (std^2) from ActorNN
            std = np.sqrt(sigma_tensor.detach().numpy())
            action = np.random.normal(mu_tensor.detach().numpy(),std)

            # Acion must be between -1,1
            actions.append(np.clip(action,-1,1))

        return actions

    def train(self, buffers: deque, actor_lr = None, return_grads = False):
        epochs = self.epochs

        if actor_lr is not None:
            self.learning_rate_actor = actor_lr

        T = len(buffers)
        # Gts = deque() # for debug, delete after -> acces data Gts[i][t]
        Git = np.zeros([self.n_agents, T]) # Git[i,t]
        Qjsum_estim = np.zeros([self.n_agents, T]) # Qjsum_estim[i,t]
        p_old_it = np.zeros([self.n_agents, T]) # p_old_it[i,t]
        

        # Agents LOOP, to create required variables
        for i in range(self.n_agents):
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Calculate the simulated Q value (target), Monte carlo Gt
            # Going backwards, G(t) = gamma * G(t+1) + r(t), with G(T)=r(T)
            # T = len(rewards)
            Gt_array = np.zeros([T])
            Gt_array[-1] = rewards[-1]
            for t in range(T-2,-1,-1):
                Gt_array[t]  = Gt_array[t+1]*self.discount + rewards[t]

            Git[i,:] = Gt_array
        
        # Calculate the advantage estimator (attempt), and old p of a
        for i in range(self.n_agents):
            # separate data from experience buffer
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)

            # Create input tensor, This one: input = [s,a] tensor -> Q(s,a) [200x8]. If instead V(s), input = [s]
            # inputs = torch.tensor(np.column_stack((states,actions)), dtype=torch.float32)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
           
            p_old_it[i,:] = self.probability_of_ai(states_tensor, actions_tensor, i).detach().numpy()
            Vi_baselines = self.criticsNN[i](states_tensor).squeeze().detach().numpy()
            # advantage_estim[i,:] = -Vi_baselines

            for t in range(T):
                Nit = buffers.buffers[i][t].Ni
                for j in Nit: # i included here, only uses local i
                    Qjsum_estim[i,t] += Git[j,t]


        ### Training LOOP, per agent: ###
        for i in range(self.n_agents):
            buffer = buffers.buffers[i]
            states, actions, rewards, new_states, Ni, finished = zip(*buffer)
            Gt = torch.tensor(Git[i,:],dtype=torch.float32)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)

            Vi_baselines = self.criticsNN[i](states_tensor).squeeze()
            Adv = Qjsum - Vi_baselines
            Qjsum =  torch.tensor(Qjsum_estim[i,:], dtype=torch.float32)

            # criticNN = self.criticsNN[i]
            # critic_optimizer = self.critic_optimizers[i]
            # actorNN = self.actorsNN[i]
            # actor_optimizer = self.actor_optimizers[i]

            # Perform training for epochs
            for m in range(epochs):

                ### CRITIC update:
                # Set gradient to zero
                self.critic_optimizers[i].zero_grad() 
                # value function: # calculate the approximated V(s) = NN(input)
                V_approx = self.criticsNN[i](states_tensor).squeeze()
                # Compute MSE loss, as E[Gt-V(s) = A(s,a)] = 0
                loss = nn.functional.mse_loss(V_approx, Gt)
                # Compute gradient
                loss.backward()
                # Clip gradient norm to avoid infinite gradient
                nn.utils.clip_grad_norm_(self.criticsNN[i].parameters(), max_norm=10) 
                # Update
                self.critic_optimizers[i].step()

                ### ACTOR update
                # Set gradient to zero
                self.actor_optimizers[i].zero_grad()
                pi_old = torch.tensor(p_old_it[i,:], dtype=torch.float32)
                # Compute new advantage with updated critic
                # Compute r_theta ratio of probabilities
                current_pi = self.probability_of_ai(states_tensor,actions_tensor, i)
                r_theta = current_pi/pi_old
                # Comute loss function - 1/N (min...)
                left_min = r_theta * Adv
                right_min = torch.clamp(r_theta, 1 - self.epsilon, 1 + self.epsilon) * Adv #clipping method for tensors
                loss = - torch.mean(torch.min(left_min,right_min))
                # compute gradient
                loss.backward()
                # Clip gradient norm to avoid infinite gradient
                nn.utils.clip_grad_norm_(self.actorsNN[i].parameters(), max_norm=10) 
                # Update
                self.actor_optimizers[i].step()


    def probability_of_ai(self,states_i,actions_i, agent:int):
        """ all variables are tensors. the time vector for current agent=i
            assume that the two action components are not correlated,
            thus P of both happening is p1*p2
        """
        i = agent
        mu , sigma = self.actorsNN[i](states_i)

        # p_tensor = (2*np.pi*sigma**2)**(-1/2) * torch.exp(-(actions-mu)**2/(2*sigma**2))
        # p = p_tensor[:,0]*p_tensor[:,1] # assuming uncorrelated action

        p1 = torch.pow(2 * np.pi * sigma[:,0], -0.5) * torch.exp(-(actions_i[:,0] - mu[:,0])**2 / (2 * sigma[:,0]))
        p2 = torch.pow(2 * np.pi * sigma[:,1], -0.5) * torch.exp(-(actions_i[:,1] - mu[:,1])**2 / (2 * sigma[:,1]))
        p = p1*p2

        return p      

    def save(self,filename = "network"):
        folder ="models"
        cirtic_name = filename + "-PP0critics.pth"
        actors_name = filename + "-PP0actors.pth"

        torch.save(self.criticsNN, os.path.join(folder,cirtic_name))
        print(f'Saved Critic NNs as {cirtic_name}')
        torch.save(self.actorsNN, os.path.join(folder,actors_name))
        print(f'Saved Actors List as {actors_name}')