
from distutils.log import error
from os import stat
import random
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math

### TO DO ############
"""
- Verify that the z states work as intended
- Proper null state for the z state
- GitHub repo
"""
######################

# Spatial dimension
dim = 2 
# time step in seconds
dt = 0.01 
# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
YELLOW       = "#FFFF00";
BLUE         = '#98F5FF';

def num_to_rgb(val, max_val):
    if (val > max_val):
        raise ValueError("val must not be greater than max_val")
    if (val < 0 or max_val < 0):
        raise ValueError("arguments may not be negative")

    i = (val * 255 / max_val);
    r = round(math.sin(0.024 * i + 0) * 127 + 128)/255;
    g = round(math.sin(0.024 * i + 2) * 127 + 128)/255;
    b = round(math.sin(0.024 * i + 4) * 127 + 128)/255;
    return (r,g,b)

class drones:
    
    def __init__(self, n_agents: int, n_obstacles: int, grid: list, end_formation: str, k_closest = 2, deltas: np.ndarray = None, simplify_zstate = False) -> None:
        """[summary]

        Args:
            i_agents (int): number of agents, integer
            n_obstacles (int): number of obstacles integer
            grid (list): list of the two axis limits delimiting the grid
            end_formation(str): what formation to reach e.g "O" is a circle
            k_closest(int): how many agents to account for in the Delta disk, for having a finite localized state (implementation)
            deltas(ndarray, column vector): vector of Deltas for each agent sensing radius
        """
        self.n_agents = n_agents
        self.grid = grid
        self.goal = self.grid
        self.k_closest = k_closest
        self.deltas = deltas
        self.simplify_zstate = simplify_zstate

        # State space dynamics. For now, all agents have same A,B
        self.A = np.eye(dim)
        self.B = np.eye(dim)*dt

        # Initialize agents and obstacles
        self.obstacles = self.create_obstacles(n_obstacles)
        self.state= self.init_agents(n_agents)
        self.end_points = self.generate_formation(end_formation)

        # Calculate localized states z (uing the reward funciton)
        _, _, z_states = self.rewards(self.state, self.end_points, self.n_agents, self.d_safety, self.deltas)
        self.z_states = z_states

        # self.trajectory = []
        # self.trajectory.append(self.state.copy())

    def reset(self, renew_obstacles = True):
        self.state = self.init_agents(self.n_agents)
        if renew_obstacles == True:
           self.obstacles = self.create_obstacles(self.n_obstacles)

    
    def __str__(self):
        print("Grid size: [x_lim, y_lim]\n",self.grid)
        print("State: [x, y, vx, vy, r]\n", self.state)
        print("Obstacles [x, y, r]:\n",self.obstacles)
        return ""

    def generate_formation(self,end_formation):
        """generate end point coordinates from a description of the end formation shape

        Args:
            end_formation (str): a label given to a shape of final formation, i.e "O" for circle

        Returns:
            formation: [xF1^T, xF2^T,...]^T. Column vector
        """
        if end_formation == "O":
            size = min(self.grid)/2
            angle_step = 2*np.pi/self.n_agents
            formation = np.zeros([self.n_agents*dim,1])

            for i in range(self.n_agents):
                formation[dim*i,0] = np.cos(i * angle_step)*0.9*self.grid[0]/2 + self.grid[0]/2
                formation[dim*i+1,0] = np.sin(i * angle_step)*0.9*self.grid[1]/2 + self.grid[1]/2

        else:
            error(str(end_formation) +" is Not a valid end formation identifier")

        # find maximum allowed d_safety (\hat{d}_i): di <= min(|| xFi -xFj || - li - lj) for all j!=i
        d_safety = np.zeros([self.n_agents,1])

        for i in range(self.n_agents):
            xFi = formation[dim*i:(i+1)*dim,0]
            li = self.state[i,2*dim]
            d_i = np.infty
            for j in range(self.n_agents): 
                if j != i:
                    xFj = formation[dim*j:(j+1)*dim,0]
                    lj = self.state[j,2*dim]
                    d_ij = np.linalg.norm(xFi-xFj) -li -lj
                    d_i = min([d_i,d_ij])

            d_safety[i] = d_i
        self.d_safety = np.floor(d_safety*100)/100

        # formation: [xF1^T, xF2^T,...]^T. Column vector
        return formation

    def create_obstacles(self,n_obstacles):
        self.n_obstacles = n_obstacles

        max_size = 0.1*np.max(self.grid)
        min_size = 0.05*max_size

        # generate random obstacles
        # Generate the obstacle coordinates and size
        # first column: x, second: y, third: r
        obstacles = np.random.rand(n_obstacles,dim+1)
        obstacles[:,0] = obstacles[:,0]*self.grid[0]
        obstacles[:,1] = obstacles[:,1]*self.grid[1]
        obstacles[:,dim] = obstacles[:,dim]*(max_size-min_size) + min_size 

        return obstacles

    def init_agents(self,n_agents):
        # initialize state array:
        # column 1: x, col 2: y, col 3: vx, col 4 vy, col 5: l
        l = 0.1 # 10 cm
        self.n_agents  = n_agents
        self.global_state_space  = n_agents*(2*dim + 1) # x,y, vx,vy, radius

        grid = self.grid

        if self.simplify_zstate:
            # Only take into account position x,y (remove vx vy l)
            self.local_state_space = (dim)*(1+self.k_closest)
        else:
            self.local_state_space = (2*dim+1)*(1+self.k_closest)

        self.global_action_space = n_agents*dim # vx,vy (or ax,ay in the future)
        self.local_action_space = dim

        state = np.zeros([n_agents,5])
        state[:,4] = l

        # Create a grid and choose random nodes without repetition
        delta_l = 2*1.1*l
        divisions = np.floor(np.array(grid)/delta_l)
        possible_coord = []

        for idx in range(int(divisions[0])):
            for jdx in range(int(divisions[1])):
                coord = [idx*delta_l, jdx*delta_l]
                possible_coord.append(coord)

        random_coord = np.array(random.sample(possible_coord, n_agents))
        state[:,0:dim] = random_coord

        return state
    
    def step(self,actions):
        """_summary_

        Args:
            actions (_type_):List of actions, each entry one agent [u1^T, u2^T, ...].Assuming each actions is a row vector with an entry for each dimension for each agent.

        Returns:
            state: new state after applying action
            z_states (list): localized states (from the new state)
            r_vec (row vector): vector for each localized reward for each agent. Average is total reward 
        """

        # Update state: s -> s' with the system dynamics
        for i in range(self.n_agents):

            Ai = self.A
            Bi = self.B

            xi = np.transpose(self.state[i,0:dim])
            ui = actions[i]

            next_xi = np.matmul(Ai,xi) + np.matmul(Bi,ui)

            self.state[i,0:dim] = np.transpose(next_xi)
            self.state[i,dim:2*dim] = np.transpose(ui)


        # Calculate new individual reward [r1(s,a), r2,...] vector, plus related distance dependent values
        r_vec, n_collisions, z_states = self.rewards(self.state, self.end_points, self.n_agents, self.d_safety, self.deltas)
        self.z_states = z_states

        # SHould return (s', r(s,a), n_collisions(s') ,finished)
        finished = False

        # TO DO: Proper is_finished
        return self.state, z_states, r_vec, n_collisions, finished

    def rewards(self, state, end_points, n_agents, d_safety, deltas):
        '''
        state: [column 1: x, col 2: y, col 3: vx, col 4 vy, col 5: r ] np.array[i,2*dim+1]
        end_points: column [x1, y1, x2, y2, ... ]^T
        d_safety: column d_i [d1, d2, ...]^T
        '''
        n_agents = np.size(state,0)

        # weights: q|xi-xF|^2 + b log(d_i/d_ij)
        q = 1 
        b = 1

        xF = np.reshape(end_points,[n_agents,dim])
        xi = state[:,0:dim]

        # row vector, q|xi-xF|^2
        to_goal_cost = q*np.power(np.linalg.norm(xF-xi,axis=1),2)

        # Collision cost
        if deltas == None:
            # In case of no deltas, we assume Dleta = d_safety, i.e no simplification
            deltas = d_safety

        d_ij, log_d, N_delta, collisions = self.distance_data(state,deltas,d_safety)

        collision_cost = b*np.sum(log_d*N_delta,1)
        n_collisions = np.sum(collisions)

        # These are the approximated localized rewards
        reward_vector = -(to_goal_cost+collision_cost)

        # Calculate localized z states
        z_states = self.localized_states(state, end_points, N_delta, d_ij)

        return reward_vector, n_collisions, z_states

    def distance_data(self,state,deltas,d_safety):
        '''Return matrix of clipped distances matrix d[ij]
           Also returns normalized distance matrix
           Also returns collision binary graph 
           Also Delta proximity neighbourhood
           
           deltas must be a column!
           graph N includes i as neighbour 
           '''

        n_agents = np.size(state,0)
        d_ij = np.zeros([n_agents,n_agents])
        d_ij_norm = np.zeros_like(d_ij)

        for i in range(n_agents):
            xi = state[i,0:dim]
            li = state[i,2*dim]
            for j in range(n_agents):
                if j != i:
                    xj = state[j,0:dim]
                    lj = state[j,2*dim]

                    # Calculate agents relative distance
                    d_ij[i,j] = min(np.linalg.norm(xi-xj) -li -lj, d_safety[i])
                    d_ij_norm[i,j] = d_safety[i]/d_ij[i,j]
                else:
                    d_ij[i,j] = min(-li -li, d_safety[i])
                    # Just to be safe, the distance to itself in the normalized case i make it =1, as log(1)=0 so it is neutral
                    d_ij_norm[i,j] = 1

        collisions = d_ij_norm <= 0
        N_delta = d_ij <= deltas
        # Handling negative logarithms (only for d normalized, to use in logarithms)
        d_ij_norm[collisions] = 9.99E16
        log_d = np.log(d_ij_norm)
        log_d[collisions] = 9.99E16

        return d_ij, log_d, N_delta, collisions

    def localized_states(self, state, end_points, N_delta, d_ij):
        n_agents = np.size(state,0)
        sorted_idx = np.argsort(d_ij,1)
        k = self.k_closest

        z = []

        for i in range(n_agents):
            # How many agents are in Delta range, minus itself
            in_range = np.sum(N_delta[i,:])-1
            sorted_agents = sorted_idx[i,:]

            xi = state[i,0:dim]
            xFi = end_points[i*dim:(i+1)*dim]

            # Adding zii as the first row
            Zi = np.zeros([k+1,2*dim+1])
            Zi[0,:] = state[i,:].copy()
            # print(Zi,xFi.flaten() ,xi)
            Zi[0,0:dim] = xFi.flatten() - xi

            for kth in range(1,k+1):
            # kth = 1,2,...k
            
                if kth <= in_range:
                    # There exist a kth neighbour inside Delta
                    j = sorted_agents[kth]
                    xj = state[j,0:dim]
                    zj = state[j,:].copy()
                    zj[0:dim] = xj-xi
                    # print(f"{kth}th closest agent is {j}, coord {xj}, rel coord {xj-xi}")

                else: 
                    # There is no neigbhour, thus using a null state (or state that should almost not add value)
                    # I try for now to just add the next state in order, as if to just add the two closest even if outside the Delta range
                    # Hopping that the NN learns that agents outside delta do not contribute to Q
                    # Probably, the proper thing would be to project this next closest to the Delta boundary
                    j = sorted_agents[kth]
                    xj = state[j,0:dim]
                    zj = state[j,:].copy()
                    zj[0:dim] = xj-xi

                Zi[kth,:] = zj

            if self.simplify_zstate:
                # Remove parts of the satte that overcomplicate:
                # No (vx,vy,l)
                z.append(Zi[:,0:dim])
            else:
                z.append(Zi)
        # z is a list of arrays. Each array is the localized Delta state for each agent
        return z

    # %% Plotting methods 
    def show(self, state = None, not_animate = True):

        if not_animate:
            state = self.state

        fig, ax = plt.subplots(); # note we must use plt.subplots, not plt.subplot
        # (or if you have an existing figure)
        # fig = plt.gcf()
        # ax = fig.gca()

        ax.set_xlim((0, self.grid[0]));
        ax.set_ylim((0, self.grid[1]));
        ax.grid(True);

        for i in range(self.n_obstacles):
            circle = plt.Circle((self.obstacles[i,0], self.obstacles[i,1]), self.obstacles[i,2], color='black');
            ax.add_patch(circle);

        for i in range(self.n_agents):
            agent_color = num_to_rgb(i,self.n_agents-1)
            circle = plt.Circle((state[i,0], state[i,1]), state[i,4], color=agent_color, fill=False, label=f"{i+1}");
            ax.add_patch(circle);
            # end point in plot
            ax.plot(self.end_points[i*dim,0],self.end_points[i*dim+1,0],color=agent_color,marker = "*");
        
        ax.legend()

        if not_animate:
            plt.show()
        else:
            return fig;

    def animate(self, trajectory,frame_time = 0.2, frames = 20):

        good_frame = 0
        each_frame = len(trajectory)/frames
        for n_frame,state in enumerate(trajectory):
            if each_frame > 1 and round(good_frame) == n_frame:
                good_frame+=each_frame
            else:
                continue
            fig = self.show(state, not_animate=False);
            display.display(fig);
            display.clear_output(wait=True)
            time.sleep(frame_time)

    def plot(self, trajectory):

        # Create trajectory matrices -> time,  [x(t=0), x(t+1), x(t+2)],  | agent
        times = len(trajectory)
        x_cord = np.zeros([self.n_agents, times])
        y_cord = np.zeros([self.n_agents, times])
        collision_table = np.full([self.n_agents, times], False)

        for t,state in enumerate(trajectory):
            x_cord[:,t] = state[:,0]
            y_cord[:,t] = state[:,1]

            # Collision calc
            for i in range(self.n_agents):
                xi = state[i,0:dim]
                li = state[i,2*dim]
                for j in range(self.n_agents):
                    if j != i:
                        xj = state[j,0:dim]
                        lj = state[j,2*dim]

                        dij = np.linalg.norm(xi-xj) -li -lj
                        if dij<=0:
                            collision_table[i,t] = True

        collisions = np.sum(collision_table)

        # Plot obstacles and final state of agents
        fig, ax = plt.subplots(); # note we must use plt.subplots, not plt.subplot
        # (or if you have an existing figure)
        # fig = plt.gcf()
        # ax = fig.gca()

        ax.set_xlim((0, self.grid[0]));
        ax.set_ylim((0, self.grid[1]));
        ax.grid(True);

        # Plot obstacles
        for i in range(self.n_obstacles):
            circle = plt.Circle((self.obstacles[i,0], self.obstacles[i,1]), self.obstacles[i,2], color='black');
            ax.add_patch(circle);

        # Plot agents and collisions
        for i in range(self.n_agents):
            agent_color = num_to_rgb(i,self.n_agents-1)
            circle = plt.Circle((state[i,0], state[i,1]), state[i,4], color=agent_color, fill=False, label=f"{i+1}");
            ax.add_patch(circle);

            ax.plot(x_cord[i,:],y_cord[i,:],color=agent_color);
            ax.plot(self.end_points[i*dim,0],self.end_points[i*dim+1,0],color=agent_color,marker = "*");

            collisions_xcord = x_cord[i,collision_table[i,:]]
            collisions_ycord = y_cord[i,collision_table[i,:]]
            total_markers  = len(collisions_xcord)
            ax.plot(collisions_xcord,collisions_ycord,color=agent_color, marker = "v",fillstyle = "none", markevery=np.floor(total_markers/10));
            ax.set_title(f"{self.n_agents} agents, collisions = {collisions}")


        ax.legend()
        plt.show()


# other control functions
def gradient_control(state,env, u_max = 1):
    """Given a global state, calculates the direction of gradient of the cost (logarithm barrier)
        for each agent separately, taking into account all neighbours

    Args:
        state (np.array): each row is an agent state (x,y, vx,vy, l)
        env (_type_): drone environment

    Returns:
        actions: list of row vectors [u1^T, u2^T,...]
    """

    # maximum value of control component 
    # u_max m/s
    b = 0.1
    q = 1
    actions = []

    for i in range(env.n_agents):
        xi = np.transpose(state[i,0:dim])
        ri = state[i,4]
        xF = env.end_points[i*dim:(i+1)*dim,0]
        di_hat = env.d_safety[i]

        term1 = 2*(xi-xF)

        Ni = [j for j in range(env.n_agents) if j != i] # Complete graph (except itself), global knowledge
        # print(Ni)
        term2 = np.zeros_like(xi)
        for j in Ni:
            xj = np.transpose(state[j,0:2])
            rj = state[j,4]
            dij = np.linalg.norm(xi-xj)-ri-rj

            if dij <= di_hat:
                term2+= (xi-xj) / (dij*np.linalg.norm(xi-xj))

        grad = q*term1 - b*term2
        ui = np.clip(-grad , -u_max,u_max)
        actions.append(ui) 

    return actions

def proportional_control(state,env):

    # maximum value of control component 
    u_max = 1 # m/s
    k_gain = 1
    actions = []

    for i in range(env.n_agents):
        xi = np.transpose(state[i,0:dim])
        ri = state[i,4]
        xF = env.end_points[i*dim:(i+1)*dim,0]
        # di_hat = env.d_safety[i]

        # zi = xF_i - xi (vector to goal)
        zi = xF-xi
        ui = k_gain * zi

        ui_norm = np.linalg.norm(ui)
        if ui_norm > u_max:
            # Cap the control norm to u_max
            ui = ui/ui_norm*u_max

        actions.append(ui)

    return actions

# Plotting functions
def running_average(x, N = 50):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Plot Rewards and steps
def plot_rewards(episode_reward_list, collision_list, n_ep_running_average=50):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)],
               episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(episode_reward_list)+1)],
               collision_list, label='Collisions per episode')
    ax[1].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
        collision_list, n_ep_running_average), label='Avg. number of collisions per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of collisions')
    ax[1].set_title('Total number of collisions vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
 

        




    


        
        



