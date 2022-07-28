
import math
import os
import random
import time
from distutils.log import error

import matplotlib.pyplot as plt
from matplotlib import markers, animation
import numpy as np
from IPython import display

### TO DO ############
"""
- Put back collision cost b = 
- Change back from not using Critic NN in training(),benchmark_critic()SACAgent (now trying Gt)
- TrainedAgent for the actor loading
- Add collisions to the animation. Add action arrows to animation?
- (inactive now) Remove fixed random seed to initialize agents
- 
- Plot of the Q function field for a fixed kth satates (varying xi)
- Maybe clip the reward for near collision distances
"""
######################

# Spatial dimension
dim = 2 
# time step in seconds
dt = 0.05
max_time_steps = 200
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
        self.simplify_zstate = simplify_zstate
        self.internal_t = 0
        self.collision_weight = 0.2 # Weight of collision cost per unit of time. -r = q|xi-xF|^2 + b log(d_i/d_ij)

        # Other geometry parameters
        self.drone_radius = np.ones(n_agents)*0.1 # radius of each drone in m

        # State space dynamics. For now, all agents have same A,B
        self.A = np.eye(dim)
        self.B = np.eye(dim)*dt

        # Initialize agents and obstacles, and check if delta is a correct value
        self.obstacles = self.create_obstacles(n_obstacles)
        self.end_points, self.d_safety = self.generate_formation(end_formation)

        if deltas is None:
            # In case of no deltas, we assume Dleta = d_safety, i.e no simplification (maximum deltas allowed)
            self.deltas = self.d_safety
        else:
            self.deltas = np.minimum(deltas, self.d_safety)
            if not np.all(deltas <= self.d_safety):
                print("Some deltas are greater than the final minimum distance between end positions. Using minimum distance between end positions for those cases instead.",f"deltas = {self.deltas}")

        self.state, self.z_states = self.init_agents(n_agents)

        # self.trajectory = []
        # self.trajectory.append(self.state.copy())

    def reset(self, renew_obstacles = True):
        self.state, self.z_states = self.init_agents(self.n_agents)
        self.internal_t = 0
        if renew_obstacles == True:
           self.obstacles = self.create_obstacles(self.n_obstacles)

    
    def __str__(self):
        print("Grid size: [x_lim, y_lim]\n",self.grid)
        print("State: [x, y, vx, vy, r]\n", self.state)
        print(f"z_sattes for k_closest = {self.k_closest}: simplify? {self.simplify_zstate}")
        print("safety distance for each agent:\n", self.d_safety)
        print("Deltas disk radius for each agent: \n", self.deltas)
        # print("Obstacles [x, y, r]:\n",self.obstacles)
        print(f"Collision cost weight (per unit of time) = {self.collision_weight} ")
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
        d_safety = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            xFi = formation[dim*i:(i+1)*dim,0]
            li = self.drone_radius[i]
            d_i = np.infty
            for j in range(self.n_agents): 
                if j != i:
                    xFj = formation[dim*j:(j+1)*dim,0]
                    lj = self.drone_radius[j]
                    d_ij = np.linalg.norm(xFi-xFj) -li -lj
                    d_i = min([d_i,d_ij])

            d_safety[i] = d_i

        # formation: [xF1^T, xF2^T,...]^T. Column vector, d_safety (distance to closest end position from agent's end position)
        return formation, np.floor(d_safety*100)/100

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

        # REMOVE THIS seed FOR true random initial state
        # random.seed(1)
        random_coord = np.array(random.sample(possible_coord, n_agents))
        state[:,0:dim] = random_coord

        # Calculate localized states z (uing the reward funciton)
        _, _, z_states, Ni = self.rewards(state, self.end_points, self.n_agents, self.d_safety, self.deltas)
        # Update the Ni graph
        self.Ni = Ni

        return state, z_states
    
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
        r_vec, n_collisions, z_states, Ni = self.rewards(self.state, self.end_points, self.n_agents, self.d_safety, self.deltas)
        # Update the z and Ni graph
        self.z_states = z_states
        self.Ni = Ni

        # SHould return (s', r(s,a), n_collisions(s') ,finished)
        end_points = np.reshape(self.end_points,np.shape(self.state[:,0:dim]))
        error_from_end = np.linalg.norm(end_points-self.state[:,0:dim],axis = 1)

        if np.all(error_from_end <=0.2) or self.internal_t>= max_time_steps-1:
            finished = True
        else:
            finished = False

        self.internal_t += 1
        # TO DO: Proper is_finished
        return self.state, z_states, r_vec, n_collisions, finished

    def rewards(self, state, end_points, n_agents, d_safety, deltas):
        '''
        state: [column 1: x, col 2: y, col 3: vx, col 4 vy, col 5: r ] np.array[i,2*dim+1]
        end_points: column [x1, y1, x2, y2, ... ]^T
        d_safety: column d_i [d1, d2, ...]^T
        '''
        n_agents = np.size(state,0)

        # weights: q|xi-xF|^2 + b log(d_i/d_ij). I multiply per dt as i assume is cost/time
        q = 2*dt
        b = self.collision_weight*dt

        xF = np.reshape(end_points,[n_agents,dim])
        xi = state[:,0:dim]

        # row vector, q|xi-xF|^2
        to_goal_cost = q*np.power(np.linalg.norm(xF-xi,axis=1),2)

        # Collision cost

        d_ij, log_d, N_delta, collisions = self.distance_data(state,deltas,d_safety)

        collision_cost = b*np.sum(log_d*N_delta,1)
        real_collision_cost = b*np.sum(log_d,1)
        n_collisions = np.sum(collisions)

        # These are the approximated localized rewards
        reward_vector = -(to_goal_cost+collision_cost)

        # Calculate localized z states
        z_states, Ni = self.localized_states(state, end_points, N_delta, d_ij)

        return reward_vector, n_collisions, z_states, Ni

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
            li = self.drone_radius[i]
            for j in range(n_agents):
                if j != i:
                    xj = state[j,0:dim]
                    lj = self.drone_radius[j]

                    # Calculate agents relative distance
                    d_ij[i,j] = min(np.linalg.norm(xi-xj) -li -lj, d_safety[i])
                    if d_ij[i,j] == 0: # Handle unlikely case of exactly 0, if not then the coming division would be error
                        d_ij[i,j] = -10**-6
                    d_ij_norm[i,j] = d_safety[i]/d_ij[i,j]
                else:
                    d_ij[i,j] = min(-li -li, d_safety[i])
                    # Just to be safe, the distance to itself in the normalized case i make it =1, as log(1)=0 so it is neutral
                    d_ij_norm[i,j] = 1

        collisions = d_ij_norm <= 0
        N_delta = d_ij <= deltas
        # Handling negative logarithms (only for d normalized, to use in logarithms)
        d_ij_norm[collisions] = 9.99E3
        log_d = np.log(d_ij_norm)
        log_d[collisions] = 9.99E3

        return d_ij, log_d, N_delta, collisions

    def localized_states(self, state, end_points, N_delta, d_ij):
        n_agents = np.size(state,0)
        sorted_idx = np.argsort(d_ij,1)
        k = self.k_closest

        z = []
        Ni_list = []

        for i in range(n_agents):
            # How many agents are in Delta range, minus itself
            in_range = np.sum(N_delta[i,:])-1
            sorted_agents = sorted_idx[i,:]
            Ni = [i]

            xi = state[i,0:dim]
            xFi = end_points[i*dim:(i+1)*dim]

            # Adding zii as the first row
            Zi = np.zeros([k+1,2*dim+1])
            Zi[0,:] = state[i,:].copy()
            # print(Zi,xFi.flaten() ,xi)
            Zi[0,0:dim] = -(xFi.flatten() - xi)

            for kth in range(1,k+1):
            # kth = 1,2,...k
            
                if kth <= in_range:
                    # There exist a kth neighbour inside Delta
                    j = sorted_agents[kth]
                    Ni.append(j)
                    xj = state[j,0:dim].copy()
                    zj = state[j,:].copy()
                    zj[0:dim] = xj-xi
                    # zj[0:dim] = xj-xFi.flatten()
                    # print(f"{kth}th closest agent is {j}, coord {xj}, rel coord {xj-xi}")

                else: 
                    # There is no neigbhour, thus using a null state (or state that should almost not add value)
                    # I try for now to just add the next state in order, as if to just add the two closest even if outside the Delta range
                    # Hopping that the NN learns that agents outside delta do not contribute to Q
                    # Probably, the proper thing would be to project this next closest to the Delta boundary
                    # j = sorted_agents[kth]
                    # xj = state[j,0:dim].copy()
                    # zj = state[j,:].copy()
                    # zj[0:dim] = xj-xi

                    # Create a "ghost" agent that is just behind agent, at a distance 1.1*Delta in the direction to the goal
                    j = sorted_agents[kth]
                    zi = Zi[0,0:dim]
                    zj = state[j,:].copy()
                    zj[0:dim] = zi/np.linalg.norm(zi) * self.deltas[i]*1.1

                Zi[kth,:] = zj
            
            if self.simplify_zstate:
                # Remove parts of the satte that overcomplicate:
                # No (vx,vy,l)
                z.append(Zi[:,0:dim])
            else:
                z.append(Zi)
            
            Ni_list.append(Ni)

        # z is a list of arrays. Each array is the localized Delta state for each agent
        # Add: Ni_list = list of neighbours for each agent
        return z, Ni_list

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

    def animate_basic(self, trajectory,frame_time = 0.2, frames = 20):

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

        # ax.set_xlim((0, self.grid[0]));
        # ax.set_ylim((0, self.grid[1]));
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
            # problem when it is 0, markevery=np.floor(total_markers/2))
            ax.plot(collisions_xcord,collisions_ycord,color=agent_color, marker = "v",fillstyle = "none", markevery=2);
            ax.set_title(f"{self.n_agents} agents, collisions = {collisions}")


        ax.legend()
        plt.show()
    
    def animate(self, trajectory, z_trajectory , deltas, name = "test", format ="gif"):

        if format == "mp4":
            # plt.rcParams['animation.ffmpeg_path'] ='D:\\Programes portables\\ffmpeg\\bin\\ffmpeg.exe'
            plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\\Andreu\\OneDrive - KTH\\programes\\ffmpeg\\bin\\ffmpeg.exe'


        fig, ax = plt.subplots(); # note we must use plt.subplots, not plt.subplot
        ax.set_xlim((-1, self.grid[0]+1));
        ax.set_ylim((-1, self.grid[1]+1));
        # ax.grid(True)
        ax.set_title(f"Delta = {deltas[0]}");
        circles = []
        d_circles = []
        arrows = []

        states = trajectory[0]
        z_states = z_trajectory[0]
        for i in range(self.n_agents):
            xi = states[i,0:dim]
            xFi = self.end_points[i*dim:(i+1)*dim,0].flatten()
            agent_color = num_to_rgb(i,self.n_agents-1)
            ax.plot(xFi[0],xFi[1],color=agent_color,marker = "*");
            circle = plt.Circle((states[i,0], states[i,1]), states[i,4], color=agent_color, fill=False, label=f"{i+1}");
            circles.append(ax.add_patch(circle))
            
            delta_circle = plt.Circle((states[i,0], states[i,1]), states[i,4] + deltas[i], color="red", fill=False, ls = "--", alpha = 0.5);
            d_circles.append(ax.add_patch(delta_circle))

            z_size = np.size(z_states[i],0)
            z_state = z_states[i]
            arrows_i = []
            for k in range(z_size):
                if k == 0:
                    star = xFi[0:dim]
                    fini = xFi[0:dim] + z_state[k,0:dim]
                    coords = np.array([star,fini])
                    arrows_i.append(ax.plot(coords[:,0], coords[:,1] , color = agent_color, lw = 0.5, alpha = 0.2))
                else:
                    star = xi[0:dim]
                    fini = xi[0:dim] + z_state[k,0:dim]
                    coords = np.array([star,fini])
                    arrows_i.append(ax.plot(coords[:,0], coords[:,1] , color = agent_color, lw = 0.5, alpha = 0.5))
            arrows.append(arrows_i)

        plt.legend(loc = "upper right")

        def update_objects(t:int):
            states = trajectory[t]
            z_states = z_trajectory[t]
            ax.set_title(f"Deltas = {deltas[0]}. Time = {t*dt:.1f}s")

            for i in range(self.n_agents):
                xi = states[i,0:dim]
                xFi = self.end_points[i*dim:(i+1)*dim,0].flatten()
                agent_color = num_to_rgb(i,self.n_agents-1)

                z_size = np.size(z_states[i],0)
                z_state = z_states[i]
                for k in range(z_size):
                    if k == 0:
                        star = xFi[0:dim]
                        fini = xFi[0:dim] + z_state[k,0:dim]
                        coords = np.array([star,fini])
                        arrows[i][k][0].set_data(coords[:,0], coords[:,1])
                        pass
                    else:
                        star = xi[0:dim]
                        fini = xi[0:dim] + z_state[k,0:dim]
                        coords = np.array([star,fini])
                        arrows[i][k][0].set_data(coords[:,0], coords[:,1])
                        pass
                circles[i].center = states[i,0], states[i,1]
                d_circles[i].center = states[i,0], states[i,1]

            return circles, d_circles, arrows
        
        print("\nSaving animation...")
        anim = animation.FuncAnimation(fig, update_objects, len(trajectory), interval=dt)

        if format == "gif":
            writergif = animation.PillowWriter(fps=30)
            full_name = os.path.join("videos", name + ".gif")
            anim.save(full_name, writer=writergif)
        elif format == "mp4":
            FFwriter = animation.FFMpegWriter(fps=30)
            full_name = os.path.join("videos", name + ".mp4")
            anim.save(full_name, writer = FFwriter)
        else:
            print(f"format{format} not valid")

        print(f"Animation saved as {full_name}")



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
 
def plot_grads(grad_per_episode:np.ndarray, gi_per_episode:np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    n_agents = np.size(grad_per_episode,1)
    episode_variable = [e for e in range(1, len(grad_per_episode)+1)]

    for i in range(n_agents):
        agent_color = num_to_rgb(i,n_agents-1)
        ax[0].plot(episode_variable, grad_per_episode[:,i], label=f"Agent {i+1}", color = agent_color)
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Score function gradient')
    # ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    for i in range(n_agents):
        agent_color = num_to_rgb(i,n_agents-1)
        ax[1].plot(episode_variable, gi_per_episode[:,i], label=f"Agent {i+1}", color = agent_color)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Approximated gi gradient (max norm = 100)')
    # ax[1].set_title('Total number of collisions vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
        




    


        
        



