from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import drone_env
from drone_env import running_average, plot_rewards, plot_grads
from tqdm import tqdm, trange
from SAC_agents import SA2CAgents, RandomAgent, TrainedAgent
from utils import ExperienceBuffers

plt.style.use('seaborn-dark-palette')
tex_fonts = {
    # Use LaTeX to write all text
    #     "text.usetex": True,
    "font.family": "sans-serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

plt.rcParams.update(tex_fonts)


### Set up parameters ###
n_agents = 5
deltas = np.ones(n_agents)*1
# deltas = None
env = drone_env.drones(n_agents=n_agents, n_obstacles=0, grid=[5, 5], end_formation="O", deltas=deltas ,simplify_zstate = True)
print(env)
# env.show()

N_Episodes = 1500  
plot_last = 1
save_name = "n5_E1500_Advantage"

# T = 8 # Simulate for T seconds (default dt = drone_env.dt = 0.05s) t_iter t=80
discount_factor = 0.99
alpha_critic = 10**-2
alpha_actor = 10**-4
M = 2 # Epochs, i.e steps of the SDG for the critic NN
dim_z = env.local_state_space # Dimension of the localized z_state space
dim_a = env.local_action_space # Dimension of the local action space

### 

# Initialize variables
total_collisions_per_episode = []
total_reward_per_episode = []
grad_per_episode = np.zeros([N_Episodes, n_agents])
gi_per_episode = np.zeros_like(grad_per_episode)

# times = np.arange(0, T, step=drone_env.dt) + drone_env.dt


agents = SA2CAgents(n_agents=env.n_agents, dim_local_state = dim_z, dim_local_action=dim_a, discount=discount_factor, epochs=M, learning_rate_critic=alpha_critic, learning_rate_actor=alpha_critic)
print("### Running Scalable-Actor-Critic with params: ###")
print(f"Episodes = {N_Episodes}, max Time iterations = {drone_env.max_time_steps} (T = {drone_env.max_time_steps * drone_env.dt}s, dt = {drone_env.dt}s)")
print(f"N of agents = {env.n_agents}, structure of critic NN = {agents.criticsNN[0].input_size}x{agents.criticsNN[0].L1}x{agents.criticsNN[0].L2}x{agents.criticsNN[0].output_size}")
print(f"Discount = {discount_factor}, lr for NN critical  = {alpha_critic}, lr for actor  = {alpha_actor}, epochs M = {M}")

EPISODES = trange(N_Episodes, desc='Episode: ', leave=True)
for episode in EPISODES:

    if episode >= N_Episodes-plot_last:
        # reward_history = np.zeros([len(times), env.n_agents])
        trajectory = [env.state.copy()]
        z_trajectory = [env.z_states]
    total_episode_reward = 0
    total_episode_collisions = 0
    # env.show()

    buffers = ExperienceBuffers(env.n_agents)

    # SIMULATION OVER T
    t_iter = 0
    finished = False
    while not finished:
        # Simple gradient controller u_i = -grad_i, assuming Nj = V
        state = env.state
        z_states = env.z_states
        Ni = env.Ni

        # calculate actions based on current state
        # actions = drone_env.gradient_control(state, env)
        # actions = drone_env.proportional_control(state, env)
        actions = agents.forward(z_states, Ni)
        # actions = agents.forward(z_states, Ni)

        # Update environment one time step with the actions
        new_state, new_z, rewards, n_collisions, finished = env.step(actions)
        # EXPERIECE: [z_state, action, reward, next_z, finished]
        buffers.append(z_states, actions, rewards,new_z, Ni,finished)

        total_episode_reward += np.mean(rewards)
        total_episode_collisions += n_collisions

        if episode >= N_Episodes-plot_last:
            # reward_history[t_iter,:] = reward
            trajectory.append(new_state.copy())
            z_trajectory.append(new_z)
        
        t_iter +=1

    ### END OF EPISODES
    # Train of critic with the data of the episode
    current_grad_norms, current_gi_norms = agents.train(buffers, actor_lr = alpha_actor, return_grads=True)

    # Append episodic variables/logs
    total_reward_per_episode.append(total_episode_reward)
    total_collisions_per_episode.append(total_episode_collisions)
    grad_per_episode[episode,:] = np.array(current_grad_norms)
    gi_per_episode[episode,:] = np.array(current_gi_norms)

    if episode >= N_Episodes-plot_last:
        Q_simulated, V_approx = agents.benchmark_cirtic(buffers, only_one_NN=False)

    # print(f"Episode collisions = {total_episode_collisions}")
    # env.animate(trajectory,frame_time=0.1)

    # RESET ENVIRONMENT
    env.reset(renew_obstacles=False)

    # Set progress bar description with information
    average_reward = running_average(total_reward_per_episode, 50)[-1]
    average_collisions = running_average(total_collisions_per_episode, 50)[-1]
    EPISODES.set_description(
        f"Episode {episode} - Reward/Collisions/Steps: {total_episode_reward:.1f}/{total_episode_collisions}/{t_iter} - Average: {average_reward:.1f}/{average_collisions:.2f}/{t_iter}")

    # Plot current trajectory

    if episode >= N_Episodes-plot_last:
        env.plot(trajectory)
        env.animate(trajectory, z_trajectory, deltas, name="test", format="mp4")
        times = np.arange(0, t_iter)*drone_env.dt
        plt.figure()
        for i in range(env.n_agents):
            agent_color = drone_env.num_to_rgb(i,env.n_agents-1)
            plt.plot(times,Q_simulated[i], label=f"i={i}, simulated Q (Gt)", color = agent_color)
            plt.plot(times,V_approx[i],"--" , label=f"i={i}, approx V", color = tuple(0.9*x for x in agent_color))
            print(f"Agent {i} params = {agents.actors[i].parameters}")
        plt.legend()
        plt.show()

agents.save(filename=save_name)

plot_rewards(total_reward_per_episode,total_collisions_per_episode, n_ep_running_average=50)
# plt.savefig("images/reward_training.pdf",format='pdf', bbox_inches='tight')
plot_grads(grad_per_episode,gi_per_episode)