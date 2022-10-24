from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import drone_env
from drone_env import running_average, plot_rewards, plot_grads
from tqdm import tqdm, trange
from SAC_agents import *
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
env = drone_env.drones(n_agents=n_agents, n_obstacles=0, grid=[5, 5], end_formation="O", deltas=deltas ,simplify_zstate = True)
env.collision_weight = 0.2 # old 0.2
print(env)
# env.show()

N_Episodes = 50
episodes_to_plot = [1,50]

# Initialize variables
total_collisions_per_episode = deque()
total_reward_per_episode = deque()
total_true_reward_per_episode = deque()
total_t = deque()
mean_advantage = np.zeros([env.n_agents, N_Episodes])

# times = np.arange(0, T, step=drone_env.dt) + drone_env.dt

agents = TrainedAgent(critics_name="final\\cont_n5-A2Ccritics.pth", actors_name="final\\cont_n5-A2Cactors.pth", n_agents=env.n_agents)
print("### Running Trained agent (no learning)")
print(f"Episodes = {N_Episodes}, max Time iterations = {drone_env.max_time_steps} (T = {drone_env.max_time_steps * drone_env.dt}s, dt = {drone_env.dt}s)")
print(f"N of agents = {env.n_agents}, collision weight b = {env.collision_weight}")

EPISODES = trange(N_Episodes, desc='Episode: ', leave=True)
for episode in EPISODES:

    if episode+1 in episodes_to_plot:
        # reward_history = np.zeros([len(times), env.n_agents])
        trajectory = [env.state.copy()]
        z_trajectory = [env.z_states]    
    total_episode_reward = 0
    total_true_episode_reward = 0
    total_episode_collisions = 0
    # env.show()

    buffers = ExperienceBuffers(env.n_agents)

    # SIMULATION OVER T
    t_iter = 0
    finished = False
    while not finished:

        state = env.state
        z_states = env.z_states
        Ni = env.Ni

        # calculate actions based on current state
        # actions = drone_env.gradient_control(state, env)
        # actions = drone_env.proportional_control(state, env)
        actions = agents.forward(z_states, Ni)

        # Update environment one time step with the actions
        new_state, new_z, rewards, n_collisions, finished, true_rewards = env.step(actions)
        # EXPERIECE: [z_state, action, reward, next_z, finished]
        buffers.append(z_states, actions, rewards, new_z, Ni, finished)

        total_episode_reward += np.mean(rewards)
        total_true_episode_reward += np.mean(true_rewards)
        total_episode_collisions += n_collisions

        if episode+1 in episodes_to_plot:
            # reward_history[t_iter,:] = reward
            trajectory.append(new_state.copy())
            z_trajectory.append(new_z)

        t_iter +=1

    # END OF EPISODE
    # Append episode reward
    total_reward_per_episode.append(total_episode_reward)
    total_true_reward_per_episode.append(total_true_episode_reward)
    total_collisions_per_episode.append(total_episode_collisions)
    total_t.append(t_iter)

    # Test Critic values
    Q_simulated, V_approx = agents.benchmark_cirtic(buffers, only_one_NN=False)
    advantage = [np.mean(np.power(Q_simulated[i]-V_approx[i],1)) for i in range(env.n_agents)]
    mean_advantage[:,episode] = np.array([advantage])

    # print(f"Episode collisions = {total_episode_collisions}")
    # env.animate(trajectory,frame_time=0.1)

    # RESET ENVIRONMENT
    env.reset(renew_obstacles=False)

    # Set progress bar description with information
    average_reward = running_average(total_reward_per_episode, 50)[-1]
    average_true_reward = running_average(total_true_reward_per_episode, 50)[-1]
    average_collisions = running_average(total_collisions_per_episode, 50)[-1]
    average_t = running_average(total_t, 50)[-1]
    EPISODES.set_description(
        f"Episode {episode} - Reward/Collisions/Steps: {total_episode_reward:.1f}/{total_episode_collisions}/{t_iter} - Average: {average_reward:.1f}/{average_collisions:.2f}/{average_t}. True r={average_true_reward:.1f}.")

    # Plot current trajectory

    if episode+1 in episodes_to_plot:
        env.plot(trajectory, episode)
        env.animate(trajectory, z_trajectory, deltas, episode, name=f"trained-E{episode+1}", format="mp4")
        times = np.arange(0, t_iter)*drone_env.dt
        plt.figure()
        for i in range(env.n_agents):
            agent_color = drone_env.num_to_rgb(i,env.n_agents-1)
            plt.plot(times,Q_simulated[i], label=f"i={i}, simulated Q (Gt)", color = agent_color)
            plt.plot(times,V_approx[i],"--" , label=f"i={i}, approx V", color = tuple(0.9*x for x in agent_color))
        plt.legend()
        plt.show()

plot_rewards(total_reward_per_episode, total_true_reward_per_episode, total_collisions_per_episode, n_ep_running_average=50)

plt.figure()
for i in range(env.n_agents):
        agent_color = drone_env.num_to_rgb(i,env.n_agents-1)
        plt.plot(range(N_Episodes),mean_advantage[i,:], label=f"i={i}", color = agent_color)
plt.xlabel("Episodes")
plt.ylabel("trajectory 1/T * [Q(s,a)-V(s)] = mean_T A(s,a)")
plt.legend()
plt.grid()
plt.show()
