
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import drone_env
from drone_env import running_average, plot_rewards
from tqdm import tqdm, trange
from SAC_agents import SACAgents, ExperienceBuffers, TrainedAgent

### Set up parameters ###
n_agents = 5
deltas = np.ones(n_agents)*2
env = drone_env.drones(n_agents=n_agents, n_obstacles=0, grid=[5, 5], end_formation="O", deltas=None ,simplify_zstate = True)
print(env)
# env.show()

N_Episodes = 50

T = 4 # Simulate for T seconds (default dt = drone_env.dt = 0.01s) t_iter t=500

### 

# Initialize variables
total_collisions_list = deque()
total_reward_list = deque()
mean_critic_error = np.zeros([env.n_agents, N_Episodes])
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

times = np.arange(0, T, step=drone_env.dt) + drone_env.dt
EPISODES = trange(N_Episodes, desc='Episode: ', leave=True)

agents = TrainedAgent(file_name="Q_test-critics.pth", n_agents=env.n_agents)
print("### Running Trained agent (no learning)")
print(f"Episodes = {N_Episodes}, Time iterations = {len(times)} (T = {T}s, dt = {drone_env.dt}s)")
print(f"N of agents = {env.n_agents}")

for episode in EPISODES:

    # reward_history = np.zeros([len(times), env.n_agents])
    trajectory = [env.state.copy()]
    total_episode_reward = 0
    total_episode_collisions = 0
    # env.show()

    buffers = ExperienceBuffers(env.n_agents)
    # SIMULATION OVER T
    for t_iter, time in enumerate(times):
        # Simple gradient controller u_i = -grad_i, assuming Nj = V
        state = env.state
        z_states = env.z_states
        Ni = env.Ni

        # calculate actions based on current state
        actions = drone_env.gradient_control(state, env)
        # actions = drone_env.proportional_control(state, env)

        # Update environment one time step with the actions
        new_state, new_z, rewards, n_collisions, finished = env.step(actions)
        # EXPERIECE: [z_state, action, reward, next_z, finished]
        buffers.append(z_states, actions, rewards,new_z, finished)

        total_episode_reward += np.mean(rewards)
        total_episode_collisions += n_collisions

        # reward_history[t_iter,:] = reward
        trajectory.append(new_state.copy())

    # END OF EPISODE
    # Append episode reward
    total_reward_list.append(total_episode_reward)
    total_collisions_list.append(total_episode_collisions)

    # Test Critic values
    Q_simulated, Q_approx = agents.benchmark_cirtic(buffers, only_one_NN=True)

    critic_error = [np.mean(np.power(Q_simulated[i]-Q_approx[i],2)) for i in range(env.n_agents)]

    mean_critic_error[:,episode] = np.array([critic_error])

    # print(f"Episode collisions = {total_episode_collisions}")
    # env.animate(trajectory,frame_time=0.1)

    # RESET ENVIRONMENT
    env.reset(renew_obstacles=False)

    # Set progress bar description with information
    average_reward = running_average(total_reward_list, 50)[-1]
    average_collisions = running_average(total_collisions_list, 50)[-1]
    EPISODES.set_description(
        f"Episode {episode} - Reward/Collisions/Steps: {total_episode_reward:.1f}/{total_episode_collisions}/{t_iter+1} - Average: {average_reward:.1f}/{average_collisions:.2f}/{t_iter+1}")

    # Plot current trajectory

    if episode >= N_Episodes-5:
        env.plot(trajectory)

        plt.figure()
        for i in range(env.n_agents):
            agent_color = drone_env.num_to_rgb(i,env.n_agents-1)
            plt.plot(times,Q_simulated[i], label=f"i={i}, simulated Q", color = agent_color)
            plt.plot(times,Q_approx[i],"--" , label=f"i={i}, approx Q", color = tuple(0.9*x for x in agent_color))
        plt.legend()
        plt.show()

# plot_rewards(total_reward_list,total_collisions_list, n_ep_running_average=5)

plt.figure()
for i in range(env.n_agents):
        agent_color = drone_env.num_to_rgb(i,env.n_agents-1)
        plt.plot(range(N_Episodes),mean_critic_error[i,:], label=f"i={i}", color = agent_color)
plt.xlabel("Episodes")
plt.ylabel("trajectory 1/T * (Q-Qapprox)^2")
plt.legend()
plt.grid()
plt.show()
