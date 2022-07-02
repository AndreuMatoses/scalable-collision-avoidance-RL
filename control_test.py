import numpy as np
import matplotlib.pyplot as plt
import drone_env
from drone_env import running_average, plot_rewards
from tqdm import tqdm, trange

env = drone_env.drones(n_agents=5, n_obstacles=2, grid=[5, 5], end_formation="O")
print(env)
# env.show()

T_Episodes = 2
# Simulate for T seconds (default dt = drone_env.dt = 0.01s)
T = 5

# Initialize variables
total_collisions_list = []
total_reward_list = []
times = np.arange(0, T, step=drone_env.dt) + drone_env.dt
EPISODES = trange(T_Episodes, desc='Episode: ', leave=True)

for episode in EPISODES:

    # reward_history = np.zeros([len(times), env.n_agents])
    trajectory = [env.state.copy()]
    total_episode_reward = 0
    total_episode_collisions = 0
    # env.show()

    for t_iter, t in enumerate(times):
        # Simple gradient controller u_i = -grad_i, assuming Nj = V
        state = env.state

        # calculate actions based on current state
        # actions = drone_env.gradient_control(state, env)
        actions = drone_env.proportional_control(state, env)

        # Update environment one time step with the actions
        new_state, new_z, reward, n_collisions, finished = env.step(actions)

        total_episode_reward += np.sum(reward)
        total_episode_collisions += n_collisions

        # reward_history[t_iter,:] = reward
        trajectory.append(new_state.copy())

    # Append episode reward
    total_reward_list.append(total_episode_reward)
    total_collisions_list.append(total_episode_collisions)

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
    env.plot(trajectory)

plot_rewards(total_reward_list,total_collisions_list, n_ep_running_average=5)