import gym
import numpy as np
import tensorflow as tf
import os
from dqn_agent import DQNAgent
from prioritized_replay_buffer import PrioritizedReplayBuffer
from utils import preprocess, stack_frames, plot_rewards

env = gym.make("ALE/Pong-v5", render_mode="rgb_array") #frame_skip=4
num_actions = env.action_space.n
state_shape = (84, 84, 4)

agent = DQNAgent(state_shape, num_actions)
buffer = PrioritizedReplayBuffer(100_000)

epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995
batch_size = 32
episodes = 500
update_target_freq = 5
save_model_freq = 10
beta_start = 0.4
beta_increment = 1e-3
beta = beta_start

rewards = []
os.makedirs("saved_models", exist_ok=True)

for ep in range(episodes):
    obs, _ = env.reset()
    frame = preprocess(obs)
    stacked_frames = [frame] * 4
    state = np.stack(stacked_frames, axis=2)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state, epsilon)

        total_frame_reward = 0
        for _ in range(4):  # Frame skipping
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_frame_reward += reward
            if terminated or truncated:
                break

        next_frame = preprocess(next_obs)
        next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
        buffer.add(state, action, total_frame_reward, next_state, terminated or truncated)
        state = next_state
        total_reward += total_frame_reward
        done = terminated or truncated

        if len(buffer) > batch_size:
            states, actions, rewards_, next_states, dones, indices, weights = buffer.sample(batch_size, beta=beta)
            td_errors = agent.train(states, actions, rewards_, next_states, dones, weights)
            buffer.update_priorities(indices, td_errors.numpy() + 1e-6)
            beta = min(1.0, beta + beta_increment)

    if ep % update_target_freq == 0:
        agent.update_target_network()

    if ep % save_model_freq == 0:
        agent.save(f"saved_models/model_episode_{ep}.keras")

    rewards.append(total_reward)
    #plot_rewards(rewards)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# --- Evaluation ---
obs, _ = env.reset()
frame = preprocess(obs)
stacked_frames = [frame] * 4
state = np.stack(stacked_frames, axis=2)
done = False
total_reward = 0

while not done:
    env.render()
    action = agent.act(state, epsilon=0.0)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_frame = preprocess(next_obs)
    state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
    total_reward += reward
    done = terminated or truncated

print(f"Evaluation run finished. Total reward: {total_reward}")
env.close()
