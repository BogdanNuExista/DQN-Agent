# main.py
import gym
import numpy as np
import tensorflow as tf
import os
import time
from dqn_agent import DQNAgent
from prioritized_replay_buffer import PrioritizedReplayBuffer
from utils import preprocess, stack_frames, evaluate_agent

# Environment setup
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=4, full_action_space=False)  # Use game-specific actions only
num_actions = env.action_space.n
state_shape = (84, 84, 4)

# Agent and buffer
agent = DQNAgent(state_shape, num_actions)
buffer = PrioritizedReplayBuffer(300_000)  # Optimized buffer size

# Training parameters
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
batch_size = 64
episodes = 1000
save_model_freq = 50
beta_start = 0.4
beta_increment = 1e-3
beta = beta_start

# Tracking
rewards = []
os.makedirs("saved_models", exist_ok=True)
start_time = time.time()

# Training loop
for ep in range(episodes):
    obs, _ = env.reset()
    frame = preprocess(obs)
    stacked_frames = [frame] * 4
    state = np.stack(stacked_frames, axis=2)
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = agent.act(state, epsilon)

        # Environment step with built-in frame skipping
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_frame = preprocess(next_obs)
        next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
        
        # Store transition
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1

        # Train from buffer
        if len(buffer) > batch_size:
            states, actions, rewards_, next_states, dones, indices, weights = buffer.sample(batch_size, beta)
            
            # Train agent
            td_errors = agent.train(states, actions, rewards_, next_states, dones, weights)
            
            # Update priorities
            buffer.update_priorities(indices, td_errors.numpy())
            
            # Adjust beta
            beta = min(1.0, beta + beta_increment)

    # Update exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)
    
    # Save model periodically
    if ep % save_model_freq == 0:
        agent.save(f"saved_models/model_episode_{ep}.keras")
    
    # Evaluate periodically
    if ep % 20 == 0:
        eval_reward = evaluate_agent(agent, env)
        print(f"Episode {ep}/{episodes} | "
              f"Reward: {total_reward:5.1f} | "
              f"Eval Reward: {eval_reward:5.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Beta: {beta:.3f} | "
              f"Time: {time.time()-start_time:.1f}s")
        start_time = time.time()

# Final evaluation
final_reward = evaluate_agent(agent, env)
print(f"Training complete. Final evaluation reward: {final_reward}")

# Render final performance
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

print(f"Final performance reward: {total_reward}")
env.close()