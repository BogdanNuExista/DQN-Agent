import cv2
import numpy as np
import matplotlib.pyplot as plt

# gets an RGB image from the environment
# convert to grayscale
# resize to 84x84, universal size for DQN
# normalize pixel values to [0, 1]

def preprocess(observation): 
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) 
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA) 
    return (resized / 255.0).astype(np.float32)

# stacks the last 4 frames to create a state representation
# if it's a new episode, stack the same frame 4 times
# append the new frame to the stack and remove the oldest frame

def stack_frames(stacked_frames, new_frame, is_new_episode): 
    if is_new_episode:
        stacked_frames = [new_frame] * 4
    else:
        stacked_frames.append(new_frame) 
        stacked_frames.pop(0) 
    return np.stack(stacked_frames, axis=2), stacked_frames

# Evaluates the agent by running it in the environment for a specified number of episodes

def evaluate_agent(agent, env, num_episodes=3):
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        frame = preprocess(obs)
        stacked_frames = [frame] * 4
        state = np.stack(stacked_frames, axis=2)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state, epsilon=0.0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_frame = preprocess(next_obs)
            state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            total_reward += reward
            done = terminated or truncated
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)