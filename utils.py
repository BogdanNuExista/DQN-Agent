import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (84, 84))
    return observation / 255.0

def stack_frames(stacked_frames, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frames = [new_frame] * 4 
    else:
        stacked_frames.append(new_frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=2), stacked_frames

def plot_rewards(all_rewards):
    plt.figure(figsize=(12, 5))
    plt.plot(all_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("DQN Learning Progress")
    plt.pause(0.01)
    plt.clf()
