# visualize_assault.py
import gym
import numpy as np
import argparse
import os
import time
import cv2
from dqn_agent import DQNAgent
from utils import preprocess, stack_frames

def visualize_model(model_path, episodes=5, delay=0.01):
    """
    Visualize a trained DQN agent playing Assault
    
    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to play
        delay: Time delay between frames (seconds)
    """
    # Create environment with human rendering
    env = gym.make("ALE/Assault-v5", render_mode="human", frameskip=4)
    num_actions = env.action_space.n
    state_shape = (84, 84, 4)
    
    # Create agent and load model
    agent = DQNAgent(state_shape, num_actions)
    agent.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Play episodes
    for ep in range(episodes):
        obs, _ = env.reset()
        frame = preprocess(obs)
        stacked_frames = [frame] * 4
        state = np.stack(stacked_frames, axis=2)
        total_reward = 0
        done = False
        steps = 0
        
        print(f"Starting episode {ep+1}")
        
        while not done:
            # Get action with no exploration
            action = agent.act(state, epsilon=0.0)
            
            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_frame = preprocess(next_obs)
            state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            total_reward += reward
            steps += 1
            
            # Small delay to make visualization easier to watch
            if delay > 0:
                time.sleep(delay)
            
        print(f"Episode {ep+1} finished with score: {total_reward} in {steps} steps")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained DQN agent playing Assault")
    parser.add_argument("--model", type=str, default="saved_models/model_episode_50.keras", 
                      help="Path to the saved model")
    parser.add_argument("--episodes", type=int, default=3, 
                      help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.01, 
                      help="Delay between frames (seconds)")
    
    args = parser.parse_args()
    
    # Verify the model exists
    if not os.path.exists(args.model):
        available_models = [f for f in os.listdir("saved_models") if f.endswith(".keras")]
        if available_models:
            print(f"Model {args.model} not found. Available models:")
            for model in available_models:
                print(f"  - saved_models/{model}")
            # Suggest the last saved model
            latest_model = sorted(available_models)[-1]
            print(f"\nUsing latest model: saved_models/{latest_model}")
            args.model = f"saved_models/{latest_model}"
        else:
            print("No saved models found in the saved_models directory.")
            print("Please train the model first using main.py")
            exit(1)
    
    visualize_model(args.model, args.episodes, args.delay)