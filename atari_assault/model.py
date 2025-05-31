# model.py
import tensorflow as tf
from tensorflow.keras import layers

def create_q_network(input_shape, num_actions):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),  # 84x84x4
        
        # Convolutional layers
        layers.Conv2D(32, 8, strides=4, activation='relu'),
        layers.Conv2D(64, 4, strides=2, activation='relu'),
        layers.Conv2D(64, 3, strides=1, activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_actions)  # Output: Q-values for each action
    ])
    return model