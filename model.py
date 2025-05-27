import tensorflow as tf
from tensorflow.keras import layers

def create_q_network(input_shape, num_actions):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape), # 84x84x4
        layers.Conv2D(32, 8, strides=4, activation='relu'), # 20x20x32 -> (84-8+2*0)/4+1 = 76/4+1 = 20
        layers.Conv2D(64, 4, strides=2, activation='relu'), # 9x9x64
        layers.Conv2D(64, 3, strides=1, activation='relu'), # 7x7x64
        layers.Flatten(), # 3136
        layers.Dense(512, activation='relu'), # 512
        layers.Dense(num_actions)  # q values for each action
    ])
    return model
