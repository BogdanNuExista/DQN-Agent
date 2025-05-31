# dqn_agent.py
import numpy as np
import tensorflow as tf
from model import create_q_network

class DQNAgent:
    def __init__(self, state_shape, num_actions, gamma=0.99, lr=1e-4, log_dir='logs'):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma

        # Create Q-network and target network
        self.model = create_q_network(state_shape, num_actions)
        self.target_model = create_q_network(state_shape, num_actions)
        self.update_target_network()

        # Optimizer with learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.96, staircase=True
        )

        # Logging
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_step = 0
        self.target_update_freq = 1000

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model(np.expand_dims(state, axis=0), training=False)
        return tf.argmax(q_values[0]).numpy() 

    @tf.function
    def train(self, states, actions, rewards, next_states, dones, weights):
        rewards = tf.cast(rewards, tf.float32)
        weights = tf.cast(weights, tf.float32)
        dones = tf.cast(dones, tf.bool)
        
        # Double DQN
        next_qs = self.model(next_states, training=False)
        best_actions = tf.argmax(next_qs, axis=1)
        best_actions = tf.cast(best_actions, tf.int32) 
        next_target_qs = self.target_model(next_states, training=False)
        
        batch_indices = tf.range(tf.shape(best_actions)[0], dtype=tf.int32)
        action_indices = tf.stack([batch_indices, best_actions], axis=1)
        gathered_qs = tf.gather_nd(next_target_qs, action_indices)
        
        done_mask = tf.cast(tf.logical_not(dones), tf.float32)
        target_qs = rewards + done_mask * self.gamma * gathered_qs
        
        # Compute loss
        with tf.GradientTape() as tape:
            qs = self.model(states)
            action_mask = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            action_qs = tf.reduce_sum(qs * action_mask, axis=1)
            
            td_errors = target_qs - action_qs
            huber_loss = tf.keras.losses.Huber(reduction='none')(target_qs, action_qs)
            loss = tf.reduce_mean(weights * huber_loss)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        self.optimizer.learning_rate = self.lr_schedule(self.train_step)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Log metrics
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.train_step)
            tf.summary.scalar('avg_q', tf.reduce_mean(action_qs), step=self.train_step)
        
        self.train_step += 1
        
        # Update target network
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return tf.abs(td_errors)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_network()