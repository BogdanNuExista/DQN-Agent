# dqn_agent.py
import numpy as np
import tensorflow as tf
from model import create_q_network

class DQNAgent:
    def __init__(self, state_shape, num_actions, gamma=0.99, lr=1e-4, log_dir='logs'):
        self.state_shape = state_shape # input shape (84, 84, 4)
        self.num_actions = num_actions # number of actions (6 for Pong)
        self.gamma = gamma # discount factor for future rewards

        # Create the Q-network and target network
        self.model = create_q_network(state_shape, num_actions)
        self.target_model = create_q_network(state_shape, num_actions)
        self.update_target_network()

        # optimizer with learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.96, staircase=True
        )

        # Logging and tracking
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_step = 0
        self.target_update_freq = 1000 # update target network every 1000 steps

    def update_target_network(self): # Sync target network with main network
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions) # Explore: random action
        q_values = self.model(np.expand_dims(state, axis=0), training=False)  # Exploit: best action according to Q-network
        return tf.argmax(q_values[0]).numpy() 

    @tf.function # # Compile to graph for faster execution
    def train(self, states, actions, rewards, next_states, dones, weights):
        # Convert inputs to tensorflow types
        rewards = tf.cast(rewards, tf.float32)
        weights = tf.cast(weights, tf.float32)
        dones = tf.cast(dones, tf.bool)
        
        # Double DQN: Main net selects action, target net evaluates
        next_qs = self.model(next_states, training=False)
        best_actions = tf.argmax(next_qs, axis=1)
        best_actions = tf.cast(best_actions, tf.int32) 
        next_target_qs = self.target_model(next_states, training=False)
        
        # Create indices for gathering Q-values
        batch_indices = tf.range(tf.shape(best_actions)[0], dtype=tf.int32)
        action_indices = tf.stack([batch_indices, best_actions], axis=1)
        gathered_qs = tf.gather_nd(next_target_qs, action_indices)
        
        # Calculate target Q-values
        done_mask = tf.cast(tf.logical_not(dones), tf.float32)
        target_qs = rewards + done_mask * self.gamma * gathered_qs
        
        # Compute loss with Huber
        with tf.GradientTape() as tape:
            qs = self.model(states)
            action_mask = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            action_qs = tf.reduce_sum(qs * action_mask, axis=1)
            
            td_errors = target_qs - action_qs
            huber_loss = tf.keras.losses.Huber(reduction='none')(target_qs, action_qs)
            loss = tf.reduce_mean(weights * huber_loss) # prioritized experience replay weights

        # Compute and clip gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0) # Prevent exploding gradients
        
        # Apply gradients
        self.optimizer.learning_rate = self.lr_schedule(self.train_step)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Log metrics
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.train_step)
            tf.summary.scalar('learning_rate', self.optimizer.learning_rate, step=self.train_step)
            tf.summary.scalar('avg_q', tf.reduce_mean(action_qs), step=self.train_step)
            tf.summary.scalar('avg_td_error', tf.reduce_mean(tf.abs(td_errors)), step=self.train_step)
        
        self.train_step += 1
        
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return tf.abs(td_errors) # For priority updates

    def save(self, path):
        self.model.save(path) # Save model

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_network()