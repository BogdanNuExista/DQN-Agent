import numpy as np
import tensorflow as tf
from model import create_q_network

class DQNAgent:
    def __init__(self, state_shape, num_actions, gamma=0.99, lr=1e-4, log_dir='logs'):
        self.state_shape = state_shape # (84, 84, 4) for Pong
        self.num_actions = num_actions # 6 actions in Pong
        self.gamma = gamma # Discount factor for future rewards

        self.model = create_q_network(state_shape, num_actions)
        self.target_model = create_q_network(state_shape, num_actions)
        self.update_target_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.Huber()
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( 
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.96, staircase=True
        )

        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_step = 0

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions) # if exploring, choose random action
        q_values = self.model(np.expand_dims(state, axis=0), training=False)
        return tf.argmax(q_values[0]).numpy() # if exploiting, choose action with highest Q-value

    def train(self, states, actions, rewards, next_states, dones, weights):
        # Q(s,a) = r + γ * max Q(s',a') 
        next_qs = self.model.predict(next_states) # main network selects actions for next states
        best_actions = tf.argmax(next_qs, axis=1)
        next_target_qs = self.target_model.predict(next_states) # target network evaluates next states
        target_qs = rewards + (1 - dones) * self.gamma * tf.gather(next_target_qs, best_actions, batch_dims=1) # Bellman equation, when dones is True, next state is terminal, so we don't add future rewards

        with tf.GradientTape() as tape: # compute loss and gradients
            qs = self.model(states)
            action_qs = tf.reduce_sum(qs * tf.one_hot(actions, self.num_actions), axis=1)
            td_errors = target_qs - action_qs
            loss = tf.reduce_mean(weights * tf.square(td_errors))

        grads = tape.gradient(loss, self.model.trainable_variables)
        # Filter out None gradients
        valid_grads = [(grad, var) for grad, var in zip(grads, self.model.trainable_variables) if grad is not None]
        clipped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in valid_grads]
        
        # update model weights
        self.optimizer.learning_rate = self.lr_schedule(self.train_step)
        self.optimizer.apply_gradients(clipped_grads)
        
        # tensorboard logging
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.train_step)
            tf.summary.scalar('learning_rate', self.optimizer.learning_rate, step=self.train_step)
        self.train_step += 1
        
        return tf.abs(td_errors)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_network()
