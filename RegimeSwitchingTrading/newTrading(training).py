import os 
import collections
import re
#from turtle import Shape
import numpy as np
import statistics
import tensorflow as tf
import tqdm
#from matplotlib import pyplot as plt
from keras import layers
#from tensorflow.python.keras.layers import Dense
from typing import Any, List, Sequence, Tuple
import pandas as pd
from keras.initializers import random_uniform
#from tensorflow.initializers import random_uniform
#import matplotlib.pyplot as plt
from newTradingModelTraining import TradingEnv
import tensorflow_probability as tfp


tf.compat.v1.enable_eager_execution()

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

env = TradingEnv()


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)


num_actions =12  # env.action_space # 2
num_hidden_units = 128*12

model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray, episode: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action, episode)
  return (np.array(state,np.float32), 
          np.array(reward, np.float32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor, episode: int) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action, episode], 
                           [tf.float32, tf.int32, tf.int32])


def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int, episode: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  stockAllocations = np.ones([12], dtype=np.int16)
  stockAllocations =[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220  ]

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
  
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
  
    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    dist = tfp.distributions.Categorical(probs=action_probs_t, dtype=tf.int32)
    rank = dist.sample()
  
    #rank = np.argmax(action_probs_t)
    #print("\n rank:", rank)
    
    action = stockAllocations[rank[0]]  
    #print('\n episode:', episode)
    #print('\n logits:', action_logits_t)
    #print('\n action: ', action )


    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action, episode)
    state.set_shape(initial_state_shape)
    
    #print('\n reward:', reward)
    #print('\n state:', state)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()
  
  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
   """Compute expected returns per timestep."""

   n = tf.shape(rewards)[0]
   #returns = tf.TensorArray(dtype=tf.float32, size=n)
   returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
   # Start from the end of `rewards` and accumulate reward sums
   # into the `returns` array
   rewards = tf.cast(rewards[::-1], dtype=tf.float32)
   discounted_sum = tf.constant(0.0, shape=(6,))
   indexes = np.zeros(shape = (1000,), dtype= np.int32)
   discounted_sum_shape = discounted_sum.shape
   for i in tf.range(n):
     reward = rewards[i]
     np.put(indexes, i, i, mode='raise') 
     discounted_sum = reward + gamma * discounted_sum
     # discounted_sum.set_shape(discounted_sum_shape)
     a = discounted_sum[0]
     b = discounted_sum[1]
     returns = returns.write(i, discounted_sum)
     #returns.write(i, discounted_sum)
     #returns = returns.stack()[::-1]

   if standardize:
        returns = returns.gather(indexes)
        returns = ((returns - tf.math.reduce_mean(returns)) / 
                 (tf.math.reduce_std(returns) + eps))
        

   return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


#@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int, episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode, episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)
      # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    print('\n loss : ', loss)

     # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward

#%%time

min_episodes_criterion = 50     #100
max_episodes = 100     # 1000
max_steps_per_episode = 100     #1000


# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
  for i in t:
    anm = env.reset(episode=i)
    episode = i
    initial_state = tf.constant(env.reset(episode=i)[0], dtype=tf.float32)
    temp  = train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode, episode)
    episode_reward = int(temp)
    
    print('episode: ', i,  ', reward :', episode_reward)

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)
  
    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)
  
    # Show average episode reward every 10 episodes
    #if i % 10 == 0:
    #  pass # print(f'Episode {i}: average reward: {avg_reward}')
  
    #if running_reward > reward_threshold and i >= min_episodes_criterion:  
    #    break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


# Render an episode and save as a GIF file

from IPython import display as ipythondisplay
from PIL import Image
'''
from pyvirtualdisplay import Display


display = Display(visible=0, size=(400, 300))
display.start()


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]
  
  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))
  
    if done:
      break
  
  return images

# Save GIF image
images = render_episode(env, model, max_steps_per_episode)
image_file = 'cartpole-v0.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)

import tensorflow_docs.vis.embed as embed
embed.embed_file(image_file)
'''

