import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
from BestTradingModelTandT import TradingEnv 
from BestTradingWithSwitchingDataTesting import TradingExtraDataTestingEnv
from BestTradingModelWithSwitchingdata import TradingExtraDataEnv
##env= gym.make("LunarLander-v2")

env = TradingExtraDataEnv()
#env = TradingEnv()

low = env.observation_space.low
high = env.observation_space.high

class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(2048,activation='relu')
    self.d2 = tf.keras.layers.Dense(1536,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    x = self.d2(x)
    v = self.v(x)
    return v
    

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(2048,activation='relu')
    self.d2 = tf.keras.layers.Dense(1536,activation='relu')
    self.a = tf.keras.layers.Dense(12,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    x = self.d2(x)
    a = self.a(x)
    return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None
        self.stockAllocations = np.ones([12], dtype=np.float64)
        self.stockAllocations =[-30, -20,-10, 0, 10, 20, 40, 60, 70, 80, 90, 100 ]

    def act(self,state):
        #tem = np.asarray([state])
        shape = len(state)
        if shape == 2:
          data = tf.convert_to_tensor([state[0]], dtype= np.float32)
        else:
          data = tf.convert_to_tensor([state], dtype= np.float32)       
        
        #andy =  np.array([state])
        ## print(' data: ', data)
        ## print(' andy:', andy)
        ## print( ' tem:', tem)
        ##andy  = state[0]
        state0 = data     ## tf.convert_to_tensor([andy], dtype= np.float32)
        prob = self.actor(state0)
        #new_state=tf.constant(state, dtype=tf.float32)
        #prob = self.actor(new_state)
        #print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        realAct=int(action.numpy()[0])
        alloc = self.stockAllocations[realAct] 
        return alloc
        # action = np.random.choice([i for i in range(env.action_space.n)], 1, p=prob[0])
        # log_prob = tf.math.log(prob[0][action]).numpy()
        # self.log_prob = log_prob[0]
        # #print(self.log_prob)
        # return action[0]


    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        index =  np.where(self.stockAllocations == action)
        log_prob = dist.log_prob(index)
        loss = -log_prob*td
        return loss



    def learn(self, state, action, reward, next_state, done):

        shape = len(state)
        if shape == 2:
          data = tf.convert_to_tensor([state[0]], dtype= np.float32)
        else:
          data = tf.convert_to_tensor([state], dtype= np.float32)
        state0 = data
        state = np.array([state[0]])
        ##state0 = tf.convert_to_tensor(state[0], dtype= np.float32)
        next_state0 = tf.convert_to_tensor([next_state], dtype= np.float32)
        next_state = np.array([next_state])
       ## next_state0 = tf.convert_to_tensor(next_state[0], dtype= np.float32)
        #self.gamma = tf.convert_to_tensor(0.99, dtype=tf.double)
        #d = 1 - done
        #d = tf.convert_to_tensor(d, dtype=tf.double)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state0, training=True)
             
            #p = self.actor(state, training=True).numpy()[0][action]
            #p = tf.convert_to_tensor([[p]], dtype=tf.float32)
            #print(p)
            v =  self.critic(state0,training=True)
            #v = tf.dtypes.cast(v, tf.double)

            vn = self.critic(next_state0, training=True)
            #vn = tf.dtypes.cast(vn, tf.double)
            td = reward + self.gamma*vn*(1-int(done)) - v
            #print(td)
            #td = tf.math.subtract(tf.math.add(reward, tf.math.multiply(tf.math.multiply(self.gamma, vn), d)), v)
            #a_loss = -self.log_prob*td
            a_loss = self.actor_loss(p, action, td)
            #a_loss = -tf.math.multiply(tf.math.log(p),td)
            #a_loss = tf.keras.losses.categorical_crossentropy(td, p)
            #a_loss = -tf.math.multiply(self.log_prob,td)
            c_loss = td**2
            #c_loss = tf.math.pow(td,2)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

'''
Training the neural network
'''
agentoo7 = agent()
steps = 12
total = 200 #batch size
nber=0
for s in range(steps):
  
  done = False
  state = env.reset(s)
  total_reward = 0
  all_aloss = []
  all_closs = []
  step = 0
  while step<total:
    #env.render()
    action = agentoo7.act(state)
    #print(action)
    temp =  s*total + step
    nber = temp % 1523
    #ans =  env.step(action, nber)
    next_state, reward, done, _ = env.step(action, nber)
    aloss, closs = agentoo7.learn(state, action, reward, next_state, done)
    all_aloss.append(aloss)
    all_closs.append(closs)
    state = next_state
    total_reward += reward
    step = step + 1
    
    
    if step>=200:
      
      #print("total step for this episord are {}".format(t))
      print("total reward after {} steps is {}",s, total_reward)
      #rk = nber
      #env.plotter(rk)
      
      
      ''' Testing'''
      
testingEnv = TradingExtraDataTestingEnv() 
agentoo7 = agent()
steps = 12
total = 200 # batch size
nber=0
for s in range(steps):
  
  done = False
  state = testingEnv.reset(s)
  total_reward = 0
  all_aloss = []
  all_closs = []
  step = 0
  while step<total:
    #env.render()
    action = agentoo7.act(state)
    #print(action)
    temp =  s*total + step
    nber = temp % 1523
    #ans =  env.step(action, nber)
    next_state, reward, done, _ = testingEnv.stepTesting(action, nber)
    aloss, closs = agentoo7.learn(state, action, reward, next_state, done)
    all_aloss.append(aloss)
    all_closs.append(closs)
    state = next_state
    total_reward += reward
    step = step + 1
    
    
    if step>=200:
      
      #print("total step for this episord are {}".format(t))
      print("total reward after {} steps is {}",s, total_reward)
      rk = nber
      testingEnv.plotter_Test(rk)
      
      
      '''
    if s == 200 : 
      print("total reward after {} steps is {}",s, total_reward)
      rk = s
      env.plotter(rk)
      '''
  
      

