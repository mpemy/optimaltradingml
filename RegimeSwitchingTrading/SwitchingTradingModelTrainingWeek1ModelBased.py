import array
import os 
import numpy as np
import pandas as pd
import gym
import math
#from google.colab import drive
#drive.mount('/content/drive',force_remount=True)
import keras
from keras.initializers import random_uniform

import tensorflow as tf
#from tensorflow.initializers import random_uniform
import matplotlib.pyplot as plt
##import pickle

##tf.compat.v1.disable_eager_execution()

def load_csv(filepath):
    data =  []
    col = []
    checkcol = False
    with open(filepath) as f:
        for val in f.readlines():
            val = val.replace("\n","")
            val = val.split(',')
            if checkcol is False:
                col = val
                checkcol = True
            else:
                data.append(val)
    df = pd.DataFrame(data=data, columns=col)
    return df

Episode = 0


dfAAPL5 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/AAPLIntradayOf9-5-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf5-1-2017.csv")
dfGOOG5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/GOOGIntradayOf9-5-2017.csv")
dfT5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/TIntradayOf9-5-2017.csv")
dfVZ5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/VZIntradayOf9-5-2017.csv")
dfXOM5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/XOMIntradayOf9-5-2017.csv")
dfIBM5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/IBMIntradayOf9-5-2017.csv")
dfGSPC5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^GSPCIntradayOf9-5-2017.csv")
dfIXIC5 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^IXICIntradayOf9-5-2017.csv")

dfAAPL6 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/AAPLIntradayOf9-6-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf6-1-2017.csv")
dfGOOG6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/GOOGIntradayOf9-6-2017.csv")
dfT6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/TIntradayOf9-6-2017.csv")
dfVZ6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/VZIntradayOf9-6-2017.csv")
dfXOM6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/XOMIntradayOf9-6-2017.csv")
dfIBM6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/IBMIntradayOf9-6-2017.csv")
dfGSPC6 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^GSPCIntradayOf9-6-2017.csv")

dfAAPL7 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/AAPLIntradayOf9-7-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf7-1-2017.csv")
dfGOOG7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/GOOGIntradayOf9-7-2017.csv")
dfT7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/TIntradayOf9-7-2017.csv")
dfVZ7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/VZIntradayOf9-7-2017.csv")
dfXOM7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/XOMIntradayOf9-7-2017.csv")
dfIBM7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/IBMIntradayOf9-7-2017.csv")
dfGSPC7 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^GSPCIntradayOf9-7-2017.csv")


# dfAAPL8 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/AAPLIntradayOf9-8-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf8-1-2017.csv")
# dfGOOG8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/GOOGIntradayOf9-8-2017.csv")
# dfT8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/TIntradayOf9-8-2017.csv")
# dfVZ8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/VZIntradayOf9-8-2017.csv")
# dfXOM8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/XOMIntradayOf9-8-2017.csv")
# dfIBM8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/IBMIntradayOf9-6-2017.csv")
# dfGSPC8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^GSPCIntradayOf9-8-2017.csv")

resultgspc = [dfGSPC5, dfGSPC6, dfGSPC7] # dfGSPC4 is a labor day, so it is empty
resultaapl = [dfAAPL5, dfAAPL6, dfAAPL7]
resultgoog = [dfGOOG5, dfGOOG6, dfGOOG7]
resultT = [dfT5, dfT6, dfT7]
resultvz = [dfVZ5, dfVZ6, dfVZ7]
resultxom = [dfXOM5, dfXOM6, dfXOM7]
resultibm = [dfIBM5, dfIBM6, dfIBM7]

def AddingDataFrame (list: pd.DataFrame):
    temp = list[0]
    for i in range(1, 3):
        temp = temp.append(list[i])
    return temp

test0 = dfGSPC5.add(dfGSPC6)

test = AddingDataFrame(resultgspc)
test3= pd.concat(resultgspc)
print(test3.size)

# dfGSPC =AddingDataFrame(resultgspc)
# dfAAPL =AddingDataFrame(resultaapl)
# dfGOOG =AddingDataFrame(resultgoog)
# dfT =AddingDataFrame(resultT)
# dfVZ =AddingDataFrame(resultvz)
# dfXOM =AddingDataFrame(resultxom)
# dfIBM =AddingDataFrame(resultibm)


dfGSPC =pd.concat(resultgspc, ignore_index= True)
dfAAPL =pd.concat(resultaapl, ignore_index= True)
dfGOOG =pd.concat(resultgoog, ignore_index= True)
dfT =pd.concat(resultT, ignore_index= True)
dfVZ =pd.concat(resultvz, ignore_index= True)
dfXOM =pd.concat(resultxom, ignore_index= True)
dfIBM =pd.concat(resultibm, ignore_index= True)


def find_absolute_extrema(df, column_name='Price'):

  # Convert 'Trade Time' to datetime if it's in string format
  # Convert 'Trade Time' to datetime if it's in string format
  if df['Trade Time'].dtype == 'O':
      df['Trade Time'] = pd.to_datetime(df['Trade Time'])

  # Exclude the specified number of rows from the beginning and end
   # df_excluding_ends = df.iloc[exclude_start: -exclude_end]
  df_excluding_ends = df.iloc[500:-1]


# Find the absolute maximum and minimum values and their indices
  absolute_max_index = df_excluding_ends[column_name].idxmax()
  absolute_max = (absolute_max_index,
                    df_excluding_ends.loc[absolute_max_index][column_name])


  absolute_min_index = df_excluding_ends[column_name].idxmin()
  absolute_min = (absolute_min_index,
                    df_excluding_ends.loc[absolute_min_index][column_name])

  return absolute_max, absolute_min


# dfGSPC is our DataFrame with a 'Trade Time' column and a 'Price' column
absolute_max,absolute_min = find_absolute_extrema(dfGSPC)

# Record positions where absolute max and min happen
position_absolute_max = absolute_max[0]
position_absolute_min = absolute_min[0]



# Print the results
print("Absolute Maximum:")
print(f"At Position Index {absolute_max[0]}: The absolute max of GSPC is {absolute_max[1]}")

print("\nAbsolute Minimum:")
print(f"At Position Index {absolute_min[0]}: The absolute min of GSPC is {absolute_min[1]}")


# Extract the subset of data from the starting point to the absolute minimum
subset_dfGSPC_start_to_min = dfGSPC.iloc[:position_absolute_min]
subset_dfAAPL_start_to_min = dfAAPL.iloc[:position_absolute_min]
subset_dfGOOG_start_to_min = dfGOOG.iloc[:position_absolute_min]
subset_dfIBM_start_to_min = dfIBM.iloc[:position_absolute_min]
subset_dfT_start_to_min = dfT.iloc[:position_absolute_min]
subset_dfVZ_start_to_min = dfVZ.iloc[:position_absolute_min]
subset_dfXOM_start_to_min = dfXOM.iloc[:position_absolute_min]

# Extract the subset of data from the absolute minimum to the absolute maximum
subset_dfGSPC_min_to_max = dfGSPC.iloc[position_absolute_min:position_absolute_max]
subset_dfAAPL_min_to_max = dfAAPL.iloc[position_absolute_min:position_absolute_max]
subset_dfGOOG_min_to_max = dfGOOG.iloc[position_absolute_min:position_absolute_max]
subset_dfIBM_min_to_max = dfIBM.iloc[position_absolute_min:position_absolute_max]
subset_dfT_min_to_max = dfT.iloc[position_absolute_min:position_absolute_max]
subset_dfVZ_min_to_max = dfVZ.iloc[position_absolute_min:position_absolute_max]
subset_dfXOM_min_to_max = dfXOM.iloc[position_absolute_min:position_absolute_max]

# Extract the subset of data from the absolute maximum to the endpoint (last) price
subset_dfGSPC_max_to_end = dfGSPC.iloc[position_absolute_max:]
subset_dfAAPL_max_to_end = dfAAPL.iloc[position_absolute_max:]
subset_dfGOOG_max_to_end = dfGOOG.iloc[position_absolute_max:]
subset_dfIBM_max_to_end = dfIBM.iloc[position_absolute_max:]
subset_dfT_max_to_end = dfT.iloc[position_absolute_max:]
subset_dfVZ_max_to_end = dfVZ.iloc[position_absolute_max:]
subset_dfXOM_max_to_end = dfXOM.iloc[position_absolute_max:]





gspcPrices = dfGSPC["Price"]
#print("Price=", gspcPrices[0])
#print("Price=", gspcPrices)

def MovingAverage (j: int, n: int, data: any):
    sum = 0
    for k in range(j, n+j) :
        #print("data[", k,"]=", data[k])
        #temp = data[k]
        sum = sum + data[k]
    result = sum / n

    return result
gspcAverages = []
batch = 25 
for n in range(len(gspcPrices)- batch) :  #
    temp = MovingAverage(n, batch, gspcPrices)
    gspcAverages.append(temp)

def MovingAverages(data):
    result = []
    for n in range(len(data) - batch) :
        x = MovingAverage(n, batch, data)
        result.append(x)
    return result

def Slopes(data):
    result = []
    for n in range(len(data) - batch) :
        m = (data[n + batch] - data[n] ) / batch
        result.append(m)
    return result

gspc = MovingAverages(dfGSPC["Price"])

mslopes = Slopes(gspc)
dates_pos = [1]
dates_neg = [1]
def SlopeSigns(data):
    pos_pos = 0
    pos_neg = 0
    neg_pos = 0
    neg_neg = 0
    for n in range(len(data)-1):
        if data[n] >= 0 and data[n+1] >= 0 :
            pos_pos += 1
            dates_pos.append(n+1)
        if data[n] >= 0 and data[n+1] <= 0 :
            pos_neg += 1
            dates_neg.append(n+1)
        if data[n] <= 0 and data[n+1] >= 0 :
            neg_pos += 1
            dates_pos.append(n+1)
        if data[n] <= 0 and data[n+1] <= 0 :
            neg_neg += 1
            dates_neg.append(n+1)
    sumEvents = pos_pos + pos_neg + neg_pos + neg_neg
    result = [pos_pos, pos_neg, neg_pos, neg_neg]
    Probabilities = [pos_pos /sumEvents, float(pos_neg) /sumEvents,
                 float(neg_pos) /sumEvents, float(neg_neg) /sumEvents ]
    return Probabilities
    
signs = SlopeSigns(mslopes)


DiscountFactor = 0.99
BaseVolume = 10

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.float32)
    
    def  store_transition(self, state, action, reward, state_, done):
        index =  self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] =state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1-int(done)
        self.mem_cntr +=1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states =self.state_memory[batch]
        new_states =self.new_state_memory[batch]
        actions =self.action_memory[batch]
        rewards =self.reward_memory[batch]
        terminal =self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminal
    
        
class Actor(object) :
    def __init__(self, lr, n_actions, name, input_dims, sess, fcl_dims,
                 fc2_dims, action_bound, batch_size=64, chkpt_dir='C:\temp\TradingPaper\SpyderCodes'):
         
         self.input_dims = input_dims
         self.lr = lr
         self.n_actions = n_actions
         self.name = name
         self.fcl_dims = fcl_dims
         self.fc2_dims = fc2_dims
         self.sess = sess
         self.batch_size = batch_size
         self.action_bound = action_bound
         self.chkpt_dir = chkpt_dir
         self.build_network()
         self.params = tf.compat.v1.trainable_variables(scope=self.name)
         self.saver = tf.compat.v1.train.Saver()
         self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.chkpt')
         
         self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
         self.actor_gradients = list(map(lambda x: tf.compat.v1.div(x, self.batch_size), 
                                self.unnormalized_actor_gradients))
         self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).\
                         apply_gradients(zip(self.actor_gradients, self.params))
         
     
    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape = (self.input_dims,),
                                        name='input')
            self.action_gradient =tf.compat.v1.placeholder(tf.float32, 
                                        shape = (self.n_actions,),
                                        name='input')
            f1 = 1/np.sqrt(self.fcl_dims)
            dense1 = tf.keras.layers.Dense( units=6, 
                                     kernel_initializer =random_uniform(-f1, f1),
                                     bias_initializer = random_uniform(-f1, f1))
          #  batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(dense1)
            
            f2 =1 / np.sqrt(self.fc2_dims)
            dense2 = tf.keras.layers.Dense(layer1_activation, units=self.fc2_dims, 
                                     kernel_initializer = random_uniform(-f2, f2),
                                     bias_initializer = random_uniform(-f2, f2))
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.compat.v1.nn.relu(batch2)
            
            f3=0.003
            mu = tf.compat.v1.layers.dense(layer2_activation, units=self.n_actions,
                                 activation ='tanh',
                                 kernel_initializer = random_uniform(-f3, f3),
                                 bias_initializer = random_uniform(-f3, f3))
            self.mu = tf.compat.v1.multiply(mu,self.action_bound)
            
    def predict(self, inputs):
        self.inputs= inputs
        return self.sess.run(self.mu, feed_dict={self.input:inputs})
    
    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,
                                                self.action_gradient: gradients})
    def save_checkpoint(self):
        print('..saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)
        
    def load_checkpoint(self):
        print('....loading checkpoint...')
        self.saver.restore(self.sess, self.checkpoint_file)
                                 
            
class Agent(object):
    def  __init__(self, alpha, input_dims, tau, env, gamma=0.99, n_actions=2,
                  max_size =1000000, layer1_size=400, layer2_size=300, batch_size = 64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, [input_dims], n_actions)
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high )
        self.target_actor = Actor(alpha, n_actions, 'Target_Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        
        self.update_actor = \
        [self.target_actor.params[i].assign(tf.multiply(self.actor.params[i], self.tau)
                 +tf.multiply(self.target_actor.params[i], 1- self.tau ))
        for i in range(len(self.target_actor.params))]
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        self.update_network_parameters(first=True)
    
    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_actor.sess.run(self.update_actor)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, state):
      #  state = state[np.newaxis,:]
        state = state[0]
        mu =self.actor.predict(state)
        
        return mu[0]
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size :
            return
        state, action, reward, new_state, done =\
        self.memory.sample_buffer(self.batch_size)
        
        actor_value_ =self.target_actor.predict(new_state, self.target_actor.predict(new_state))

        target = []
        for j in range(self.batch_size):
          target.append(reward[j]+self.gamma * actor_value_[j]*done[j] )
        target = np.reshape(target, (self.batch_size, 1))

        a_outs = self.actor.predict(state)
        grads = self.actor_gradients 
        self.actor.train(state, grads[0]) 

        self.updated_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint();
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
      
TransitionProbabilities =  SlopeSigns(mslopes)
 

Episode=0
startPoint = 10
S = []
tempAAPL =[]
for j in range(startPoint, len(dfAAPL["Price"])):
    tempAAPL.append(dfAAPL["Price"][j])

tempGOOG =[]
for j in range(startPoint, len(dfGOOG["Price"])):
    tempGOOG.append(dfGOOG["Price"][j]) 

tempIBM=[]    
for j in range(startPoint, len(dfIBM["Price"])):
    tempIBM.append(dfIBM["Price"][j])     

tempT=[]
for j in range(startPoint, len(dfT["Price"])):
    tempT.append(dfT["Price"][j])     

tempVZ=[]
for j in range(startPoint, len(dfVZ["Price"])):
    tempVZ.append(dfVZ["Price"][j])     

tempXOM=[]
for j in range(startPoint, len(dfXOM["Price"])):
    tempXOM.append(dfXOM["Price"][j])         

S.append(tempAAPL)
S.append(tempGOOG)
S.append(tempIBM)
S.append(tempT)
S.append(tempVZ)
S.append(tempXOM)

# define a function to extract the stock prices from start to abs min, from abs min to abs max, and from abs max to the end
def Extract(data, start, end):
    result = []
    for j in range(start, end):
        result.append(data[j])
    return result

S1 = []
S2 =[]
S3= []


AAPL1 = Extract(dfAAPL["Price"],startPoint,position_absolute_min)
AAPL2 =Extract(dfAAPL["Price"],position_absolute_min,position_absolute_max)
AAPL3 = Extract(dfAAPL["Price"],position_absolute_max,len(dfAAPL))

GOOG1 = Extract(dfGOOG["Price"],startPoint,position_absolute_min)
GOOG2 =Extract(dfGOOG["Price"],position_absolute_min,position_absolute_max)
GOOG3 = Extract(dfGOOG["Price"],position_absolute_max,len(dfGOOG))

IBM1 = Extract(dfIBM["Price"],startPoint,position_absolute_min)
IBM2 =Extract(dfIBM["Price"],position_absolute_min,position_absolute_max)
IBM3 = Extract(dfIBM["Price"],position_absolute_max,len(dfIBM))


T1 = Extract(dfT["Price"],startPoint,position_absolute_min)
T2 =Extract(dfT["Price"],position_absolute_min,position_absolute_max)
T3 = Extract(dfT["Price"],position_absolute_max,len(dfT))


VZ1 = Extract(dfVZ["Price"],startPoint,position_absolute_min)
VZ2 =Extract(dfVZ["Price"],position_absolute_min,position_absolute_max)
VZ3 = Extract(dfVZ["Price"],position_absolute_max,len(dfVZ))

XOM1 = Extract(dfXOM["Price"],startPoint,position_absolute_min)
XOM2 =Extract(dfXOM["Price"],position_absolute_min,position_absolute_max)
XOM3 = Extract(dfXOM["Price"],position_absolute_max,len(dfXOM))

S1.append(AAPL1)
S1.append(GOOG1)
S1.append(IBM1)
S1.append(T1)
S1.append(VZ1)
S1.append(XOM1)



S2.append(AAPL2)
S2.append(GOOG2)
S2.append(IBM2)
S2.append(T2)
S2.append(VZ2)
S2.append(XOM2)


S3.append(AAPL3)
S3.append(GOOG3)
S3.append(IBM3)
S3.append(T3)
S3.append(VZ3)
S3.append(XOM3)




# define a function to find the corresponding stock prices at specific dates
def StockPrices(data, dates):
    result = []
    for i in dates:
        if i<len(data):
            result.append(data[i])
    return result


# define a function to compute the average return rates for each of the stock:
def MeanReturn(data):
    return_rates = []
    for i in range(len(data)):
        r = (data[i] - data[i-1])/data[i-1]
        return_rates.append(r)
    result = sum(return_rates)/len((return_rates))
    return result


# meanAAPL1 =  MeanReturn(AAPL1)
# meanGOOG1 = MeanReturn(GOOG1)
# meanIBM1 = MeanReturn(IBM1)
# meanT1 =  MeanReturn(T1)
# meanVZ1 = MeanReturn(VZ1)
# meanXOM1 = MeanReturn(XOM1)

# print(meanAAPL1)
# print(meanGOOG1)

# define a function to compute the return rates for each of the stock at different dats as a list
def Returns(data, dates):
    return_rates= []
    for i in dates:
        if i < len(data):
            r = (data[i] - data[i-1])/data[i-1]
            return_rates.append(r)     
    return return_rates

rtempAAPL = []
for i in range(1, len(tempAAPL)):
    x = (tempAAPL[i] - tempAAPL[i-1])/tempAAPL[i-1]
    rtempAAPL.append(x)

rtempGOOG = []
for i in range(1, len(tempGOOG)):
    x = (tempGOOG[i] - tempGOOG[i-1])/tempGOOG[i-1]
    rtempGOOG.append(x)

rtempIBM = []
for i in range(1, len(tempIBM)):
    x = (tempIBM[i] - tempIBM[i-1])/tempIBM[i-1]
    rtempIBM.append(x)

rtempVZ = []
for i in range(1, len(tempVZ)):
    x = (tempVZ[i] - tempVZ[i-1])/tempVZ[i-1]
    rtempVZ.append(x)

rtempT = []
for i in range(1, len(tempT)):
    x = (tempT[i] - tempT[i-1])/tempT[i-1]
    rtempT.append(x)

rtempXOM = []
for i in range(1, len(tempXOM)):
    x = (tempXOM[i] - tempXOM[i-1])/tempXOM[i-1]
    rtempXOM.append(x)

meanAAPL = sum(rtempAAPL)/len(rtempAAPL)
meanGOOG = sum(rtempGOOG)/len(rtempGOOG)
meanIBM = sum(rtempIBM)/len(rtempIBM)
meanT = sum(rtempT)/len(rtempT)
meanVZ = sum(rtempVZ)/len(rtempVZ)
meanXOM = sum(rtempXOM)/len(rtempXOM)


# get the return rates for each of the stock from the start to the abs min index, from the abs min index to the abs max index,
# and from the abs max index to the end

rtempAAPL1 = Extract(rtempAAPL,1,position_absolute_min)
rtempAAPL2 =Extract(rtempAAPL,position_absolute_min,position_absolute_max)
rtempAAPL3 = Extract(rtempAAPL,position_absolute_max,len(rtempAAPL))

rtempGOOG1 = Extract(rtempGOOG,1,position_absolute_min)
rtempGOOG2 =Extract(rtempGOOG,position_absolute_min,position_absolute_max)
rtempGOOG3 = Extract(rtempGOOG,position_absolute_max,len(rtempGOOG))

rtempIBM1 = Extract(rtempIBM,1,position_absolute_min)
rtempIBM2 =Extract(rtempIBM,position_absolute_min,position_absolute_max)
rtempIBM3 = Extract(rtempIBM,position_absolute_max,len(rtempIBM))

rtempT1 = Extract(rtempT,1,position_absolute_min)
rtempT2 =Extract(rtempT,position_absolute_min,position_absolute_max)
rtempT3 = Extract(rtempT,position_absolute_max,len(rtempT))

rtempXOM1 = Extract(rtempXOM,1,position_absolute_min)
rtempXOM2 =Extract(rtempXOM,position_absolute_min,position_absolute_max)
rtempXOM3 = Extract(rtempXOM,position_absolute_max,len(rtempXOM))

rtempVZ1 = Extract(rtempVZ,1,position_absolute_min)
rtempVZ2 =Extract(rtempVZ,position_absolute_min,position_absolute_max)
rtempVZ3 = Extract(rtempVZ,position_absolute_max,len(rtempVZ))


meanAAPL1 = sum(rtempAAPL1)/len(rtempAAPL1)
meanGOOG1 = sum(rtempGOOG1)/len(rtempGOOG1)
meanIBM1= sum(rtempIBM1)/len(rtempIBM1)
meanT1 = sum(rtempT1)/len(rtempT1)
meanVZ1 = sum(rtempVZ1)/len(rtempVZ1)
meanXOM1 = sum(rtempXOM1)/len(rtempXOM1)


meanAAPL2 = sum(rtempAAPL2)/len(rtempAAPL2)
meanGOOG2 = sum(rtempGOOG2)/len(rtempGOOG2)
meanIBM2= sum(rtempIBM2)/len(rtempIBM2)
meanT2 = sum(rtempT2)/len(rtempT2)
meanVZ2 = sum(rtempVZ2)/len(rtempVZ2)
meanXOM2 = sum(rtempXOM2)/len(rtempXOM2)



meanAAPL3 = sum(rtempAAPL3)/len(rtempAAPL3)
meanGOOG3 = sum(rtempGOOG3)/len(rtempGOOG3)
meanIBM3= sum(rtempIBM3)/len(rtempIBM3)
meanT3 = sum(rtempT3)/len(rtempT3)
meanVZ3 = sum(rtempVZ3)/len(rtempVZ3)
meanXOM3 = sum(rtempXOM3)/len(rtempXOM3)


covAAPLAPPL1 = sum((a - meanAAPL1) * (b - meanAAPL1) for (a,b) in zip(rtempAAPL1, rtempAAPL1)) / min(len(rtempAAPL1), len(rtempAAPL1)) 
covGOOGAAPL1= sum((a - meanGOOG1) * (b - meanAAPL1) for (a,b) in zip(rtempGOOG1, rtempAAPL1)) / min(len(rtempGOOG1), len(rtempAAPL1)) 
covIBMAAPL1 = sum((a - meanIBM1) * (b - meanAAPL1) for (a,b) in zip(rtempIBM1, rtempAAPL1)) / min(len(rtempIBM1), len(rtempAAPL1)) 
covTAPPL1 = sum((a - meanT1) * (b - meanAAPL1) for (a,b) in zip(rtempT1, rtempAAPL1)) / min(len(rtempT), len(rtempAAPL1)) 
covVZAPPL1 = sum((a - meanVZ1) * (b - meanAAPL1) for (a,b) in zip(rtempVZ1, rtempAAPL1)) / min(len(rtempVZ1), len(rtempAAPL1)) 
covXOMAPPL1 = sum((a - meanXOM1) * (b - meanAAPL1) for (a,b) in zip(rtempXOM1, rtempAAPL1)) / min(len(rtempXOM1), len(rtempAAPL1)) 


covAAPLGOOG1 = sum((a - meanAAPL1) * (b - meanGOOG1) for (a,b) in zip(rtempAAPL1, rtempGOOG1)) / min(len(rtempAAPL1), len(rtempGOOG1)) 
covGOOGGOOG1 = sum((a - meanGOOG1) * (b - meanGOOG1) for (a,b) in zip(rtempGOOG1, rtempGOOG1)) / min(len(rtempGOOG1), len(rtempGOOG1)) 
covIBMGOOG1 = sum((a - meanIBM1) * (b - meanGOOG1) for (a,b) in zip(rtempIBM1, rtempGOOG)) / min(len(rtempIBM1), len(rtempGOOG1)) 
covTGOOG1 = sum((a - meanT1) * (b - meanGOOG1) for (a,b) in zip(rtempT1, rtempGOOG1)) / min(len(rtempT1), len(rtempGOOG1)) 
covVZGOOG1 = sum((a - meanVZ1) * (b - meanGOOG1) for (a,b) in zip(rtempVZ1, rtempGOOG1)) / min(len(rtempVZ1), len(rtempGOOG1)) 
covXOMGOOG1 = sum((a - meanXOM1) * (b - meanGOOG1) for (a,b) in zip(rtempXOM1, rtempGOOG1)) / min(len(rtempXOM1), len(rtempGOOG1)) 

covAAPLIBM1 = sum((a - meanAAPL1) * (b - meanIBM1) for (a,b) in zip(rtempAAPL1, rtempIBM1)) / min(len(rtempAAPL1), len(rtempIBM1)) 
covGOOGIBM1 = sum((a - meanGOOG1) * (b - meanIBM1) for (a,b) in zip(rtempGOOG1, rtempIBM1)) / min(len(rtempGOOG1), len(rtempIBM1)) 
covIBMIBM1 = sum((a - meanIBM1) * (b - meanIBM1) for (a,b) in zip(rtempIBM1, rtempIBM1)) / min(len(rtempIBM1), len(rtempIBM1)) 
covTIBM1 = sum((a - meanT1) * (b - meanIBM1) for (a,b) in zip(rtempT1, rtempIBM1)) / min(len(rtempT1), len(rtempIBM1)) 
covVZIBM1= sum((a - meanVZ1) * (b - meanIBM1) for (a,b) in zip(rtempVZ1, rtempIBM1)) / min(len(rtempVZ1), len(rtempIBM1))
covXOMIBM1 = sum((a - meanXOM1) * (b - meanIBM1) for (a,b) in zip(rtempXOM1, rtempIBM1)) / min(len(rtempXOM1), len(rtempIBM1)) 

covAAPLT1 = sum((a - meanAAPL1) * (b - meanT1) for (a,b) in zip(rtempAAPL1, rtempT1)) / min(len(rtempAAPL1), len(rtempT1)) 
covGOOGT1 = sum((a - meanGOOG1) * (b - meanT1) for (a,b) in zip(rtempGOOG1, rtempT1)) / min(len(rtempGOOG1), len(rtempT1)) 
covIBMT1 = sum((a - meanIBM1) * (b - meanT1) for (a,b) in zip(rtempIBM1, rtempT1)) / min(len(rtempIBM1), len(rtempT1)) 
covTT1 = sum((a - meanT1) * (b - meanT1) for (a,b) in zip(rtempT1, rtempT1)) / min(len(rtempT1), len(rtempT1)) 
covVZT1 = sum((a - meanVZ1) * (b - meanT1) for (a,b) in zip(rtempVZ1, rtempT1)) / min(len(rtempVZ1), len(rtempT1)) 
covXOMT1 = sum((a - meanXOM1) * (b - meanT1) for (a,b) in zip(rtempXOM1, rtempT1)) / min(len(rtempXOM1), len(rtempT1)) 


covAAPLVZ1 = sum((a - meanAAPL1) * (b - meanVZ1) for (a,b) in zip(rtempAAPL1, rtempVZ1)) / min(len(rtempAAPL1), len(rtempVZ1)) 
covGOOGVZ1 = sum((a - meanGOOG1) * (b - meanVZ1) for (a,b) in zip(rtempGOOG1, rtempVZ1)) / min(len(rtempGOOG1), len(rtempVZ1)) 
covIBMVZ1 = sum((a - meanIBM1) * (b - meanVZ1) for (a,b) in zip(rtempIBM1, rtempVZ1)) / min(len(rtempIBM1), len(rtempVZ1)) 
covTVZ1 = sum((a - meanT1) * (b - meanVZ1) for (a,b) in zip(rtempT1, rtempVZ1)) / min(len(rtempT1), len(rtempVZ1)) 
covVZVZ1 = sum((a - meanVZ1) * (b - meanVZ1) for (a,b) in zip(rtempVZ1, rtempVZ1)) / min(len(rtempVZ1), len(rtempVZ1)) 
covXOMVZ1 = sum((a - meanXOM1) * (b - meanVZ1) for (a,b) in zip(rtempXOM1, rtempVZ1)) / min(len(rtempXOM1), len(rtempVZ1)) 



covAAPLXOM1 = sum((a - meanAAPL1) * (b - meanXOM1) for (a,b) in zip(rtempAAPL1, rtempXOM1)) / min(len(rtempAAPL1), len(rtempXOM1)) 
covGOOGXOM1 = sum((a - meanGOOG1) * (b - meanXOM1) for (a,b) in zip(rtempGOOG1, rtempXOM1)) / min(len(rtempGOOG1), len(rtempXOM)) 
covIBMXOM1 = sum((a - meanIBM1) * (b - meanXOM1) for (a,b) in zip(rtempIBM1, rtempXOM1)) / min(len(rtempIBM1), len(rtempXOM1)) 
covTXOM1 = sum((a - meanT1) * (b - meanXOM1) for (a,b) in zip(rtempT1, rtempXOM1)) / min(len(rtempT1), len(rtempXOM1)) 
covVZXOM1 = sum((a - meanVZ1) * (b - meanXOM1) for (a,b) in zip(rtempVZ1, rtempXOM1)) / min(len(rtempVZ1), len(rtempXOM1)) 
covXOMXOM1 = sum((a - meanXOM1) * (b - meanXOM1) for (a,b) in zip(rtempXOM1, rtempXOM1)) / min(len(rtempXOM1), len(rtempXOM1)) 


covAAPLAPPL2 = sum((a - meanAAPL2) * (b - meanAAPL2) for (a,b) in zip(rtempAAPL2, rtempAAPL2)) / min(len(rtempAAPL2), len(rtempAAPL2)) 
covGOOGAAPL2= sum((a - meanGOOG2) * (b - meanAAPL2) for (a,b) in zip(rtempGOOG2, rtempAAPL2)) / min(len(rtempGOOG2), len(rtempAAPL2)) 
covIBMAAPL2 = sum((a - meanIBM2) * (b - meanAAPL2) for (a,b) in zip(rtempIBM2, rtempAAPL2)) / min(len(rtempIBM2), len(rtempAAPL2)) 
covTAPPL2 = sum((a - meanT2) * (b - meanAAPL2) for (a,b) in zip(rtempT2, rtempAAPL2)) / min(len(rtempT), len(rtempAAPL2)) 
covVZAPPL2 = sum((a - meanVZ2) * (b - meanAAPL2) for (a,b) in zip(rtempVZ2, rtempAAPL2)) / min(len(rtempVZ2), len(rtempAAPL2)) 
covXOMAPPL2 = sum((a - meanXOM2) * (b - meanAAPL2) for (a,b) in zip(rtempXOM2, rtempAAPL2)) / min(len(rtempXOM2), len(rtempAAPL2)) 


covAAPLGOOG2 = sum((a - meanAAPL2) * (b - meanGOOG2) for (a,b) in zip(rtempAAPL2, rtempGOOG2)) / min(len(rtempAAPL2), len(rtempGOOG2)) 
covGOOGGOOG2 = sum((a - meanGOOG2) * (b - meanGOOG2) for (a,b) in zip(rtempGOOG2, rtempGOOG2)) / min(len(rtempGOOG2), len(rtempGOOG2)) 
covIBMGOOG2 = sum((a - meanIBM2) * (b - meanGOOG2) for (a,b) in zip(rtempIBM2, rtempGOOG2)) / min(len(rtempIBM2), len(rtempGOOG2)) 
covTGOOG2 = sum((a - meanT2) * (b - meanGOOG2) for (a,b) in zip(rtempT2, rtempGOOG2)) / min(len(rtempT2), len(rtempGOOG2)) 
covVZGOOG2 = sum((a - meanVZ2) * (b - meanGOOG2) for (a,b) in zip(rtempVZ2, rtempGOOG2)) / min(len(rtempVZ2), len(rtempGOOG2)) 
covXOMGOOG2 = sum((a - meanXOM2) * (b - meanGOOG2) for (a,b) in zip(rtempXOM2, rtempGOOG2)) / min(len(rtempXOM2), len(rtempGOOG2)) 

covAAPLIBM2 = sum((a - meanAAPL2) * (b - meanIBM2) for (a,b) in zip(rtempAAPL2, rtempIBM2)) / min(len(rtempAAPL2), len(rtempIBM2)) 
covGOOGIBM2 = sum((a - meanGOOG2) * (b - meanIBM2) for (a,b) in zip(rtempGOOG2, rtempIBM2)) / min(len(rtempGOOG2), len(rtempIBM2)) 
covIBMIBM2 = sum((a - meanIBM2) * (b - meanIBM2) for (a,b) in zip(rtempIBM2, rtempIBM2)) / min(len(rtempIBM2), len(rtempIBM2)) 
covTIBM2 = sum((a - meanT2) * (b - meanIBM2) for (a,b) in zip(rtempT2, rtempIBM2)) / min(len(rtempT2), len(rtempIBM2)) 
covVZIBM2 = sum((a - meanVZ2) * (b - meanIBM2) for (a,b) in zip(rtempVZ2, rtempIBM2)) / min(len(rtempVZ2), len(rtempIBM2))
covXOMIBM2 = sum((a - meanXOM2) * (b - meanIBM2) for (a,b) in zip(rtempXOM2, rtempIBM2)) / min(len(rtempXOM2), len(rtempIBM2)) 

covAAPLT2 = sum((a - meanAAPL2) * (b - meanT2) for (a,b) in zip(rtempAAPL2, rtempT2)) / min(len(rtempAAPL2), len(rtempT2)) 
covGOOGT2 = sum((a - meanGOOG2) * (b - meanT2) for (a,b) in zip(rtempGOOG2, rtempT2)) / min(len(rtempGOOG2), len(rtempT2)) 
covIBMT2 = sum((a - meanIBM2) * (b - meanT2) for (a,b) in zip(rtempIBM2, rtempT2)) / min(len(rtempIBM2), len(rtempT2)) 
covTT2 = sum((a - meanT2) * (b - meanT2) for (a,b) in zip(rtempT2, rtempT2)) / min(len(rtempT2), len(rtempT2)) 
covVZT2 = sum((a - meanVZ2) * (b - meanT2) for (a,b) in zip(rtempVZ2, rtempT2)) / min(len(rtempVZ2), len(rtempT2)) 
covXOMT2 = sum((a - meanXOM2) * (b - meanT2) for (a,b) in zip(rtempXOM2, rtempT2)) / min(len(rtempXOM2), len(rtempT2)) 


covAAPLVZ2 = sum((a - meanAAPL2) * (b - meanVZ2) for (a,b) in zip(rtempAAPL2, rtempVZ2)) / min(len(rtempAAPL2), len(rtempVZ2)) 
covGOOGVZ2 = sum((a - meanGOOG2) * (b - meanVZ2) for (a,b) in zip(rtempGOOG2, rtempVZ2)) / min(len(rtempGOOG2), len(rtempVZ2)) 
covIBMVZ2 = sum((a - meanIBM2) * (b - meanVZ2) for (a,b) in zip(rtempIBM2, rtempVZ2)) / min(len(rtempIBM2), len(rtempVZ2)) 
covTVZ2 = sum((a - meanT2) * (b - meanVZ2) for (a,b) in zip(rtempT2, rtempVZ2)) / min(len(rtempT2), len(rtempVZ2)) 
covVZVZ2 = sum((a - meanVZ2) * (b - meanVZ2) for (a,b) in zip(rtempVZ2, rtempVZ2)) / min(len(rtempVZ2), len(rtempVZ2)) 
covXOMVZ2 = sum((a - meanXOM2) * (b - meanVZ2) for (a,b) in zip(rtempXOM2, rtempVZ2)) / min(len(rtempXOM2), len(rtempVZ2)) 



covAAPLXOM2 = sum((a - meanAAPL2) * (b - meanXOM2) for (a,b) in zip(rtempAAPL2, rtempXOM2)) / min(len(rtempAAPL2), len(rtempXOM2)) 
covGOOGXOM2 = sum((a - meanGOOG2) * (b - meanXOM2) for (a,b) in zip(rtempGOOG2, rtempXOM2)) / min(len(rtempGOOG2), len(rtempXOM2)) 
covIBMXOM2 = sum((a - meanIBM2) * (b - meanXOM2) for (a,b) in zip(rtempIBM2, rtempXOM2)) / min(len(rtempIBM2), len(rtempXOM2)) 
covTXOM2 = sum((a - meanT2) * (b - meanXOM2) for (a,b) in zip(rtempT2, rtempXOM2)) / min(len(rtempT2), len(rtempXOM2)) 
covVZXOM2 = sum((a - meanVZ2) * (b - meanXOM2) for (a,b) in zip(rtempVZ2, rtempXOM2)) / min(len(rtempVZ2), len(rtempXOM2)) 
covXOMXOM2 = sum((a - meanXOM2) * (b - meanXOM2) for (a,b) in zip(rtempXOM2, rtempXOM2)) / min(len(rtempXOM2), len(rtempXOM2)) 


covAAPLAPPL3 = sum((a - meanAAPL3) * (b - meanAAPL3) for (a,b) in zip(rtempAAPL3, rtempAAPL3)) / min(len(rtempAAPL3), len(rtempAAPL3)) 
covGOOGAAPL3= sum((a - meanGOOG3) * (b - meanAAPL3) for (a,b) in zip(rtempGOOG3, rtempAAPL3)) / min(len(rtempGOOG3), len(rtempAAPL3)) 
covIBMAAPL3 = sum((a - meanIBM3) * (b - meanAAPL3) for (a,b) in zip(rtempIBM3, rtempAAPL3)) / min(len(rtempIBM3), len(rtempAAPL3)) 
covTAPPL3 = sum((a - meanT3) * (b - meanAAPL3) for (a,b) in zip(rtempT3, rtempAAPL3)) / min(len(rtempT3), len(rtempAAPL3)) 
covVZAPPL3 = sum((a - meanVZ3) * (b - meanAAPL3) for (a,b) in zip(rtempVZ3, rtempAAPL3)) / min(len(rtempVZ3), len(rtempAAPL3)) 
covXOMAPPL3 = sum((a - meanXOM3) * (b - meanAAPL3) for (a,b) in zip(rtempXOM3, rtempAAPL3)) / min(len(rtempXOM3), len(rtempAAPL3)) 


covAAPLGOOG3 = sum((a - meanAAPL3) * (b - meanGOOG3) for (a,b) in zip(rtempAAPL3, rtempGOOG3)) / min(len(rtempAAPL3), len(rtempGOOG3)) 
covGOOGGOOG3 = sum((a - meanGOOG3) * (b - meanGOOG3) for (a,b) in zip(rtempGOOG3, rtempGOOG3)) / min(len(rtempGOOG3), len(rtempGOOG3)) 
covIBMGOOG3 = sum((a - meanIBM3) * (b - meanGOOG3) for (a,b) in zip(rtempIBM3, rtempGOOG3)) / min(len(rtempIBM3), len(rtempGOOG3)) 
covTGOOG3 = sum((a - meanT3) * (b - meanGOOG3) for (a,b) in zip(rtempT3, rtempGOOG3)) / min(len(rtempT3), len(rtempGOOG3)) 
covVZGOOG3 = sum((a - meanVZ3) * (b - meanGOOG3) for (a,b) in zip(rtempVZ3, rtempGOOG3)) / min(len(rtempVZ3), len(rtempGOOG3)) 
covXOMGOOG3 = sum((a - meanXOM3) * (b - meanGOOG3) for (a,b) in zip(rtempXOM3, rtempGOOG3)) / min(len(rtempXOM3), len(rtempGOOG3)) 

covAAPLIBM3 = sum((a - meanAAPL3) * (b - meanIBM3) for (a,b) in zip(rtempAAPL3, rtempIBM3)) / min(len(rtempAAPL3), len(rtempIBM3)) 
covGOOGIBM3 = sum((a - meanGOOG3) * (b - meanIBM3) for (a,b) in zip(rtempGOOG3, rtempIBM3)) / min(len(rtempGOOG3), len(rtempIBM3)) 
covIBMIBM3 = sum((a - meanIBM3) * (b - meanIBM3) for (a,b) in zip(rtempIBM3, rtempIBM3)) / min(len(rtempIBM3), len(rtempIBM3)) 
covTIBM3 = sum((a - meanT3) * (b - meanIBM3) for (a,b) in zip(rtempT3, rtempIBM3)) / min(len(rtempT3), len(rtempIBM3)) 
covVZIBM3= sum((a - meanVZ3) * (b - meanIBM3) for (a,b) in zip(rtempVZ3, rtempIBM3)) / min(len(rtempVZ3), len(rtempIBM3))
covXOMIBM3 = sum((a - meanXOM3) * (b - meanIBM3) for (a,b) in zip(rtempXOM3, rtempIBM3)) / min(len(rtempXOM3), len(rtempIBM3)) 

covAAPLT3 = sum((a - meanAAPL3) * (b - meanT3) for (a,b) in zip(rtempAAPL3, rtempT3)) / min(len(rtempAAPL3), len(rtempT3)) 
covGOOGT3 = sum((a - meanGOOG3) * (b - meanT3) for (a,b) in zip(rtempGOOG3, rtempT3)) / min(len(rtempGOOG3), len(rtempT3)) 
covIBMT3 = sum((a - meanIBM3) * (b - meanT3) for (a,b) in zip(rtempIBM3, rtempT3)) / min(len(rtempIBM3), len(rtempT3)) 
covTT3 = sum((a - meanT3) * (b - meanT3) for (a,b) in zip(rtempT3, rtempT3)) / min(len(rtempT3), len(rtempT3)) 
covVZT3 = sum((a - meanVZ3) * (b - meanT3) for (a,b) in zip(rtempVZ3, rtempT3)) / min(len(rtempVZ3), len(rtempT3)) 
covXOMT3 = sum((a - meanXOM3) * (b - meanT3) for (a,b) in zip(rtempXOM3, rtempT3)) / min(len(rtempXOM3), len(rtempT3)) 


covAAPLVZ3 = sum((a - meanAAPL3) * (b - meanVZ3) for (a,b) in zip(rtempAAPL3, rtempVZ3)) / min(len(rtempAAPL3), len(rtempVZ3)) 
covGOOGVZ3 = sum((a - meanGOOG3) * (b - meanVZ3) for (a,b) in zip(rtempGOOG3, rtempVZ3)) / min(len(rtempGOOG3), len(rtempVZ3)) 
covIBMVZ3 = sum((a - meanIBM3) * (b - meanVZ3) for (a,b) in zip(rtempIBM3, rtempVZ3)) / min(len(rtempIBM3), len(rtempVZ3)) 
covTVZ3 = sum((a - meanT3) * (b - meanVZ3) for (a,b) in zip(rtempT3, rtempVZ3)) / min(len(rtempT3), len(rtempVZ3)) 
covVZVZ3 = sum((a - meanVZ3) * (b - meanVZ3) for (a,b) in zip(rtempVZ3, rtempVZ3)) / min(len(rtempVZ3), len(rtempVZ3)) 
covXOMVZ3 = sum((a - meanXOM3) * (b - meanVZ3) for (a,b) in zip(rtempXOM3, rtempVZ3)) / min(len(rtempXOM3), len(rtempVZ3)) 



covAAPLXOM3 = sum((a - meanAAPL3) * (b - meanXOM3) for (a,b) in zip(rtempAAPL3, rtempXOM3)) / min(len(rtempAAPL3), len(rtempXOM3)) 
covGOOGXOM3 = sum((a - meanGOOG3) * (b - meanXOM3) for (a,b) in zip(rtempGOOG3, rtempXOM3)) / min(len(rtempGOOG3), len(rtempXOM3)) 
covIBMXOM3 = sum((a - meanIBM3) * (b - meanXOM3) for (a,b) in zip(rtempIBM3, rtempXOM3)) / min(len(rtempIBM3), len(rtempXOM3)) 
covTXOM3 = sum((a - meanT3) * (b - meanXOM3) for (a,b) in zip(rtempT3, rtempXOM3)) / min(len(rtempT3), len(rtempXOM3)) 
covVZXOM3 = sum((a - meanVZ3) * (b - meanXOM3) for (a,b) in zip(rtempVZ3, rtempXOM3)) / min(len(rtempVZ3), len(rtempXOM3)) 
covXOMXOM3 = sum((a - meanXOM3) * (b - meanXOM3) for (a,b) in zip(rtempXOM3, rtempXOM3)) / min(len(rtempXOM3), len(rtempXOM3)) 




mu1= [ meanAAPL1, meanGOOG1, meanIBM1, meanT1, meanVZ1, meanXOM1]
mu2= [ meanAAPL2, meanGOOG2, meanIBM2, meanT2, meanVZ2, meanXOM2]
mu3= [ meanAAPL3, meanGOOG3, meanIBM3, meanT3, meanVZ3, meanXOM3]

sigma1=[]
sigma2=[]
sigma3=[]

sigma1.append([covAAPLAPPL1, covGOOGAAPL1, covIBMAAPL1, covTAPPL1, covVZAPPL1, covXOMAPPL1])
sigma1.append([covAAPLGOOG1, covGOOGGOOG1, covIBMGOOG1, covTGOOG1, covVZGOOG1, covXOMGOOG1])
sigma1.append([covAAPLIBM1, covGOOGIBM1, covIBMIBM1, covTIBM1, covVZIBM1, covXOMIBM1])
sigma1.append([covAAPLT1, covGOOGT1, covIBMT1, covTT1, covVZT1, covXOMT1])
sigma1.append([covAAPLVZ1, covGOOGVZ1, covIBMVZ1, covTVZ1, covVZVZ1, covXOMVZ1])
sigma1.append([covAAPLXOM1, covGOOGXOM1, covIBMXOM1, covTXOM1, covVZXOM1, covXOMXOM1])

sigma2.append([covAAPLAPPL2, covGOOGAAPL2, covIBMAAPL2, covTAPPL2, covVZAPPL2, covXOMAPPL2])
sigma2.append([covAAPLGOOG2, covGOOGGOOG2, covIBMGOOG2, covTGOOG2, covVZGOOG2, covXOMGOOG2])
sigma2.append([covAAPLIBM2, covGOOGIBM2, covIBMIBM2, covTIBM2, covVZIBM2, covXOMIBM2])
sigma2.append([covAAPLT2, covGOOGT2, covIBMT2, covTT2, covVZT2, covXOMT2])
sigma2.append([covAAPLVZ2, covGOOGVZ2, covIBMVZ2, covTVZ2, covVZVZ2, covXOMVZ2])
sigma2.append([covAAPLXOM2, covGOOGXOM2, covIBMXOM2, covTXOM2, covVZXOM2, covXOMXOM2])


sigma3.append([covAAPLAPPL3, covGOOGAAPL3, covIBMAAPL3, covTAPPL3, covVZAPPL3, covXOMAPPL3])
sigma3.append([covAAPLGOOG3, covGOOGGOOG3, covIBMGOOG3, covTGOOG3, covVZGOOG3, covXOMGOOG3])
sigma3.append([covAAPLIBM3, covGOOGIBM3, covIBMIBM3, covTIBM2, covVZIBM2, covXOMIBM2])
sigma3.append([covAAPLT3, covGOOGT3, covIBMT3, covTT3, covVZT3, covXOMT3])
sigma3.append([covAAPLVZ3, covGOOGVZ3, covIBMVZ3, covTVZ3, covVZVZ3, covXOMVZ3])
sigma3.append([covAAPLXOM3, covGOOGXOM3, covIBMXOM3, covTXOM3, covVZXOM3, covXOMXOM3])

# define a function to calculate the model based stock prices
#Calculate the price of the i-th stock at time k+1.
def stock_price_model(S_current, f, action, mu, sigma, x_i, alpha, tau):
  drift_term = (f * action+ mu) * tau
  diffusion_term = np.sqrt(tau) * np.dot(sigma, x_i) * S_current
  S_next = S_current + (drift_term * S_current) + diffusion_term

  return S_next

# define a function to compute stock prices for a basket of stocks based on their mean return and covariance matrix
def compute_stock_prices(S_current_vec, f_vec, action_vec, mu_vec, sigma, x_i, alpha, tau):  #mu_vec is the mean return vector for different stocks
    S_next_vec = np.zeros_like(S_current_vec)
    for i, S_current in enumerate(S_current_vec):
        S_next_vec[i] = stock_price_model(S_current, f_vec[i], action_vec[i], mu_vec[i], sigma[i], x_i, alpha, tau)
    return S_next_vec




# covAAPLAPPL = sum((a - meanAAPL) * (b - meanAAPL) for (a,b) in zip(rtempAAPL, rtempAAPL)) / min(len(rtempAAPL), len(rtempAAPL)) 
# covAAPLGOOG = sum((a - meanAAPL) * (b - meanGOOG) for (a,b) in zip(rtempAAPL, rtempGOOG)) / min(len(rtempAAPL), len(rtempGOOG)) 
# covAAPLIBM = sum((a - meanAAPL) * (b - meanIBM) for (a,b) in zip(rtempAAPL, rtempIBM)) / min(len(rtempAAPL), len(rtempIBM)) 
# covGOOGGOOG = sum((a - meanGOOG) * (b - meanGOOG) for (a,b) in zip(rtempGOOG, rtempGOOG)) / min(len(rtempGOOG), len(rtempGOOG)) 
# covIBMIBM = sum((a - meanIBM) * (b - meanIBM) for (a,b) in zip(rtempIBM, rtempIBM)) / min(len(rtempIBM), len(rtempIBM)) 
# covIBMGOOG = sum((a - meanIBM) * (b - meanGOOG) for (a,b) in zip(rtempIBM, rtempGOOG)) / min(len(rtempIBM), len(rtempGOOG)) 

covAAPLAPPL = sum((a - meanAAPL) * (b - meanAAPL) for (a,b) in zip(rtempAAPL, rtempAAPL)) / min(len(rtempAAPL), len(rtempAAPL)) 
covGOOGAAPL= sum((a - meanGOOG) * (b - meanAAPL) for (a,b) in zip(rtempGOOG, rtempAAPL)) / min(len(rtempGOOG), len(rtempAAPL)) 
covIBMAAPL = sum((a - meanIBM) * (b - meanAAPL) for (a,b) in zip(rtempIBM, rtempAAPL)) / min(len(rtempIBM), len(rtempAAPL)) 
covTAPPL = sum((a - meanT) * (b - meanAAPL) for (a,b) in zip(rtempT, rtempAAPL)) / min(len(rtempT), len(rtempAAPL)) 
covXOMAPPL = sum((a - meanXOM) * (b - meanAAPL) for (a,b) in zip(rtempXOM, rtempAAPL)) / min(len(rtempXOM), len(rtempAAPL)) 
covVZAPPL = sum((a - meanVZ) * (b - meanAAPL) for (a,b) in zip(rtempVZ, rtempAAPL)) / min(len(rtempVZ), len(rtempAAPL))  


covAAPLVZ = sum((a - meanAAPL) * (b - meanVZ) for (a,b) in zip(rtempAAPL, rtempVZ)) / min(len(rtempAAPL), len(rtempVZ)) 
covGOOGVZ = sum((a - meanGOOG) * (b - meanVZ) for (a,b) in zip(rtempGOOG, rtempVZ)) / min(len(rtempGOOG), len(rtempVZ)) 
covIBMVZ = sum((a - meanIBM) * (b - meanVZ) for (a,b) in zip(rtempIBM, rtempVZ)) / min(len(rtempIBM), len(rtempVZ)) 
covTVZ = sum((a - meanT) * (b - meanVZ) for (a,b) in zip(rtempT, rtempVZ)) / min(len(rtempT), len(rtempVZ)) 
covXOMVZ = sum((a - meanXOM) * (b - meanVZ) for (a,b) in zip(rtempXOM, rtempVZ)) / min(len(rtempXOM), len(rtempVZ)) 
covVZVZ = sum((a - meanVZ) * (b - meanVZ) for (a,b) in zip(rtempVZ, rtempVZ)) / min(len(rtempVZ), len(rtempVZ)) 

covAAPLT = sum((a - meanAAPL) * (b - meanT) for (a,b) in zip(rtempAAPL, rtempT)) / min(len(rtempAAPL), len(rtempT)) 
covGOOGT = sum((a - meanGOOG) * (b - meanT) for (a,b) in zip(rtempGOOG, rtempT)) / min(len(rtempGOOG), len(rtempT)) 
covIBMT = sum((a - meanIBM) * (b - meanT) for (a,b) in zip(rtempIBM, rtempT)) / min(len(rtempIBM), len(rtempT)) 
covTT = sum((a - meanT) * (b - meanT) for (a,b) in zip(rtempT, rtempT)) / min(len(rtempT), len(rtempT)) 
covXOMT = sum((a - meanXOM) * (b - meanT) for (a,b) in zip(rtempXOM, rtempT)) / min(len(rtempXOM), len(rtempT)) 
covVZT = sum((a - meanVZ) * (b - meanT) for (a,b) in zip(rtempVZ, rtempT)) / min(len(rtempVZ), len(rtempT)) 

covAAPLXOM = sum((a - meanAAPL) * (b - meanXOM) for (a,b) in zip(rtempAAPL, rtempXOM)) / min(len(rtempAAPL), len(rtempXOM)) 
covGOOGXOM = sum((a - meanGOOG) * (b - meanXOM) for (a,b) in zip(rtempGOOG, rtempXOM)) / min(len(rtempGOOG), len(rtempXOM)) 
covIBMXOM = sum((a - meanIBM) * (b - meanXOM) for (a,b) in zip(rtempIBM, rtempXOM)) / min(len(rtempIBM), len(rtempXOM)) 
covTXOM = sum((a - meanT) * (b - meanXOM) for (a,b) in zip(rtempT, rtempXOM)) / min(len(rtempT), len(rtempXOM)) 
covXOMXOM = sum((a - meanXOM) * (b - meanXOM) for (a,b) in zip(rtempXOM, rtempXOM)) / min(len(rtempXOM), len(rtempXOM)) 
covVZXOM = sum((a - meanVZ) * (b - meanXOM) for (a,b) in zip(rtempVZ, rtempXOM)) / min(len(rtempVZ), len(rtempXOM)) 

covAAPLIBM = sum((a - meanAAPL) * (b - meanIBM) for (a,b) in zip(rtempAAPL, rtempIBM)) / min(len(rtempAAPL), len(rtempIBM)) 
covGOOGIBM = sum((a - meanGOOG) * (b - meanIBM) for (a,b) in zip(rtempGOOG, rtempIBM)) / min(len(rtempGOOG), len(rtempIBM)) 
covIBMIBM = sum((a - meanIBM) * (b - meanIBM) for (a,b) in zip(rtempIBM, rtempIBM)) / min(len(rtempIBM), len(rtempIBM)) 
covTIBM = sum((a - meanT) * (b - meanIBM) for (a,b) in zip(rtempT, rtempIBM)) / min(len(rtempT), len(rtempIBM)) 
covXOMIBM = sum((a - meanXOM) * (b - meanIBM) for (a,b) in zip(rtempXOM, rtempIBM)) / min(len(rtempXOM), len(rtempIBM)) 
covVZIBM= sum((a - meanVZ) * (b - meanIBM) for (a,b) in zip(rtempVZ, rtempIBM)) / min(len(rtempVZ), len(rtempIBM)) 

covAAPLGOOG = sum((a - meanAAPL) * (b - meanGOOG) for (a,b) in zip(rtempAAPL, rtempGOOG)) / min(len(rtempAAPL), len(rtempGOOG)) 
covGOOGGOOG = sum((a - meanGOOG) * (b - meanGOOG) for (a,b) in zip(rtempGOOG, rtempGOOG)) / min(len(rtempGOOG), len(rtempGOOG)) 
covIBMGOOG = sum((a - meanIBM) * (b - meanGOOG) for (a,b) in zip(rtempIBM, rtempGOOG)) / min(len(rtempIBM), len(rtempGOOG)) 
covTGOOG = sum((a - meanT) * (b - meanGOOG) for (a,b) in zip(rtempT, rtempGOOG)) / min(len(rtempT), len(rtempGOOG)) 
covXOMGOOG = sum((a - meanXOM) * (b - meanGOOG) for (a,b) in zip(rtempXOM, rtempGOOG)) / min(len(rtempXOM), len(rtempGOOG)) 
covVZGOOG = sum((a - meanVZ) * (b - meanGOOG) for (a,b) in zip(rtempVZ, rtempGOOG)) / min(len(rtempVZ), len(rtempGOOG)) 


mu= [ meanAAPL, meanGOOG, meanIBM, meanT, meanVZ, meanXOM]
print(mu)
print()
sigma=[]
sigma.append([covAAPLAPPL, covGOOGAAPL, covIBMAAPL, covTAPPL, covVZAPPL, covXOMAPPL])
sigma.append([covAAPLGOOG, covGOOGGOOG, covIBMGOOG, covTGOOG, covVZGOOG, covXOMGOOG])
sigma.append([covAAPLIBM, covGOOGIBM, covIBMIBM, covTIBM, covVZIBM, covXOMIBM])
sigma.append([covAAPLT, covGOOGT, covIBMT, covTT, covVZT, covXOMT])
sigma.append([covAAPLVZ, covGOOGVZ, covIBMVZ, covTVZ, covVZVZ, covXOMVZ])
sigma.append([covAAPLXOM, covGOOGXOM, covIBMXOM, covTXOM, covVZXOM, covXOMXOM])

print(sigma)


corAAPLGOOG  =  covAAPLGOOG / ((covAAPLAPPL ** 0.5) * (covGOOGGOOG ** 0.5))


Returns_AAPL_pos = Returns(tempAAPL, dates_pos)
Returns_GOOG_pos = Returns(tempGOOG, dates_pos)







V = []
vtempAAPL =[]
for j in range(startPoint, len(dfAAPL["Volume"])):
    vtempAAPL.append(dfAAPL["Volume"][j] - dfAAPL["Volume"][j-1])

vtempGOOG =[]
for j in range(startPoint, len(dfGOOG["Volume"])):
    vtempGOOG.append(dfGOOG["Volume"][j] - dfGOOG["Volume"][j-1] ) 

vtempIBM=[]    
for j in range(startPoint, len(dfIBM["Volume"])):
    vtempIBM.append(dfIBM["Volume"][j] - dfIBM["Volume"][j-1])     

vtempT=[]
for j in range(startPoint, len(dfT["Volume"])):
    vtempT.append(dfT["Volume"][j]- dfT["Volume"][j-1])     

vtempVZ=[]
for j in range(startPoint, len(dfVZ["Volume"])):
    vtempVZ.append(dfVZ["Volume"][j] - dfVZ["Volume"][j-1])     

vtempXOM=[]
for j in range(startPoint, len(dfXOM["Volume"])):
    vtempXOM.append(dfXOM["Volume"][j] - dfXOM["Volume"][j-1])         

V.append(vtempAAPL)
V.append(vtempGOOG)
V.append(vtempIBM)
V.append(vtempT)
V.append(vtempVZ)
V.append(vtempXOM)



V1 = []
V2 = []
V3 = []


vtempAAPL1 = Extract(vtempAAPL,1,position_absolute_min)
vtempAAPL2 =Extract(vtempAAPL,position_absolute_min,position_absolute_max)
vtempAAPL3 = Extract(vtempAAPL,position_absolute_max,len(vtempAAPL))

vtempGOOG1 = Extract(vtempGOOG,1,position_absolute_min)
vtempGOOG2 =Extract(vtempGOOG,position_absolute_min,position_absolute_max)
vtempGOOG3 = Extract(vtempGOOG,position_absolute_max,len(vtempGOOG))

vtempIBM1 = Extract(vtempIBM,1,position_absolute_min)
vtempIBM2 =Extract(vtempIBM,position_absolute_min,position_absolute_max)
vtempIBM3 = Extract(vtempIBM,position_absolute_max,len(vtempIBM))

vtempT1 = Extract(vtempT,1,position_absolute_min)
vtempT2 =Extract(vtempT,position_absolute_min,position_absolute_max)
vtempT3 = Extract(vtempT,position_absolute_max,len(vtempT))

vtempXOM1 = Extract(vtempXOM,1,position_absolute_min)
vtempXOM2 =Extract(vtempXOM,position_absolute_min,position_absolute_max)
vtempXOM3 = Extract(vtempXOM,position_absolute_max,len(vtempXOM))

vtempVZ1 = Extract(vtempVZ,1,position_absolute_min)
vtempVZ2 =Extract(vtempVZ,position_absolute_min,position_absolute_max)
vtempVZ3 = Extract(vtempVZ,position_absolute_max,len(vtempVZ))



V1.append(vtempAAPL1)
V1.append(vtempGOOG1)
V1.append(vtempIBM1)
V1.append(vtempT1)
V1.append(vtempVZ1)
V1.append(vtempXOM1)


V2.append(vtempAAPL2)
V2.append(vtempGOOG2)
V2.append(vtempIBM2)
V2.append(vtempT2)
V2.append(vtempVZ2)
V2.append(vtempXOM2)


V3.append(vtempAAPL3)
V3.append(vtempGOOG3)
V3.append(vtempIBM3)
V3.append(vtempT3)
V3.append(vtempVZ3)
V3.append(vtempXOM3)

plt.style.use('default')  
fig,ax=plt.subplots()
plt.style.use('ggplot')
ax.plot(vtempAAPL)
ax.set_title("Apple Trading Volumes")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()



def MarketVWAP(i, k):
    sumN = 0
    sumD = 0

    if k in range(position_absolute_min - startPoint):
        for j in range(0, k):
            sumN += S1[i][j] * abs(V1[i][j])
            sumD += abs(V1[i][j])
    elif k in range(position_absolute_min - startPoint, position_absolute_max - startPoint):
        l=k-(position_absolute_min -startPoint)
        for j in range(0, l):
            sumN += S2[i][j] * abs(V2[i][j])
            sumD += abs(V2[i][j])
    else:
        l = k-(position_absolute_max -startPoint)
        for j in range(0,l):
            sumN += S3[i][j] * abs(V3[i][j])
            sumD += V3[i][j]

    result = 0 if sumD == 0 else sumN / sumD

    return result




# define a function to compute the volume for each of the stock
def Volumes(data, dates):
    V_dates= []
    for j in dates:
        if j < len(data):
            V_dates.append(data["Volume"][j] - data["Volume"][j-1])     
    return V_dates

V_pos = []
Volumes_AAPL_pos = Volumes(dfAAPL, dates_pos)
Volumes_GOOG_pos = Volumes(dfGOOG, dates_pos)




# APPLTrades =[]
# APPLVWAP =[]
# for j in range(200, len(S[2])):
#     APPLTrades.append(j)
#     APPLVWAP.append(MarketVWAP(2,j))
    
#plt.style.use('default')  
#fig,ax=plt.subplots()
#plt.style.use('ggplot')
#ax.plot(APPLTrades, APPLVWAP)
#ax.set_title("test")
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#plt.show()


Actions=[]
ActionAAPL =[]
ActionAAPL.append(1)
Actions.append(ActionAAPL)  

ActionGOOG =[]
ActionGOOG.append(1)
Actions.append(ActionGOOG) 
   
ActionIBM =[]
ActionIBM.append(1)
Actions.append(ActionIBM)

ActionT =[]
ActionT.append(1)
Actions.append(ActionT) 

ActionVZ =[]
ActionVZ.append(1)
Actions.append(ActionVZ) 

ActionXOM =[]
ActionXOM.append(1)
Actions.append(ActionXOM) 





def TraderVWAP(i, k):
    
    sumN =0
    sumD =0
    if k in range(position_absolute_min - startPoint):
        for j in range(0,k):
            sumN = sumN + S1[i][j]*Actions[i][j]
            sumD = sumD + Actions[i][j]
    elif k in range(position_absolute_min - startPoint, position_absolute_max - startPoint):
        l= k - (position_absolute_min -startPoint)
        for j in range(0,l):
            sumN = sumN + S2[i][j]*Actions[i][j]
            sumD = sumD + Actions[i][j]
    else:
        l= k - (position_absolute_max - startPoint)
        for j in range(0,l):
            sumN = sumN + S3[i][j]*Actions[i][j]
            sumD = sumD + Actions[i][j]

    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result

# def TraderVWAP(i, k):
    
#     sumN =0
#     sumD =0
#     for j in range(0, k):
#         sumN = sumN + S1[i][j]*Actions[i][j]
#         sumD = sumD + Actions[i][j]
    
#     if sumD == 0:
#         result =0
#     else:
#         result = sumN / sumD
    
#     return result

def Reward(i, k):    
    
    return (DiscountFactor ** k) * (TraderVWAP(i, k) - MarketVWAP(i, k))

def RewardTotalAnte(k):
    sum = 0
    for i in range(0, 6):
        sum = sum + Reward(i, k)
       
    return sum

def QValue(s, a, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardTotalAnte(k)
       
    return sum
def TraderVWAPReal(i, k, a, s):
    temp = TraderVWAP(i,k)
    sumD =3
    for j in range(0, k):
        sumD = sumD + Actions[i][j]
    
    if a[i] > V[i][k] :
        h=4
           
    Num =  temp * sumD + a[i] * s[i]
    Dem = sumD + a[i]
    
    if Dem == 0:
        result = 0
    else:    
        result= Num/Dem
    
    return result

def RewardReal(i, k, a, s):    
    
    return (DiscountFactor ** k) * (TraderVWAPReal(i, k, a, s) - MarketVWAP(i, k+1))

def RewardTotal(k, a, s):
    sum = 0
    for i in range(0, 6):
        sum = sum + RewardReal(i, k, a, s)
       
    return sum

def QValueRealAll(s, a, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardTotal(k, a, s)
       
    return sum

def QValueReal(i, s, a, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardReal(i, k, a, s)
       
    return sum

def ArgQValueReal(s, p):
    result =np.ones([6], dtype=float)
    for i in range(0, 6):
        v= V[i][p]
        if v<BaseVolume :
            result[i]=-1
        else:
            count = int(v/BaseVolume)   
            a=np.ones(count, dtype=float)
            q=np.ones(count, dtype=float)
            for j in range(0, count):
                a[j]= 0 + BaseVolume * j
                q[j]= QValueReal(i, s[i], a[j], p)
            temp = np.argmax(q)
            val = a[temp]
            result[i]=val  
            
    return result

# Actions is a public list
#Answer = QValueReal(0,250,20, 2)
#aa = ArgQValueReal([100,150,252,100,250,90], 3)
#ab = ArgQValueReal([300,150,252,100,250,90], 2)

Rewards=[]
RewardAAPL =[]
Rewards.append(RewardAAPL)  

RewardGOOG =[]
Rewards.append(RewardGOOG) 
   
RewardIBM =[]
Rewards.append(RewardIBM)

RewardT =[]
Rewards.append(RewardT) 

RewardVZ =[]
Rewards.append(RewardVZ) 

RewardXOM =[]
Rewards.append(RewardXOM) 
'''
for k in range(0, 10):
        stock = np.ones([6], dtype=float)
        stock =[S[0][k], S[1][k], S[2][k], S[3][k], S[4][k], S[5][k] ]
        tempAction = ArgQValueReal(stock, k)        
        Actions[0].append(tempAction[0]) 
        Rewards[0].append(RewardReal(0, k,tempAction[0] , S[0][k]))
        Actions[1].append(tempAction[1])
        Rewards[1].append( RewardReal(1, k,tempAction[1] , S[1][k]))
        Actions[2].append(tempAction[2])
        Rewards[2].append( RewardReal(2, k,tempAction[2] , S[2][k]))
        Actions[3].append(tempAction[3])
        Rewards[3].append( RewardReal(3, k,tempAction[3] , S[3][k]))
        Actions[4].append(tempAction[4])
        Rewards[4].append( RewardReal(4, k,tempAction[4] , S[4][k]))
        Actions[5].append(tempAction[5])
        Rewards[5].append( RewardReal(5, k,tempAction[5] , S[5][k]))       
'''

class BasicEnv(gym.Env):
  def __init__(self):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)
  def step(self, action):
        state = 1
    
        if action == 2:
            reward = 1
        else:
            reward = -1
            
        done = True
        info = {}
        return state, reward, done, info
  def reset(self):
        state = 0
        return state



class SwitchingTradingTrainingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self):
       
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(6,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)

    def step(self, action, episode):
        
        M0 = 200.0
        M1 = 50.0
        M2 = 200.0
        M3 = 500.0
        M4 = 500.0
        M5 = 200.0
        tau = 6.5 *60 *60*4/len(S[1])
        f = [0.0001, 0.0001, -0.0001, -0.0001, 0.0001, -0.0001]
        AllocVect  = np.ones([6], dtype=float)
        AllocVect =[M0, M1, M2, M3, M4, M5]

        k= episode
        Actions[0].append(action * M0/100.0)
        Actions[1].append(action * M1/100.0)
        Actions[2].append(action * M2/100.0)
        Actions[3].append(action * M3/100.0)
        Actions[4].append(action * M4/100.0)
        Actions[5].append(action * M5/100.0) 
        stock = np.ones([6], dtype=float)
        s0 = np.max(S[0]) 
        s1 = np.max(S[1])
        s2 = np.max(S[2])
        s3 = np.max(S[3])
        s4 = np.max(S[4])
        s5 = np.max(S[5])
        #stock =[S[0][k]/s0, S[1][k]/s1, S[2][k]/s2, S[3][k]/s3, S[4][k]/s4, S[5][k]/s5 ]
        realAlloc = [M0*action /100.0, M1 *  action /100.0 , M2* action /100.0, M3* action /100.0, M4* action /100.0, M5* action /100.0]  
        # reward=  RewardTotal(episode, realAlloc, stock)
        newstock = np.ones([6], dtype=float) # initialize new stock as an array
        reward = 0 # initialize reward as zero
        done = 0
        random_noise = np.random.normal(0,1,size=(6,1)) # random noises to compute model based stock prices
        if k+1 in range(position_absolute_min -startPoint):
            stock =[S1[0][k]/s0, S1[1][k]/s1, S1[2][k]/s2, S1[3][k]/s3, S1[4][k]/s4, S1[5][k]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock =compute_stock_prices(stock, f, realAlloc, mu1, sigma1, random_noise,-1, tau)
        elif k+1 in range(position_absolute_min - startPoint, position_absolute_max - startPoint):
            l=k-(position_absolute_min - startPoint)
            stock =[S2[0][l]/s0, S2[1][l]/s1, S2[2][l]/s2, S2[3][l]/s3, S2[4][l]/s4, S2[5][l]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock = compute_stock_prices(stock, f, realAlloc, mu2, sigma2, random_noise,-1, tau)
            #newstock =[S2[0][l+1]/s0, S2[1][l+1]/s1,S2[2][l+1]/s2, S2[3][l+1]/s3, S2[4][l+1]/s4, S2[5][l+1]/s5 ]  
        elif k+1 >= position_absolute_max - startPoint and k < min(len(S3[1]), len(S3[2]), len(S3[3]), len(S3[4]), len(S3[5]), len(S3[0])):
            l=k-(position_absolute_max - startPoint)
            stock =[S3[0][l]/s0, S3[1][l]/s1, S3[2][l]/s2, S3[3][l]/s3, S3[4][l]/s4, S3[5][l]/s5]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock = compute_stock_prices(stock, f,realAlloc, mu3, sigma3, random_noise,-1, tau)
            #newstock =[S3[0][l+1]/s0, S3[1][l+1]/s1, S3[2][l+1]/s2, S3[3][l+1]/s3,S3[4][l+1]/s4, S[5][l+1]/s5 ]                     
        #if k in dates_pos :
        #   Actions_pos[0].append(action * M0/100.0)
        #   Actions_pos[1].append(action * M1/100.0)
        #   Actions_pos[2].append(action * M2/100.0)
        #   Actions_pos[3].append(action * M3/100.0)
        #   Actions_pos[4].append(action * M4/100.0)
        #   Actions_pos[5].append(action * M5/100.0) 
        #  reward=  RewardRealTotal_pos(episode, Actions, stock)
        #else:
        #     Actions_neg[0].append(action * M0/100.0)
        #     Actions_neg[1].append(action * M1/100.0)
        #     Actions_neg[2].append(action * M2/100.0)
        #     Actions_neg[3].append(action * M3/100.0)
        #     Actions_neg[4].append(action * M4/100.0)
        #     Actions_neg[5].append(action * M5/100.0)
        #      reward=  RewardRealTotal_neg(episode, Actions, stock)
        # done = 0   # np.array_equal(self._agent_location, self._target_location)
        # newstock = np.ones([6], dtype=float)
        # newstock =[S[0][k+1]/s0, S[1][k+1]/s1, S[2][k+1]/s2, S[3][k+1]/s3, S[4][k+1]/s4, S[5][k+1]/s5 ]
        observation = newstock
        info = self._get_info()

        
               
    
        return observation, reward, done, info 
    
                


    def newplotter(self, episode): 
        name =''
        if episode in range(position_absolute_min -startPoint):
            for i in range (6):
                Trades  = []
                mVWAP = []
                tVWAP = []
                if i == 0:
                    name ='AAPL when the market is down'
                elif i == 1:
                    name ='GOOG when the market is down'
                elif i == 2:
                    name ='IBM when the market is down'
                elif i == 3:
                    name ='AT&T when the market is down'
                elif i == 4:
                    name ='VZ when the market is down'
                elif i == 5:
                    name ='Exxon Mobil when the market is down'
                for j in range(1, episode):
                    Trades.append(j)
                    mVWAP.append(MarketVWAP(i, j))
                    tVWAP.append(TraderVWAP(i, j))
                SwitchingTradingTrainingEnv.graph(Trades, tVWAP, mVWAP, name)   
        elif episode in range(position_absolute_min - startPoint, position_absolute_max - startPoint):
            for i in range (6):
                Trades  = []
                mVWAP = []
                tVWAP = []
                if i == 0:
                    name ='AAPL when the market is up'
                elif i == 1:
                    name ='GOOG when the market is up'
                elif i == 2:
                    name ='IBM when the market is up'
                elif i == 3:
                    name ='AT&T when the market is up'
                elif i == 4:
                    name ='VZ when the market is up'
                elif i == 5:
                    name ='Exxon Mobil when the market is down'
                for j in range(position_absolute_min - startPoint, episode):
                    Trades.append(j)
                    mVWAP.append(MarketVWAP(i, j))
                    tVWAP.append(TraderVWAP(i, j))
                SwitchingTradingTrainingEnv.graph(Trades, tVWAP, mVWAP, name)   
        elif episode >= position_absolute_max - startPoint and episode < min(len(S3[1]), len(S3[2]), len(S3[3]), len(S3[4]), len(S3[5]), len(S3[0])):
            for i in range (6):
                Trades  = []
                mVWAP = []
                tVWAP = []
                if i == 0:
                    name ='AAPL when the market is down'
                elif i == 1:
                    name ='GOOG when the market is down'
                elif i == 2:
                    name ='IBM when the market is down'
                elif i == 3:
                    name ='AT&T when the market is down'
                elif i == 4:
                    name ='VZ when the market is down'
                elif i == 5:
                    name ='Exxon Mobil when the market is down'
                for j in range(position_absolute_max - startPoint, episode):
                    Trades.append(j)
                    mVWAP.append(MarketVWAP(i, j))
                    tVWAP.append(TraderVWAP(i, j))
                SwitchingTradingTrainingEnv.graph(Trades, tVWAP, mVWAP, name)   

    def plotter(self, episode): 
        name =''
        for i in range(0,6):
            Trades  = []
            mVWAP = []
            tVWAP = []
            if i==0 :
                name ='AAPL'
            elif i== 1:
                name = 'GOOG'
            elif i== 2: 
                name = 'IBM'
            elif i==3:
                name ='VZ'
            elif i==4:
                name ='AT&T'
            elif i== 5:
                name ='Exxon Mobil'
            for j in range(1, episode):
                Trades.append(j)
                mVWAP.append(MarketVWAP(i, j))
                tVWAP.append(TraderVWAP(i, j))
            SwitchingTradingTrainingEnv.graph(Trades, tVWAP, mVWAP, name)
            
           

    def reset(self, episode, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        k= episode
        stock = np.ones([6], dtype=float)
        s0 = np.max(S[0]) 
        s1 = np.max(S[1])
        s2 = np.max(S[2])
        s3 = np.max(S[3])
        s4 = np.max(S[4])
        s5 = np.max(S[5])
        stock =[S[0][k]/s0, S[1][k]/s1, S[2][k]/s2, S[3][k]/s3, S[4][k]/s4, S[5][k]/s5 ]
        observation = stock
        info = self._get_info()
        return (observation, info) 
    
    def _get_info(self):
        return {"ModelVWAP vs MarketVWAP":15}
        
    def _get_obs(self):
        return {"Model":15}
    
    def graph(trades, traderVWAP, MarketVWAP, stockLabel):
        plt.style.use('default')  
        fig,ax=plt.subplots()
        plt.style.use('ggplot')
        #ax.plot(trades, traderVWAP, MarketVWAP )
        ax.set_title(stockLabel)
        line1, = ax.plot(trades, traderVWAP, label='trader VWAP')
        line2, = ax.plot(trades, MarketVWAP, label='market VWAP')
        ax.legend(handles=[line1, line2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()





