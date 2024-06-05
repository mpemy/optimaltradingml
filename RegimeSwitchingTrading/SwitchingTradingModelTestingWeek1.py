import array
import os 
import numpy as np
import pandas as pd
import gym
import time 
import datetime
from datetime import timedelta
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

# dfAAPL31 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/AAPLIntradayOf10-31-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf8-1-2017.csv")
# dfGOOG31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/GOOGIntradayOf10-31-2017.csv")
# dfT31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/TIntradayOf10-31-2017.csv")
# dfVZ31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/VZIntradayOf10-31-2017.csv")
# dfXOM31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/XOMIntradayOf10-31-2017.csv")
# dfIBM31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/IBMIntradayOf10-31-2017.csv")
# dfGSPC31 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Code/^GSPCIntradayOf10-31-2017.csv")
dfAAPL8 =pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/AAPLIntradayOf9-8-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf8-1-2017.csv")
dfGOOG8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/GOOGIntradayOf9-8-2017.csv")
dfT8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/TIntradayOf9-8-2017.csv")
dfVZ8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/VZIntradayOf9-8-2017.csv")
dfXOM8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/XOMIntradayOf9-8-2017.csv")
dfIBM8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/IBMIntradayOf9-8-2017.csv")
dfGSPC8 = pd.read_csv("/Users/nazhang/Desktop/Research/MachingLearning-Financial Math/SwitchingTradingPaper/Codes/^GSPCIntradayOf9-8-2017.csv")

resultgspc = [dfGSPC8] # dfGSPC4 is a labor day, so it is empty
resultaapl = [dfAAPL8]
resultgoog = [dfGOOG8]
resultT = [dfT8]
resultvz = [dfVZ8]
resultxom = [dfXOM8]
resultibm = [dfIBM8]

# def AddingDataFrame (list: pd.DataFrame):
#     temp = list[0]
#     for i in range(1, 4):
#         temp = temp.append(list[i])
#     return temp

#test0 = dfGSPC5.add(dfGSPC6)

# test = AddingDataFrame(resultgspc)
# test3= pd.concat(resultgspc)
# print(test3.size)

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
    temp = len(data)
    for j in range(start, min(end, temp)):
        result.append(data[j])
    return result




# define a function to find the corresponding stock prices at specific dates
def StockPrices(data, dates):
    result = []
    for i in dates:
        if i<len(data):
            result.append(data[i])
    return result




# define a function to compute the average return rates for each of the stock:
def MeanReturn(data, dates):
    return_rates = []
    for i in dates:
        if i< len(data):
            r = (data[i] - data[i-1])/data[i-1]
            return_rates.append(r)
    result = sum(return_rates)/len(dates)
    return result

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


corAAPLGOOG  =  covAAPLGOOG / ((covAAPLAPPL ** 0.5) * (covGOOGGOOG ** 0.5))




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


plt.style.use('default')  
fig,ax=plt.subplots()
plt.style.use('ggplot')
ax.plot(vtempAAPL)
ax.set_title("Apple Trading Volumes")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()




def MarketVWAP(i, k):
    
    sumN =0
    sumD =0

    for j in range(0, k):
        sumN = sumN + S[i][j]*abs(V[i][j])
        sumD = sumD + abs(V[i][j])
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result


# define a function to compute the volume for each of the stock
def Volumes(data, dates):
    V_dates= []
    for j in dates:
        if j < len(data):
            V_dates.append(data["Volume"][j] - data["Volume"][j-1])     
    return V_dates





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



Actions_Test=[]
Action_TestAAPL =[]
Action_TestAAPL.append(1)
Actions_Test.append(Action_TestAAPL)  

Action_TestGOOG =[]
Action_TestGOOG.append(1)
Actions_Test.append(Action_TestGOOG) 
   
Action_TestIBM =[]
Action_TestIBM.append(1)
Actions_Test.append(Action_TestIBM)

Action_TestT =[]
Action_TestT.append(1)
Actions_Test.append(Action_TestT) 

Action_TestVZ =[]
Action_TestVZ.append(1)
Actions_Test.append(Action_TestVZ) 

Action_TestXOM =[]
Action_TestXOM.append(1)
Actions_Test.append(Action_TestXOM) 


def TraderVWAP(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S[i][j]*Actions[i][j]#
        sumD = sumD + Actions[i][j]
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result

def TraderVWAPTesting(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S[i][j]*Actions_Test[i][j]
        sumD = sumD + Actions_Test[i][j]
    
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
        sumD = sumD + Actions_Test[i][j]
    
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


class SwitchingTradingTestingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self):
       
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(6,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)

    def stepTesting(self, action, episode):
        
        M0 = 200.0
        M1 = 50.0
        M2 = 200.0
        M3 = 500.0
        M4 = 500.0
        M5 = 200.0
        
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
        realAlloc = [M0*action /100.0, M1 *  action /100.0 , M2* action /100.0, M3* action /100.0, M4* action /100.0, M5* action /100.0]  
        newstock = np.ones([6], dtype=float) # initialize new stock as an array
        reward = 0 # initialize reward as zero
        done = 0
        if k+1 in range(len(S) -startPoint):
            stock =[S[0][k]/s0, S[1][k]/s1, S[2][k]/s2, S[3][k]/s3, S[4][k]/s4, S[5][k]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock =[S[0][k+1]/s0, S[1][k+1]/s1, S[2][k+1]/s2, S[3][k+1]/s3, S[4][k+1]/s4, S[5][k+1]/s5 ]      
    
       
        observation = newstock
        info = self._get_info()

        return observation, reward, done, info 
    
    def newplotter(self, episode): 
        name =''
        for i in range(0,6):
            Trades  = []
            mVWAP = []
            tVWAP = []
            if i==0 :
                name ='AAPL when the market(SP500) is down'
            elif i== 1:
                name = 'GOOG when the market(SP500) is down'
            elif i== 2: 
                name = 'IBM when the market(SP500) is down'
            elif i==3:
                name = 'AT&T when the market(SP500) is down'
            elif i==4:
                name = 'VZ when the market(SP500) is down'
            elif i== 5:
                name ='Exxon Mobil when the market(SP500) is down'
            for j in range(1, episode):
                Trades.append(j)
                mVWAP.append(MarketVWAP(i, j))
                tVWAP.append(TraderVWAP(i, j))
            SwitchingTradingTestingEnv.newgraph(Trades, tVWAP, mVWAP, name)
                                
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
            SwitchingTradingTestingEnv.graph(Trades, tVWAP, mVWAP, name)
            
           

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
    
    def newgraph(trades, traderVWAP, MarketVWAP, stockLabel):
        plt.style.use('default')  
        fig,ax=plt.subplots()
        plt.style.use('ggplot')
        #ax.plot(trades, traderVWAP, MarketVWAP )
        ax.set_title(stockLabel)
        Times = []
        date_and_time = datetime.datetime(2017, 9, 8, 9, 30, 0)
        #date_and_time = datetime.time(9, 30, 0)
        #date_and_time = datetime.datetime(0, 0, 0, 9, 30, 0)
        day = 6.5 *60 *60 # each trade day starts from 9:30 to 4pm, 6.5 hours in total
        for i in range(len(trades)):
            step = day/len(trades) # in seconds
            now = 0 + i *step
            time_change = datetime.timedelta(seconds = int(now))  # originally it is datetime.timedelta
            new_time =date_and_time + time_change
            Times.append(new_time)

        line1, = ax.plot(Times, traderVWAP, label='trader VWAP')
        line2, = ax.plot(Times, MarketVWAP, label='market VWAP')
        ax.legend(handles=[line1, line2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()

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



