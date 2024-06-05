import array
import os 
import numpy as np
import pandas as pd
import gym
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
    sumEvents_pos = pos_pos + pos_neg
    sumEvents_neg = neg_pos + neg_neg
    #sumEvents = pos_pos + pos_neg + neg_pos + neg_neg
    result = [pos_pos, pos_neg, neg_pos, neg_neg]
    Probabilities = [pos_pos /sumEvents_pos, float(pos_neg) /sumEvents_pos,
                 float(neg_pos) /sumEvents_neg, float(neg_neg) /sumEvents_neg ]
    return Probabilities
    
transitionmatrix = SlopeSigns(mslopes)

print(transitionmatrix)

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


S_pos = []
S_neg = []
S_pos.append(StockPrices(dfAAPL["Price"], dates_pos))
S_pos.append(StockPrices(dfGOOG["Price"], dates_pos))
S_pos.append(StockPrices(dfIBM["Price"], dates_pos))    
S_pos.append(StockPrices(dfT["Price"], dates_pos))
S_pos.append(StockPrices(dfVZ["Price"], dates_pos))
S_pos.append(StockPrices(dfXOM["Price"], dates_pos))


S_neg.append(StockPrices(dfAAPL["Price"], dates_neg))
S_neg.append(StockPrices(dfGOOG["Price"], dates_neg))
S_neg.append(StockPrices(dfIBM["Price"], dates_neg))
S_neg.append(StockPrices(dfT["Price"], dates_neg))
S_neg.append(StockPrices(dfVZ["Price"], dates_neg))
S_neg.append(StockPrices(dfXOM["Price"], dates_neg))


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


Returns_AAPL_pos = Returns(tempAAPL, dates_pos)
Returns_GOOG_pos = Returns(tempGOOG, dates_pos)
Returns_T_pos = Returns(tempT, dates_pos)
Returns_VZ_pos = Returns(tempVZ, dates_pos)
Returns_XOM_pos = Returns(tempXOM, dates_pos)
Returns_IBM_pos = Returns(tempIBM, dates_pos)

Returns_AAPL_neg = Returns(tempAAPL, dates_neg)
Returns_GOOG_neg = Returns(tempGOOG, dates_neg)
Returns_T_neg = Returns(tempT, dates_neg)
Returns_VZ_neg = Returns(tempVZ, dates_neg)
Returns_XOM_neg = Returns(tempXOM, dates_neg)
Returns_IBM_neg = Returns(tempIBM, dates_neg)



meanAAPL_pos = MeanReturn(tempAAPL, dates_pos)    
meanGOOG_pos = MeanReturn(tempGOOG, dates_pos)  
meanT_pos = MeanReturn(tempT, dates_pos)  
meanVZ_pos = MeanReturn(tempVZ, dates_pos)  
meanXOM_pos = MeanReturn(tempXOM, dates_pos)  
meanIBM_pos = MeanReturn(tempIBM, dates_pos)  

meanAAPL_neg = MeanReturn(tempAAPL, dates_neg)    
meanGOOG_neg = MeanReturn(tempGOOG, dates_neg)  
meanT_neg = MeanReturn(tempT, dates_neg)  
meanVZ_neg = MeanReturn(tempVZ, dates_neg)  
meanXOM_neg = MeanReturn(tempXOM, dates_neg)  
meanIBM_neg = MeanReturn(tempIBM, dates_neg) 



covAAPLAPPL_pos = sum((a - meanAAPL_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_AAPL_pos)) / min(len(Returns_AAPL_pos), len(Returns_AAPL_pos)) 
covGOOGAAPL_pos= sum((a - meanGOOG_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_AAPL_pos)) / min(len(Returns_GOOG_pos), len(Returns_AAPL_pos)) 
covTAPPL_pos = sum((a - meanT_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_T_pos, Returns_AAPL_pos)) / min(len(Returns_T_pos), len(Returns_AAPL_pos)) 
covVZAPPL_pos = sum((a - meanVZ_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_VZ_pos, Returns_AAPL_pos)) / min(len(Returns_VZ_pos), len(Returns_AAPL_pos))  
covXOMAPPL_pos = sum((a - meanXOM_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_XOM_pos, Returns_AAPL_pos)) / min(len(Returns_XOM_pos), len(Returns_AAPL_pos)) 
covIBMAAPL_pos = sum((a - meanIBM_pos) * (b - meanAAPL_pos) for (a,b) in zip(Returns_IBM_pos, Returns_AAPL_pos)) / min(len(Returns_IBM_pos), len(Returns_AAPL_pos)) 

covAAPLGOOG_pos = sum((a - meanAAPL_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_GOOG_pos)) / min(len(Returns_AAPL_pos), len(Returns_GOOG_pos)) 
covGOOGGOOG_pos = sum((a - meanGOOG_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_GOOG_pos)) / min(len(Returns_GOOG_pos), len(Returns_GOOG_pos)) 
covTGOOG_pos = sum((a - meanT_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_T_pos, Returns_GOOG_pos)) / min(len(Returns_T_pos), len(Returns_GOOG_pos)) 
covVZGOOG_pos = sum((a - meanVZ_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_VZ_pos, Returns_GOOG_pos)) / min(len(Returns_VZ_pos), len(Returns_GOOG_pos)) 
covXOMGOOG_pos = sum((a - meanXOM_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_XOM_pos, Returns_GOOG_pos)) / min(len(Returns_XOM_pos), len(Returns_GOOG_pos)) 
covIBMGOOG_pos = sum((a - meanIBM_pos) * (b - meanGOOG_pos) for (a,b) in zip(Returns_IBM_pos, Returns_GOOG_pos)) / min(len(Returns_IBM_pos), len(Returns_GOOG_pos))
 

covAAPLT_pos = sum((a - meanAAPL_pos) * (b - meanT_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_T_pos)) / min(len(Returns_AAPL_pos), len(Returns_T_pos)) 
covGOOGT_pos = sum((a - meanGOOG_pos) * (b - meanT_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_T_pos)) / min(len(Returns_GOOG_pos), len(Returns_T_pos)) 
covTT_pos = sum((a - meanT_pos) * (b - meanT_pos) for (a,b) in zip(Returns_T_pos, Returns_T_pos)) / min(len(Returns_T_pos), len(Returns_T_pos)) 
covVZT_pos = sum((a - meanVZ_pos) * (b - meanT_pos) for (a,b) in zip(Returns_VZ_pos, Returns_T_pos)) / min(len(Returns_VZ_pos), len(Returns_T_pos))
covXOMT_pos = sum((a - meanXOM_pos) * (b - meanT_pos) for (a,b) in zip(Returns_XOM_pos, Returns_T_pos)) / min(len(Returns_XOM_pos), len(Returns_T_pos))  
covIBMT_pos = sum((a - meanIBM_pos) * (b - meanT_pos) for (a,b) in zip(Returns_IBM_pos, Returns_T_pos)) / min(len(Returns_IBM_pos), len(Returns_T_pos)) 

covAAPLVZ_pos = sum((a - meanAAPL_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_VZ_pos)) / min(len(Returns_AAPL_pos), len(Returns_VZ_pos)) 
covGOOGVZ_pos = sum((a - meanGOOG_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_VZ_pos)) / min(len(Returns_GOOG_pos), len(Returns_VZ_pos)) 
covTVZ_pos = sum((a - meanT_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_T_pos, Returns_VZ_pos)) / min(len(Returns_T_pos), len(Returns_VZ_pos)) 
covVZVZ_pos = sum((a - meanVZ_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_VZ_pos, Returns_VZ_pos)) / min(len(Returns_VZ_pos), len(Returns_VZ_pos)) 
covXOMVZ_pos = sum((a - meanXOM_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_XOM_pos, Returns_VZ_pos)) / min(len(Returns_XOM_pos), len(Returns_VZ_pos)) 
covIBMVZ_pos = sum((a - meanIBM_pos) * (b - meanVZ_pos) for (a,b) in zip(Returns_IBM_pos, Returns_VZ_pos)) / min(len(Returns_IBM_pos), len(Returns_VZ_pos)) 



covAAPLXOM_pos = sum((a - meanAAPL_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_XOM_pos)) / min(len(Returns_AAPL_pos), len(Returns_XOM_pos)) 
covGOOGXOM_pos = sum((a - meanGOOG_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_XOM_pos)) / min(len(Returns_GOOG_pos), len(Returns_XOM_pos)) 
covTXOM_pos = sum((a - meanT_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_T_pos, Returns_XOM_pos)) / min(len(Returns_T_pos), len(Returns_XOM_pos))
covVZXOM_pos = sum((a - meanVZ_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_VZ_pos, Returns_XOM_pos)) / min(len(Returns_VZ_pos), len(Returns_XOM_pos))  
covXOMXOM_pos = sum((a - meanXOM_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_XOM_pos, Returns_XOM_pos)) / min(len(Returns_XOM_pos), len(Returns_XOM_pos)) 
covIBMXOM_pos = sum((a - meanIBM_pos) * (b - meanXOM_pos) for (a,b) in zip(Returns_IBM_pos, Returns_XOM_pos)) / min(len(Returns_IBM_pos), len(Returns_XOM_pos)) 


covAAPLIBM_pos = sum((a - meanAAPL_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_AAPL_pos, Returns_IBM_pos)) / min(len(Returns_AAPL_pos), len(Returns_IBM_pos)) 
covGOOGIBM_pos = sum((a - meanGOOG_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_GOOG_pos, Returns_IBM_pos)) / min(len(Returns_GOOG_pos), len(Returns_IBM_pos)) 
covTIBM_pos = sum((a - meanT_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_T_pos, Returns_IBM_pos)) / min(len(Returns_T_pos), len(Returns_IBM_pos)) 
covVZIBM_pos = sum((a - meanVZ_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_VZ_pos, Returns_IBM_pos)) / min(len(Returns_VZ_pos), len(Returns_IBM_pos)) 
covXOMIBM_pos = sum((a - meanXOM_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_XOM_pos, Returns_IBM_pos)) / min(len(Returns_XOM_pos), len(Returns_IBM_pos)) 
covIBMIBM_pos = sum((a - meanIBM_pos) * (b - meanIBM_pos) for (a,b) in zip(Returns_IBM_pos,Returns_IBM_pos)) / min(len(Returns_IBM_pos), len(Returns_IBM_pos)) 


covAAPLAPPL_neg = sum((a - meanAAPL_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_AAPL_neg)) / min(len(Returns_AAPL_neg), len(Returns_AAPL_neg)) 
covGOOGAAPL_neg= sum((a - meanGOOG_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_AAPL_neg)) / min(len(Returns_GOOG_neg), len(Returns_AAPL_neg)) 
covTAPPL_neg = sum((a - meanT_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_T_neg, Returns_AAPL_neg)) / min(len(Returns_T_neg), len(Returns_AAPL_neg)) 
covVZAPPL_neg = sum((a - meanVZ_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_VZ_neg, Returns_AAPL_neg)) / min(len(Returns_VZ_neg), len(Returns_AAPL_neg))  
covXOMAPPL_neg = sum((a - meanXOM_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_XOM_neg, Returns_AAPL_neg)) / min(len(Returns_XOM_neg), len(Returns_AAPL_neg)) 
covIBMAAPL_neg = sum((a - meanIBM_neg) * (b - meanAAPL_neg) for (a,b) in zip(Returns_IBM_neg, Returns_AAPL_neg)) / min(len(Returns_IBM_neg), len(Returns_AAPL_neg)) 

covAAPLGOOG_neg = sum((a - meanAAPL_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_GOOG_neg)) / min(len(Returns_AAPL_neg), len(Returns_GOOG_neg)) 
covGOOGGOOG_neg = sum((a - meanGOOG_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_GOOG_neg)) / min(len(Returns_GOOG_neg), len(Returns_GOOG_neg)) 
covTGOOG_neg = sum((a - meanT_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_T_neg, Returns_GOOG_neg)) / min(len(Returns_T_neg), len(Returns_GOOG_neg)) 
covVZGOOG_neg = sum((a - meanVZ_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_VZ_neg, Returns_GOOG_neg)) / min(len(Returns_VZ_neg), len(Returns_GOOG_neg)) 
covXOMGOOG_neg = sum((a - meanXOM_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_XOM_neg, Returns_GOOG_neg)) / min(len(Returns_XOM_neg), len(Returns_GOOG_neg)) 
covIBMGOOG_neg = sum((a - meanIBM_neg) * (b - meanGOOG_neg) for (a,b) in zip(Returns_IBM_neg, Returns_GOOG_neg)) / min(len(Returns_IBM_neg), len(Returns_GOOG_neg))
 

covAAPLT_neg = sum((a - meanAAPL_neg) * (b - meanT_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_T_neg)) / min(len(Returns_AAPL_neg), len(Returns_T_neg)) 
covGOOGT_neg = sum((a - meanGOOG_neg) * (b - meanT_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_T_neg)) / min(len(Returns_GOOG_neg), len(Returns_T_neg)) 
covTT_neg = sum((a - meanT_neg) * (b - meanT_neg) for (a,b) in zip(Returns_T_neg, Returns_T_neg)) / min(len(Returns_T_neg), len(Returns_T_neg)) 
covVZT_neg = sum((a - meanVZ_neg) * (b - meanT_neg) for (a,b) in zip(Returns_VZ_neg, Returns_T_neg)) / min(len(Returns_VZ_neg), len(Returns_T_neg))
covXOMT_neg = sum((a - meanXOM_neg) * (b - meanT_neg) for (a,b) in zip(Returns_XOM_neg, Returns_T_neg)) / min(len(Returns_XOM_neg), len(Returns_T_neg))  
covIBMT_neg = sum((a - meanIBM_neg) * (b - meanT_neg) for (a,b) in zip(Returns_IBM_neg, Returns_T_neg)) / min(len(Returns_IBM_neg), len(Returns_T_neg)) 

covAAPLVZ_neg = sum((a - meanAAPL_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_VZ_neg)) / min(len(Returns_AAPL_neg), len(Returns_VZ_neg)) 
covGOOGVZ_neg = sum((a - meanGOOG_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_VZ_neg)) / min(len(Returns_GOOG_neg), len(Returns_VZ_neg)) 
covTVZ_neg = sum((a - meanT_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_T_neg, Returns_VZ_neg)) / min(len(Returns_T_neg), len(Returns_VZ_neg)) 
covVZVZ_neg = sum((a - meanVZ_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_VZ_neg, Returns_VZ_neg)) / min(len(Returns_VZ_neg), len(Returns_VZ_neg)) 
covXOMVZ_neg = sum((a - meanXOM_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_XOM_neg, Returns_VZ_neg)) / min(len(Returns_XOM_neg), len(Returns_VZ_neg)) 
covIBMVZ_neg = sum((a - meanIBM_neg) * (b - meanVZ_neg) for (a,b) in zip(Returns_IBM_neg, Returns_VZ_neg)) / min(len(Returns_IBM_neg), len(Returns_VZ_neg)) 



covAAPLXOM_neg = sum((a - meanAAPL_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_XOM_neg)) / min(len(Returns_AAPL_neg), len(Returns_XOM_neg)) 
covGOOGXOM_neg = sum((a - meanGOOG_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_XOM_neg)) / min(len(Returns_GOOG_neg), len(Returns_XOM_neg)) 
covTXOM_neg = sum((a - meanT_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_T_neg, Returns_XOM_neg)) / min(len(Returns_T_neg), len(Returns_XOM_neg))
covVZXOM_neg = sum((a - meanVZ_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_VZ_neg, Returns_XOM_neg)) / min(len(Returns_VZ_neg), len(Returns_XOM_neg))  
covXOMXOM_neg = sum((a - meanXOM_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_XOM_neg, Returns_XOM_neg)) / min(len(Returns_XOM_neg), len(Returns_XOM_neg)) 
covIBMXOM_neg = sum((a - meanIBM_neg) * (b - meanXOM_neg) for (a,b) in zip(Returns_IBM_neg, Returns_XOM_neg)) / min(len(Returns_IBM_neg), len(Returns_XOM_neg)) 


covAAPLIBM_neg = sum((a - meanAAPL_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_AAPL_neg, Returns_IBM_neg)) / min(len(Returns_AAPL_neg), len(Returns_IBM_neg)) 
covGOOGIBM_neg = sum((a - meanGOOG_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_GOOG_neg, Returns_IBM_neg)) / min(len(Returns_GOOG_neg), len(Returns_IBM_neg)) 
covTIBM_neg = sum((a - meanT_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_T_neg, Returns_IBM_neg)) / min(len(Returns_T_neg), len(Returns_IBM_neg)) 
covVZIBM_neg = sum((a - meanVZ_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_VZ_neg, Returns_IBM_neg)) / min(len(Returns_VZ_neg), len(Returns_IBM_neg)) 
covXOMIBM_neg = sum((a - meanXOM_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_XOM_neg, Returns_IBM_neg)) / min(len(Returns_XOM_neg), len(Returns_IBM_neg)) 
covIBMIBM_neg = sum((a - meanIBM_neg) * (b - meanIBM_neg) for (a,b) in zip(Returns_IBM_neg,Returns_IBM_neg)) / min(len(Returns_IBM_neg), len(Returns_IBM_neg)) 



corAAPLGOOG_pos  =  covAAPLGOOG_pos / ((covAAPLAPPL_pos ** 0.5) * (covGOOGGOOG_pos ** 0.5))
corAAPLGOOG_neg  =  covAAPLGOOG_neg / ((covAAPLAPPL_neg ** 0.5) * (covGOOGGOOG_neg ** 0.5))


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


# def MarketVWAP(i, k):
    
#     sumN =0
#     sumD =0

#     for j in range(0, k):
#         sumN = sumN + S[i][j]*V[i][j]
#         sumD = sumD + V[i][j]
    
#     if sumD == 0:
#         result =0
#     else:
#         result = sumN / sumD
    
#     return result


# define a function to compute the volume for each of the stock
def Volumes(data, dates):
    V_dates= []
    for j in dates:
        if j < len(data):
            V_dates.append(data["Volume"][j] - data["Volume"][j-1])     
    return V_dates

V_pos = []
V_neg = []
Volumes_AAPL_pos = Volumes(dfAAPL, dates_pos)
Volumes_GOOG_pos = Volumes(dfGOOG, dates_pos)
Volumes_T_pos = Volumes(dfT, dates_pos)
Volumes_VZ_pos = Volumes(dfVZ, dates_pos)
Volumes_XOM_pos = Volumes(dfXOM, dates_pos)
Volumes_IBM_pos = Volumes(dfIBM, dates_pos)

Volumes_AAPL_neg = Volumes(dfAAPL, dates_neg)
Volumes_GOOG_neg = Volumes(dfGOOG, dates_neg)
Volumes_T_neg = Volumes(dfT, dates_neg)
Volumes_VZ_neg = Volumes(dfVZ, dates_neg)
Volumes_XOM_neg = Volumes(dfXOM, dates_neg)
Volumes_IBM_neg = Volumes(dfIBM, dates_neg)

V_pos.append(Volumes_AAPL_pos)
V_pos.append(Volumes_GOOG_pos)
V_pos.append(Volumes_IBM_pos)
V_pos.append(Volumes_T_pos)
V_pos.append(Volumes_VZ_pos)
V_pos.append(Volumes_XOM_pos)



V_neg.append(Volumes_AAPL_neg)
V_neg.append(Volumes_GOOG_neg)
V_neg.append(Volumes_IBM_neg)
V_neg.append(Volumes_T_neg)
V_neg.append(Volumes_VZ_neg)
V_neg.append(Volumes_XOM_neg)





def MarketVWAP_pos(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S_pos[i][j]*V_pos[i][j]
        sumD = sumD + V_pos[i][j]
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result


def MarketVWAP_neg(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S_neg[i][j]*V_neg[i][j]
        sumD = sumD + V_neg[i][j]
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result


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

## New logic

Actions_pos=[]
ActionAAPL_pos =[]
ActionAAPL_pos.append(1)
Actions_pos.append(ActionAAPL_pos)  

ActionGOOG_pos =[]
ActionGOOG_pos.append(1)
Actions_pos.append(ActionGOOG_pos) 
   
ActionIBM_pos =[]
ActionIBM_pos.append(1)
Actions_pos.append(ActionIBM_pos)

ActionT_pos =[]
ActionT_pos.append(1)
Actions_pos.append(ActionT_pos) 

ActionVZ_pos =[]
ActionVZ_pos.append(1)
Actions_pos.append(ActionVZ_pos) 

ActionXOM_pos =[]
ActionXOM_pos.append(1)
Actions_pos.append(ActionXOM_pos) 

Actions_neg=[]
ActionAAPL_neg =[]
ActionAAPL_neg.append(1)
Actions_neg.append(ActionAAPL_neg)  

ActionGOOG_neg =[]
ActionGOOG_neg.append(1)
Actions_neg.append(ActionGOOG_neg) 
   
ActionIBM_neg =[]
ActionIBM_neg.append(1)
Actions_neg.append(ActionIBM_neg)

ActionT_neg =[]
ActionT_neg.append(1)
Actions_neg.append(ActionT_neg) 

ActionVZ_neg =[]
ActionVZ_neg.append(1)
Actions_neg.append(ActionVZ_neg) 

ActionXOM_neg =[]
ActionXOM_neg.append(1)
Actions_neg.append(ActionXOM_neg) 


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
    
    return (DiscountFactor ** k) * (TraderVWAPReal(i, k, a, s) - 1.05*MarketVWAP(i, k+1))

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

def TraderVWAP_pos(i, k, actions): # actions is an array of array
    
    sumN =0
    sumD =0
    for day in dates_pos:
        if day <= k :
            sumN = sumN + S[i][day]*actions[day]
            sumD = sumD + actions[day]
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result


def TraderVWAP_neg(i, k, actions):
    
    sumN =0
    sumD =0
    for day in dates_neg:
        if day <= k :
            sumN = sumN + S[i][day]*actions[day]
            sumD = sumD + actions[day]
    
    if sumD == 0:
        result =0
    else:
        result = sumN / sumD
    
    return result

def Reward_pos(i, k, actions):    
    
    return (DiscountFactor ** k) * (TraderVWAP_pos(i, k, actions) - MarketVWAP_pos(i, k))

def Reward_neg(i, k, actions):    
    
    return (DiscountFactor ** k) * (TraderVWAP_neg(i, k, actions) - MarketVWAP_neg(i, k))

def RewardTotalAnte_pos(k, actions):
    sum = 0
    for i in range(0, 6):
        sum = sum + Reward_pos(i, k, actions)
       
    return sum

def RewardTotalAnte_neg(k, actions):
    sum = 0
    for i in range(0, 6):
        sum = sum + Reward_neg(i, k, actions)
       
    return sum

def QValue_pos(s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardTotalAnte_pos(k, actions) 
    return sum

def QValue_neg(s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardTotalAnte_neg(k, actions)
       
    return sum

def TraderVWAPReal_pos(i, k, actions, s):
    temp = TraderVWAP_pos(i,k, actions)
    # sumD =3
    # for j in range(0, k):
    #     sumD = sumD + actions[j]
    # Num =  temp * sumD + actions[i] * s[i]
    # Dem = sumD + actions[i]
    
    # if Dem == 0:
    #     result = 0
    # else:    
    #     result= Num/Dem
    result = temp
    
    return result

def TraderVWAPReal_neg(i, k, actions, s):
    temp = TraderVWAP_neg(i,k, actions)
    # sumD =3
    # for j in range(0, k):
    #     sumD = sumD + actions[j]
    # Num =  temp * sumD + actions[i] * s[i]
    # Dem = sumD + actions[i]
    
    # if Dem == 0:
    #     result = 0
    # else:    
    #     result= Num/Dem
    result = temp
    
    return result


def RewardReal_pos(i, k, actions, s):    
    
    return (DiscountFactor ** k) * (TraderVWAPReal_pos(i, k, actions, s) - MarketVWAP_pos(i, k+1))

def RewardReal_neg(i, k, actions, s):    
    
    return (DiscountFactor ** k) * (TraderVWAPReal_neg(i, k, actions, s) - MarketVWAP_neg(i, k+1))

def RewardRealTotal_pos(k, actions, s):
    sum = 0
    for i in range(0, 6):
        sum = sum + RewardReal_pos(i, k, actions[i], s)
       
    return sum

def RewardRealTotal_neg(k, actions, s):
    sum = 0
    for i in range(0, 6):
        sum = sum + RewardReal_neg(i, k, actions[i], s)
       
    return sum

def QValueRealAll_pos(s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardRealTotal_pos(k, actions, s)
       
    return sum

def QValueRealAll_neg(s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardRealTotal_neg(k, actions, s)
       
    return sum

def QValueReal_pos(i, s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardReal_pos(i, k, actions, s)
       
    return sum

def QValueReal_neg(i, s, actions, p):
    sum = 0
    for k in range(0, p):
        sum = sum + RewardReal_neg(i, k, actions, s)
       
    return sum

def ArgQValueReal_pos(s, p):
    result =np.ones([6], dtype=float)
    for i in range(0, 6):
        v= V_pos[i][p]
        if v<BaseVolume :
            result[i]=-1
        else:
            count = int(v/BaseVolume)   
            a=np.ones(count, dtype=float)
            q=np.ones(count, dtype=float)
            for j in range(0, count):
                a[j]= 0 + BaseVolume * j
                q[j]= QValueReal_pos(i, s[i], a[j], p)
            temp = np.argmax(q)
            val = a[temp]
            result[i]=val  
            
    return result


def ArgQValueReal_neg(s, p):
    result =np.ones([6], dtype=float)
    for i in range(0, 6):
        v= V_neg[i][p]
        if v<BaseVolume :
            result[i]=-1
        else:
            count = int(v/BaseVolume)   
            a=np.ones(count, dtype=float)
            q=np.ones(count, dtype=float)
            for j in range(0, count):
                a[j]= 0 + BaseVolume * j
                q[j]= QValueReal_neg(i, s[i], a[j], p)
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
        if k+1 in range(position_absolute_min -startPoint):
            stock =[S1[0][k]/s0, S1[1][k]/s1, S1[2][k]/s2, S1[3][k]/s3, S1[4][k]/s4, S1[5][k]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock =[S1[0][k+1]/s0, S1[1][k+1]/s1, S1[2][k+1]/s2, S1[3][k+1]/s3, S1[4][k+1]/s4, S1[5][k+1]/s5 ]
        elif k+1 in range(position_absolute_min - startPoint, position_absolute_max - startPoint):
            l=k-(position_absolute_min - startPoint)
            stock =[S2[0][l]/s0, S2[1][l]/s1, S2[2][l]/s2, S2[3][l]/s3, S2[4][l]/s4, S2[5][l]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock =[S2[0][l+1]/s0, S2[1][l+1]/s1,S2[2][l+1]/s2, S2[3][l+1]/s3, S2[4][l+1]/s4, S2[5][l+1]/s5 ]  
        elif k+1 >= position_absolute_max - startPoint and k < min(len(S3[1]), len(S3[2]), len(S3[3]), len(S3[4]), len(S3[5]), len(S3[0])):
            l=k-(position_absolute_max - startPoint)
            stock =[S3[0][l]/s0, S3[1][l]/s1, S3[2][l]/s2, S3[3][l]/s3, S3[4][l]/s4, S3[5][l]/s5 ]
            reward=  RewardTotal(episode, realAlloc, stock)
            done = 0
            newstock =[S3[0][l+1]/s0, S3[1][l+1]/s1, S3[2][l+1]/s2, S3[3][l+1]/s3,S3[4][l+1]/s4, S[5][l+1]/s5 ]                     
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
    
    def switchplotter(self, episode):
        nameUp =''
        nameDown =''
        for i in range(0,6):
            Trades_pos  = []
            mVWAP_pos = []
            tVWAP_pos = []
            Trades_neg  = []
            mVWAP_neg = []
            tVWAP_neg = []
            if i==0 :
                nameUp ='AAPL Uptrend'
                nameDown = 'AAPL Downtrend'
            elif i== 1:
                nameUp = 'GOOG Uptrend'
                nameDown = 'GOOG Downtrend'
            elif i== 2:
                nameUp = 'IBM Uptrend'
                nameDown = 'IBM Downtrend'
            elif i==3:
                nameUp ='VZ Uptrend'
                nameDown ='VZ Downtrend'
            elif i==4:
                nameUp ='AT&T Uptrend'
                nameDown = 'Downtrend'
            elif i== 5:
                nameUp ='Exxon Mobil Uptrend'
                nameDown = 'Exxon Mobil Downtrend'
            for j in range(1, episode):
                if j in dates_pos:
                    Trades_pos.append(j)
                    mVWAP_pos.append(MarketVWAP_pos(i, j))
                    tVWAP_pos.append(TraderVWAP_pos(i, j, Actions[i]))        
                else:
                    Trades_neg.append(j)
                    mVWAP_neg.append(MarketVWAP_neg(i, j))
                    tVWAP_neg.append(TraderVWAP_neg(i, j, Actions[i]))
            SwitchingTradingTrainingEnv.graph(Trades_pos, tVWAP_pos, mVWAP_pos, nameUp)
            SwitchingTradingTrainingEnv.graph(Trades_neg, tVWAP_neg, mVWAP_neg, nameDown) 
                  


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





