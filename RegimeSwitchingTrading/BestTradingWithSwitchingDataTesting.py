import os 
import numpy as np
import pandas as pd
import gym
import keras
from keras.initializers import random_uniform

import tensorflow as tf
import matplotlib.pyplot as plt


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


dfAAPL8 =pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/AAPLIntradayOf9-8-2017.csv") #    pd.read_csv("C:/Temp/TradingPaper/AAPLIntradayOf8-1-2017.csv")
dfGOOG8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/GOOGIntradayOf9-8-2017.csv")
dfT8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/TIntradayOf9-8-2017.csv")
dfVZ8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/VZIntradayOf9-8-2017.csv")
dfXOM8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/XOMIntradayOf9-8-2017.csv")
dfIBM8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/IBMIntradayOf9-6-2017.csv")
dfGSPC8 = pd.read_csv("/home/mouspem/codes/optimaltradingml/RegimeSwitchingTrading/^GSPCIntradayOf9-8-2017.csv")

#resultgspc = [dfGSPC5, dfGSPC6, dfGSPC7, dfGSPC8]
resultaapl = [dfAAPL8]
resultgoog = [dfGOOG8]
resultT = [dfT8]
resultvz = [dfVZ8]
resultxom = [dfXOM8]
resultibm = [dfIBM8]

dfAAPL = pd.concat(resultaapl, ignore_index=True)
dfGOOG = pd.concat(resultgoog, ignore_index=True)
dfT = pd.concat(resultT, ignore_index=True)
dfVZ =pd.concat(resultvz, ignore_index=True)
dfXOM =pd.concat(resultxom, ignore_index=True)
dfIBM =pd.concat(resultibm, ignore_index=True)


S = []
tempAAPL =[]
tempAAPLV =[]
VolAccepted = 50000
V = []
tempGOOG = []
tempGOOGV = []
tempIBM=[]
tempIBMV=[]     
tempT=[]
tempTV=[]
        

tempVZ=[]
tempVZV=[]      

tempXOM=[]
tempXOMV=[]

#   Start Here


for j in range(1, len(dfAAPL["Volume"])):
    temp = dfAAPL["Volume"][j] - dfAAPL["Volume"][j-1]
    if( temp < VolAccepted) : 
        tempAAPLV.append(temp)
        tempAAPL.append(dfAAPL["Price"][j])

tempGOOGV =[]
for j in range(1, len(dfGOOG["Volume"])):
    temp = dfGOOG["Volume"][j] - dfGOOG["Volume"][j-1]
    if( temp < VolAccepted) :
        tempGOOGV.append( temp)
        tempGOOG.append(dfGOOG["Price"][j])  

tempIBM=[]    
for j in range(1, len(dfIBM["Volume"])):
    temp = dfIBM["Volume"][j] - dfIBM["Volume"][j-1]
    if(temp < VolAccepted) :
        tempIBMV.append(temp)
        tempIBM.append(dfIBM["Price"][j])      

tempT=[]
for j in range(1, len(dfT["Volume"])):
    temp = dfT["Volume"][j]- dfT["Volume"][j-1]
    if(temp < VolAccepted) :
        tempTV.append(temp)
        tempT.append(dfT["Price"][j])      

tempVZ=[]
for j in range(1, len(dfVZ["Volume"])):
    temp = dfVZ["Volume"][j] - dfVZ["Volume"][j-1]
    if (temp < VolAccepted ) :
        tempVZV.append(temp)
        tempVZ.append(dfVZ["Price"][j])        

for j in range(1, len(dfXOM["Volume"])):
    temp = dfXOM["Volume"][j] - dfXOM["Volume"][j-1]
    if( temp< VolAccepted) :
        tempXOMV.append(temp)
        tempXOM.append(dfXOM["Price"][j])          

V.append(tempAAPLV)
V.append(tempGOOGV)
V.append(tempIBMV)
V.append(tempTV)
V.append(tempVZV)
V.append(tempXOMV)       

S.append(tempAAPL)
S.append(tempGOOG)
S.append(tempIBM)
S.append(tempT)
S.append(tempVZ)
S.append(tempXOM)



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

lastmVWAP = [0]

# plot the trading Volume


def MarketVWAP(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S[i][j]*abs(V[i][j]) #S[i][j]
        sumD = sumD+ abs(V[i][j]) #sumD + V[i][j]     
    if sumD == 0:
        result =0
    # else:
    #     result = sumN / sumD
    # if lastmVWAP[0] > 0 and result > 10*lastmVWAP[0]:
    #     result = lastmVWAP[0]
    # else:
    #     lastmVWAP[0] = result

    result = sumN /sumD
   

    return result


def TraderVWAP(i, k):
    
    sumN =0
    sumD =0
    for j in range(0, k):
        sumN = sumN + S[i][j]*Actions_Test[i][j]#
        sumD = sumD + Actions_Test[i][j]
    
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

DiscountFactor = 0.99
BaseVolume = 10

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


class TradingExtraDataTestingEnv(gym.Env):
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
        Actions_Test[0].append(action * M0/100.0)
        Actions_Test[1].append(action * M1/100.0)
        Actions_Test[2].append(action * M2/100.0)
        Actions_Test[3].append(action * M3/100.0)
        Actions_Test[4].append(action * M4/100.0)
        Actions_Test[5].append(action * M5/100.0) 
        stock = np.ones([6], dtype=float)
        s0 = np.max(S[0]) 
        s1 = np.max(S[1])
        s2 = np.max(S[2])
        s3 = np.max(S[3])
        s4 = np.max(S[4])
        s5 = np.max(S[5])
        stock =[S[0][k]/s0, S[1][k]/s1, S[2][k]/s2, S[3][k]/s3, S[4][k]/s4, S[5][k]/s5 ]
        realAlloc = [M0*action /100.0, M1 *  action /100.0 , M2* action /100.0, M3* action /100.0, M4* action /100.0, M5* action /100.0]  
        reward=  RewardTotal(episode, realAlloc, stock) if episode >3 else 0
        done = 0   # np.array_equal(self._agent_location, self._target_location)
        newstock = np.ones([6], dtype=float)
        newstock =[S[0][k+1]/s0, S[1][k+1]/s1, S[2][k+1]/s2, S[3][k+1]/s3, S[4][k+1]/s4, S[5][k+1]/s5 ]
        observation = newstock
        info = self._get_info()

        return observation, reward, done, info 
               
    
    def plotterMarket(self, episode): 
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
            for j in range(10, episode):
                Trades.append(j)
                mVWAP.append(MarketVWAP(i, j))
            TradingExtraDataTestingEnv.graphSingle(Trades, mVWAP, name)       
    

    def plotter_Test(self, episode): 
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
                tVWAP.append(TraderVWAPTesting(i, j))
            TradingExtraDataTestingEnv.graph(Trades, tVWAP, mVWAP, name)
            
           

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

    def graphSingle(trades, MarketVWAP, stockLabel):
        plt.style.use('default')  
        fig,ax=plt.subplots()
        plt.style.use('ggplot')
        #ax.plot(trades, traderVWAP, MarketVWAP )
        ax.set_title(stockLabel)
        #line1, = ax.plot(trades, traderVWAP, label='trader VWAP')
        line2, = ax.plot(trades, MarketVWAP, label='market VWAP')
        ax.legend(handles=[line2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()


# put the enviroment in the same python notebook
#env = TradingExtraDataEnv()
#env.plotterMarket(5000)