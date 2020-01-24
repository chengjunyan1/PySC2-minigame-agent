import numpy as np
import os
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import pysc2.agents.Q.maincontroller as MC


N_ACTIONS = {}                  # 动作空间大小
N_ACTIONS["d"] = 16             # 动作空间大小
N_ACTIONS["t"] = 13             # 动作空间大小
N_ACTIONS["r"] = 3              # 动作空间大小
N_STATES = 35                   # 观测空间大小
MEMORY_CAPACITY={}              # 记忆库大小
MEMORY_CAPACITY["d"] = 100000   # d记忆库大小
MEMORY_CAPACITY["t"] = 100000   # t记忆库大小
MEMORY_CAPACITY["r"] = 500      # r记忆库大小
MEMORY_START = {}               # 记忆库训练阈值
MEMORY_START["d"] = 5000        # d记忆库训练阈值
MEMORY_START["t"] = 5000        # t记忆库训练阈值
MEMORY_START["r"] = 30          # r记忆库训练阈值
TARGET_REPLACE_ITER = 500       # QNet的更新频率
GAMMA = 0.99                    # 奖励递减参数
LR = 0.00025                    # ONet学习率
BETAS = (0.9, 0.999)            # Adam优化器参数
BATCH_SIZE={}                   # 每次抽取记忆大小
BATCH_SIZE["d"] = 32            # 每次抽取记忆大小
BATCH_SIZE["t"] = 32            # 每次抽取记忆大小
BATCH_SIZE["r"] = 2             # 每次抽取记忆大小
EPS_START = 0.9                 # ONet起始探索率
EPS_END = 0.05                  # ONet终止探索率
EPS_DECAY = {}                  # 探索率衰减系数
EPS_DECAY["d"] = 10000          # 探索率衰减系数
EPS_DECAY["t"] = 10000          # 探索率衰减系数
EPS_DECAY["r"] = 100            # 探索率衰减系数
UPDATE_RATE = 50                # 对手更新周期

RANDOM_FLAG = 0                 # 随机探索标签
RANDOM_OPEN = 1                 # 是否开启随机探索(关闭则固定为终止探索率)
TEST_MODE = 1                   # 是否开启测试模式(打印胜率,关闭随机)

CUDA_OPEN = MC.CUDA_OPEN
DEVICE_IDS = MC.DEVICE_IDS

Main_Directory = MC.Main_Directory

def prepare_data(x,requires_grad=True):

    if isinstance(x,np.ndarray):
        x=Variable(torch.from_numpy(x),requires_grad=requires_grad)
    if isinstance(x,np.float64):
        x=Variable(torch.Tensor([[x.item()]]),requires_grad=requires_grad)
    if isinstance(x,int):
        x=Variable(torch.Tensor([[x]]),requires_grad=requires_grad)
    x=x.float()
    if x.data.dim()==1:
        x=x.unsqueeze(0)
    return x

class QNet(nn.Module):
    Qdim_input=0
    Qdim_output=0
    Qdim_hidden=0
    
    def __init__(self,Qdim_hidden=16):
        super(QNet,self).__init__()
        
        #创建QNet
        self.Qdim_input=N_STATES
        self.Qdim_hidden=Qdim_hidden

        self.QNet = torch.nn.Sequential(
            torch.nn.Linear(self.Qdim_input, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.SELU(),  
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.SELU(), 
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.SELU(), 
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.SELU(), 
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
        )

        self.dNet = torch.nn.Sequential(
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(), 
            torch.nn.Linear(self.Qdim_hidden, N_ACTIONS["d"]), 
        )

        self.tNet = torch.nn.Sequential(
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(), 
            torch.nn.Linear(self.Qdim_hidden, N_ACTIONS["t"]), 
        )

        self.rNet = torch.nn.Sequential(
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(), 
            torch.nn.Linear(self.Qdim_hidden, N_ACTIONS["r"]), 
        )

    def Qforward(self,x,T):
        x=prepare_data(x)
        if CUDA_OPEN: 
            x=x.cuda()
        x=self.QNet(x)
        if T=="t":
            x=self.tNet(x)
        elif T=="r":
            x=self.rNet(x)
        elif T=="d":
            x=self.dNet(x)
        return x
    
    def __call__(self,state,T):
        state=prepare_data(state)
        pred=self.Qforward(state,T)
        if CUDA_OPEN: 
            pred=pred.cpu()
        return pred.data.numpy()

    def save_QNet(self):
        print(self.QNet.state_dict())
        torch.save(self.QNet.state_dict(), Main_Directory+'Net/NetDQN/Q-net.pkl')
        torch.save(self.tNet.state_dict(), Main_Directory+'Net/NetDQN/t-net.pkl')
        torch.save(self.rNet.state_dict(), Main_Directory+'Net/NetDQN/r-net.pkl')
        torch.save(self.dNet.state_dict(), Main_Directory+'Net/NetDQN/d-net.pkl')


class QAgent():

    def __init__(self):

        self.eval_net, self.target_net = QNet(), QNet()
        if os.path.isfile(Main_Directory+'Net/NetDQN/Q-net.pkl'):
            self.eval_net.QNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/Q-net.pkl'))
            self.target_net.QNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/Q-net.pkl'))
            print("QNet Loaded!")
        if os.path.isfile(Main_Directory+'Net/NetDQN/t-net.pkl'):
            self.eval_net.tNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/t-net.pkl'))
            self.target_net.tNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/t-net.pkl'))
            print("tNet Loaded!")
        if os.path.isfile(Main_Directory+'Net/NetDQN/r-net.pkl'):
            self.eval_net.rNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/r-net.pkl'))
            self.target_net.rNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/r-net.pkl'))
            print("rNet Loaded!")
        if os.path.isfile(Main_Directory+'Net/NetDQN/d-net.pkl'):
            self.eval_net.dNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/d-net.pkl'))
            self.target_net.dNet.load_state_dict(torch.load(Main_Directory+'Net/NetDQN/d-net.pkl'))
            print("dNet Loaded!")
        if CUDA_OPEN:
            self.eval_net.cuda()
            self.target_net.cuda()
            #多GPU加速bug太多已放弃
            #if torch.cuda.device_count()>1:  
            #    self.eval_net=nn.DataParallel(self.eval_net,device_ids=DEVICE_IDS)
            #    self.target_net=nn.DataParallel(self.target_net,device_ids=DEVICE_IDS)

        #经验回放
        self.memory = {}                # 初始化记忆库
        self.temp_memory = {}           # 初始化临时记忆库
        self.memory_counter = {}        # 记忆库记数
        self.learn_step_counter = {}    # 用于 target 更新计时
        self.counter = {}               # 实时记录动作数
        #训练记录
        self.start_record={}            # 开始记录的标志
        self.cost_record={}             # 记录cost变化
        self.qvalue_record={}           # 记录平局Q值
        self.winrate_record=[]          # 记录胜率
        self.checkpoint_cost={}         # 保存每局游戏的结束点
        self.checkpoint_qvalue={}       # 保存每局游戏的结束点
              
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, betas=BETAS)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
        self.memory["t"] = np.zeros((MEMORY_CAPACITY["t"], N_STATES * 2 + 2))     
        self.memory["r"] = np.zeros((MEMORY_CAPACITY["r"], N_STATES * 2 + 2))    
        self.memory["d"] = np.zeros((MEMORY_CAPACITY["d"], N_STATES * 2 + 2))    
        self.temp_memory["t"] = np.zeros(N_STATES * 2 + 2)                 
        self.temp_memory["r"] = np.zeros(N_STATES * 2 + 2)    
        self.temp_memory["d"] = np.zeros(N_STATES * 2 + 2) 
        self.learn_step_counter["t"] = 0    
        self.learn_step_counter["r"] = 0  
        self.learn_step_counter["d"] = 0 
        self.memory_counter["t"] = 0  
        self.memory_counter["r"] = 0  
        self.memory_counter["d"] = 0 
        self.counter["t"] = 0
        self.counter["r"] = 0
        self.counter["d"] = 0

        self.start_record['t']=0
        self.cost_record['t']=[]             
        self.qvalue_record['t']=[]           
        self.checkpoint_cost['t']=[]         
        self.checkpoint_qvalue['t']=[] 
        self.start_record['r']=0      
        self.cost_record['r']=[]             
        self.qvalue_record['r']=[]           
        self.checkpoint_cost['r']=[]         
        self.checkpoint_qvalue['r']=[] 
        self.start_record['d']=0     
        self.cost_record['d']=[]             
        self.qvalue_record['d']=[]           
        self.checkpoint_cost['d']=[]         
        self.checkpoint_qvalue['d']=[]      


    #——————————————Train—————————————#

    def choose_action(self, T, x):
        global RANDOM_FLAG
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # 这里只输入一个 sample
        if RANDOM_OPEN:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.counter[T] / EPS_DECAY[T])
        else:
            eps_threshold = EPS_END
        if TEST_MODE:
            eps_threshold = 0
        if np.random.uniform() < eps_threshold:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS[T])
            RANDOM_FLAG=1
        else:   # 选最优动作
            actions_value = self.eval_net(x,T)
            action = np.argmax(actions_value)     # return the argmax
            RANDOM_FLAG=0
            # Record
            if self.start_record[T]:
                self.qvalue_record[T].append(actions_value.mean())
        self.counter[T]+=1
        return action

    def store_transition(self, T, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter[T] % MEMORY_CAPACITY[T]
        self.memory[T][index, :] = transition
        self.memory_counter[T] += 1

    def store_sparse(self, T, s, a, s_):
        #稀疏奖励下，临时存储
        transition = np.hstack((s, [a, 0], s_))
        self.temp_memory[T] = np.row_stack((self.temp_memory[T],transition))
    
    def store_transfer(self, T, r):
        if len(self.temp_memory[T].shape)!=1:
            count=0 #跳过第一个数据
            for i in range(self.temp_memory[T].shape[0]):
                if count!=0:
                    temp=self.temp_memory[T][i]
                    s = temp[:N_STATES]
                    a = temp[N_STATES:N_STATES+1].astype(int)
                    s_= temp[-N_STATES:]
                    self.store_transition(T, s, a, r, s_)
                count=1
        self.temp_memory[T] = np.zeros(N_STATES * 2 + 2)

    def learn(self,T):
        # target net 参数更新
        if self.learn_step_counter[T] % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter[T] += 1

        # 抽取记忆库中的批数据
        MEMORY_POOL=min(self.memory_counter[T],MEMORY_CAPACITY[T])
        sample_index = np.random.choice(MEMORY_POOL, BATCH_SIZE[T])
        b_memory = self.memory[T][sample_index, :]
        b_s = b_memory[:, :N_STATES]
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = b_memory[:, -N_STATES:]
        if CUDA_OPEN:
            b_a=b_a.cuda()
            b_r=b_r.cuda()

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net.Qforward(b_s,T).gather(1, b_a)   # shape (batch, 1)
        q_next = self.target_net.Qforward(b_s_,T).detach()      # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].unsqueeze(1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        print("Loss:",loss)
        if self.start_record[T]:
            self.cost_record[T].append(float(loss))
        if CUDA_OPEN:
            loss=loss.cuda()
        
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def read_record(self):
        Record_Path=Main_Directory+'Net/RecordDQN/'

        if os.path.isfile(Record_Path+"tCOST.txt"):
            self.cost_record['t']=np.loadtxt(Record_Path+"tCOST.txt")
            self.cost_record['t']=self.cost_record['t'].tolist()
        if os.path.isfile(Record_Path+"tQVALUE.txt"):
            self.qvalue_record['t']=np.loadtxt(Record_Path+"tQVALUE.txt")
            self.qvalue_record['t']=self.qvalue_record['t'].tolist()
        if os.path.isfile(Record_Path+"tCOSTCP.txt"):
            self.checkpoint_cost['t']=np.loadtxt(Record_Path+"tCOSTCP.txt")
            self.checkpoint_cost['t']=self.checkpoint_cost['t'].tolist()
        if os.path.isfile(Record_Path+"tQVALUECP.txt"):
            self.checkpoint_qvalue['t']=np.loadtxt(Record_Path+"tQVALUECP.txt")
            self.checkpoint_qvalue['t']=self.checkpoint_qvalue['t'].tolist()
            
        if os.path.isfile(Record_Path+"rCOST.txt"):
            self.cost_record['r']=np.loadtxt(Record_Path+"rCOST.txt")
            self.cost_record['r']=self.cost_record['r'].tolist()
        if os.path.isfile(Record_Path+"rQVALUE.txt"):
            self.qvalue_record['r']=np.loadtxt(Record_Path+"rQVALUE.txt")
            self.qvalue_record['r']=self.qvalue_record['r'].tolist()
        if os.path.isfile(Record_Path+"rCOSTCP.txt"):
            self.checkpoint_cost['r']=np.loadtxt(Record_Path+"rCOSTCP.txt")
            self.checkpoint_cost['r']=self.checkpoint_cost['r'].tolist()
        if os.path.isfile(Record_Path+"rQVALUECP.txt"):
            self.checkpoint_qvalue['r']=np.loadtxt(Record_Path+"rQVALUECP.txt")
            self.checkpoint_qvalue['r']=self.checkpoint_qvalue['r'].tolist()

        if os.path.isfile(Record_Path+"dCOST.txt"):
            self.cost_record['d']=np.loadtxt(Record_Path+"dCOST.txt")
            self.cost_record['d']=self.cost_record['d'].tolist()
        if os.path.isfile(Record_Path+"dQVALUE.txt"):
            self.qvalue_record['d']=np.loadtxt(Record_Path+"dQVALUE.txt")
            self.qvalue_record['d']=self.qvalue_record['d'].tolist()
        if os.path.isfile(Record_Path+"dCOSTCP.txt"):
            self.checkpoint_cost['d']=np.loadtxt(Record_Path+"dCOSTCP.txt")
            self.checkpoint_cost['d']=self.checkpoint_cost['d'].tolist()
        if os.path.isfile(Record_Path+"dQVALUECP.txt"):
            self.checkpoint_qvalue['d']=np.loadtxt(Record_Path+"dQVALUECP.txt")
            self.checkpoint_qvalue['d']=self.checkpoint_qvalue['d'].tolist()

        for i in self.cost_record:
            if type(self.cost_record[i])==float:
                self.cost_record[i]=[self.cost_record[i]]
        for i in self.qvalue_record:
            if type(self.qvalue_record[i])==float:
                self.qvalue_record[i]=[self.qvalue_record[i]]
        for i in self.checkpoint_cost:
            if type(self.checkpoint_cost[i])==float:
                self.checkpoint_cost[i]=[self.checkpoint_cost[i]]
        for i in self.checkpoint_qvalue:
            if type(self.checkpoint_qvalue[i])==float:
                self.checkpoint_qvalue[i]=[self.checkpoint_qvalue[i]]

    def record(self):
        Record_Path=Main_Directory+'Net/RecordDQN/'

        np.savetxt(Record_Path+"tCOST.txt", np.array(self.cost_record['t']))
        np.savetxt(Record_Path+"tQVALUE.txt", np.array(self.qvalue_record['t']))
        np.savetxt(Record_Path+"tCOSTCP.txt", np.array(self.checkpoint_cost['t']))
        np.savetxt(Record_Path+"tQVALUECP.txt", np.array(self.checkpoint_qvalue['t']))

        np.savetxt(Record_Path+"rCOST.txt", np.array(self.cost_record['r']))
        np.savetxt(Record_Path+"rQVALUE.txt", np.array(self.qvalue_record['r']))
        np.savetxt(Record_Path+"rCOSTCP.txt", np.array(self.checkpoint_cost['r']))
        np.savetxt(Record_Path+"rQVALUECP.txt", np.array(self.checkpoint_qvalue['r']))

        np.savetxt(Record_Path+"dCOST.txt", np.array(self.cost_record['d']))
        np.savetxt(Record_Path+"dQVALUE.txt", np.array(self.qvalue_record['d']))
        np.savetxt(Record_Path+"dCOSTCP.txt", np.array(self.checkpoint_cost['d']))
        np.savetxt(Record_Path+"dQVALUECP.txt", np.array(self.checkpoint_qvalue['d']))