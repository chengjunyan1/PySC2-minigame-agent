import numpy as np
import os
import math
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import pysc2.agents.O.OneNet as O


N_ACTIONS = O.Action_Space  # 动作空间大小
N_STATES = O.Observe_Space  # 观测空间大小
MEMORY_CAPACITY = {}        # 记忆库大小
MEMORY_START = {}           # 训练起始阈值
TARGET_REPLACE_ITER = 1000  # QNet的更新频率
GAMMA = 0.99                # 奖励递减参数
LR = 0.00025                # ONet学习率
BETAS = (0.9, 0.999)        # Adam优化器参数
BATCH_SIZE = {}             # 每次抽取记忆大小
EPS_START = 0.9             # ONet起始探索率
EPS_END = 0.1               # ONet终止探索率
EPS_DECAY = {}              # 探索率衰减系数
UPDATE_RATE = 20            # 对手更新周期(直接指定)
UPDATE_SPEED1 = 1000        # 对手更新周期系数一(stair rate)
UPDATE_SPEED2 = 1000        # 对手更新周期系数二

RANDOM_FLAG=0               # 随机探索标签
RANDOM_OPEN=1               # 是否开启随机探索(关闭则固定为0)

Main_Directory=O.Main_Directory

class OAgent():

    def __init__(self,Task):

        self.eval_net, self.target_net = O.ONet(Task), O.ONet(Task)
        self.eval_net.load_ONet()
        self.target_net.load_ONet()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, betas=BETAS)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

        #经验回放
        self.memory = {}                # 初始化记忆库
        self.temp_memory = {}           # 初始化临时记忆库
        self.learn_step_counter = {}    # 用于 target 更新计时
        self.memory_counter = {}        # 记忆库记数
        self.counter = {}               # 实时记录动作次数
        #训练记录
        self.Task=Task
        self.start_record={}            # 开始记录的标志
        self.cost_record={}             # 记录cost变化
        self.qvalue_record={}           # 记录平局Q值
        self.winrate_record=[]          # 记录胜率
        self.checkpoint_cost={}         # 保存每局游戏的结束点
        self.checkpoint_qvalue={}       # 保存每局游戏的结束点
        #初始化
        for i in Task:
            self.memory[i] = np.zeros((MEMORY_CAPACITY[i], N_STATES * 2 + 2))     # 初始化记忆库
            self.temp_memory[i] = np.zeros(N_STATES * 2 + 2)                      # 初始化临时记忆库
            self.learn_step_counter[i] = 0     # 用于 target 更新计时
            self.memory_counter[i] = 0         # 记忆库记数
            self.counter[i] = 0
            self.start_record[i]=0
            self.cost_record[i]=[]
            self.qvalue_record[i]=[]
            self.checkpoint_cost[i]=[] 
            self.checkpoint_qvalue[i]=[] 


    #——————————————Train—————————————#

    def choose_action(self, Task, x):
        global RANDOM_FLAG,RANDOM_OPEN
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # 这里只输入一个 sample
        if RANDOM_OPEN:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.counter[Task] / EPS_DECAY[Task])
        else:
            eps_threshold = 0
        if np.random.uniform() < eps_threshold:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS[Task])
            RANDOM_FLAG=1
        else:   # 选最优动作
            actions_value = self.eval_net(Task)
            action = np.argmax(actions_value)     # return the argmax
            RANDOM_FLAG=0
            # Record
            if self.start_record[Task]:
                self.qvalue_record[Task].append(actions_value.mean())
        self.counter[Task]+=1
        return action

    def store_transition(self, Task, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter[Task] % MEMORY_CAPACITY[Task]
        self.memory[Task][index, :] = transition
        self.memory_counter[Task] += 1
    
    def store_sparse(self, Task, s, a, s_):
        #稀疏奖励下，临时存储
        transition = np.hstack((s, [a, 0], s_))
        self.temp_memory[Task] = np.row_stack((self.temp_memory[Task],transition))
    
    def store_transfer(self, Task, r):
        if len(self.temp_memory[Task].shape)!=1:
            count=0 #跳过第一个数据
            for i in range(self.temp_memory[Task].shape[0]):
                if count!=0:
                    temp=self.temp_memory[Task][i]
                    s = temp[:N_STATES]
                    a = temp[N_STATES:N_STATES+1].astype(int)
                    s_= temp[-N_STATES:]
                    self.store_transition(Task, s, a, r, s_)
                count=1
        self.temp_memory[Task] = np.zeros(N_STATES * 2 + 2)

    def learn(self, Task):
        # target net 参数更新
        if self.learn_step_counter[Task] % TARGET_REPLACE_ITER == 0:
            self.target_net=copy.deepcopy(self.eval_net)
        self.learn_step_counter[Task] += 1

        # 抽取记忆库中的批数据
        MEMORY_POOL=min(self.memory_counter[Task],MEMORY_CAPACITY[Task])
        sample_index = np.random.choice(MEMORY_POOL, BATCH_SIZE[Task])
        b_memory = self.memory[Task][sample_index, :]
        b_s = b_memory[:, :N_STATES]
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = b_memory[:, -N_STATES:]

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(Task, b_s).gather(1, b_a)        # shape (batch, 1)
        q_next = self.target_net(Task, b_s_).detach()           # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].unsqueeze(1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        print("Task:",Task," Loss:",loss)
        if self.start_record[Task]:
            self.cost_record[Task].append(float(loss))
        if O.CUDA_OPEN:
            loss=loss.cuda()

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def read_record(self,Task):
        Record_Path=Main_Directory+'Net/Record/'+Task+'/'
        if os.path.isfile(Record_Path+"WINRATE.txt"):
            self.winrate_record=np.loadtxt(Record_Path+"WINRATE.txt")
        for i in self.Task:
            if os.path.isfile(Record_Path+i+"COST.txt"):
                self.cost_record[i]=np.loadtxt(Record_Path+i+"COST.txt")
                self.cost_record[i]=self.cost_record[i].tolist()
                if type(self.cost_record[i])==float:
                    self.cost_record[i]=[self.cost_record[i]]
            if os.path.isfile(Record_Path+i+"QVALUE.txt"):
                self.qvalue_record[i]=np.loadtxt(Record_Path+i+"QVALUE.txt")
                self.qvalue_record[i]=self.qvalue_record[i].tolist()
                if type(self.qvalue_record[i])==float:
                    self.qvalue_record[i]=[self.qvalue_record[i]]
            if os.path.isfile(Record_Path+i+"COSTCP.txt"):
                self.checkpoint_cost[i]=np.loadtxt(Record_Path+i+"COSTCP.txt")
                self.checkpoint_cost[i]=self.checkpoint_cost[i].tolist()
                if type(self.checkpoint_cost[i])==float:
                    self.checkpoint_cost[i]=[self.checkpoint_cost[i]]
            if os.path.isfile(Record_Path+i+"QVALUECP.txt"):
                self.checkpoint_qvalue[i]=np.loadtxt(Record_Path+i+"QVALUECP.txt")
                self.checkpoint_qvalue[i]=self.checkpoint_qvalue[i].tolist() 
                if type(self.checkpoint_qvalue[i])==float:
                    self.checkpoint_qvalue[i]=[self.checkpoint_qvalue[i]]

    def record(self,Task):
        Record_Path=Main_Directory+'Net/Record/'+Task+'/' #TASK为文件夹名
        np.savetxt(Record_Path+"WINRATE.txt", np.array(self.winrate_record))
        for i in self.Task:
            np.savetxt(Record_Path+i+"COST.txt", np.array(self.cost_record[i]))
            np.savetxt(Record_Path+i+"QVALUE.txt", np.array(self.qvalue_record[i]))
            np.savetxt(Record_Path+i+"COSTCP.txt", np.array(self.checkpoint_cost[i]))
            np.savetxt(Record_Path+i+"QVALUECP.txt", np.array(self.checkpoint_qvalue[i])) 