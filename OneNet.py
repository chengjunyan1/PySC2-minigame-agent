import torch.nn as nn
import numpy as np
import os
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random

import pysc2.agents.O.utilities as ut

"""—————————————————————————————— M E M ——————————————————————————————"""

###################################
# Data Structure For DR/NLPGM
# EFrag: one root and multiple leaves,Representing the accessible infomations
# RFrag: multiple roots and one leave,Representing the reasoning of infos
# 1. Define Template
# 2. Generate Graph from Template
# 3. Reasoning and Learning
# Entity class and Reason class are template for EFrag and RFrag respectively
# EFrag and RFrag are instantiation for Entity and Reason respectively
# Entity和Reason的实例在整个运行过程中有效
# 变量的全局修饰全部在创建时自动进行,人输入时不需要输入修饰后的变量名
###################################

Current_Factors=[] # 记录当前可用的事实/属性,在EFrag聚合后和RFrag创建时更新,必要缺省值节点名字前带*
Current_RFrags={} # 记录当前的RFrag,以便激活对应的神经网络,{Type:ID}
Root_RFrags=[] # 记录哪些节点是根节点,即只需要感知的RFrag,存储ID
DefaultValue_Table={} #保存属性的缺省值,{attrib:default}
Dimention_List={} #保存每个属性的维度信息,{attrib:dimention}
    
#______________________EFrag________________________#

Entity_List={} #实体表,根据实体名字索引,{Type:Entity}
Aggregate_Policy={} #聚合函数表,{Type:AFunc},每类Entity有一个
EFrag_Count={} #EFrag计数器,根据实体名索引 *记录每种实例化实体的数量
EFrag_List={} #EFrag实例表,{ID:EFrag}
EValue_Table={} #保存每个EFrag的值,{Type:VALUE},VALUE={ID{attrib:value}}
EOutput_List={} #EFrag输出表,每种实体一张表 *感知部分的最终输出 {attrib:value}
EOutput_Set=[] #保存所有Entity输出的集合

def DefaultAggregate(Type): #默认聚合函数,给定要聚合的实体类型,返回聚合后的属性表
    ValueTable=EValue_Table[Type] #取出同类下所有片段
    temp={}
    for i in ValueTable:
        temp=ut.dict_add(temp,ValueTable[i]) #直接对所有片段的值表求和
    return temp

# Entity对象仅在初始化阶段创建,EFrag创建时copy Entity的对象
# Entity对象名义上是静态的，使用时只修改他在EFrag中的copy
class Entity:
    entity_name='' #实体名称
    attrib_set=[] #属性集合
    attrib_list={} #属性表,属性名：值,输入时自动修饰成全局名称
    attrib_dict={} #属性字典，属性名：索引
    attrib_index={} #属性索引 索引：属性名
    
    def __init__(self,name,attrib,constants): #attrib属性名数组，constants numpy
        self.entity_name=name
        self.attrib_set=attrib #可暂时不进行修饰
        self.attrib_index=attrib #建立属性索引
        self.attrib_value=constants #初始化值列表,赋默认值
        count=0
        for i in attrib:
            if constants!='NoValue':
                self.attrib_list[i]=constants[count]
            self.attrib_dict[i]=count #建立属性字典
            count+=1
                
    def Update_attrib(self,attrib): #更新目标属性,attrib={attrib:value}
        for i in attrib:
            self.attrib_list[i]=attrib[i]

class EFrag:
    Type='' #EFrag类型,要与实体名一致
    Name='' #EFrag实例名称,方便区分不同实体
    ID='' #分配一个ID,名字+计数值 
    Entity=0 #存储一个实体

    def __init__(self,Type,value,Name):
        self.Type=Type
        self.Name=Name
        self.Entity=copy.copy(Entity_List[Type])
        self.ID=Type+str(EFrag_Count[Type])
        if value!='NoValue':
            self.Entity.Update_attrib(value)
        self.Updata_ValueTable()
    
    def Updata_ValueTable(self):
        EValue_Table[self.Type][self.ID]=self.Entity.attrib_list

# ———— PACKAGING ———— #   

def Create_Entity(name,attrib,attrib_dim,constants='NoValue',AFunc=DefaultAggregate): #创建Entity,同时进行各种操作
    attrib=ut.factor_decorator('E',name,attrib) #修饰为全局名称
    ut.updata_list(EOutput_Set,attrib) #将输出属性添加到输出集合表
    temp=Entity(name,attrib,constants) #创建实体
    Entity_List[name]=temp #更新实体表
    EValue_Table[name]={} #在EValue_Table中加入条目
    EFrag_Count[name]=0 #创建一个实体片段计数器
    Aggregate_Policy[name]=AFunc #创建对应的聚合函数 
    for i in attrib_dim: #输出属性维度表,用属性名索引
        i_name=ut.factor_decorator('E',name,i) #修饰得到全局属性名
        Dimention_List[i_name]=attrib_dim[i] #将维数填入表中
    return temp
    
def Create_EFrag(Type,value='NoValue',Name=''): #创建EFrag,每个EFrag包含一个Entity的copy
    EFrag_Count[Type]+=1
    temp=EFrag(Type,value,Name) #创建片段
    EFrag_List[temp.ID]=temp
    return temp

def EFrag_Rename(ID,Name):
    EFrag_list[ID].Name=Name
    
# 感知数据处理,首先为每个观测到的实体标签创建EFrag,然后为每个EFrag更新值,最后聚合输出

def Observer(Type,value='NoValue'): #为观察到的标签创建一个EFrag,创建时不赋新值
    temp=Create_EFrag(Type,value)
    return temp

def Updata_EFrag(ID,value): #修改某个EFrag的值,value为字典数{attrib:value}
    EFrag_List[ID].Entity.Update_attrib(value)
    EFrag_List[ID].Updata_ValueTable()

def EOutput(): #聚合同类EFrag的输出值,放入表中{attrib:value}
    for i in Entity_List: #取出Entity名字
        if EFrag_Count[i]!=0: #计数器中有这一项才聚合(表示存在至少一个EFrag)
            outcome=Aggregate_Policy[i](i)
            #ut.dict_merge(EOutput_List,outcome) #将聚合后的结果放入表中
            Entityi=Entity_List[i]
            factor_set=Entityi.attrib_set #取出得到的属性集合
            for attrib in outcome:
                E_value=ut.prepare_data(outcome[attrib])
                Current_Values[attrib]=E_value
        else: #默认值
            Entityi=Entity_List[i]
            factor_set=Entityi.attrib_set #取出得到的属性集合
            for attrib in factor_set:
                E_value=ut.prepare_data(DefaultValue_Table[attrib])
                Current_Values[attrib]=E_value


#______________________RFrag________________________#

Reason_List={}  #推理表,类型索引,{Type:Reason}
Context_List={} #上下文检查函数表,类型索引,{Type:CFunc},每类Reason有一个
RFrag_Count={}  #RFrag计数器
RFrag_List={}   #RFrag实例表,ID索引,{ID:RFrag}
RStack_List={}  #推理栈表

RDim_Hidden=10 #隐藏层神经元数量

def StrictContext(Type): #严格上下文,当前属性中存在全部输入属性即满足
    Object_Reason=Reason_List[Type] #取出待操作的Reason
    InputFactor=Object_Reason.InputReasons #取出输入要求
    return ut.subsetof_list(InputFactor,Current_Values)

def DefaultContext(Type): #默认上下文,当前属性中存在至少一个实体输入属性且包含所有推理属性即满足
    Object_Reason=Reason_List[Type] #取出待操作的Reason
    E_set=[]
    R_set=[]
    InputFactor=Object_Reason.InputReasons #取出输入要求
    for i in InputFactor:
        if i not in SV_Table:
            [F,T,A]=ut.globalfactor_parser(i)
            if F=='E':
                E_set.append(i)
            elif F=='R':
                R_set.append(i)
    if E_set==[]:
        return ut.subsetof_list(R_set,Current_Values)
    else:
        return ut.intersect_list(E_set,Current_Values)!=[] and ut.subsetof_list(R_set,Current_Values)

# 正常情况下一个Reason只创建一个RFrag,Reason中的属性必须先进行修饰,修饰器见utilities.py
# 每个Reason包含一个神经网络,同类RFrag共用同一个神经网络推理

class Reason(nn.Module):
    reason_name='' #推理名
    InputReasons=[] #输入推理,推理/实体属性名表,字符串数组  ** 必须为修饰后的全局名称 **
    OutputReason=0 #输出推理,推理名,字符串  ** 必须为修饰后的全局名称，在创建时修饰 **

    def __init__(self,name,inputs,output,Norm):
        super(Reason,self).__init__()
        self.reason_name=name
        self.InputReasons=inputs
        self.OutputReason=output
        self.Norm=Norm
        
        # 神经网络区
        self.dim_hidden=RDim_Hidden
        self.inputsize=0
        for i in self.InputReasons:
            self.inputsize+=Dimention_List[i]
        self.outputsize=0
        for i in self.OutputReason:
            self.outputsize+=Dimention_List[i]

        if self.Norm:
            self.Net = torch.nn.Sequential(
                torch.nn.LayerNorm(self.inputsize),
                torch.nn.Linear(self.inputsize, self.dim_hidden), 
                torch.nn.Dropout(),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.dim_hidden, self.outputsize),
            )
        else:
            self.Net = torch.nn.Sequential(
                torch.nn.Linear(self.inputsize, self.dim_hidden), 
                torch.nn.Dropout(),
                torch.nn.SELU(),
                torch.nn.Linear(self.dim_hidden, self.dim_hidden), 
                torch.nn.Dropout(),
                torch.nn.SELU(),
                torch.nn.Linear(self.dim_hidden, self.outputsize),
            )
        
        if CUDA_OPEN_M: 
            self.Net=self.Net.cuda()

    def __call__(self,x):
        '''
        Residual=x
        x=self.Net[0](x)
        x=self.Net[1](x)
        x=self.Net[2](x)
        out=x+Residual
        x=self.Net[3](out)
        x=self.Net[4](x)
        '''
        if CUDA_OPEN_M: 
            x=x.cuda()
        x=self.Net(x)
        return x
        

class RFrag: #RFrag直接实现推理的功能,推理图中不可成环
    Type='' #片段类型,与ReasonName一致
    Name='' #片段名称,方便区分不同片段
    ID='' #片段ID,名字+计数值
    Reason=0 #对应的事实
    Next=[] #下一个RFrag的集合,可能连接不止一个RFrag,在全部RFrag创建完毕后生成
    Previous=[] #上一个RFrag集合,在全部RFrag创建完毕后生成

    def __init__(self,Type,Name):
        self.Type=Type
        self.Name=Name
        self.Reason=copy.copy(Reason_List[Type])
        if Type not in RFrag_Count:
            RFrag_Count[Type]=0 #创建一个推理片段计数器
        else:
            RFrag_Count[Type]+=1
        self.ID=Type+str(RFrag_Count[Type])
        
    def __call__(self,x):
        x=self.Reason(x)
        return x.data.numpy()


class RStack: #推理栈

    def __init__(self):
        self.Top=0       #栈顶指针
        self.Sequence=[] #推理顺序
        self.Size=0      #片段总数

    def reset(self):
        self.Top=self.Size  #重新指向栈顶
    
    def push(self,x):
        self.Sequence.append(x)
        self.Size+=1
        self.Top=self.Size
    
    def pop(self):
        data=self.Sequence[self.Top-1]
        self.Top-=1
        return data
    
    def bottom(self):
        return self.Top==0

    def delete(self,x):
        i=self.Sequence.index(x)
        del self.Sequence[i]
        self.Size-=1


# ———— PACKAGING ———— #   
        
def Create_Reason(Type,inputs,output,inputsizes,outputsize,Norm=True,CFunc=DefaultContext):
    output=ut.factor_decorator('R',Type,output) #对输出属性进行修饰
    for i in inputsizes:
        Dimention_List[i]=inputsizes[i] #将维数填入表中
    for i in outputsize:
        i_name=ut.factor_decorator('R',Type,i)
        Dimention_List[i_name]=outputsize[i] #将维数填入表中
    temp=Reason(Type,inputs,output,Norm)
    Reason_List[Type]=temp
    RFrag_Count[Type]=0
    Context_List[Type]=CFunc #为Reason创建上下文检查函数
    return temp
    
def Create_RFrag(Type,Name=''):
    temp=RFrag(Type,Name) #创建片段
    RFrag_List[temp.ID]=temp
    return temp

def RFrag_Rename(ID,Name):
    RFrag_List[ID].Name=Name

def RFrag_Expand(RType,Task):
    Reasoni=Reason_List[RType]
    Inputi=Reasoni.InputReasons
    for i in Inputi:
        if i not in SV_Table: #无此属性
            [F,T,A]=ut.globalfactor_parser(i)
            if F!='E':
                check_T=ut.dict_chack2d(Current_RFrags,T)
                if check_T=="NOMATCH":
                    temp=Create_RFrag(T) #创建RFrag(父节点)
                    Current_RFrags[Task][T]=temp.ID #将RFrag的ID加入到当前片段表中
                    if i in Current_Factors:
                        RStack_List[Task].delete(temp.ID)
                    RStack_List[Task].push(temp.ID)
                else:
                    Current_RFrags[Task][T]=Current_RFrags[check_T][T]
                    if i in Current_Factors:
                        RStack_List[Task].delete(Current_RFrags[check_T][T])
                    RStack_List[Task].push(Current_RFrags[check_T][T])
                    temp=RFrag_List[Current_RFrags[Task][T]]
                output_factor=temp.Reason.OutputReason #取出该RFrag生产的推理结果类型
                ut.updata_list(Current_Factors,output_factor,True) #更新已知因素表
                RFrag_Expand(T,Task) #为推理片段展开父节点

def Task_Prereason(Task):
    global Current_Factors
    Current_Factors=[]
    Current_RFrags[Task]={}
    for i in QAccept[Task]: #根据任务输入逆向推理出推理路径
        if i not in SV_Table: #不是单值
            parse=ut.globalfactor_parser(i)
            if parse[1] in Reason_List:
                T=parse[1]
                check_T=ut.dict_chack2d(Current_RFrags,T)
                if check_T=="NOMATCH":
                    temp=Create_RFrag(T) #创建RFrag(父节点)
                    Current_RFrags[Task][T]=temp.ID #将RFrag的ID加入到当前片段表中
                    RStack_List[Task].push(temp.ID)
                else:
                    Current_RFrags[Task][T]=Current_RFrags[check_T][T]
                    RStack_List[Task].push(Current_RFrags[check_T][T])
                    temp=RFrag_List[Current_RFrags[Task][T]]
                output_factor=temp.Reason.OutputReason #取出该RFrag生产的推理结果类型
                ut.updata_list(Current_Factors,output_factor,True) #更新已知因素表
                RFrag_Expand(T,Task) #为推理片段展开父节点  
    

"""—————————————————————————————— N E T ——————————————————————————————"""

###################################
# Main Net
# Dynamic Generate Layers
# Concluding ONet and QNet
# ONet for reasoning QNet for planning
###################################

RFrag_Checked=[]    #已推理的RFrag
State_Dimention=0   #状态总维度
Current_Values={}   #已知属性表,{attrib:value}
Main_Directory='C:/Users/cheng/Anaconda3/Lib/site-packages/pysc2/agents/O/'

class QNet(nn.Module):
    # 决策部分
    Qdim_input=0
    Qdim_output=0
    Qdim_hidden=0
    
    def __init__(self,QAccept,Qdim_output,Qdim_hidden):
        super(QNet,self).__init__()
        self.Qdim_output=Qdim_output
        
        #创建QNet
        self.QAccept=QAccept
        for i in QAccept:
            length=Dimention_List[i]
            self.Qdim_input+=length
        self.Qdim_hidden=Qdim_hidden

        self.Net = torch.nn.Sequential(
            torch.nn.Linear(self.Qdim_input, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),  
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),  
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_hidden), 
            torch.nn.Dropout(),
            torch.nn.ReLU(),  
            torch.nn.Linear(self.Qdim_hidden, self.Qdim_output), 
        )
    
    def __call__(self,x):
        if CUDA_OPEN: 
            x=x.cuda()
        x=self.Net(x)
        return x

class ONet(nn.Module):
    # 推理部分
    Task_List=[] #任务集
    Net_List={} #ONet片段网络表,{ID:value}
    
    def __init__(self,Task_List):
        super(ONet,self).__init__()
        self.Task_List=Task_List
        for R in Reason_List: #初始化时链接每个片段的网络,计算时直接激活
            self.add_net(R)
        
        #创建QNet
        for i in Task_List:
            self.Net_List[i]=Task_Net[i]
            setattr(self, i+'-Net.0', self.Net_List[i].Net[0])
            setattr(self, i+'-Net.1', self.Net_List[i].Net[1])
            setattr(self, i+'-Net.2', self.Net_List[i].Net[2])
            setattr(self, i+'-Net.3', self.Net_List[i].Net[3])
            setattr(self, i+'-Net.4', self.Net_List[i].Net[4])
            setattr(self, i+'-Net.5', self.Net_List[i].Net[5])
            setattr(self, i+'-Net.6', self.Net_List[i].Net[6])
            setattr(self, i+'-Net.7', self.Net_List[i].Net[7])
            setattr(self, i+'-Net.8', self.Net_List[i].Net[8])
            setattr(self, i+'-Net.9', self.Net_List[i].Net[9])

            if CUDA_OPEN: 
                self.Net_List[i]=self.Net_List[i].cuda()
                #if torch.cuda.device_count()>1:  
                #    self.QNet=nn.DataParallel(self.QNet,device_ids=DEVICE_IDS)

    #——————————————ONet—————————————#
    
    #添加网络，根据Type决定
    def add_net(self,Type):
        Reasoni=Reason_List[Type] #取出该Reason的实例
        self.Net_List[Type]=Reasoni #不仅包含了网络,同时也包含了输入输出的相关信息
        if Reasoni.Norm:
            setattr(self, Type+'-Net.0', self.Net_List[Type].Net[0])
            setattr(self, Type+'-Net.1', self.Net_List[Type].Net[1])
            setattr(self, Type+'-Net.2', self.Net_List[Type].Net[2])
            setattr(self, Type+'-Net.3', self.Net_List[Type].Net[3])
            setattr(self, Type+'-Net.4', self.Net_List[Type].Net[4])
        else:
            setattr(self, Type+'-Net.0', self.Net_List[Type].Net[0])
            setattr(self, Type+'-Net.1', self.Net_List[Type].Net[1])
            setattr(self, Type+'-Net.2', self.Net_List[Type].Net[2])
            setattr(self, Type+'-Net.3', self.Net_List[Type].Net[3])
            setattr(self, Type+'-Net.4', self.Net_List[Type].Net[4])
            setattr(self, Type+'-Net.5', self.Net_List[Type].Net[5])
            setattr(self, Type+'-Net.6', self.Net_List[Type].Net[6])
    
    #动态激活网络,根据网络ID决定
    def activate(self,ID):
        Reasoni=self.Net_List[ID]
        x=torch.randn(0)
        for i in Reasoni.InputReasons:
            value=Current_Values[i]
            if CUDA_OPEN_M:
                value=value.cpu()
            x=torch.cat((x,value),1)
        if CUDA_OPEN_M:
            x=x.cuda()
        y=self.Net_List[ID](x) #调用该Reason的网络进行推理
        for i in range(len(y)):
            Current_Values[Reasoni.OutputReason[i]]=y[i].unsqueeze(0)
        return y

    #——————————————QNet—————————————#
    
    #前馈
    def Qforward(self,x,Task):
        x=self.Net_List[Task](x)
        return x
    
    #——————————————Control—————————————#

    #根据RFrag确定的推理路径进行推理
    #推理路径:首先添加全部必要缺省值,再从根节点开始推理,直至全部RFrag推理完毕,最后进入QNet进行决策
    def forwardO(self,Task):
        #ONet
        #缺省值和观测值的填入在每个时间片初始化时完成,仅缺省观测值
        while not RStack_List[Task].bottom():
            RFragid=RStack_List[Task].pop()
            RFragi=RFrag_List[RFragid]
            Output=RFragi.Reason.OutputReason
            Type=RFragi.Reason.reason_name
            if Context_List[Type](Type):
                self.activate(Type)
                RFrag_Checked.append(Type)
        RStack_List[Task].reset() #前馈完成重置栈顶指针
    
    def forwardQ(self,Task):
        #QNet
        QInput=torch.randn(0) #创建一个空tensor用来装QNet的输入
        for i in QAccept[Task]:
            data=Current_Values[i]
            if CUDA_OPEN_M: 
                data=data.cpu()
            QInput=torch.cat((QInput,data),1) # # # 此处可能出错！调试时需注意 # # #
        if CUDA_OPEN: 
            QInput=QInput.cuda()
        Qo=self.Qforward(QInput,Task)
        return Qo
    
    def forward(self,Task):
        self.forwardO(Task)
        O=self.forwardQ(Task)
        return O
    
    def __call__(self,Task,state='vState'):
        if state=='vState':
            pred=self.forward(Task)
            if CUDA_OPEN: 
                pred=pred.cpu()
            return pred.data.numpy()
        else:
            temp=torch.randn(0)
            for s in state:
                Init(s)
                pred=self.forward(Task)
                if CUDA_OPEN: 
                    pred=pred.cpu()
                temp=torch.cat((temp,pred),0)
            return temp

    def save_ONet(self):
        for i in self.Net_List:
            torch.save(self.Net_List[i].Net.state_dict(), Main_Directory+'Net/NetO/'+i+'-net.pkl')

    def load_ONet(self):
        for i in self.Net_List:
            if os.path.isfile(Main_Directory+'Net/NetO/'+i+'-net.pkl'):
                self.Net_List[i].Net.load_state_dict(torch.load(Main_Directory+'Net/NetO/'+i+'-net.pkl'))
                print(i,"Net Loaded!")


"""—————————————————————————————— C O N S O L E ——————————————————————————————"""

# Setting for Agent
Entity_Table=["Adept","Banshee","Battlecruiser","Carrier","Colossus","DarkTemplar","Ghost","Immortal","Liberator","Marine","SiegeTank","Stalker","Tempest","Thor","VoidRay","Zealot"]
Reason_Table=["InfantryLevel","ShipLevel","VehicleLevel"]
MSC_Units=["Adept","Battlecruiser","Carrier","Colossus","Immortal","Liberator","Marine","SiegeTank","SiegeTankSieged","Stalker","Thor","VoidRay","Zealot"]
DSC_Units=["Adept","Banshee","Battlecruiser","Carrier","Colossus","DarkTemplar","Ghost","Immortal","Liberator","Marine","SiegeTank","SiegeTankSieged","Stalker","Tempest","Thor","Zealot"]
MSC_Price=[0,60,400,400,300,200,150,40,200,100,350,250,80]
DSC_Price=[0,60,145,400,400,300,160,200,200,150,40,200,100,500,350,80]


#正确版
Possitive_Combat={
    "Adept":["Marine"],
    "Banshee":["SiegeTank","Colossus"],
    "Battlecruiser":["Thor","Carrier"],
    "Carrier":["Thor"],
    "Colossus":["Marine","Zealot"],
    "DarkTemplar":[],
    "Ghost":[],
    "Immortal":["SiegeTank"],
    "Liberator":["Banshee"],
    "Marine":["Immortal"],
    "SiegeTank":["Marine"],
    "Stalker":["Banshee"],
    "Tempest":["Colossus"],
    "Thor":["Marine"],
    "VoidRay":["Battlecruiser","Tempest"],
    "Zealot":["Immortal"]
}
Negative_Combat={
    "Adept":["Zealot"],
    "Banshee":["Liberator"],
    "Battlecruiser":["VoidRay"],
    "Carrier":["VoidRay"],
    "Colossus":["Immortal"],
    "DarkTemplar":[],
    "Ghost":[],
    "Immortal":["Zealot","Marine"],
    "Liberator":["Carrier","Battlecruiser"],
    "Marine":["SiegeTank","Colossus"],
    "SiegeTank":["Banshee","Immortal"],
    "Stalker":["Immortal"], 
    "Tempest":["VoidRay"],
    "Thor":["Immortal"],
    "VoidRay":["Liberator"],
    "Zealot":["Colossus"]
}

'''
#错误版
Possitive_Combat={
    "Adept":["Battlecruiser"],
    "Banshee":["SiegeTank","Colossus"],
    "Battlecruiser":["Thor","Carrier"],
    "Carrier":["Thor"],
    "Colossus":["Thor","Zealot"],
    "DarkTemplar":[],
    "Ghost":[],
    "Immortal":["Tempest"],
    "Liberator":["Banshee"],
    "Marine":["Immortal"],
    "SiegeTank":["Marine"],
    "Stalker":["Carrier"],
    "Tempest":["Colossus"],
    "Thor":["Marine"],
    "VoidRay":["Battlecruiser","Tempest"],
    "Zealot":["Banshee"]
}
Negative_Combat={
    "Adept":["Banshee"],
    "Banshee":["Stalker"],
    "Battlecruiser":["VoidRay"],
    "Carrier":["Battlecruiser"],
    "Colossus":["Battlecruiser"],
    "DarkTemplar":[],
    "Ghost":[],
    "Immortal":["Zealot","Marine"],
    "Liberator":["Zealot","Battlecruiser"],
    "Marine":["SiegeTank","Colossus"],
    "SiegeTank":["Banshee","Marine"],
    "Stalker":["Immortal"], 
    "Tempest":["Stalker"],
    "Thor":["Immortal"],
    "VoidRay":["Liberator"],
    "Zealot":["Thor"]
}
'''

Research_Influence={
    "InfantryLevel":["Adept","DarkTemplar","Marine","Zealot"],
    "ShipLevel":["Battlecruiser","Carrier","Liberator","VoidRay"],
    "VehicleLevel":["Colossus","Immortal","SiegeTank","Thor"]
}
SV_Table=["InfantryLevel","ShipLevel","VehicleLevel","Minerals","Foods","Self_HP","Enemy_HP"]

# Init fot Agent
Units_List=[311,55,57,79,4,76,50,83,689,48,33,32,74,496,52,80,73]
Units_Table=["Adept","Banshee","Battlecruiser","Carrier","Colossus","DarkTemplar","Ghost","Immortal","Liberator","Marine","SiegeTank","SiegeTankSieged","Stalker","Tempest","Thor","VoidRay","Zealot"]
Unit_Size=[12,21,69,69,37,9,9,21,21,9,32,32,12,69,37,37,12]
Action_Space={}
Observe_Space=35
Tank_index=10
QAccept={}
Task_Net={}

# Configurations
DEVICE_IDS=[0,1]
#全局使用用GPU会导致内存到显存交换次数过多,需要优化再考虑使用
CUDA_OPEN=0 #torch.cuda.is_available()
CUDA_OPEN_M=0 #torch.cuda.is_available()

# 算法启动
def Start(TASK):

    #这些不用了
    '''
    Dimention_List["Minerals"]=1
    Dimention_List["Foods"]=1
    Dimention_List["Self_HP"]=1
    Dimention_List["Enemy_HP"]=1
    '''

    # 输入Entity
    for i in Entity_Table:
        Create_Entity(i,["Quantity"],{"Quantity":1}) #己方信息
        Create_Entity('C'+i,["Quantity"],{"Quantity":1}) #对手信息
        DefaultValue_Table[ut.factor_decorator('E',i,'Quantity')]=0 #填入缺省值
        DefaultValue_Table[ut.factor_decorator('E','C'+i,'Quantity')]=0 #填入缺省值

    # 输入Reason
    for i in Entity_Table:
        if i != "DarkTemplar" and i != "SiegeTank" and i != "Ghost" and i != "Stalker":
            if i == "Medivac":
                decS=''
                decC='C'
            else:
                decS='C'
                decC=''
            inputs=[ut.factor_decorator('E',i,'Quantity')]
            inputs_dimention={ut.factor_decorator('E',i,'Quantity'):1}
            for j in Possitive_Combat[i]:
                inputs.append(ut.factor_decorator('E',decS+j,'Quantity')) #针对对手
                inputs_dimention[ut.factor_decorator('E',decS+j,'Quantity')]=1
            Create_Reason('P'+i,inputs,['P'+i],inputs_dimention,{'P'+i:1}) #己方优势

            inputs=[ut.factor_decorator('E','C'+i,'Quantity')]
            inputs_dimention={ut.factor_decorator('E','C'+i,'Quantity'):1}
            for j in Possitive_Combat[i]:
                inputs.append(ut.factor_decorator('E',decC+j,'Quantity')) #针对己方
                inputs_dimention[ut.factor_decorator('E',decC+j,'Quantity')]=1
            Create_Reason('CP'+i,inputs,['CP'+i],inputs_dimention,{'CP'+i:1}) #对方优势

            inputs=[ut.factor_decorator('E',i,'Quantity')]
            inputs_dimention={ut.factor_decorator('E',i,'Quantity'):1}
            for j in Negative_Combat[i]:
                inputs.append(ut.factor_decorator('E','C'+j,'Quantity')) #针对对手
                inputs_dimention[ut.factor_decorator('E','C'+j,'Quantity')]=1
            Create_Reason('N'+i,inputs,['N'+i],inputs_dimention,{'N'+i:1}) #己方劣势

            inputs=[ut.factor_decorator('E','C'+i,'Quantity')]
            inputs_dimention={ut.factor_decorator('E','C'+i,'Quantity'):1}
            for j in Negative_Combat[i]:
                inputs.append(ut.factor_decorator('E',j,'Quantity')) #针对己方
                inputs_dimention[ut.factor_decorator('E',j,'Quantity')]=1
            Create_Reason('CN'+i,inputs,['CN'+i],inputs_dimention,{'CN'+i:1}) #对方劣势

            '''
            inputP=ut.factor_decorator('R','P'+i,'P'+i)
            inputN=ut.factor_decorator('R','N'+i,'N'+i)
            Create_Reason('S'+i,[inputP,inputN],['S'+i],{inputP:1,inputN:1},{'S'+i:1}) #己方综合

            inputP=ut.factor_decorator('R','CP'+i,'CP'+i)
            inputN=ut.factor_decorator('R','CN'+i,'CN'+i)
            Create_Reason('CS'+i,[inputP,inputN],['CS'+i],{inputP:1,inputN:1},{'CS'+i:1}) #对方综合
            '''

            if "DSC" in QAccept and i in DSC_Units:
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('R','P'+i,'P'+i))
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('R','CP'+i,'CP'+i))
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('R','N'+i,'N'+i))
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('R','CN'+i,'CN'+i))
            if "MSCt" in QAccept and i in MSC_Units:
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('R','P'+i,'P'+i))
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('R','CP'+i,'CP'+i))
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('R','N'+i,'N'+i))
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('R','CN'+i,'CN'+i))
            if "MSCr" in QAccept and i in MSC_Units:
                ut.updata_list(QAccept["MSCr"],ut.factor_decorator('R','P'+i,'P'+i))
                ut.updata_list(QAccept["MSCr"],ut.factor_decorator('R','CP'+i,'CP'+i))
                ut.updata_list(QAccept["MSCr"],ut.factor_decorator('R','N'+i,'N'+i))
                ut.updata_list(QAccept["MSCr"],ut.factor_decorator('R','CN'+i,'CN'+i))

        else:
            if "DSC" in QAccept and i in DSC_Units:
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('E',i,'Quantity'))
                ut.updata_list(QAccept["DSC"],ut.factor_decorator('E','C'+i,'Quantity'))
            if "MSCt" in QAccept and i in MSC_Units:
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('E',i,'Quantity'))
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('E','C'+i,'Quantity'))
            if "MSCr" in QAccept and i in MSC_Units:
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('E',i,'Quantity'))
                ut.updata_list(QAccept["MSCt"],ut.factor_decorator('E','C'+i,'Quantity'))
    
    for i in Reason_Table:
        inputs=[i]
        inputs_dimention={i:1}
        for j in Research_Influence[i]:
            if j != "DarkTemplar" and j != "SiegeTank" and j != "Ghost" and j != "Stalker":
                inputs.append(ut.factor_decorator('R','P'+j,'P'+j))
                inputs_dimention[ut.factor_decorator('R','P'+j,'P'+j)]=1
                inputs.append(ut.factor_decorator('R','N'+j,'N'+j))
                inputs_dimention[ut.factor_decorator('R','N'+j,'N'+j)]=1
            else:
                inputs.append(ut.factor_decorator('E',j,'Quantity'))
                inputs_dimention[ut.factor_decorator('E',j,'Quantity')]=1
        Create_Reason('T'+i,inputs,['T'+i],inputs_dimention,{'T'+i:1},Norm=False) #科技优势
        if "MSCt" in QAccept:
            ut.updata_list(QAccept["MSCt"],ut.factor_decorator('R','T'+i,'T'+i))
        if "MSCr" in QAccept:
            ut.updata_list(QAccept["MSCr"],ut.factor_decorator('R','T'+i,'T'+i))

    # 为任务生成推理路径
    for i in TASK:
        RStack_List[i]=RStack()
        Task_Prereason(i)
        Task_Net[i]=QNet(QAccept[i],Action_Space[i],Qdim_hidden=16)
    print(RStack_List["MSCt"].Sequence)

# 每轮计算前的初始化
def Init(s):
    
    # 初始化变量
    for i in EFrag_Count:
        EFrag_Count[i]=0 #EFrag计数器,根据实体名索引 *记录每种实例化实体的数量
    EFrag_List={} #EFrag实例表,{ID:EFrag}
    for i in EValue_Table: #保存每个EFrag的值,{Type:VALUE},VALUE={ID:value}
        EValue_Table[i]={}
    EOutput_List={} #EFrag输出表,每种实体一张表 *感知部分的最终输出 {Type:{attrib:value}}
    
    #初始化主网络
    global Current_Values,RFrag_Checked
    Current_Values={} #已知属性表,{attrib:value}
    RFrag_Checked=[] #已推理的RFrag
    
    #处理观察到的状态
    count=0
    for i in s:
        if count<16:
            Type=Entity_Table[count]
            if Type != "DarkTemplar" and Type != "SiegeTank" and Type != "Ghost" and Type != "Stalker":
                temp=Observer(Type,{ut.factor_decorator('E',Type,'Quantity'):i}) #为观察到的标签创建一组EFrag
            else:
                Current_Values[ut.factor_decorator('E',Type,'Quantity')]=ut.prepare_data(i) #特殊单位不创建EFrag,直接添加到已知量表中
        elif count>=16 and count<32:
            Type=Entity_Table[count-16]
            if Type != "DarkTemplar" and Type != "SiegeTank" and Type != "Ghost" and Type != "Stalker":
                temp=Observer('C'+Type,{ut.factor_decorator('E','C'+Type,'Quantity'):i}) #为观察到的标签创建一组EFrag
            else:
                Current_Values[ut.factor_decorator('E','C'+Type,'Quantity')]=ut.prepare_data(i) #特殊单位不创建EFrag,直接添加到已知量表中
        else:
            break
        count+=1
    '''
    Current_Values["Self_HP"]=ut.prepare_data(s[32])
    Current_Values["Enemy_HP"]=ut.prepare_data(s[33])
    Current_Values["Minerals"]=ut.prepare_data(s[34])
    Current_Values["Foods"]=ut.prepare_data(s[35])
    '''
    Current_Values["InfantryLevel"]=ut.prepare_data(s[32])
    Current_Values["ShipLevel"]=ut.prepare_data(s[33])
    Current_Values["VehicleLevel"]=ut.prepare_data(s[34])
    EOutput() #聚合同类EFrag的输出值,放入表中{Type:{reason:value}}