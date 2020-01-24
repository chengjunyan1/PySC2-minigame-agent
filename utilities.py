import numpy as np
import torch
from torch.autograd import Variable

# Utilities

def dict_add(dic_A,dic_B): #相同索引字典数求和
    temp={}
    if dic_A=={}:
        return dic_B
    if dic_B=={}:
        return dic_A
    for i in dic_A:
        temp[i]=dic_A[i]+dic_B[i]
    return temp

def dict_chack2d(dic,target): #查看一个二级dict中有无目标值,查到返回一级目录
    for i in dic:
        for j in dic[i]:
            if j==target:
                return i
    return "NOMATCH"

def dict_merge(dic_A,dic_B): #将dic_B并入dic_A
    for i in dic_B:
        dic_A[i]=dic_B[i]

def updata_list(object_list,data,default_open=False): #输入要加入的值和待添加的list,此list相当于集合,default_open消除型号开头的缺省值
    if type(data)==str:
        if data not in object_list:
            if default_open:
                default='*'+data
                if default in object_list: #若表中存在缺省项则先删除
                    default_index=object_list.index(default)
                    del object_list[default_index]
                    object_list.append(data)
                else:
                    object_list.append(data)
            else:
                object_list.append(data)
    elif type(data)==list:
        for i in range(len(data)):
            if data[i] not in object_list:
                if default_open:
                    default='*'+data[i]
                    if default in object_list: #若表中存在缺省项则先删除
                        default_index=object_list.index(default)
                        del object_list[default_index]
                        object_list.append(data[i])
                    else:
                        object_list.append(data[i])
                else:
                    object_list.append(data[i])
                
def subsetof_list(data,object_list): #data是object_list的子集
    return set(data).issubset(object_list)

def intersect_list(data,object_list): #返回data和object_list的交集
    intersect=[]
    for i in data:
        if i in object_list:
            intersect.append(i)
    return intersect

def factor_decorator(Flag,Type,factors): #属性修饰器,将局部属性名转换为全局属性名
    if type(factors)==str:
        return Flag+':'+Type+':'+factors #Flag表示'R'或'E',Type为类型名,factors为属性名
    elif type(factors)==list:
        table=[]
        for i in factors:
            temp=Flag+':'+Type+':'+i
            table.append(temp)
        return table

def globalfactor_parser(factor): #将全局变量名拆分为Flag,Type,Factor
    return factor.split(':')

def finish_check(current,target): #current为list,target为dict,完成为1未完成为0
    for i in target:
        if i not in current:
            return 0
    return 1

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