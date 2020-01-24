# DEFINITIONS

import torch

# Setting for Agent
#Task_Set=["d","t","r"]
Actions_List={}
Actions_List['d']=[0,457,459,460,461,462,465,468,473,475,477,492,493,495,496,503]
Actions_List['t']=[0,457,460,461,462,473,475,477,492,493,496,500,503]
Actions_List['r']=[410,414,423]

Action_Table={}
Action_Table['d']=["No Operation","Train Adept","Train Banshee","Train Battlecruiser","Train Carrier","Train Colossus","Train DarkTemplar","Train Ghost","Train Immortal","Train Liberator","Train Marine","Train SiegeTank","Train Stalker","Train Tempest","Train Thor","Train Zealot"]
Action_Table['t']=["No Operation","Train Adept","Train Battlecruiser","Train Carrier","Train Colossus","Train Immortal","Train Liberator","Train Marine","Train SiegeTank","Train Stalker","Train Thor","Train VoidRay","Train Zealot"]
Action_Table['r']=["Research InfantryLevel","Research ShipLevel","Research VehicleLevel"]

# Init fot Agent
Units_List=[311,55,57,79,4,76,50,83,689,48,33,32,74,496,52,80,73]
Units_Table=["Adept","Banshee","Battlecruiser","Carrier","Colossus","DarkTemplar","Ghost","Immortal","Liberator","Marine","SiegeTank","SiegeTankSieged","Stalker","Tempest","Thor","VoidRay","Zealot"]
Units_Size=[12,21,69,69,37,9,9,21,21,9,32,32,12,69,37,37,37,12]
DSC_Price=[0,60,145,400,400,300,160,200,200,150,40,200,100,500,350,80]
Tank_index=10
PF_Center=[0,26]

# Configurations
DEVICE_IDS=[0,1]
CUDA_OPEN=torch.cuda.is_available()
Main_Directory='C:/Users/cheng/Anaconda3/Lib/site-packages/pysc2/agents/Q/'