from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import math
import datetime
import os

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import pysc2.agents.Q.DQN.agent as QA
import pysc2.agents.Q.maincontroller as MC


_PlanetaryFortress = 130 
_UNIT_TYPE=features.SCREEN_FEATURES .unit_type.index

Actions_List=MC.Actions_List
Units_List=MC.Units_List
Units_Table=MC.Units_Table
Units_Size=MC.Units_Size
Tank_index=MC.Tank_index


def sigmoid_rate(x,speed1=100,speed2=1000):
    y=np.floor(100.0/(1+np.exp(-x/speed1)))-math.floor(80.0/(1+np.exp(x/speed2)))
    y=np.floor(y/10)*10
    return y

class DSCAgent(base_agent.BaseAgent):
	""" 
	DSC Agent 1 step edition(三倍速) Used as component,cannot learn,update every few game loops
	agent for Dynamic Strategic Combat minigame
	Per action need 1 steps (observe infomation directly from the screen and take action instantly)
	Could observe vague infomation about the army,but could instantly take actions
	"""
	def __init__(self):
		super(DSCAgent, self).__init__()

		self.Game_Loop=0
		self.Game_Win=0
		self.Game_Tied=0
		self.Game_Lose=0
		self.Game_Score=0

		self.InfantryLevel=0
		self.ShipLevel=0
		self.VehicleLevel=0

		self.Q=QA.QAgent()
		#QA.RANDOM_OPEN=0
		Record_Path=QA.Main_Directory+'Net/RecordDQN/'
		self.Action_Record=np.zeros(15)
		if os.path.isfile(Record_Path+"WINRATEOP(DSC).txt"):
			self.Q.winrate_record=np.loadtxt(Record_Path+"WINRATEOP(DSC).txt")
			self.Q.winrate_record=self.Q.winrate_record.tolist()
			if type(self.Q.winrate_record)==float:
				self.Q.winrate_record=[self.Q.winrate_record]
		if os.path.isfile(Record_Path+"ActionRecord(DSC).txt"):
			self.Action_Record=np.loadtxt(Record_Path+"ActionRecord(DSC).txt")
	
	def step(self, obs):
		super(DSCAgent, self).step(obs)
        #update network in a rate
		UPDATE_RATE=QA.UPDATE_RATE #sigmoid_rate(self.Game_Loop)
		if obs.last():
			self.Game_Loop+=1
			if self.Game_Loop%UPDATE_RATE==0 and os.path.isfile(QA.Main_Directory+'Net/NetDQN/Q-net.pkl'):
				self.Q.eval_net.QNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetDQN/Q-net.pkl'))
				self.Q.target_net.QNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetDQN/Q-net.pkl'))
			if self.Game_Loop%UPDATE_RATE==0 and os.path.isfile(QA.Main_Directory+'Net/NetDQN/d-net.pkl'):
				self.Q.eval_net.dNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetDQN/d-net.pkl'))
				self.Q.target_net.dNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetDQN/d-net.pkl'))

			self.Game_Score+=obs.reward # Victory:1 Tied(Timeout):0 Defeated:-1
			self.Game_Win+=(obs.reward==1) 
			self.Game_Tied+=(obs.reward==0)
			self.Game_Lose+=(obs.reward==-1)

			if QA.TEST_MODE:
				print("\n")
				print("——————Game Info——————")
				print("Game Loops",self.Game_Loop)
				print("Game Win",self.Game_Win)
				print("Game Lose",self.Game_Lose)
				print("Game Tied",self.Game_Tied)
				print("Win Rate",float(self.Game_Win)/self.Game_Loop)
				print("\n")

			self.Q.winrate_record.append(float(self.Game_Win)/self.Game_Loop)
			Record_Path=QA.Main_Directory+'Net/RecordDQN/'
			np.savetxt(Record_Path+"WINRATEOP(DSC).txt", np.array(self.Q.winrate_record))
			np.savetxt(Record_Path+"ActionRecord(DSC).txt", np.array(self.Action_Record))
			function_id=0
		else:
			#Observe
			unit_type_table=obs.observation["feature_screen"][_UNIT_TYPE]
			player_relative=obs.observation["feature_screen"].player_relative
			state_count=0
			v_state=np.zeros(QA.N_STATES)
			#Observe Army
			for i in range(17):
				self_units=unit_type_table*(player_relative==1)
				unit_y,unit_x=(self_units==Units_List[i]).nonzero()
				Unit_Count_Army=int(math.ceil(len(unit_y))/Units_Size[i])
				if Units_Table[i]=="SiegeTankSieged":
					v_state[Tank_index]+=Unit_Count_Army
				else:
					v_state[state_count]=Unit_Count_Army
					state_count+=1
			#Observe Enemies
			for i in range(17):
				enemy_units=unit_type_table*(player_relative==4)
				unit_y,unit_x=(enemy_units==Units_List[i]).nonzero()
				Unit_Count_Enemy=int(math.ceil(len(unit_y))/Units_Size[i])
				if Units_Table[i]=="SiegeTankSieged":
					v_state[Tank_index+15]+=Unit_Count_Enemy
				else:
					v_state[state_count]=Unit_Count_Enemy
					state_count+=1

			unit_hit_points=obs.observation["feature_screen"].unit_hit_points
			unit_type=obs.observation["feature_screen"].unit_type
			#Obtain hit points of enemy's base
			enemy_hit_points=unit_hit_points*(unit_type==_PlanetaryFortress)*(player_relative==4)
			enemy_hp_y,enemy_hp_x=enemy_hit_points.nonzero()
			enemy_hp=np.sum(enemy_hit_points)/len(enemy_hp_y)
			#Obtain hit points of our base
			self_hit_points=unit_hit_points*(unit_type==_PlanetaryFortress)*(player_relative==1)
			self_hp_y,self_hp_x=self_hit_points.nonzero()
			self_hp=np.sum(self_hit_points)/len(self_hp_y)

			minerals=obs.observation["player"][1] 	#remained minerals
			vespene=obs.observation["player"][2]	#remained vespene
			foods=60-obs.observation["player"][3] 	#remained food supply

			v_state[32]=self.InfantryLevel
			v_state[33]=self.ShipLevel
			v_state[34]=self.VehicleLevel
			
            #Choose Action
			Action=self.Q.choose_action('d',v_state) #Generate actions by algorithm

			if Actions_List['d'][Action] in obs.observation["available_actions"]:
				function_id = Actions_List['d'][Action]
				if Action!=0:
					self.Action_Record[Action-1]+=1
			else: #lack resources
				function_id = 0

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[function_id].args]

		return actions.FunctionCall(function_id, args)