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
Action_Table=MC.Action_Table
Units_Size=MC.Units_Size
Tank_index=MC.Tank_index


class DSCAgent(base_agent.BaseAgent):
	""" 
	DSC Agent 1 step edition(三倍速)
	agent for Dynamic Strategic Combat minigame
	Per action need 1 steps (observe infomation directly from the screen and take action instantly)
	Could observe vague infomation about the army,but could instantly take actions
	"""
	def __init__(self):
		super(DSCAgent, self).__init__()

		self.pre_enemyhp=5000
		self.pre_selfhp=5000
		self.save_state=np.zeros(39) #0-14 our situation, 15-29 enemy situation,30 our hp, 31 enemy hp,32 our minierals
		self.save_acton=0
		self.prestep_time=0

		self.InfantryLevel=0
		self.ShipLevel=0
		self.VehicleLevel=0

		self.Game_Loop=0
		self.Game_Win=0
		self.Game_Tie=0
		self.Game_Lose=0
		self.Game_Score=0
		self.memory_counter=0 	#记录产生了多少记忆
		self.step_counter=0 	#记录共完成了多少步

		self.Q=QA.QAgent()
		self.Q.read_record()
		Record_Path=QA.Main_Directory+'Net/RecordDQN/'
		if os.path.isfile(Record_Path+"WINRATE(DSC).txt"):
			self.Q.winrate_record=np.loadtxt(Record_Path+"WINRATE(DSC).txt")
			self.Q.winrate_record=self.Q.winrate_record.tolist()
			if type(self.Q.winrate_record)==float:
				self.Q.winrate_record=[self.Q.winrate_record]

	def step(self, obs):
		super(DSCAgent, self).step(obs)

		step_time=datetime.datetime.now() #计时
		if obs.last():
			#Sparse reward
			#记忆转储到记忆库中,根据游戏结果进行奖励
			self.Q.store_transfer('d',obs.reward)
			#Record
			if self.Q.start_record['d']==1:
				self.Q.checkpoint_cost['d'].append(len(self.Q.cost_record['d'])) 
				self.Q.checkpoint_qvalue['d'].append(len(self.Q.qvalue_record['d']))

			#Game info
			self.Game_Loop+=1
			self.Game_Score+=obs.reward # Victory:1 Tie(Timeout):0 Defeated:-1
			self.Game_Win+=(obs.reward==1) 
			self.Game_Tie+=(obs.reward==0)
			self.Game_Lose+=(obs.reward==-1)
			print("\n")
			print("——————Game Info——————")
			print("Game Loops",self.Game_Loop)
			print("Game Win",self.Game_Win)
			print("Game Lose",self.Game_Lose)
			print("Game Tie",self.Game_Tie)
			print("Win Rate",float(self.Game_Win)/self.Game_Loop)
			print("\n")
			self.Q.eval_net.save_QNet() # save the net params

			if self.Q.start_record['d']==1:
				self.Q.winrate_record.append(float(self.Game_Win)/self.Game_Loop)
				Record_Path=QA.Main_Directory+'Net/RecordDQN/'
				np.savetxt(Record_Path+"WINRATE(DSC).txt", np.array(self.Q.winrate_record))
				self.Q.record()
			function_id=0
		else:
			#_______________Observe_______________#
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

			#state(37):contain enemy and our situation of 15 units(30), enemy and our hp(2),our minerals(1),our food supply(1)
			'''
			v_state[32]=self_hp
			v_state[33]=enemy_hp
			v_state[34]=minerals
			v_state[35]=foods
			'''
			v_state[32]=self.InfantryLevel
			v_state[33]=self.ShipLevel
			v_state[34]=self.VehicleLevel

			#_______________AGENT_______________#
			Action=self.Q.choose_action('d',v_state) #Generate actions by algorithm
			self.save_action=Action

			Not_Save=0
			if Actions_List['d'][Action] in obs.observation["available_actions"]:
				function_id = Actions_List['d'][Action]
				save_rate=lambda x:1 / (1 + np.exp(2-x/40)) #部分动作不保存,根据价格计算出的出现概率
				if Action!=0 and np.random.uniform()>save_rate(MC.DSC_Price[Action]):
					Not_Save=1
			else: #lack resources
				function_id = 0
				self.save_action = 0
				reward = 0
				Not_Save=1

			#experience replay:st=self.save_state,at=self.save_action,r=reward,st_=v_state
			self.step_counter+=1
			if not obs.first():
				if not Not_Save:
					self.Q.store_sparse('d',self.save_state,self.save_action,v_state)
					self.memory_counter+=1
					print("+———————————————————————————————+")
					print("Memory Count:",self.memory_counter)
					print("Step Count:",self.step_counter)
					print("Action:",Action_Table['d'][self.save_action])
					if QA.RANDOM_FLAG:
						print("*Random Generated Action")
					print("Time Cost:",step_time-self.prestep_time)
					print("+———————————————————————————————+")
			
			self.prestep_time=step_time
			self.save_state=v_state
			
			if self.Q.memory_counter['d'] > QA.MEMORY_START['d']:
				self.Q.start_record['d']=1
				self.Q.learn('d') # 记忆库满了就进行学习

			print(self.Q.eval_net.QNet.state_dict())

			#_______________COEND_______________#

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[function_id].args]

		return actions.FunctionCall(function_id, args)