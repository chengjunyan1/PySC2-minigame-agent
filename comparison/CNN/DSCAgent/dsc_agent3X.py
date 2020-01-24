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

import pysc2.agents.Q.CNN.agent as QA
import pysc2.agents.Q.maincontroller as MC


_PlanetaryFortress = 130 

Actions_List=MC.Actions_List
Action_Table=MC.Action_Table


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
		self.save_state=0 #0-14 our situation, 15-29 enemy situation,30 our hp, 31 enemy hp,32 our minierals
		self.save_acton=0
		self.prestep_time=0

		self.Game_Loop=0
		self.Game_Win=0
		self.Game_Tie=0
		self.Game_Lose=0
		self.Game_Score=0
		self.memory_counter=0 	#记录产生了多少记忆
		self.step_counter=0 	#记录共完成了多少步

		self.Q=QA.QAgent()
		self.Q.read_record()
		Record_Path=QA.Main_Directory+'Net/RecordCNN/'
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
				Record_Path=QA.Main_Directory+'Net/RecordCNN/'
				np.savetxt(Record_Path+"WINRATE(DSC).txt", np.array(self.Q.winrate_record))
				self.Q.record()
			function_id=0
		else:
			#_______________Observe_______________#
			unit_type=obs.observation["feature_screen"].unit_type
			player_relative=obs.observation["feature_screen"].player_relative
			v_state=np.array([unit_type*(-1.0*(player_relative==4))]) #对方为负,己方为正,1x84x84

			#_______________AGENT_______________#
			Action=self.Q.choose_action('d',v_state) #Generate actions by algorithm
			self.save_action=Action

			Not_Save=0
			if Actions_List['d'][Action] in obs.observation["available_actions"]:
				function_id = Actions_List['d'][Action]
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
					print("QMemory Count:",self.Q.memory_counter['d'])
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

			print(self.Q.eval_net.Res1.state_dict())

			#_______________COEND_______________#

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[function_id].args]

		return actions.FunctionCall(function_id, args)