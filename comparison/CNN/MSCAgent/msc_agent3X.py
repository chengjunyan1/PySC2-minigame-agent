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


class MSCAgent(base_agent.BaseAgent):
	""" 
	DSC Agent 1 step edition(三倍速)
	agent for Dynamic Strategic Combat minigame
	Per action need 1 steps (observe infomation directly from the screen and take action instantly)
	Could observe vague infomation about the army,but could instantly take actions
	"""
	def __init__(self):
		super(MSCAgent, self).__init__()

		self.pre_enemyhp=5000
		self.pre_selfhp=5000
		self.save_state={}
		self.save_state["t"]=0
		self.save_state["r"]=0
		self.save_action={}
		self.save_action["t"]=0
		self.save_action["r"]=0
		self.prestep_time=0

		self.InfantryLevel=0
		self.ShipLevel=0
		self.VehicleLevel=0
		self.first_research=1

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
		if os.path.isfile(Record_Path+"WINRATE(MSC).txt"):
			self.Q.winrate_record=np.loadtxt(Record_Path+"WINRATE(MSC).txt")
			self.Q.winrate_record=self.Q.winrate_record.tolist()
			if type(self.Q.winrate_record)==float:
				self.Q.winrate_record=[self.Q.winrate_record]

	def step(self, obs):
		super(MSCAgent, self).step(obs)

		step_time=datetime.datetime.now() #计时
		if obs.last():
			#Sparse reward
			#记忆转储到记忆库中,根据游戏结果进行奖励
			self.Q.store_transfer("r",obs.reward)
			self.Q.store_transfer("t",obs.reward)
			#Record
			if self.Q.start_record['t']==1:
				self.Q.checkpoint_cost['t'].append(len(self.Q.cost_record['t'])) 
				self.Q.checkpoint_qvalue['t'].append(len(self.Q.qvalue_record['t']))
			if self.Q.start_record['r']==1:
				self.Q.checkpoint_cost['r'].append(len(self.Q.cost_record['r'])) 
				self.Q.checkpoint_qvalue['r'].append(len(self.Q.qvalue_record['r']))

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

			if self.Q.start_record['t']==1:
				self.Q.winrate_record.append(float(self.Game_Win)/self.Game_Loop)
				Record_Path=QA.Main_Directory+'Net/RecordCNN/'
				np.savetxt(Record_Path+"WINRATE(MSC).txt", np.array(self.Q.winrate_record))
			function_id=0
		else:
			#_______________Observe_______________#
			unit_type=obs.observation["feature_screen"].unit_type
			player_relative=obs.observation["feature_screen"].player_relative
			v_state=np.array([unit_type*(-1.0*(player_relative==4))]) #对方为负,己方为正,1x84x84
			vespene=obs.observation["player"][2]	#remained vespene
			if obs.first():
				self.first_research=1
			
			#_______________AGENT_______________#
			if vespene>=300:
				action=self.Q.choose_action("r",v_state)
				function_id=Actions_List['r'][action]
				if action==0:
					self.InfantryLevel+=1
				elif action==1:
					self.ShipLevel+=1
				elif action==2:
					self.VehicleLevel+=1
				if not self.first_research:
					self.Q.store_sparse("r",self.save_state["r"],self.save_action["r"],v_state) #临时存储
					print("++++++++++++++++++++++++++")
					print("Research:",Action_Table['r'][self.save_action["r"]])
					print("Memory:",self.Q.memory_counter["r"])
					print("InfantryLevel",self.InfantryLevel)
					print("ShipLevel",self.ShipLevel)
					print("VehicleLevel",self.VehicleLevel)
					if QA.RANDOM_FLAG:
						print("*Random Generated Action")
					print("++++++++++++++++++++++++++")
				self.save_action["r"]=action
				self.save_state["r"]=v_state
				self.first_research=0
			else:
				Action=self.Q.choose_action("t",v_state) #Generate actions by algorithm
				self.save_action["t"]=Action

				Not_Save=0
				if Actions_List['t'][Action] in obs.observation["available_actions"]:
					function_id = Actions_List['t'][Action]
				else: #lack resources
					function_id = 0
					self.save_action["t"] = 0
					Not_Save=1

				#experience replay:st=self.save_state,at=self.save_action,r=reward,st_=v_state
				self.step_counter+=1
				if not obs.first():
					if not Not_Save:
						self.Q.store_sparse("t",self.save_state["t"],self.save_action["t"],v_state) #临时存储
						self.memory_counter+=1
						print("+———————————————————————————————+")
						print("Memory Count:",self.memory_counter)
						print("Step Count:",self.step_counter)
						print("Action:",Action_Table['t'][self.save_action["t"]])
						if QA.RANDOM_FLAG:
							print("*Random Generated Action")
						print("Time Cost:",step_time-self.prestep_time)
						print("+———————————————————————————————+")
			
			self.prestep_time=step_time
			self.save_state["t"]=v_state
			
			# 记忆库满了就进行学习
			if self.Q.memory_counter["t"] > QA.MEMORY_START["t"]:
				self.Q.start_record['t']=1
				self.Q.learn("t")
			if self.Q.memory_counter["r"] > QA.MEMORY_START["r"]:
				self.Q.start_record['r']=1
				self.Q.learn("r")

			#_______________COEND_______________#

			print(self.Q.eval_net.tNet.state_dict())

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[function_id].args]

		return actions.FunctionCall(function_id, args)