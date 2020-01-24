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


def sigmoid_rate(x,speed1=100,speed2=1000):
    y=np.floor(100.0/(1+np.exp(-x/speed1)))-math.floor(80.0/(1+np.exp(x/speed2)))
    y=np.floor(y/10)*10
    return y

class MSCAgent(base_agent.BaseAgent):
	""" 
	MSC Agent 1 step edition(三倍速) Used for self-play,cannot learn,update every few game loops
	agent for Dynamic Strategic Combat minigame
	Per action need 1 steps (observe infomation directly from the screen and take action instantly)
	Could observe vague infomation about the army,but could instantly take actions
	"""
	def __init__(self):
		super(MSCAgent, self).__init__()

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
		Record_Path=QA.Main_Directory+'Net/RecordCNN/'
		if os.path.isfile(Record_Path+"WINRATEOP(MSC).txt"):
			self.Q.winrate_record=np.loadtxt(Record_Path+"WINRATEOP(MSC).txt")
			self.Q.winrate_record=self.Q.winrate_record.tolist()
			if type(self.Q.winrate_record)==float:
				self.Q.winrate_record=[self.Q.winrate_record]
	
	def step(self, obs):
		super(MSCAgent, self).step(obs)
        #update network in a rate
		UPDATE_RATE=QA.UPDATE_RATE #sigmoid_rate(self.Game_Loop)
		if obs.last():
			self.Game_Loop+=1
			if self.Game_Loop%UPDATE_RATE==0 and os.path.isfile(QA.Main_Directory+'Net/NetMSC/Conv-net.pkl'):
				self.Q.eval_net.Conv.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/Conv-net.pkl'))
				self.Q.target_net.Conv.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/Conv-net.pkl'))
			if self.Game_Loop%UPDATE_RATE==0 and os.path.isfile(QA.Main_Directory+'Net/NetMSC/r-net.pkl'):
				self.Q.eval_net.rNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/r-net.pkl'))
				self.Q.target_net.rNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/r-net.pkl'))
			if self.Game_Loop%UPDATE_RATE==0 and os.path.isfile(QA.Main_Directory+'Net/NetMSC/t-net.pkl'):
				self.Q.eval_net.tNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/t-net.pkl'))
				self.Q.target_net.tNet.load_state_dict(torch.load(QA.Main_Directory+'Net/NetMSC/t-net.pkl'))

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
			Record_Path=QA.Main_Directory+'Net/RecordCNN/'
			np.savetxt(Record_Path+"WINRATEOP(MSC).txt", np.array(self.Q.winrate_record))
			function_id=0
		else:
			#Observe
			unit_type=obs.observation["feature_screen"].unit_type
			player_relative=obs.observation["feature_screen"].player_relative
			v_state=np.array([unit_type*(-1.0*(player_relative==4))]) #对方为负,己方为正,1x84x84
			vespene=obs.observation["player"][2]	#remained vespene
			
            #Choose Action
			if vespene>=300:
				action=self.Q.choose_action("r",v_state)
				function_id=Actions_List['r'][action]
				if action==0:
					self.InfantryLevel+=1
				elif action==1:
					self.ShipLevel+=1
				elif action==2:
					self.VehicleLevel+=1
			else:
				Action=self.Q.choose_action("t",v_state) #Generate actions by algorithm

				if Actions_List['t'][Action] in obs.observation["available_actions"]:
					function_id = Actions_List['t'][Action]
				else: #lack resources
					function_id = 0

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[function_id].args]

		return actions.FunctionCall(function_id, args)