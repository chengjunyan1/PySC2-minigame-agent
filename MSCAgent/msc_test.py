from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from pysc2.agents import base_agent
from pysc2.lib import actions

Train_for_MSC=[0,457,460,461,462,473,475,477,492,493,496,500,503]
Research_for_MSC=[0,410,414,423]

class TestAgent(base_agent.BaseAgent):
    """ 
    DSC Agent 3 steps edition
    agent for Dynamic Strategic Combat minigame
    Per action need 3 steps (first observe army,second select base,third take action)
    Could observe full infomation about the army,but waste 2 steps per action
    """

    def step(self, obs):
        super(TestAgent, self).step(obs)
        
        vespene=obs.observation["player"][2] #remained vespene
        function_id = 0
        if vespene>=120:
            action=np.random.randint(0,4)
            if Research_for_MSC[action] in obs.observation["available_actions"]:
                function_id = Research_for_MSC[action]
            else: #lack resources or max research
                function_id = 0
        if function_id==0: #lack resources
            action=np.random.randint(0,13)
            if Train_for_MSC[action] in obs.observation["available_actions"]:
                function_id = Train_for_MSC[action]
            else: #lack resources or max research
                function_id = 0
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]

        return actions.FunctionCall(function_id, args)
