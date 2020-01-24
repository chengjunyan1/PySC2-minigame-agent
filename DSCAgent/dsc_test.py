from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from pysc2.agents import base_agent
from pysc2.lib import actions

Style = [
    1.0,    # Noop
    1.0,    # Adept
    1.0,    # Banshee
    1.0,    # Battlecruiser
    1.0,    # Carrier
    1.0,    # Colossus
    1.0,    # DarkTemplar
    1.0,    # Ghost
    1.0,    # Immortal
    1.0,    # Liberator
    1.0,    # Marine
    1.0,    # SiegeTank
    1.0,    # Stalker
    1.0,    # Tempest
    1.0,    # Thor 
    1.0,    # Zealot
]
Style=np.array(Style)

Actions_for_DSC=[0,457,459,460,461,462,465,468,473,475,477,492,493,495,496,503]

class TestAgent(base_agent.BaseAgent):
  """ 
  DSC Agent 3 steps edition
  agent for Dynamic Strategic Combat minigame
  Per action need 3 steps (first observe army,second select base,third take action)
  Could observe full infomation about the army,but waste 2 steps per action
  """

  def step(self, obs):
    super(TestAgent, self).step(obs)

    action_seeds=np.random.rand(16)*Style
    action=np.argmax(action_seeds)
    if Actions_for_DSC[action] in obs.observation["available_actions"]:
        function_id = Actions_for_DSC[action]
    else: #lack resources
        function_id = 0
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]

    return actions.FunctionCall(function_id, args)
