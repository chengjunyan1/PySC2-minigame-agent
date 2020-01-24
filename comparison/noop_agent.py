from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions


class NoopAgent(base_agent.BaseAgent):
  """No op agent for human operate."""

  def step(self, obs):
    super(NoopAgent, self).step(obs)
    function_id = 0
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
