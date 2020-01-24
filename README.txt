目录

DSCAgent      任务一的Agent
MSCAgent     任务二的Agent
Net 	     存放网络参数和训练记录

测试

#Single Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent

#Multi Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent --agent2 pysc2.agents.random_agent.RandomAgent 