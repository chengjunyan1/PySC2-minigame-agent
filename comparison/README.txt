目录

CNN	    图像输入，用CNN处理的agent
DQN 	    数组输入，用DQN处理的agent
A3C 	    没做完，还不能用（不打算做了）
Net 	    存网络参数，训练记录
Map 	    存地图
Repository    保存各种agent的代码（不用于训练，需更新以适用于新地图）

测试

#Single Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent

#Multi Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent --agent2 pysc2.agents.random_agent.RandomAgent 