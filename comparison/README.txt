Ŀ¼

CNN	    ͼ�����룬��CNN�����agent
DQN 	    �������룬��DQN�����agent
A3C 	    û���꣬�������ã����������ˣ�
Net 	    �����������ѵ����¼
Map 	    ���ͼ
Repository    �������agent�Ĵ��루������ѵ������������������µ�ͼ��

����

#Single Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent

#Multi Player
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent --agent2 pysc2.agents.random_agent.RandomAgent 