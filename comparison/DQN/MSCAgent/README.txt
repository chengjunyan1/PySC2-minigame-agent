！！！！agent vs agent！！！！

#With Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3X.MSCAgent  --agent_race terran --agent2 pysc2.agents.Q.DQN.MSCAgent.msc_agent3XC.MSCAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3XC.MSCAgent  --agent_race terran --agent2 pysc2.agents.Q.DQN.MSCAgent.msc_agent3X.MSCAgent --agent2_race protoss

！！！！agent vs random！！！！

#With Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3X.MSCAgent --agent_race terran --agent2 pysc2.agents.O.msc_test.TestAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3XC.MSCAgent --agent_race terran --agent2 pysc2.agents.O.msc_test.TestAgent --agent2_race protoss

！！！！agent vs human！！！！

#With Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3X.MSCAgent  --agent_race terran --agent2 pysc2.agents.Q.noop_agent.NoopAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple96 --agent pysc2.agents.Q.DQN.MSCAgent.msc_agent3XC.MSCAgent  --agent_race terran --agent2 pysc2.agents.Q.noop_agent.NoopAgents --agent2_race protoss