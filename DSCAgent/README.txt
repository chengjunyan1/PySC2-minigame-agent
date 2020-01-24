！！！！agent vs agent！！！！

#With Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.O.DSCAgent.dsc_agent3X.DSCAgent --agent_race terran --agent2 pysc2.agents.C.DSCAgent.dsc_agent3XO.DSCAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.C.DSCAgent.dsc_agent3XO.DSCAgent --agent_race terran --agent2 pysc2.agents.C.DSCAgent.dsc_agent3XO.DSCAgent --agent2_race protoss

！！！！agent vs random！！！！

#With Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.O.DSCAgent.dsc_agent3X.DSCAgent --agent_race terran --agent2 pysc2.agents.O.DSCAgent.dsc_test.TestAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.C.DSCAgent.dsc_agent3XO.DSCAgent --agent_race terran --agent2 pysc2.agents.O.DSCAgent.dsc_test.TestAgent --agent2_race protoss

！！！！agent vs human！！！！

#With Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.O.DSCAgent.dsc_agent3X.DSCAgent --agent_race terran --agent2 pysc2.agents.O.noop_agent.NoopAgent --agent2_race protoss

#Without Train
python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.C.DSCAgent.dsc_agent3XO.DSCAgent --agent_race terran --agent2 pysc2.agents.O.noop_agent.NoopAgent --agent2_race protoss