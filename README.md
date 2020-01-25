# PySC2-minigame-learning-agent
A PySC minigame learning agent

*by Cheng Jun-Yan*

**Prerequisite:** mainly PyTorch 0.4.1 and PySC2

This is a research project supervised by Prof. Zhao Liu from 2018.3 to 2018.7. This is not a group project. It contains a learning system designed by myself by integrating the Multi Entity Beyasian Network and reinforcement learning. In comparison folder, I implemented other agents based on DDQN, CNN for comparison. The Draft folder included some my interesting drafts which showed my original thoughts about this project.

## System Architecture

![image](https://github.com/chengjunyan1/PySC2-minigame-learning-agent/raw/master/sd.png)

#### This figure shows the infomation flow of my system. Some details may not be able to be showed in the figure. Include:

#### - The forgetting mechanism of the Facts base. 
For this PySC2 minigame, I choose to clear the facts in every time step. 

#### - The way action selected. 
It is assumed that once the inference finished, there must be a fragment output action probability vector activated, then the action could directly selected by this vector.  

#### - The way update the networks. 
The network is trained based on Experience Replay technique (Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236).

#### - The way activated fragments form a "big" network. 
The inference engine will store a inference stack that stored all the visited fragments when inferencing in a time step. Thus a graph composed of these fragments with the perception as the top layer and the action vector as the leaf could be obtained by the inference stack. Since each fragment is actually compsosed of a neural network, all these network could form a lerge *end-to-end* neural network.   

## About the method

The basic idea of the method I used is introducing the human knowledge into the neural network. In probabilistic graph models, humans can transform their knowledge into the structure of probabilistic graph, while the probabilistic models could be learned from data. On the other side, the neural network have troumendous ability of fitting the functions. So an simple idea is to use the neural networks to relace the probabilistic models in a PGM, while the structure of the PGM is given from human's knowledge. 

Specifically, the graph model in my method is based on Multi Entity Beyasian Network (Kathryn Blackmond Laskey, MEBN: A language for first-order Bayesian knowledge bases, Artificial Intelligence, Volume 172, Issues 2–3, 2008: https://www.sciencedirect.com/science/article/pii/S0004370207001312). A graph is composed of many disjoint fragments which are smaller graphs, each fragment could be regarded as a function maps the "root" nodes to the leaves. The probabilitic model inside the fragment is represented by a neural network. When reasoning, the planner will automatically find the fragments that could map from the input perceptions to the desired result which is the action for the agent to execute in a time step. 

The structure of the fragment should be given by human expert while the probabilitic model is learned by the agent itself. For example, I may design a fragment input the number of marines and output the possibility of buying a medivac next step because I think the more marines we have the more medivac we need. 

There are two rules of activating fragments, by target or by perception. For example, we have three fragment. the root of fragment 1 are [a,b], its leaves are [c,d,e], fragment 2 is [f] to [g,h], fragment 3 is [e,g] to [i]. If we activate by target and set target as [i], the planner will automatically build a reasoning route, firstly use 1,2 then 3. If we activate by perception, when the system percept [a,b,f], fragments 1,2 will be activated, then, [e,g] generated by 1,2 will activate 3. These two rules could also be combined, firstly use target to restrict the potential fragents we need for a task when a task assigned, then activate fragments in each time step based on the perception.

The training of the agent is similar to DDQN. The agent could be trained by playing game with itself, or playing with other agent. 

## About the game

PySC2 is a dynamic reinforcement learning environment on StarCraft II developed by Deepmind, this is their lated result on PySC2, AlphaStar:https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning. 

The agent is not aimed to challenge the complete StarCraft II game which is too complex. I chosed to challenge a minigame based on SC2. The minigame is called Dynamical Strategic Combat (and a variant called MSC, the difference is that, in MSC, you can use your resources to upgrade the technologies that make the units more powerful). This is a 1v1 game. The target of this minigame is to destroy enemy's base. The resources in each player's hands increase automatically over time, and players can use these resources to build thier army. The newly bought units will automatically attack on the enemy's base (go a straight line to enemie's base). However, it may also fight with the enemy's army on the road. The unit and fight mechanism in SC2 is very complex which is impossible to simply described here.

The map of the minigame is contained on Map folder.

## About the environment

The camera is fixed, and the units are not controllable to the agent. The number of available actions for the agent in each step is restrict to 15 (building 15 different units. The input of the agent is the location of the units and their status. The enemy's units may not be observable all the time which is restrited by the *"war fog"* mechanism in SC2. The state space is largely reduced compared to the complete SC2 game. However, because of the highly complex mechanism of SC2, it is still very challenging (even human player who have no experience of such kind of game can not getting start in a short time!).

*Sample image of the minigame

![image](https://github.com/chengjunyan1/PySC2-minigame-learning-agent/raw/master/dsc.png)

## About the result

In the experiment, the behavior of the agent is still looked not very smart. Although it could defeat the random agent and some very simple agent. The agent always tends to keep use one or two kinds of unit. The strategy only changed a little when the situation changed.

Roughly speaking, I think there are several main causes:

Firstly, the minigame itself is not very fair, I adjusted the data of each unit for many times, but there are still many units not useful or economy. Testing the algorithm in a incomplete environment is not very rational. 

Secondly, the design of reward is difficult, the agent cannot get an accurate feedback of every action it made. The agent is only rewarded at the end of the game which make it difficult to aware which action is good or not. 

Thirdly, defining the structure of the graph is also not easy. Even myself cannot judge whether the structure is rational enough because I'm also not an expert of this game. In most cases, I give the structure by intuition. 

Moreover, although StarCraft II is an extremely intresting environment, but researching a new method in such a complex environment is not very rational without a large team and long term effort. Too much mechanisms inside the game. The processing of input data is already very difficult, all inputs are represented as rgb map and feature maps. We need to build a huge intelligent system that processing different kinds of infomation to support the running of the algorithm.
