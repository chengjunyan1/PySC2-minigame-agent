from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib

'''

Actions for DSC (defined in pysc2.lib.actions)

    Function.ui_func(0, "no_op", no_op),
    Function.ui_func(2, "select_point", select_point),
    Function.ui_func(3, "select_rect", select_rect),
    Function.ui_func(5, "select_unit", select_unit,
                     lambda obs: obs.ui_data.HasField("multi")),
    Function.ui_func(7, "select_army", select_army,
                     lambda obs: obs.player_common.army_count > 0),

    Function.ability(457, "Train_Adept_quick", cmd_quick, 922),
    Function.ability(459, "Train_Banshee_quick", cmd_quick, 621),
    Function.ability(460, "Train_Battlecruiser_quick", cmd_quick, 623),
    Function.ability(461, "Train_Carrier_quick", cmd_quick, 948),
    Function.ability(462, "Train_Colossus_quick", cmd_quick, 978),
    Function.ability(465, "Train_DarkTemplar_quick", cmd_quick, 920),
    Function.ability(473, "Train_Immortal_quick", cmd_quick, 979),
    Function.ability(475, "Train_Liberator_quick", cmd_quick, 626),
    Function.ability(477, "Train_Marine_quick", cmd_quick, 560),
    Function.ability(478, "Train_Medivac_quick", cmd_quick, 620),
    Function.ability(492, "Train_SiegeTank_quick", cmd_quick, 591),
    Function.ability(495, "Train_Tempest_quick", cmd_quick, 955),
    Function.ability(496, "Train_Thor_quick", cmd_quick, 594),
    Function.ability(500, "Train_VoidRay_quick", cmd_quick, 950),
    Function.ability(503, "Train_Zealot_quick", cmd_quick, 916),

Obs for DSC (defined in pysc2.lib.features)

    obs.observation["available_actions"]
    obs.observation["player"]=[
        player_id
        minerals
        vespene
        food used (otherwise known as supply)
        food cap
        food used by army
        food used by workers
        idle worker count
        army count
        warp gate count (for protoss)
        larva count (for zerg)
    ]
    obs.observation["multi_select"]=[
        unit type
        player_relative
        health
        shields
        energy
        transport slot taken if it's in a transport
        build progress as a percentage if it's still being buil t
    ]

    obs.observation["feature_screen"].xxxxxx
    SCREEN_FEATURES:
        height_map : Show the terrain levels.
        visibility : It shows which part of the map is hidden, have been seen, or currently visible.
        creep : It show which part of the minimap is covered by creeps.
        power : Which parts have protoss power, only shows your power.
        player_id : Who owns the units, with absolute ids.
        player_relative : Which units are friendly vs hostile. Takes values in [0, 4], denoting [background, self, ally, neutral, enemy] units respectively.
        unit_type : A unit type id
        selected : Which units are selected.
        hit_points : How many hit points the unit has.
        energy : How much energy the unit has.
        shields : How much shields the unit has. Only for protoss units.
        unit_density : How many units are in this pixel.
        unit_density_aa : An anti-aliased version of unit_density with a maximum of 16 per unit per pixel. This gives you sub-pixel unit location and size.

Tutorials

    SC2 AI Wiki:http://wiki.sc2ai.net/Main_Page
    How to Locate and Select Units in PySC2:https://itnext.io/how-to-locate-and-select-units-in-pysc2-2bb1c81f2ad3

Samples

    unit_type_table = obs.observation["feature_screen"][ _UNIT_TYPE ]
    player_relative=obs.observation["feature_screen"].player_relative
    Unit_Count=np.zeros(17)
    #Observe Enemies
    #print("_______obs________")
    for i in range(17):
      enemy_units=unit_type_table*(player_relative==4)
      unit_y,unit_x =(enemy_units==Units_for_DSC[i]).nonzero()
      Unit_Count[i]=int(math.ceil(len(unit_y))/Unit_Size[i])
      #print(Units_Table[i],Unit_Count[i])

    #Observe Army
    #Option 1:Select Army(By using action 7,full infomation but waste 2 steps)
    army_situation=obs.observation["multi_select"]
    #Option 2:Directly Observe
    for i in range(17):
      self_units=unit_type_table*=(player_relative==1)
      unit_y,unit_x =(self_units==Units_for_DSC[i]).nonzero()
      Unit_Count[i]=int(math.ceil(len(unit_y))/Unit_Size[i])
      #print(Units_Table[i],Unit_Count[i])


    unit_hit_points=obs.observation["feature_screen"].unit_hit_points
    player_relative=obs.observation["feature_screen"].player_relative
    unit_type=obs.observation["feature_screen"].unit_type
    #Get enemy's hp
    enemy_hit_points=unit_hit_points*(unit_type==_PlanetaryFortress)*(player_relative==4)
    enemy_ht_y,enemy_ht_x=enemy_hit_points.nonzero()
    enemy_pf_ht=numpy.sum(enemy_hit_points)/len(enemy_ht_y)

#Unit Types for DSC (defined in pysc2.lib.units)

_PlanetaryFortress = 130 
_Adept = 311
_Banshee = 55
_Battlecruiser = 57
_Carrier = 79
_Colossus = 4
_DarkTemplar = 76
_Immortal = 83
_Liberator = 689
_Marine = 48
_Medivac = 54
_SiegeTank = 33
_SiegeTankSieged = 32
_Tempest = 496
_Thor = 52
_VoidRay = 80
_Zealot = 73

Actions_for_DSC=[0,457,459,460,461,462,465,473,475,477,478,492,495,496,500,503]
Units_for_DSC=[311,55,57,79,4,76,83,689,48,54,33,32,496,52,80,73]
Units_Table=["Adept","Banshee","Battlecruiser","Carrier","Colossus","DarkTemplar","Immortal","Liberator","Marine","Medivac","SiegeTank","SiegeTankSieged","Tempest","Thor","VoidRay","Zealot"]
Action_Table=["No Operation","Train Adept","Train Banshee","Train Battlecruiser","Train Carrier","Train Colossus","Train DarkTemplar","Train Immortal","Train Liberator","Train Marine","Train Medivac","Train SiegeTank","Train Tempest","Train Thor","Train VoidRay","Train Zealot"]
Unit_Size=[12,21,69,69,37,9,21,21,9,21,32,32,69,37,37,12]
PF_Center=[0,26]
_UNIT_TYPE=features.SCREEN_FEATURES .unit_type.index

'''

class DSC(lib.Map):
  directory = "C:\\Users\\cheng\\Documents\\StarCraft II\\Maps"
  players = 2
  game_steps_per_episode = 16 * 60 * 8  # 8 minute limit.

DSC_maps = ["DynamicStrategicCombat"]

for name in DSC_maps:
  globals()[name] = type(name, (DSC,), dict(filename=name))
