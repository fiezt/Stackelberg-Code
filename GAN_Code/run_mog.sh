#!/bin/sh
python MoG_Simulation.py sim3 gan simgrad diamond tanh normal big 2 1
python MoG_Simulation.py sim9 gan stack diamond tanh fast big 2 1
python MoG_Simulation.py sim14 nsgan simgrad circle relu normal big 2 1
python MoG_Simulation.py sim19 nsgan stack circle relu fast big 2 1