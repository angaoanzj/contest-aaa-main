import random, itertools, copy
import functools
import math
import os
import sys
import tensorflow as tf
import numpy as np

ls = ['B', 'r', 'g', 'b', 'w']

# print(set(itertools.permutations('11100', 5)))

# RETURN_DICT = [('g', 'w', 'y'), ('r', 'g', 'y'), ('b', 'w', 'w'), ('B', 'r', 'r'), ('r', 'b', 'y'), ('B', 'w', 'w'), ('g', 'g', 'g'), 
#                ('r', 'r', 'b'), ('B', 'g', 'y'), ('g', 'y', 'y'), ('g', 'b', 'b'), ('r', 'y', 'y'), ('r', 'b', 'b'), ('w', 'y', 'y'), 
#                ('b', 'w', 'y'), ('B', 'w', 'y'), ('r', 'r', 'r'), ('r', 'r', 'y'), ('g', 'b', 'y'), ('r', 'r', 'g'), ('y', 'y', 'y'), 
#                ('w', 'w', 'y'), ('g', 'g', 'b'), ('r', 'r', 'w'), ('b', 'b', 'w'), ('g', 'g', 'y'), ('B', 'B', 'w'), ('r', 'g', 'g'), 
#                ('B', 'g', 'g'), ('w', 'w', 'w'), ('B', 'b', 'y'), ('B', 'B', 'y'), ('b', 'b', 'y'), ('r', 'w', 'w'), ('r', 'w', 'y'), 
#                ('B', 'B', 'g'), ('g', 'w', 'w'), ('B', 'B', 'r'), ('B', 'b', 'b'), ('b', 'b', 'b'), ('B', 'y', 'y'), ('B', 'B', 'B'), 
#                ('B', 'B', 'b'), ('b', 'y', 'y'), ('g', 'g', 'w'), ('B', 'r', 'y'), ('B', 'b'), ('B', 'B'), ('b', 'b'), ('y', 'y'), 
#                ('g', 'y'), ('r', 'b'), ('g', 'w'), ('B', 'y'), ('b', 'y'), ('b', 'w'), ('B', 'w'), ('r', 'g'), ('w', 'w'), ('B', 'r'), 
#                ('g', 'b'), ('B', 'g'), ('r', 'w'), ('r', 'y'), ('r', 'r'), ('g', 'g'), ('w', 'y'), 
#                ('B'), ('b'), ('r'), ('y'), ('w'), ('g')]


index2bi = ['11100', '11010', '11001', '10110', '10101', '10011', '01110', '01101', '01011', '00111' ]
COLORS = ['black', 'red', 'green', 'blue', 'white']
COLOURS = {'B':'black', 'r':'red', 'y':'yellow', 'g':'green', 'b':'blue', 'w':'white'}
ls = ['B', 'r', 'g', 'b', 'w']
ls2 = ['B', 'r', 'g', 'b', 'w', 'y']


# def generate_return_combos(collected_list):
#     total_return_combos = []
#     for binary in collected_list:
#         return_combos = []
#         for i in range(5):
#             collected_gems = {COLORS[i]:1 for i in range(5) if (binary[i]=='1')}
#         total_gems_list = [i for i in COLOURS.values() if i not in collected_gems.keys()]
            
#         for num_return in range(0,3+1):
#             for combo in set(itertools.combinations_with_replacement(total_gems_list, num_return)):
#                 #Filter out colours with zero gems, and append.
#                 returned_gems = {c:0 for c in COLOURS.values()}
#                 for colour in combo:
#                     returned_gems[colour] += 1
                    
#                 return_combos.append(dict({i for i in returned_gems.items() if i[-1]>0}))
#         # print(return_combos)
#         total_return_combos.append(return_combos)
                
#     return total_return_combos

def generate_return_combos(collected_gems, num_return):
    temp_return_combos = []
    return_combos = []
    total_gems_list = [i for i in COLOURS.values() if i not in collected_gems.keys()]
        
    for num in range(0,num_return+1):
        for combo in set(itertools.combinations_with_replacement(total_gems_list, num)):
            #Filter out colours with zero gems, and append.
            returned_gems = {c:0 for c in COLOURS.values()}
            for colour in combo:
                returned_gems[colour] += 1
            temp_return_combos.append(dict({i for i in returned_gems.items() if i[-1]>0}))
            
    for d in temp_return_combos:
        return_combos.append(dict(sorted(d.items(), key=lambda x:x[0])))
        return_combos = sorted(return_combos, key=lambda x: list(x.keys()))  
    return return_combos

return_combos = generate_return_combos({'black':1, 'red':1, 'green':1},3)
print(return_combos)
print(len(return_combos))

# cd = os.path.dirname(os.path.abspath('agents'))
# print(cd)
# print(tf.test.is_gpu_available())

def generate_collect_combos():
    temp_collect_combos = []
    collect_combos = []
    available_colours = [c for c in COLOURS.values() if c !='yellow']
    for combo_length in range(1, min(len(available_colours), 3) + 1):
        for combo in itertools.combinations(available_colours, combo_length):
            collected_gems = {c:0 for c in COLOURS.values() if c != 'yellow'}
            for colour in combo:
                collected_gems[colour] += 1
            temp_collect_combos.append(dict({i for i in collected_gems.items() if i[-1]>0})) 
    
    for d in temp_collect_combos:
        collect_combos.append(dict(sorted(d.items(), key=lambda x:x[0])))
        collect_combos = sorted(collect_combos, key=lambda x: len(x.keys()))  
    return collect_combos

collect_combos = generate_collect_combos()
print("Collect Combos: ", collect_combos)
print(len(collect_combos))

# test = {c:0 for c in COLOURS.values()}
# retured_gems = dict({i for i in test.items() if i[-1]>0})
# print(dict({i for i in test.items() if i[-1]>0}))
# print(type(retured_gems)==dict)
