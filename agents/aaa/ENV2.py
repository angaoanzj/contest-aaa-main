# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    04/01/2021
# Purpose: Implements a Game class to run implemented games for this framework.

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import random, copy, time, itertools
# from template import GameState, Agent
# from func_timeout import func_timeout, FunctionTimedOut
# import traceback
import os
import sys
from copy import deepcopy
# from agents.aaa.dqn_splendor2 import STATE_DIM
cd = os.path.dirname(os.path.abspath('agents'))
sys.path.append(cd)
from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import *
import numpy as np
import math

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

FREEDOM = True #Whether or not to penalise agents for incorrect moves and timeouts. Useful for debugging.
WARMUP  = 15    #Warmup period (time given to each agent on their first turn).
NUM_AGENTS    = 2
GEMS_DICT = {'black':0, 'red':1, 'yellow':2, 'green':3, 'blue':4, 'white':5}
COLORS = ['black', 'red', 'green', 'blue', 'white']
NOBLE_DICT = {'4g4r':1, '3w3r3B':2, '3b3g3r':3, '3w3b3g':4, '4w4b':5, '4w4B':6, '3w3b3B':7, '4r4B':8, '4b4g':9, '3g3r3B':10}
COLOURS = {'B':'black', 'r':'red', 'y':'yellow', 'g':'green', 'b':'blue', 'w':'white'}
GEMS = {c:0 for c in COLOURS.values()}
bi2index = {'11100':0, '11010':1, '11001':2, '10110':3, '10101':4, '10011':5, '01110':6, '01101':7, '01011':8, '00111':9}
index2bi = ['11100', '11010', '11001', '10110', '10101', '10011', '01110', '01101', '01011', '00111' ]

NOBLES = [('4g4r', {'green': 4, 'red': 4}), 
          ('3w3r3B', {'white': 3, 'red': 3, 'black': 3}), 
          ('3b3g3r', {'blue': 3, 'green': 3, 'red': 3}), 
          ('3w3b3g', {'white': 3, 'blue': 3, 'green': 3}), 
          ('4w4b', {'white': 4, 'blue': 4}), 
          ('4w4B', {'white': 4, 'black': 4}), 
          ('3w3b3B', {'white': 3, 'blue': 3, 'black': 3}), 
          ('4r4B', {'red': 4, 'black': 4}), 
          ('4b4g', {'blue': 4, 'green': 4}), 
          ('3g3r3B', {'green': 3, 'red': 3, 'black': 3}),
          None
          ]

STATE_DIM = 980
# angents' card with no points
NonePointCard = 0

def DoAction(agent_state,action):
    ''' fg reward
    input: action
    output: action cost and the changed cards
    '''
    action_cost = deepcopy(GEMS)
    mychange_card = deepcopy(GEMS)
    # Determine the score of the three action scenarios, if going to the same/different color
    action_cost['score'] = 0
    if action['type'] == 'collect_diff' or action['type'] == 'collect_same':
        # Initialization score is 0
        action_cost['score'] = 0
        for key in GEMS.keys():
            # Iterate through all the colors
            if key in action['collected_gems'].keys():
                temp = action['collected_gems'][key]
            else:
                temp = 0
            if key in action['returned_gems'].keys():
                temp -= action['returned_gems'][key]
            action_cost[key] = temp

    elif action['type'] == 'reserve': 
        # If reserve, get a gold gem
        action_cost['reserve'] = 1  
        for key in GEMS.keys():
            action_cost[key] = 0            

    elif action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
        action_cost['score']+=action['card'].points
        if action['card'].points == 0:
            global NonePointCard
            NonePointCard += 1
        for key in GEMS.keys():
            if key in action['returned_gems'].keys():
                action_cost[key] -= action['returned_gems'][key]
        mychange_card[action['card'].colour] = 1
    if action['noble'] != None:
        action_cost['score']+=3
    # Returns my existing cards, and if the action is to buy cards, it also returns the added cards
    # A value when the cost of the action or the return value
    return action_cost, mychange_card

def getAgentState(game_state,agent):
    card_num = {}
    agent_state = {}
    total_value = {}
    '''
    # agent_state stores the current state, with 3 values: score, card_num(card), total_value(card+gem)
    # The first value is the current agent's score
    # agent.gems the number of gems the player has
    # agent.card The player's existing cards'''
    agent_state['score'] = agent.score
    for key in GEMS.keys():
        # Determine if there is a card of this color
        if key in agent.cards.keys():
             # mystate The second value is the player's existing color value, 
             # equal to the sum of the number of existing gems and the number of gem cards
            total_value[key] = agent.gems[key] + len(agent.cards[key])
            # card_num indicates the number of cards per color
            card_num[key] = len(agent.cards[key])
        else:
            # If the number of cards is 0, then it is the value
            total_value[key] = agent.gems[key]
            card_num[key] = 0
    agent_state['card_num'] = card_num
    agent_state['total_value'] = total_value
    return agent_state


class SplendorEnv():

    def __init__(self):
        self.game_rule = SplendorGameRule(NUM_AGENTS)
        self.state = np.zeros(STATE_DIM)     
        self.done = False
        # self.valid = True
    
    def game2vec(self, game_state):
        self.state = np.zeros(STATE_DIM)
        for index, (gem, count) in enumerate(game_state.board.gems.items()):
            self.state[0 + GEMS_DICT[gem]] = count

# for each level in dealt cards, discretize. Each level consists of 4 cards.
        for index, level in enumerate(game_state.board.dealt):
            for ind, card in enumerate(level):
                if card != None:
                    for colour, cost in card.cost.items():
                        self.state[6 + 11*ind*index+1 + GEMS_DICT[colour]] = cost
                    self.state[6 + 11*ind*index+1 + 5] = card.points
                    self.state[6 + 11*ind*index+1 + 6 + GEMS_DICT[card.colour]] = 1

# nobles has a fixed score of 3, so we log the noble code instead. Num agents =2, total board nobles = 3. Next starting index = 156.
        for index, noble in enumerate(game_state.board.nobles):
            for colour, cost in noble[1].items():
                self.state[138 + 6*index + GEMS_DICT[colour]] = cost
            self.state[138 + 6*index + 5] = NOBLE_DICT[noble[0]]

#  Num agents=2. Enumerate over all reserved and purchased cards. Log their colour, costs and points
        for index, agent in enumerate(game_state.agents):
            self.state[156 + 388*index] = agent.score
            for gem, count in agent.gems.items():
                self.state[156 + 388*index + 1 + GEMS_DICT[colour]] = count
            for ind, (card_colour, cards) in enumerate(agent.cards.items()):
                for card in cards:
                    for colour, cost in card.cost.items():
                        self.state[163 + 388*index + 7 + 11*ind + GEMS_DICT[colour]] = cost
                    self.state[156 + 388*index + 7 + 11*ind + 5 + GEMS_DICT[card_colour]] = 1
                    self.state[156 + 388*index + 7 + 11*ind + 10] = card.points
            for ind, noble in enumerate(agent.nobles):
                for colour, cost in noble[1].items():
                    self.state[156 + 388*index + 370 + 6*ind + GEMS_DICT[colour]] = cost
                self.state[156 + 388*index + 370 + 6*ind + 5] = NOBLE_DICT[noble[0]]

        return self.state

    def vec2action(self, act, game_state, agent):
        # label = 0
        # action = {}
        # cnt = 0
        # for i in act:
        #     if (i == 1):
        #         label = cnt
        #         label = int(label)
        #         break
        #     cnt += 1
        label = act
            
        potential_nobles = []        
        for noble in game_state.board.nobles:
            if self.noble_visit(agent, noble):
                potential_nobles.append(noble)
                    
# theoretical maximum combinations when collecting 3 different gems: 25 collect combinations (1, 2, and 3 gems), 
# 56 valid return combos for each 1 collect gem, 35 valid return combos for each 2 collect gem combos, 
# and 20 valid return combos for each 3 collect gem combos. 11 Possible noble combinations including None.

        if label < 3080:
            action = None
            collect_label = math.floor(label/(56*11))
            return_label = math.floor((label - (collect_label*(56*11)))/11)
            noble_label = math.floor((label - (collect_label*(56*11))) - return_label*11)
            
            # if COLORS[collect_label] != 'yellow' and game_state.board.gems[COLORS[collect_label]] > 0:
                
            collect_combos = self.generate_collect_combos()
            collected_gems = collect_combos[collect_label]
            return_combos = self.generate_return_combos(collected_gems, num_return=3)
            returned_gems = return_combos[return_label]
            
            # if len(potential_nobles) > noble_label:
            #     noble = NOBLES[noble_label]
            # else:
            #     noble = None
                
            noble = NOBLES[noble_label]
            
            action = {'type': 'collect_diff', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}   
        
        elif label < 6930:
            action = None
            collect_label = math.floor((label-3080)/(35*11))
            return_label = math.floor((label-3080 - (collect_label*(35*11)))/11)
            noble_label = math.floor((label-3080 - (collect_label*(35*11))) - return_label*11)
            
            # if COLORS[collect_label] != 'yellow' and game_state.board.gems[COLORS[collect_label]] > 0:
                
            collect_combos = self.generate_collect_combos()
            # binary = index2bi[collect_label]
            # collected_gems = {COLORS[i]:1 for i in range(5) if (binary[i]=='1')}
            collected_gems = collect_combos[collect_label]
            return_combos = self.generate_return_combos(collected_gems, num_return=3)
            returned_gems = return_combos[return_label]
                    
            # if len(potential_nobles) > noble_label:
            #     noble = NOBLES[noble_label]
            # else:
            #     noble = None
                
            noble = NOBLES[noble_label]
            
            action = {'type': 'collect_diff', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}   
        
        elif label < 9130:
            action = None
            collect_label = math.floor((label-6930)/(20*11))
            return_label = math.floor((label-6930-(collect_label*(20*11)))/11)
            noble_label = math.floor((label-6930-(collect_label*(20*11))) - return_label*11)
            
            # if COLORS[collect_label] != 'yellow' and game_state.board.gems[COLORS[collect_label]] > 0:
                
            collect_combos = self.generate_collect_combos()
            # binary = index2bi[collect_label]
            # collected_gems = {COLORS[i]:1 for i in range(5) if (binary[i]=='1')}
            collected_gems = collect_combos[collect_label]
            return_combos = self.generate_return_combos(collected_gems, num_return=3)
            returned_gems = return_combos[return_label]
                    
            # if len(potential_nobles) > noble_label:
            #     noble = NOBLES[noble_label]
            # else:
            #     noble = None
                
            noble = NOBLES[noble_label]
            
            action = {'type': 'collect_diff', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}
    
# 21 return actions * 5 combinations * 11 potential nobles = 1155.
        elif (label < 10285):
            action = None
            collect_label = math.floor((label-9130)/(21*11))
            return_label = math.floor((label-9130 - (collect_label*(21*11)))/11)
            noble_label = math.floor((label-9130 - (collect_label*(21*11)))- return_label*11)
            
            # if COLORS[collect_label] != 'yellow' and game_state.board.gems[COLORS[collect_label]] >= 4:
                
            collected_gems = {COLORS[collect_label]:2}
            return_combos = self.generate_return_combos(collected_gems, num_return=2)
            returned_gems = return_combos[return_label]
                    
            # if len(potential_nobles) > noble_label:
            #     noble = NOBLES[noble_label]
            # else:
            #     noble = None
                
            noble = NOBLES[noble_label]
            
            action = {'type': 'collect_same', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}

# 12 reservation available, if exceed agent gem limit during reservation, 5 return actions available, leading to a theoretical max of 60 actions. 10 + 1 None
# nobles. Maximum 11 potential nobles exist, hence total action space = 660.
        elif (label < 10945):
            action = None
            reserve_label = math.floor((label-10285)/(5*11))
            return_label = math.floor((label-10285 -(reserve_label*(5*11)))/11)
            noble_label = math.floor((label - 10285 - (reserve_label*(5*11))- return_label*11))
            
            # if len(agent.cards) < 3:
                
            collected_gems = {'yellow':1} if game_state.board.gems['yellow']>0 else {}
            return_combos = self.generate_return_combos(collected_gems, num_return=1)
            returned_gems = return_combos[return_label]
            
            # if len(potential_nobles) > noble_label:
            #     noble = NOBLES[noble_label]
            # else:
            #     noble = None
                
            noble = NOBLES[noble_label]

            if (len(game_state.board.dealt_list()) > reserve_label):
                card = game_state.board.dealt_list()[reserve_label]
                action = {'type': 'reserve',
                            'card': card,
                            'collected_gems': collected_gems,
                            'returned_gems': returned_gems,
                            'noble': noble}
                    
# 12 buy + 3 buy reserved, 11 nobles = 165 combinations
        elif (label < 11110):
            action = None
            buy_label = math.floor((label-10945)/11)
            noble_label = math.floor((label-10945)- buy_label*11)
            cards = game_state.board.dealt_list() + agent.cards['yellow']
            if (len(cards) > buy_label):
                card = cards[buy_label]
                
                # if card or len(agent.cards[card.colour]) < 7:
                returned_gems = self.game_rule.resources_sufficient(agent, card.cost) #Check if this card is affordable.
                if type(returned_gems)==dict:
                    new_nobles = []
                    for noble in game_state.board.nobles:
                        agent_post_action = copy.deepcopy(agent)
                        #Give the card featured in this action to a copy of the agent.
                        agent_post_action.cards[card.colour].append(card)
                        #Use this copied agent to check whether this noble can visit.
                        if self.noble_visit(agent_post_action, noble):
                            new_nobles.append(noble) #If so, add noble to the new list.
                            
                    # if len(new_nobles) > noble_label:
                    #     noble = NOBLES[noble_label]
                    # else:
                    #     noble = None
                        
                    noble = NOBLES[noble_label]
            
                    action={'type': 'buy_reserve' if card in agent.cards['yellow'] else 'buy_available',
                                    'card': card,
                                    'returned_gems': returned_gems,
                                    'noble': noble}
        elif (label < 11120):
            noble_label = label - 11110
            if len(potential_nobles) > noble_label:
                noble = NOBLES[noble_label]
            else:
                noble = None
            action = {'type': 'pass', 'noble': noble}
                        
        return action
    
    def noble_visit(self, agent, noble):
        _,costs = noble
        for colour,cost in costs.items():
            if not len(agent.cards[colour]) >= cost:
                return False
        return True
    
    def generate_collect_combos(self):
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
                
    
    def generate_return_combos(self, collected_gems, num_return):
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
    
    def generateSuccessor(self, state, action, agent_id):        
        agent,board = state.agents[agent_id],state.board
        agent.last_action = action #Record last action such that other agents can make use of this information.
        score = 0
        reward = 0.0
        
        if 'card' in action:
            card = action['card']
        
        if 'collect' in action['type'] or action['type']=='reserve':
            #Decrement board gem stacks by collected_gems. Increment player gem stacks by collected_gems.
            for colour,count in action['collected_gems'].items():
                if colour == 'yellow':
                    reward += 0.2
                else:
                    reward += 0.1
                    
                board.gems[colour] -= count
                agent.gems[colour] += count

                    
            #Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
            for colour,count in action['returned_gems'].items():
                agent.gems[colour] -= count
                board.gems[colour] += count 
                # reward += 0.1
                
            if action['type'] == 'reserve':
                #Remove card from dealt cards by locating via unique code (cards aren't otherwise hashable).
                #Since we want to retain the positioning of dealt cards, set removed card slot to new dealt card.
                #Since the board may have None cards (empty slots that cannot be filled), check cards first.
                #Add card to player's yellow stack.
                for i in range(len(board.dealt[card.deck_id])):
                    if board.dealt[card.deck_id][i] and board.dealt[card.deck_id][i].code == card.code:
                        board.dealt[card.deck_id][i] = board.deal(card.deck_id)
                        agent.cards['yellow'].append(card)
                        
                        gem_cost = 0
                        for colour, cost in card.cost.items():
                            gem_cost += cost
                        reward += (card.points/2*gem_cost)
                        break
                    
        elif 'buy' in action['type']:
            #Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
            for colour,count in action['returned_gems'].items():
                agent.gems[colour] -= count
                board.gems[colour] += count
            #If buying one of the available cards on the board, set removed card slot to new dealt card.
            #Since the board may have None cards (empty slots that cannot be filled), check cards first.
            if 'available' in action['type']:
                for i in range(len(board.dealt[card.deck_id])):
                    if board.dealt[card.deck_id][i] and board.dealt[card.deck_id][i].code == card.code:                
                        board.dealt[card.deck_id][i] = board.deal(card.deck_id)
                        reward += 0.3
                        
                        gem_cost = 0
                        for colour, cost in card.cost.items():
                            gem_cost += cost
                        reward += (card.points/gem_cost)
                        break
            #Else, agent is buying a reserved card. Remove card from player's yellow stack.
            else:
                for i in range(len(agent.cards['yellow'])):
                    if agent.cards['yellow'][i].code == card.code:
                        del agent.cards['yellow'][i]
                        reward += 0.3
                        gem_cost = 0
                        for colour, cost in card.cost.items():
                            gem_cost += cost
                        reward += (card.points/gem_cost)
                        break                
            
            #Add card to player's stack of matching colour, and increment agent's score accordingly.
            agent.cards[card.colour].append(card)
            score += card.points
            
        if action['noble']:
            #Remove noble from board. Add noble to player's stack. Like cards, nobles aren't hashable due to possessing
            #dictionaries (i.e. resource costs). Therefore, locate and delete the noble via unique code.
            #Add noble's points to agent score.
            for i in range(len(board.nobles)):
                if board.nobles[i][0] == action['noble'][0]:
                    del board.nobles[i]
                    agent.nobles.append(action['noble'])
                    score += 3
                    
                    reward += (3/10)
                    break
                
        #Log this turn's action and any resultant score. Return updated gamestate.
        agent.agent_trace.action_reward.append((action,score))
        agent.score += score
        agent.passed = action['type']=='pass'
            
        # reward = 1/heuristic_value
        return state, reward
    
    def generate_reward(self, state, action, agent_id):
        reward = 0
        agent_state = getAgentState(state,state.agents[agent_id])        
        
        ####### noble state ######
        noble_states = []
        for noble in state.board.nobles:
            # All gems have a bonus score of 3, 
            # and each row is a dict indicating the cost of the noble, and the bonus score
            noble_state = deepcopy(GEMS)
            noble_state['score'] = 3
            for key in GEMS.keys():
                if key in noble[1].keys():
                    noble_state[key]=noble[1][key]
            noble_states.append(noble_state)
        # What needs to be returned is noblestate, 3 noobles
        next_state = deepcopy(agent_state)
        
        action_cost, card_change = DoAction(agent_state,action)

        # Initialize next_state
        next_state = deepcopy(agent_state)
        # There are seven keys: black, red, yellow, green, blue, white, score
        # If there is a value of value is not 0 means that the cost is spent, 
        # then the next action to update
        if len([v for k, v in action_cost.items() if v!=0]) != 0:
            # The action costs nothing and returns the original
            next_state['score'] = agent_state['score'] + action_cost['score']
            for key in GEMS.keys():
                # The existing cost of the player
                next_state['total_value'][key] = agent_state['total_value'][key] + action_cost[key]
        # cards updated for the next state
        for key in GEMS.keys():
            next_state['card_num'][key] = agent_state['card_num'][key] + card_change[key]

        # Make the score as big as possible, action, by buying card and noble
        # Calculate the gap to the end score
        score_loss = 15 - next_state['score']
        if score_loss > 0 :
            # get noble
            noble_loss = 0
            noble_dist = 0
            # the totol number of color gems
            color_values = 0
            for noble in noble_states:
                for key in GEMS.keys():
                    if key != 'yellow':
                    # Get the Noble Gap 
                        noble_dist += np.abs(noble[key]-next_state['card_num'][key])
                        color_values += next_state['card_num'][key]
                        noble_loss += (noble_dist*0.5) * (color_values**0.05)
            # print('noble_loss',noble_loss)
            # Calculate the cost of the action, and if there is a negative value, 
            # indicate that the gem is to be returned
            cost_gem_loss = 0
            return_gem_loss = 0
            count = 0
            #too many values with the same color
            much_same_loss=0
            for key in GEMS.keys():
                much_same_loss += (6-next_state['total_value'][key])/(color_values+1) + next_state['card_num']['yellow'] ** 0.5
                if action_cost[key] > 0:
                    cost_gem_loss += action_cost[key]
                elif action_cost[key] == 0:
                    return_gem_loss += 1.5 * action_cost[key] 
                else:    
                    count +=1
                    return_gem_loss += 1.5 * action_cost[key] 
            if count == 1:
                # get more cost equal to 1, which means reserve card
                return_gem_loss *= 1.6
            if count > 1:
                # get more cost larger than 1, which means collect gem
                return_gem_loss *= 0.8
            cost_loss =  (cost_gem_loss+return_gem_loss) / (color_values+1)

            much_tire1_loss=0
            if action['type'] == 'reserve' and action['card'].points == 0: 
                much_tire1_loss += sum(next_state['card_num'].values())
            if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                if agent_state['score']>10 and action['card'].points == 0:
                    much_tire1_loss += sum(next_state['card_num'].values())*(color_values+1) 
            global NonePointCard
            
            card_loss=0                
            if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                if NonePointCard > 3 and action['card'].points == 0:
                    card_loss += sum(next_state['card_num'].values())**0.5*(color_values**0.05)
                    #+ much_tire1_loss*0.00025  + card_loss*0.01
                    
            heuristic_value = 2 * score_loss + 0.001 * noble_loss + 0.1 * cost_loss + 0.2 * much_same_loss 
            reward = reward + 1/heuristic_value
        else: 
            reward += 0.3
        return reward
                 
    def step(self, act_vec, verify=False):          
        game_state = self.game_rule.current_game_state
        agent_id = self.game_rule.getCurrentAgentIndex()
        agent = game_state.agents[agent_id]
        game_state.agent_to_move = agent_id
        valid = False
        reward = 0.0
        pre_score = agent.score

        if type(act_vec) == dict:
            action = act_vec
        # else:
        #     # convert to Splendor Action
        #     action = self.vec2action(act_vec, game_state, agent)
            
        # if act_vec.all() == None:
        #     return self.state, reward, self.done, valid
        
        # print(action)
        legal_actions = self.game_rule.getLegalActions(game_state, agent_id)
        
        if action in legal_actions:
            valid = True
            # print("AGENT ID: {}, score: {}".format(agent_id, agent.score))
            # print(act)
            if verify:
                return self.state, reward, self.done, valid, {}
            else:
                print(action)
                dummystate = deepcopy(game_state)
                # _, reward = self.generateSuccessor(dummystate, action, agent_id)
                self.game_rule.update(action)
                next_state = self.game_rule.current_game_state
                self.state = self.game2vec(next_state)
                reward = self.generate_reward(dummystate, action, agent_id)
                # reward += (game_state.agents[agent_id].score - pre_score)*0.1
                #  - (game_state.agents[self.id + 1 % 2].score - preOppScore)
                # if game_state.agents[agent_id].score >= 15:
                #     reward += 1
                self.gameEnds(next_state)
                # if not self.done:
                #     reward -= 0.1
                
                
                # action_count = self.game_rule.action_counter
                # print("Action Counter : ",action_count)
                # print("Action : ", action)
                # print(fr"Agent {agent_id} Score: {agent.score}")
                
                # if self.done:
                    
                #     action_count = self.game_rule.action_counter
                #     print("Game Done, Action Counter =", action_count)
                #     # # Exponential decay, a*e^-bx, x = action count
                #     # reward += 100*math.exp(-0.02*action_count)
                
                #     self.game_rule.action_counter = 0
                # print(reward)
                
                print(agent.score)
                return self.state, reward, self.done, valid, {}

        else:
            return self.state, reward, self.done, valid, {}
        
    def reset(self):
        self.game_rule = SplendorGameRule(NUM_AGENTS)
        game_state = self.game_rule.initialGameState()
        self.state = self.game2vec(game_state)
        # legal_actions = self.game_rule.getLegalActions(game_state, self.game_rule.getCurrentAgentIndex())
        # print("Resetting...")
        # print("Agent Score after reset", game_state.agents[self.game_rule.getCurrentAgentIndex()].score)
        return self.state
    
    def gameEnds(self, gamestate):
        deadlock = 0
        score = 0
        for agent in gamestate.agents:
            deadlock += 1 if agent.passed else 0
            if agent.score >= 15: # and self.game_rule.current_agent_index == 0:
                score = agent.score
                self.done = True
        self.done = score >= 15
        return deadlock==len(gamestate.agents)
        
    def get_legalActions(self):
        legal_actions = self.game_rule.getLegalActions(self.game_rule.current_game_state, self.game_rule.getCurrentAgentIndex())
        return legal_actions