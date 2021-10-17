# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    04/01/2021
# Purpose: Implements a Game class to run implemented games for this framework.

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import random, copy, time, itertools
from   template     import GameState, Agent
from   func_timeout import func_timeout, FunctionTimedOut
import traceback
import os
import sys
cd = os.path.dirname(os.path.abspath('Splendor'))
sys.path.append(cd)
from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import *
import numpy as np
import math
import gym
from gym import Env, spaces
from gym.utils import seeding
# CONSTANTS ----------------------------------------------------------------------------------------------------------#

FREEDOM = True #Whether or not to penalise agents for incorrect moves and timeouts. Useful for debugging.
WARMUP  = 15    #Warmup period (time given to each agent on their first turn).
NUM_AGENTS    = 2
GEMS_DICT = {'black':0, 'red':1, 'yellow':2, 'green':3, 'blue':4, 'white':5}
COLORS = ['black', 'red', 'green', 'blue', 'white']
NOBLE_DICT = {'4g4r':1, '3w3r3B':2, '3b3g3r':3, '3w3b3g':4, '4w4b':5, '4w4B':6, '3w3b3B':7, '4r4B':8, '4b4g':9, '3g3r3B':10}
COLOURS = {'B':'black', 'r':'red', 'y':'yellow', 'g':'green', 'b':'blue', 'w':'white'}
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
          ('3g3r3B', {'green': 3, 'red': 3, 'black': 3})
          ]  

class SplendorEnv(Env):

    def __init__(self):
        self.game_rule = SplendorGameRule(NUM_AGENTS)
        self.state = np.zeros(979)
        self.next_state = copy.deepcopy(self.state)
        self.observation_space = spaces.Discrete(979)
        self.action_space = spaces.Discrete(3800)
        
        self.gamestate = self.game_rule.initialGameState()
        self.gamestate.agent_to_move = 0      

        self.done = False
        #self.valid = True
        # self.act = np.zeros(3800)
        self.agent_id = self.game_rule.getCurrentAgentIndex()
        self.agent = self.gamestate.agents[self.agent_id]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def game2vec(self, agent_id):
        self.state = np.zeros(979)

        for index, (gem, count) in enumerate(self.gamestate.board.gems.items()):
            self.state[0 + GEMS_DICT[gem]] = count

# for each level in dealt cards, discretize. Each level consists of 4 cards.
        for index, level in enumerate(self.gamestate.board.dealt):
            for ind, card in enumerate(level):
                if card != None:
                    for colour, cost in card.cost.items():
                        self.state[6 + 11*ind*index+1 + GEMS_DICT[colour]] = cost
                    self.state[6 + 11*ind*index+1 + 5] = card.points
                    self.state[6 + 11*ind*index+1 + 6 + GEMS_DICT[card.colour]] = 1

# nobles has a fixed score of 3, so we log the noble code instead. Num agents =2, total board nobles = 3. Next starting index = 156.
        for index, noble in enumerate(self.gamestate.board.nobles):
            for colour, cost in noble[1].items():
                self.state[138 + 6*index + GEMS_DICT[colour]] = cost
            self.state[138 + 6*index + 5] = NOBLE_DICT[noble[0]]

#  Num agents=2. Enumerate over all reserved and purchased cards. Log their colour, costs and points
        for index, agent in enumerate(self.gamestate.agents):
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

    def vec2action(self, label):
# theoretical maximum combinations when collecting 3 different gems: 20 valid return combos for each 3 gem combo. 11 Possible noble combinations
        if (label < 2200):
            collect_label = math.floor(label/(20*11))
            return_label = math.floor((label - (collect_label*(20*11)))/11)
            noble_label = math.floor((label - (collect_label*(20*11))) - return_label*11)
            binary = index2bi[collect_label]
            collected_gems = {COLORS[i]:1 for i in range(5) if (binary[i]=='1')}
            temp_return_combos = self.generate_return_combos(collected_gems, num_return=3)
            return_combos = []
            for d in temp_return_combos:
                return_combos.append(dict(sorted(d.items(), key=lambda x:x[0])))
            return_combos = sorted(return_combos, key=lambda x: list(x.keys()))
            returned_gems = return_combos[return_label]
            
            noble = NOBLES[noble_label]
                
            action = {'type': 'collect_diff', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}
    
# 21 return actions * 5 combinations * 11 potential nobles = 1155.
        elif (label < 3355):
            collect_label = math.floor((label-2200)/(21*11))
            return_label = math.floor((label-2200 - (collect_label*(21*11)))/11)
            noble_label = math.floor((label-2200 - (collect_label*(21*11)))- return_label*11)
            collected_gems = {COLORS[collect_label]:2}
            temp_return_combos = self.generate_return_combos(collected_gems, num_return=2)
            return_combos = []
            for d in temp_return_combos:
                return_combos.append(dict(sorted(d.items(), key=lambda x:x[0])))
            return_combos = sorted(return_combos, key=lambda x: list(x.keys()))
            returned_gems = return_combos[return_label]
            
            noble = NOBLES[noble_label]
                
            action = {'type': 'collect_same', 'collected_gems': collected_gems, 'returned_gems': returned_gems, 'noble': noble}

# 12 reservation available, if exceed agent gem limit during reservation, 5 return actions available, leading to a theoretical max of 60 actions. 10 + 1 None
# nobles. Maximum 11 potential nobles exist, hence total action space = 660.
        elif (label < 4015):
            reserve_label = math.floor((label-3355)/(5*11))
            return_label = math.floor((label-3355 -(reserve_label*(5*11)))/11)
            noble_label = math.floor((label - 3355 - (reserve_label*(5*11))- return_label*11))
            collected_gems = {'yellow':1} if self.gamestate.board.gems['yellow']>0 else {}
            temp_return_combos = self.generate_return_combos(collected_gems, num_return=1)
            return_combos = []
            for d in temp_return_combos:
                return_combos.append(dict(sorted(d.items(), key=lambda x:x[0])))
            return_combos = sorted(return_combos, key=lambda x: list(x.keys()))
            returned_gems = return_combos[return_label]
            
            noble = NOBLES[noble_label]
                
            action = None
            if (len(self.gamestate.board.dealt_list()) > reserve_label):
                card = self.gamestate.board.dealt_list()[reserve_label]
                action = {'type': 'reserve',
                            'card': card,
                            'collected_gems': collected_gems,
                            'returned_gems': returned_gems,
                            'noble': noble}

        elif (label < 4180):
            action = None
            buy_label = math.floor((label-4015)/11)
            noble_label = math.floor((label-4015)- buy_label*11)
            cards = self.gamestate.board.dealt_list() + self.gamestate.agents[self.agent_id].cards['yellow']
            if (len(cards) > buy_label):
                card = cards[buy_label]
                
            if card or len(self.gamestate.agents[self.agent_id].cards[card.colour]) < 7:
                returned_gems = self.game_rule.resources_sufficient(self.gamestate.agents[self.agent_id], card.cost) #Check if this card is affordable.
                if type(returned_gems)==dict:
                    new_nobles = []
                    for noble in self.gamestate.board.nobles:
                        agent_post_action = copy.deepcopy(self.gamestate.agents[self.agent_id])
                        #Give the card featured in this action to a copy of the agent.
                        agent_post_action.cards[card.colour].append(card)
                        #Use this copied agent to check whether this noble can visit.
                        if self.game_rule.noble_visit(agent_post_action, noble):
                            new_nobles.append(noble) #If so, add noble to the new list.
                            
                    if len(new_nobles) > noble_label:
                        noble = NOBLES[noble_label]
                    else:
                        noble = None
                
                        action=({'type': 'buy_reserve' if card in self.gamestate.agents[self.agent_id].cards['yellow'] else 'buy_available',
                                        'card': card,
                                        'returned_gems': returned_gems,
                                        'noble': noble})
        return action
    
    def generate_return_combos(self, collected_gems, num_return):
        return_combos = []
        total_gems_list = [i for i in COLOURS.values() if i not in collected_gems.keys()]
            
        for num in range(0,num_return+1):
            for combo in set(itertools.combinations_with_replacement(total_gems_list, num)):
                #Filter out colours with zero gems, and append.
                returned_gems = {c:0 for c in COLOURS.values()}
                for colour in combo:
                    returned_gems[colour] += 1
                return_combos.append(dict({i for i in returned_gems.items() if i[-1]>0}))     
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
                board.gems[colour] -= count
                agent.gems[colour] += count
                if colour == 'yellow':
                    reward += 0.2
                    
                else:
                    reward += 0.1
                    
            #Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
            for colour,count in action['returned_gems'].items():
                agent.gems[colour] -= count
                board.gems[colour] += count 
                reward += 0.1
                
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
                        reward += (card.points/gem_cost)
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
                    
                    reward += 0.3
                    break
                
        #Log this turn's action and any resultant score. Return updated gamestate.
        agent.agent_trace.action_reward.append((action,score))
        agent.score += score
        agent.passed = action['type']=='pass'
        return state, reward
                 
    def step(self, action):
        valid = False
        reward = 0.0
        
        # Action space validation
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), "Invalid Action"
        
        # convert to Splendor Action
        action = self.vec2action(action)
        
        # if act_vec.all() == None:
        #     return self.next_state, reward, self.done, valid
        
        legal_actions = self.game_rule.getLegalActions(self.gamestate, self.agent_id)
        
        if action in legal_actions:
            valid = True
            temp_state = self.gamestate
            self.gamestate, reward = self.generateSuccessor(temp_state, action, self.agent_id)
            if self.gamestate.agents[self.agent_id].score >= 15:
                reward += 10
            self.agent_id = self.game_rule.getNextAgentIndex()
            self.game_rule.update(action)
            
        # else:
        #     while valid == False:
        #         act = random.randint(0, self.action_space.n - 1)
        #         action = np.zeros(self.action_space.n)
        #         action[act] = 1
        #         if action in legal_actions:
        #             valid = True
        #             temp_state = self.gamestate
        #             self.gamestate, reward = self.generateSuccessor(temp_state, action, self.agent_id)
        #             if self.gamestate.agents[self.agent_id].score >= 15:
        #                 reward += 10
        #             self.agent_id = self.game_rule.getNextAgentIndex()
        #             self.game_rule.update(action)

        self.state = self.game2vec(self.gamestate)
        self.done = self.game_rule.gameEnds()

        print(reward)
        return self.state, reward, self.done, {}
    
    def reset(self):
        self.gamestate = self.game_rule.initialGameState()
        self.state = self.game2vec(self.gamestate)
        return self.state
    
    def render(self, mode='human'):
        pass