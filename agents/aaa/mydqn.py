from template import Agent
import random
import sys
import os
cd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cd)
from ENV2 import *
from dqn_splendor2 import *
from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import *
import numpy as np

# Tensorflow
import tensorflow as tf 
from tensorflow.python.autograph.core.unsupported_features_checker import verify
from tensorflow.python.ops.gen_math_ops import xdivy
from tensorflow.keras.initializers import VarianceScaling, Constant
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

# Global Variables
STATE_DIM = 980
ACTION_DIM = 11120
BATCH_SIZE = 256
LOAD_FROM = cd+'/agents/aaa/Splendor-saves'

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
          ('3g3r3B', {'green': 3, 'red': 3, 'black': 3}),
          None
          ]



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.agent_id = _id # Agent remember its own id.
        # Create environment
        self.env = SplendorEnv()
        # Build main and target networks
        self.agent = tf.keras.models.load_model(LOAD_FROM + '/dqn.h5', custom_objects={'tf': tf})
        # self.TARGET_DQN = build_q_network(ACTION_DIM, STATE_DIM)
        
        # self.agent = DQNAgent(self.env, self.MAIN_DQN, self.TARGET_DQN, action_dim=ACTION_DIM, state_dim=STATE_DIM)
        # self.agent.load(LOAD_FROM, load_replay_buffer=False, load_meta=False)
        
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
    
        #Checks whether a particular noble is a candidate for visiting this agent.
    def noble_visit(self, agent, noble):
        _,costs = noble
        for colour,cost in costs.items():
            if not len(agent.cards[colour]) >= cost:
                return False
        return True
    
    def resources_sufficient(self, agent, costs):
        wild = agent.gems['yellow']
        return_combo = {c:0 for c in COLOURS.values()}
        for colour,cost in costs.items():
            #If a shortfall is found, see if the difference can be made with wild/seal/yellow gems.
            available = agent.gems[colour] + len(agent.cards[colour])
            shortfall = max(cost - available, 0) #Shortfall shouldn't be negative.
            wild -= shortfall
            #If wilds are expended, the agent cannot make the purchase.
            if wild < 0:
                return False
            #Else, increment return_combo accordingly. Note that the agent should never return gems if it can afford 
            #to pay using its card stacks, and should never return wilds if it can return coloured gems instead. 
            #Although there may be strategic instances where holding on to coloured gems is beneficial (by virtue of 
            #shorting players from resources), in this implementation, this edge case is not worth added complexity.
            gem_cost                = max(cost - len(agent.cards[colour]), 0) #Gems owed.
            gem_shortfall           = max(gem_cost - agent.gems[colour], 0)   #Wilds required.
            return_combo[colour]    = gem_cost - gem_shortfall                #Coloured gems to be returned.
            return_combo['yellow'] += gem_shortfall                           #Wilds to be returned.
            
        #Filter out unnecessary colours and return dict specifying combination of gems.
        return dict({i for i in return_combo.items() if i[-1]>0})


    def SelectAction(self,actions,game_state):

        vec2action = np.load(cd+'/agents/aaa/Splendor-saves/vec2action.npy', allow_pickle=True).item()
        state = self.game2vec(game_state)
        # print(state)  
        state = np.reshape(state, [1,STATE_DIM])
        Q_values = self.agent.predict(state)
        act = np.argmax(Q_values) 
        # print(np.where(act==1))
        action = vec2action[act]
        # action = self.vec2action(act, game_state, game_state.agents[self.agent_id])
        if action in actions:
            return action
        else:
            Q_values = self.agent.predict(state)
            act = np.argmax(Q_values) 
            # print(type(vec2action))
            action = vec2action[act]
        print(np.sum(state))
        print(Q_values)
        
        if action == {}:
            return random.choice(actions)
        
        
        return action
