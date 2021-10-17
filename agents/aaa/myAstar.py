# INFORMATION ------------------------------------------------------------------------------------------------------- #

# group:  aaa
# Date:    04/10/2021
# Purpose: A star search agent.

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#
#python splendor_runner.py -r agents.ai_group_name.submited_1003 -c "agents.ai_group_name.14-15" -q -p -m 100
#python splendor_runner.py -r agents.ai_group_name.14-15 -c "agents.ai_group_name.submited_1003" -q -p -m 100
#python splendor_runner.py -r agents.ai_group_name.Q_learning_weighted -c "agents.ai_group_name.14-15" -q -p -m 100
import heapq
from template import Agent
import time
from copy import deepcopy
from collections import deque
from Splendor.splendor_model import SplendorGameRule
import numpy as np
from Splendor.splendor_utils import *
import random
THINKTIME = 0.95
currentGameRule = SplendorGameRule(2)

GEMS = {c:0 for c in COLOURS.values()}
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def GetActions(state,agent_id):
    actions = currentGameRule.getLegalActions(state,agent_id)
    myAgent = state.agents[agent_id]
    myGamNumber = sum(myAgent.gems.values())
    for action in actions:
        if myGamNumber >= 8:
            # remove the one larger than 3
            if action["type"] == 'collect_diff':
                if sum(action['collected_gems'].values()) > 10 - myGamNumber:
                    actions.remove(action)
    return actions

def DoAction(agent_state,action):
    ''' get reward
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
        # 如果分数大于7的话，就不进行选择为0分的卡
        if agent_state['score'] >= 12 and action['card'].points == 0:
            action_cost['score']+=action['card'].points
            for key in GEMS.keys():
                # if key in action['returned_gems'].keys():
                action_cost[key] -= 10
            
        else:
            action_cost['score']+=action['card'].points
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

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id # Agent remember its own id.

    def SelectAction(self, actions, gamestate):
        random.seed(1)
        start_time = time.time()
        priorqueue = PriorityQueue()  # Initialise queue. First node = root state and an empty path.
        agent = gamestate.agents[self.id]
        agent_state = getAgentState(gamestate,agent)
   
        ####### noble state ######
        noble_states = []
        for noble in gamestate.board.nobles:
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

        # before actions, select action in the begin
        if next_state['score']==0 and sum(next_state['card_num'].values())<2:
            actions = GetActions(gamestate, self.id) 
        
        for action in actions:
            if time.time() - start_time < THINKTIME:
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
                if score_loss >= 0 :
                
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
                    print('noble_loss',noble_loss)
                    # Calculate the cost of the action, and if there is a negative value, 
                    # indicate that the gem is to be returned
                    cost_gem_loss = 0
                    return_gem_loss = 0
                    count = 0
                    #too many values with the same color
                    much_same_loss=0
                    for key in GEMS.keys():
                        much_same_loss += (6-next_state['total_value'][key])/(color_values+1) + next_state['card_num']['yellow'] * 0.5
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
                    cost_loss = - (cost_gem_loss+return_gem_loss) / (color_values+1)

                    much_tire1_loss=0
                    if action['type'] == 'reserve' and action['card'].points == 0: 
                        much_tire1_loss += sum(agent_state['card_num'].values())
                    if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                        if agent_state['score']>12 and action['card'].points == 0:
                            much_tire1_loss += sum(agent_state['card_num'].values())*(color_values+1)
                                       
                    card_loss=0                
                    if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                        if sum(agent_state['card_num'].values()) <= 3 and action['card'].points > 3:
                            card_loss += sum(agent_state['card_num'].values())*(color_values+1)
                    heuristic_value = 1.5 * score_loss + 0.0001 * noble_loss + 0.4 * cost_loss + 0.4 * much_same_loss + much_tire1_loss*0.00025  + card_loss*0.01
                else: 
                    heuristic_value = 1.5
                # print(heuristic_value,score_loss,noble_loss,cost_loss,much_same_loss,heuristic_value,much_tire1_loss,much_tire1_loss*0.0001)
                priorqueue.push(action, heuristic_value)
            else:
                break
        return priorqueue.pop()
