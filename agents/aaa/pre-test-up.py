# INFORMATION ------------------------------------------------------------------------------------------------------- #

# group:  aaa
# Date:    04/10/2021
# Purpose: A star search agent.

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#
# python splendor_runner.py -r agents.ai_group_name.submited_1003 -c "agents.ai_group_name.14-15" -q -p -m 100
# python splendor_runner.py -r agents.ai_group_name.14-15 -c "agents.ai_group_name.submited_1003" -q -p -m 100
# python splendor_runner.py -r agents.ai_group_name.Q_learning_weighted -c "agents.ai_group_name.14-15" -q -p -m 100
# python splendor_runner.py -r agents.ai_group_name.myastar -c "agents.ai_group_name.hw1" -q -m 50
# python splendor_runner.py -r "agents.aaa.pre-test-up" -c "agents.aaa.myTeam" -p -q

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
splendor_game_rule = SplendorGameRule(2)
GEMS = {c: 0 for c in COLOURS.values()}
# GEMS == {'black': 0, 'red': 0, 'yellow': 0, 'green': 0, 'blue': 0, 'white': 0}
zero_score_cards_qty = 0


# the quantity of cards without score owned by the agent


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
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


def GetActions(state, agent, actions):
    """ pass, not reasonable in the middle and late stage of the game,
    when my agent has 6 gems and the adversary has 10 gems, there are 4 gems left, and no combination of 3 gems left,
    however the filter removes combination of gems less than 3 based on the quality of my agent gems only
    In the early stages of the game
    input: the whole original legal actions
    output: maximize gems quantity in collect_diff actions
    optimization: filter combinations of gems less than 3 when it's available to collect 3 gems
    actions = splendor_game_rule.getLegalActions(state, agent.id)
    """
    my_gems_sum = sum(agent.gems.values())
    legal_actions = deepcopy(actions)
    board_gems_without_yellow = deepcopy(state.board.gems)
    del board_gems_without_yellow['yellow']
    board_gems_kinds_without_yellow = sum(board_gem_qty != 0 for board_gem_qty in board_gems_without_yellow.values())
    for action in actions:
        if board_gems_kinds_without_yellow >= 3 and my_gems_sum <= 7:
            # when there are more than three kinds of gems left and my gems is less than 7,
            # legal actions of collect_diff include collect one and collect two and three,
            # and the agent should collect three gems, rather than two or one.
            if action["type"] == 'collect_diff':
                if sum(action['collected_gems'].values()) < 3:
                    legal_actions.remove(action)
        if board_gems_kinds_without_yellow >= 2 and my_gems_sum == 8:
            # when there are more than two kinds of gems left and my gems is 8,
            # legal actions of collect_diff include collect one and collect two,
            # and the agent should collect two gems, rather than one.
            if action["type"] == 'collect_diff':
                if sum(action['collected_gems'].values()) == 1:
                    legal_actions.remove(action)

    # maximum quantity of cards the agent need to hold,
    # that is the maximum cards requirement of ruby gems is three according to current level three cards and nobles,
    # there is no need to buy more ruby cards if we have held three ruby cards if ruby only need three
    # 按需囤卡 max_qty_cards_hold = {'black': 3, 'red': 3, 'green': 3, 'blue': 3, 'white': 3} 表示 各色卡牌 卡牌的 最多购买量
    print("state.board.dealt[0]", state.board.dealt[0])
    print("state.board.dealt[1]", state.board.dealt[1])
    print("state.board.dealt[2]", state.board.dealt[2])
    max_qty_cards_hold = {'black': 3, 'red': 3, 'green': 3, 'blue': 3, 'white': 3}
    print("\n\n  \n       start init max_qty_cards_hold", max_qty_cards_hold)

    # update max_qty_cards_hold according to nobles requirement
    # state.board.nobles = [('3w3b3B', {'white': 3, 'blue': 3, 'black': 3}), ('4w4B', {'white': 4, 'black': 4}), ('3w3b3g', {'white': 3, 'blue': 3, 'green': 3})]

    # there are only two kinds of nobles, 
    # three kinds of three cards, eg. {'white': 3, 'blue': 3, 'black': 3} or two kinds of four cards, eg. {'white': 4, 'black': 4}
    for noble_card in state.board.nobles:
        if len(noble_card[1]) == 2:
            max_qty_cards_hold.update(noble_card[1])
            print("max_qty_cards_hold noble", max_qty_cards_hold)

    # traverse all the cards on the dealt, update max_qty_cards_hold
    for level_one_card in state.board.dealt[0]:
        # a level one card will update max_qty_cards_hold, if it costs 4 gems, eg. {'white': 4}
        if len(level_one_card.cost) == 1 and list(level_one_card.cost.values())[0] == 4:
            max_qty_cards_hold.update(level_one_card.cost)
            print("max_qty_cards_hold level 1", max_qty_cards_hold)

    for level_two_card in state.board.dealt[1]:
        # two kinds of card.cost, eg. {'green': 6} and {'green': 5} in level two, can update max_qty_cards_hold
        if len(level_two_card.cost) == 1:
            color_of_cost, gem_qty_cost = list(level_two_card.cost.items())[0]
            if gem_qty_cost > max_qty_cards_hold[color_of_cost]:
                max_qty_cards_hold.update(level_two_card.cost)
                print("max_qty_cards_hold level 2", max_qty_cards_hold)

        # one kind of card.cost, eg.  {'red': 4, 'black': 2, 'green': 1}, it may update max_qty_cards_hold
        elif len(level_two_card.cost) == 3 and sum(level_two_card.cost.values()) == 7:
            for color_of_cost, gem_qty_cost in level_two_card.cost.items():
                if gem_qty_cost == 4 and gem_qty_cost > max_qty_cards_hold[color_of_cost]:
                    max_qty_cards_hold.update({color_of_cost: gem_qty_cost})
                    print("max_qty_cards_hold level 2 ", max_qty_cards_hold)

        # there are one kind of card.cost, eg.  {'red': 5, 'green': 3}, it may update max_qty_cards_hold
        elif len(level_two_card.cost) == 2 and sum(level_two_card.cost.values()) == 8:
            for color_of_cost, gem_qty_cost in level_two_card.cost.items():
                if gem_qty_cost == 5:
                    max_qty_cards_hold.update({color_of_cost: gem_qty_cost})
                    print("max_qty_cards_hold level 2", max_qty_cards_hold)

    for level_three_card in state.board.dealt[2]:
        # one kind of card.cost, eg. {'green': 7} in level three, can update max_qty_cards_hold
        if len(level_three_card.cost) == 1:
            max_qty_cards_hold.update(level_three_card.cost)
            print("max_qty_cards_hold level 3", max_qty_cards_hold)

        # two kinds of card.cost eg. {'white': 7, 'blue': 3} and {'black': 3, 'red': 3, 'green': 6}, can update
        elif len(level_three_card.cost) <= 3:
            for color_of_cost, gem_qty_cost in level_three_card.cost.items():
                if gem_qty_cost >= 6 and gem_qty_cost > max_qty_cards_hold[color_of_cost]:
                    max_qty_cards_hold.update({color_of_cost: gem_qty_cost})
                    print("max_qty_cards_hold level 3", max_qty_cards_hold)

        # one kind of card.cost, eg. {'green': 5, 'white': 3, 'red': 3, 'blue': 3} can update max_qty_cards_hold
        else:
            for color_of_cost, gem_qty_cost in level_three_card.cost.items():
                if gem_qty_cost == 5 and gem_qty_cost > max_qty_cards_hold[color_of_cost]:
                    max_qty_cards_hold.update({color_of_cost: gem_qty_cost})
                    print("max_qty_cards_hold level 3", max_qty_cards_hold)


    return legal_actions


def DoAction(agent_state, action):
    """
    evaluate the quantity of gems, scores and card acquired by an action
    when action is collect_diff, collect_same, or reserve, quantity of gems changes
    when action is buy_available or bue_reserve, quantity of card changes, score may change
    when noble in action is not None, score change
    return: gems_score_acquired and card_acquired
    gems_score_acquired: score and quality of gems acquired by an action
    {'black': 0, 'red': 0, 'yellow': 0, 'green': 0, 'blue': 0, 'white': 0, 'score': 0}
    card_acquired: card of a specific color acquired by an action
    {'black': 0, 'red': 0, 'yellow': 0, 'green': 0, 'blue': 0, 'white': 0}
    """
    gems_score_acquired = deepcopy(GEMS)
    card_acquired = deepcopy(GEMS)
    gems_score_acquired['score'] = 0
    # action = {'type': 'collect_diff', 'collected_gems': {'black': 1}, 'returned_gems': {}, 'noble': None}
    if action['type'] == 'collect_diff' or action['type'] == 'collect_same':
        for color in GEMS.keys():
            if color in action['collected_gems'].keys():
                temp = action['collected_gems'][color]
            else:
                temp = 0
            if color in action['returned_gems'].keys():
                temp -= action['returned_gems'][color]
            gems_score_acquired[color] = temp

    elif action['type'] == 'reserve':
        # If reserve, get a gold gem
        gems_score_acquired['reserve'] = 1
        for color in GEMS.keys():
            gems_score_acquired[color] = 0

    elif action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
        gems_score_acquired['score'] += action['card'].points
        if action['card'].points == 0:
            global zero_score_cards_qty
            zero_score_cards_qty += 1
        for color in GEMS.keys():
            if color in action['returned_gems'].keys():
                gems_score_acquired[color] -= action['returned_gems'][color]
        card_acquired[action['card'].colour] = 1

    elif action['noble'] is not None:
        gems_score_acquired['score'] += 3
    # Returns my existing cards, and if the action is to buy cards, it also returns the added cards
    # A value when the cost of the action or the return value
    return gems_score_acquired, card_acquired


def getAgentState(game_state, agent):
    """
    return: agent_state, a dictionary with three field, including score, cards_quantity, and cards_gems_sum
    score: the score of the current agent
    cards_quantity: the quantity of cards owned by the current agent
    cards_gems_sum: the total quantity of cards and gems owned by the current agent
    """
    cards_quantity = {}
    cards_gems_sum = {}
    agent_state = {'score': agent.score, 'cards_quantity': 0, 'cards_gems_sum': 0}

    for color in GEMS.keys():
        # if there is a card of this color
        if color in agent.cards.keys():
            cards_quantity[color] = len(agent.cards[color])
            cards_gems_sum[color] = agent.gems[color] + len(agent.cards[color])
        else:
            # if there isn't a card of this color
            cards_quantity[color] = 0
            cards_gems_sum[color] = agent.gems[color]
    agent_state['cards_quantity'] = cards_quantity
    agent_state['cards_gems_sum'] = cards_gems_sum
    return agent_state


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id

    def SelectAction(self, actions, gamestate):
        # random.seed(1)
        start_time = time.time()
        priorqueue = PriorityQueue()  # Initialise queue. First node = root state and an empty path.
        my_agent = gamestate.agents[self.id]
        agent_state = getAgentState(gamestate, my_agent)

        ####### noble state ######
        noble_states = []

        for noble in gamestate.board.nobles:
            # All gems have a bonus score of 3,
            # and each row is a dict indicating the cost of the noble, and the bonus score
            noble_state = deepcopy(GEMS)
            noble_state['score'] = 3
            for key in GEMS.keys():
                if key in noble[1].keys():
                    noble_state[key] = noble[1][key]
            noble_states.append(noble_state)
        # What needs to be returned is noblestate, 3 noobles
        next_state = deepcopy(agent_state)

        # before actions, select action in the begin
        # if next_state['score']==0 and sum(next_state['cards_quantity'].values())<2:
        #     actions = GetActions(gamestate, self.id)
        optimal_actions = GetActions(gamestate, my_agent, actions)

        for action in optimal_actions:

            if time.time() - start_time < THINKTIME:
                action_cost, card_change = DoAction(agent_state, action)

                # Initialize next_state
                next_state = deepcopy(agent_state)
                # There are seven keys: black, red, yellow, green, blue, white, score
                # If there is a value of value is not 0 means that the cost is spent,
                # then the next action to update
                if len([v for k, v in action_cost.items() if v != 0]) != 0:
                    # The action costs nothing and returns the original
                    next_state['score'] = agent_state['score'] + action_cost['score']
                    for key in GEMS.keys():
                        # The existing cost of the player
                        next_state['cards_gems_sum'][key] = agent_state['cards_gems_sum'][key] + action_cost[key]
                # cards updated for the next state
                for key in GEMS.keys():
                    next_state['cards_quantity'][key] = agent_state['cards_quantity'][key] + card_change[key]

                # Make the score as big as possible, action, by buying card and noble
                # Calculate the gap to the end score
                score_loss = 15 - next_state['score']
                if score_loss > 0:
                    # get noble
                    noble_loss = 0
                    noble_dist = 0
                    # the totol number of color gems
                    color_values = 0
                    for noble in noble_states:
                        for key in GEMS.keys():
                            if key != 'yellow':
                                # Get the Noble Gap
                                noble_dist += np.abs(noble[key] - next_state['cards_quantity'][key])
                                color_values += next_state['cards_quantity'][key]
                                noble_loss += (noble_dist * 0.5) * (color_values ** 0.05)
                    # print('noble_loss',noble_loss)
                    # Calculate the cost of the action, and if there is a negative value,
                    # indicate that the gem is to be returned
                    cost_gem_loss = 0
                    return_gem_loss = 0
                    count = 0
                    # too many values with the same color
                    much_same_loss = 0
                    for key in GEMS.keys():
                        much_same_loss += (6 - next_state['cards_gems_sum'][key]) / (color_values + 1) + \
                                          next_state['cards_quantity']['yellow'] ** 0.5
                        if action_cost[key] > 0:
                            cost_gem_loss += action_cost[key]
                        elif action_cost[key] == 0:
                            return_gem_loss += 1.5 * action_cost[key]
                        else:
                            count += 1
                            return_gem_loss += 1.5 * action_cost[key]
                    if count == 1:
                        # get more cost equal to 1, which means reserve card
                        return_gem_loss *= 1.6
                    if count > 1:
                        # get more cost larger than 1, which means collect gem
                        return_gem_loss *= 0.8
                    cost_loss = (cost_gem_loss + return_gem_loss) / (color_values + 1)

                    much_tire1_loss = 0
                    if action['type'] == 'reserve' and action['card'].points == 0:
                        much_tire1_loss += sum(next_state['cards_quantity'].values())
                    if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                        if agent_state['score'] > 10 and action['card'].points == 0:
                            much_tire1_loss += sum(next_state['cards_quantity'].values()) * (color_values + 1)
                    global zero_score_cards_qty

                    card_loss = 0
                    if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
                        if zero_score_cards_qty > 3 and action['card'].points == 0:
                            card_loss += sum(next_state['cards_quantity'].values()) ** 0.5 * (color_values ** 0.05)
                            # + much_tire1_loss*0.00025  + card_loss*0.01

                    heuristic_value = 2 * score_loss + 0.001 * noble_loss + 0.1 * cost_loss + 0.2 * much_same_loss
                else:
                    heuristic_value = 0
                # print(heuristic_value,score_loss,noble_loss,cost_loss,much_same_loss,heuristic_value,much_tire1_loss,much_tire1_loss*0.0001)
                priorqueue.push(action, heuristic_value)
            else:
                break
        return priorqueue.pop()
