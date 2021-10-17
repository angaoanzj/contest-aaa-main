import math
import time
from copy import deepcopy
import random, itertools, copy
from collections import deque
from Splendor.splendor_model import SplendorGameRule
from agents.aaa.mcts.multi_armed_bandits import *

COLOURS = {'B':'black', 'r':'red', 'y':'yellow', 'g':'green', 'b':'blue', 'w':'white'}


class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.mcts_obj = MCTS(self.id)

    
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        copygameRule = CopySplendorRules(2)#num_agents
        state = deepcopy(copygameRule.current_game_state )
        rootnode = self.mcts_obj.mcts(copygameRule,state,self.id,0.95)
        return rootnode.bestAction()
        



class Node():    

    # record a unique node id to distinguish duplicated states for visualisation
    nextNodeID = 0
    
    def __init__(self, copygameRule, parent, state):
        # self.mdp = mdp 
        self.coptgameRule = copygameRule
        # self.gameRules = copygameRule
        self.parent = parent
        self.state = copy.deepcopy(state)
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0

    '''
    Return the value of this node
    '''
    def getValue(self):
        return self.value


class StateNode(Node):
    
    def __init__(self,  parent, state, agent_id, reward = 0,probability = 1.0, bandit = UpperConfidenceBounds()):
        super().__init__(None, parent, state)
        
        # a dictionary from actions to an environment node
        self.children = {}

        # the reward received for this state
        self.reward = reward
        
        # the probability of this node being chosen from its parent
        self.probability = probability

        self.closeState = dict()
        # a multi-armed bandit for this node
        self.bandit = bandit

    def isFullyExpanded(self,copygameRule,state):
        validActions = copygameRule.getLegalActions(state,self.id)
        
        # not yet remove duplicate situtation
        
        
        if len(validActions) == len(self.children):
            return True
        else:
            return False
        
    def select(self,copygameRule,state):
        if not self.isFullyExpanded(copygameRule,state):
            return self
        else:
            actions = list(self.children.keys())    
            qValues = dict()
            for action in actions:
                #get the Q values from all outcome nodes
                qValues[action] = self.children[action].getValue()
            bestAction = self.bandit.select(actions, qValues)
            return self.children[bestAction].select()    

    def expand(self,copygameRule,state):
        #randomly select an unexpanded action to expand
        # legalAction = copygameRule.getLegalActions(state, self.id)
        actions = copygameRule.getLegalActions(state, self.id) - self.children.keys()
        action = random.choice(list(actions))

        # still not remove the same move

        new_state, _ = copygameRule.generateSuccessor(copy.deepcopy(state), action, self.agent_id)
        #choose an outcome
        newChild =  StateNode(self, copy.deepcopy(new_state), self.agent_id)
        newStateNode = newChild.expand()
        self.children[action] = newChild
        return newStateNode

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((reward - self.value) / self.visits)
        self.parent.backPropagate(reward * 0.9)  

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = round(self.children[action].getValue(), 3)
        return qValues
    
    def bestAction(self):
        """Choose the action with the largest q-value, random tie break."""
        q_values = self.getQFunction()
        max_q = -10000
        best_actions = []
        for (state, action), q_value in q_values.items():
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        return self.global_str2action[random.choice(best_actions)]

####
# MCTS
####
 
class MCTS():

    def __init__(self, agent_id):
        self.agent_id = agent_id
        
    def mcts(self, copygameRule,state,agent_id,timeout = 1):
        # copygameRule.current
        rootNode = StateNode(parent=None,
                             state=copy.deepcopy(state),
                             agent_id=agent_id)
        
        # different from here
        startTime = int(time.time() * 1000)
        currentTime = int(time.time() * 1000)
        while currentTime < startTime + timeout * 950:
            # find a state node to expand
            selectedNode = rootNode.select(copygameRule,copy.deepcopy(state))
            if not copygameRule.gameEnds(selectedNode.state):
                child = selectedNode.expand(copygameRule,
                                            copy.deepcopy(selectedNode.state))
                reward = self.simulate(child)
                child.backPropagate(reward)
                
            currentTime = int(time.time() * 1000)

        return rootNode
    
    def choose(self,copygameRule, state,agent_id):
        return random.choice(copygameRule.getLegalActions(state, agent_id))
         
    def simulate(self,copygameRule, node,agent_id):
        state = copy.deepcopy(node.state)
        cumulativeReward = 0.0
        depth = 0
        agent_index = agent_id
        copygameRule.current_game_state = state
        copygameRule.current_agent_index = agent_id
        while not copygameRule.game_end(state):
            #choose an action to execute
            action = self.choose(copygameRule, state, agent_index)
            
            # execute the action
            (newState, reward,agent_index) = copygameRule.execute(state, action)
            copygameRule.current_agent_index = agent_index
            # discount the reward 
            cumulativeReward += pow(0.9, depth) * reward
            depth += 1

            state = newState
            
            game_ends = False
            
            while agent_index != agent_id:

                actions = copygameRule.getLegalActions(state, agent_index)
 
                selected_action = random.choice(actions)

                try:
                    state, _, agent_index = copygameRule.execute(state, selected_action, agent_index)
                    copygameRule.current_agent_index = agent_index

                    if copygameRule.game_end(state):
                        game_ends = True
                        break
                except:
                    game_ends = True

            if game_ends:
                break
            
        return cumulativeReward         

################# 
#copy game rule and keep use those method
#####
class CopySplendorRules(SplendorGameRule):
    
    ####do action update state in a copy
    
    def execute(self, state, action,agent_id):
        reward = 0
        firstreward = calScore(state, agent_id)
        agent,board = state.agents[agent_id],state.board
        agent.last_action = action #Record last action such that other agents can make use of this information.
        score = 0
        
        if 'card' in action:
            card = action['card']
        
        if 'collect' in action['type'] or action['type']=='reserve':
            #Decrement board gem stacks by collected_gems. Increment player gem stacks by collected_gems.
            for colour,count in action['collected_gems'].items():
                board.gems[colour] -= count
                agent.gems[colour] += count
            #Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
            for colour,count in action['returned_gems'].items():
                agent.gems[colour] -= count
                board.gems[colour] += count 
            
            if action['type'] == 'reserve':
                #Remove card from dealt cards by locating via unique code (cards aren't otherwise hashable).
                #Since we want to retain the positioning of dealt cards, set removed card slot to new dealt card.
                #Since the board may have None cards (empty slots that cannot be filled), check cards first.
                #Add card to player's yellow stack.
                for i in range(len(board.dealt[card.deck_id])):
                    if board.dealt[card.deck_id][i] and board.dealt[card.deck_id][i].code == card.code:
                        board.dealt[card.deck_id][i] = board.deal(card.deck_id)
                        agent.cards['yellow'].append(card)
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
                        break
            #Else, agent is buying a reserved card. Remove card from player's yellow stack.
            else:
                for i in range(len(agent.cards['yellow'])):
                    if agent.cards['yellow'][i].code == card.code:
                        del agent.cards['yellow'][i]
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
                    break
                
        #Log this turn's action and any resultant score. Return updated gamestate.
        agent.agent_trace.action_reward.append((action,score))
        agent.score += score
        agent.passed = action['type']=='pass'
        finalreward = calScore(state, agent_id)
        return state, finalreward - firstreward, (agent_id + 1) % 2 #either 0 or 1 

    def gameEnds(self,game_state):
        deadlock = 0
        for agent in game_state.agents:
            deadlock += 1 if agent.passed else 0
            if agent.score >= 15:
                return True
        return deadlock==len(self.current_game_state.agents)   

def calScore(game_state, agent_id):
    max_score = 0
    details = []
    bought_cards = lambda a : sum([len(cards) for colour,cards in a.cards.items() if colour!='yellow'])
    for a in game_state.agents:
        details.append((a.id, bought_cards(a), a.score))
        max_score = max(a.score, max_score)
    victors = [d for d in details if d[-1]==max_score]
    if len(victors) > 1 and agent_id in [d[0] for d in victors]:
        min_cards = min([d[1] for d in details])
        if bought_cards(game_state.agents[agent_id])==min_cards:
            # Add a half point if this agent was a tied victor, and had the fewest cards.
            return game_state.agents[agent_id].score + .5
    return game_state.agents[agent_id].score    
    
 