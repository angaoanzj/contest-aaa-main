
import random
import math

class MultiArmedBandit():

    '''
        Select an action given Q-values for each action.
    '''
    def select(self, actions, qValues): abstract
    
    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''
    def runBandit(self, episodes = 2000, episodeLength = 1000, drift = True):

        #the actions available
        actions = [0, 1, 2, 3, 4]

        rewards = []
        for episode in range(0, episodes):
            self.reset()

            # The probability of receiving a payoff of 1 for each action
            probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]
        
            qValues = dict()
            N = dict()
            for action in actions:
                qValues[action] = 0.0
                N[action] = 0

            episodeRewards = []
            for step in range(0, episodeLength):

                # Halfway through the episode, change the probabilities
                if drift and step == episodeLength / 2:
                    probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]
                
                #select an action
                action = self.select(actions, qValues)

                r = random.random()
                reward = 0
                if r < probabilities[action]:
                    reward = 5

                episodeRewards += [reward]

                N[action] = N[action] + 1

                qValues[action] = qValues[action] - (qValues[action] / N[action])
                qValues[action] = qValues[action] + reward / N[action]

            rewards += [episodeRewards]

        return rewards
    
    
    
class ValueIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' Implmentation of value iteration '''
    def valueIteration(self, iterations = 100, theta = 0.001):

        # Initialise the value function V with all 0s
        values = self.initialiseValueFunction()
        for _ in range(iterations):
            delta = 0.0
            newValues = dict()
            for state in mdp.getStates():
                qValues = dict()
                for action in mdp.getActions(state):
                    # Calculate the value of Q(s,a)
                    newValue = 0.0
                    for (newState, probability) in mdp.getTransitions(state, action):
                        reward = mdp.getReward(state, action, newState)
                        newValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))
                    qValues.update({action: newValue})

                # V(s) = max_a Q(s,a)
                maxQ = max(qValues.values())
                delta = max(delta, abs(values[state] - maxQ))
                newValues.update({state: maxQ})

            values.update(newValues)

            # terminate if the value function has converged
            if delta < theta:
                break

        return values

    def initialiseValueFunction(self):
        values = dict()
        for state in self.mdp.getStates():
            values.update({state: 0.0})
        return values
    
    
class EpsilonGreedy(MultiArmedBandit):

    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def reset(self):
        None

    def select(self, actions, qValues):
        r = random.random()
        # select a random action with epsilon probability
        if r < self.epsilon:
            return random.choice(actions)
        else:
            maxActions = []
            maxValue = float('-inf')
            for action in actions:
                value = qValues[action]
                if value > maxValue:
                    maxActions = [action]
                    maxValue = value
                elif value == maxValue:
                    maxActions += [action]
                    
            # if there are multiple actions with the highest value
            # choose one randomly
            return random.choice(maxActions)    
        
class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0  #number of times a choice has been made
        self.N = dict() #number of times each action has been chosen

    def select(self, actions, qValues):

        # First execute each action one time
        for action in actions:
            if not action in self.N.keys():
                self.N[action] = 1
                self.total += 1
                return action

        maxActions = []
        maxValue = float('-inf')
        for action in actions:
            N = self.N[action]
            value = qValues[action] + math.sqrt((2 * math.log(self.total)) / N)
            if value > maxValue:
                maxActions = [action]
                maxValue = value
            elif value == maxValue:
                maxActions += [action]
                    
        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(maxActions)
        self.N[result] = self.N[result] + 1
        self.total += 1
        return result        