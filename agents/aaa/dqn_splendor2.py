from numpy.lib.function_base import append
import numpy as np 
import random
from collections import deque
import os
import sys

import tensorflow as tf 
from tensorflow.python.autograph.core.unsupported_features_checker import verify
from tensorflow.python.ops.gen_math_ops import xdivy
from tensorflow.keras.initializers import VarianceScaling, Constant
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import json
import time
import pickle

# Hyper Parameters for DQN
# GAMMA = 0.9 # discount factor for target Q 
# INITIAL_EPSILON = 0.5 # starting value of epsilon
# FINAL_EPSILON = 0.01 # final value of epsilon
# REPLAY_SIZE = 50000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch
STATE_DIM = 980
# ACTION_DIM = 11120 # Stopped at 1145
ACTION_DIM = 20000
HIDDEN_DIM = 512
GEMS_DICT = {"blue": 0, "green": 1, "red": 2, "black": 3, "white": 4, "gold": 5}
COLORS = ['blue', 'green', 'red', 'black', 'white']
bi2index = {'11100':0, '11010':1, '11001':2, '10110':3, '10101':4, '10011':5, '01110':6, '01101':7, '01011':8, '00111':9}
index2bi = ['11100', '11010', '11001', '10110', '10101', '10011', '01110', '01101', '01011', '00111' ]
cd = os.path.dirname(os.path.abspath('agents'))
sys.path.append(cd)


def build_q_network(n_actions=ACTION_DIM, learning_rate=0.0001, input_dim=STATE_DIM):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    # model_input = Input(shape=(1,))
    model_input = Input((input_dim,))
    x = Dense(HIDDEN_DIM, input_shape=(input_dim,), kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), activation='relu', 
              bias_initializer=Constant(0.01))(model_input)
    x = Dense(HIDDEN_DIM, kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), activation='relu', 
              bias_initializer=Constant(0.01))(x)
    # x = Dense(HIDDEN_DIM, kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), activation='relu', 
    #         bias_initializer=Constant(0.01))(x)
    # x = Dense(HIDDEN_DIM, kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), activation='relu', 
    #         bias_initializer=Constant(0.01))(x)


    # Split into value and advantage streams
    # val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    # val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), bias_initializer=Constant(0.01))(x)
    val = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(n_actions,))(val)


    # adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2., distribution='truncated_normal'), bias_initializer=Constant(0.01))(x)
    adv = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(n_actions,))(adv)

    # Combine streams into Q-Values
    # reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    # q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
    q_vals = Add()([val, adv])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(optimizer=Adam(learning_rate, clipvalue=1), loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])

    return model

class DQNAgent():
    # Spendor Agent
    def __init__(self, env, dqn, target_dqn, action_dim, state_dim, batch_size=BATCH_SIZE, 
                 history_length = 4, eps_initial = 0.5, eps_final = 0.01, eps_final_time_step = 0.01,
                 eps_evaluation = 0.0, eps_annealing_time_steps = 80000, replay_buffer_start_size = 5000,
                 max_time_steps = 100000, use_per = True):
        # init experience replay
        # self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.prev_step = 0
        self.epsilon = eps_initial
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_actions = list(range(0,action_dim))
        
        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size
        
        self.replay_buffer = deque()
        self.use_per = use_per
        
        # Epsilons
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_time_step = eps_final_time_step
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_time_steps = eps_annealing_time_steps
        self.epsilon_decay = 0.999
        
         # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        # self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_time_steps
        # self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        # self.slope_2 = -(self.eps_final - self.eps_final_time_step) / (self.max_time_steps - self.eps_annealing_time_steps - self.replay_buffer_start_size)
        # self.intercept_2 = self.eps_final_time_step - self.slope_2*self.max_time_steps       
        
        # init Q network structure
        self.DQN = dqn
        self.target_dqn = target_dqn
        self.env = env
        self.legal_actions = None
        self.action_dict = dict()
        self.vec2action = dict()
        self.legal_action_count = 0
        self.ddqn = True
        
        # init session
        # self.session = tf.InteractiveSession()
        # self.session.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver()
        
        # loading networks
        # self.saver = tf.train.Saver()
        # try:
        #     checkpoint = tf.train.get_checkpoint_state("agents")
        #     if checkpoint and checkpoint.model_checkpoint_path:
        #         self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        # except:
        #     return
        
    def get_replay_minibatch(self, batch_size=BATCH_SIZE):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = np.zeros((self.batch_size, self.state_dim))
        next_state_batch = np.zeros((self.batch_size, self.state_dim))
        action_batch, reward_batch, terminal_flag = [], [], []
        
        for i in range(self.batch_size):
            state_batch[i] = minibatch[i][0]
            action_batch.append(minibatch[i][1])
            reward_batch.append(minibatch[i][2])
            next_state_batch[i] = minibatch[i][3]
            terminal_flag.append(minibatch[i][4])
            
        # state_batch = [data[0] for data in minibatch]
        # action_batch = [data[1] for data in minibatch]
        # reward_batch = [data[2] for data in minibatch]
        # next_state_batch = [data[3] for data in minibatch]
        # terminal_flag = [data[4] for data in minibatch]
        
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_flag
        
    def calc_epsilon(self, time_step, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            time_step: Global time step (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif time_step < self.replay_buffer_start_size:
            return self.eps_initial
        # elif time_step >= self.replay_buffer_start_size and time_step < self.replay_buffer_start_size + self.eps_annealing_time_steps:
        #     return self.slope*time_step + self.intercept
        # elif time_step >= self.replay_buffer_start_size + self.eps_annealing_time_steps:
        #     return self.slope_2*time_step + self.intercept_2
        elif time_step >= self.replay_buffer_start_size:
            self.epsilon *= self.epsilon_decay
            return self.epsilon
        
        
    def get_action(self, time_step, state, legal_actions, env, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            time_step: global time step (used for epsilon)
            state: State to give an action for
            env: Environment
        Returns:
            An action vector containing the integer as the predicted move
        """
        # With chance epsilon, take a random action
        eps = self.calc_epsilon(time_step, evaluation)
        # print(state)
        Q_values = self.DQN.predict(state)
        act = random.choice(legal_actions)
        # act = random.randint(0, self.action_dim - 1)
        # self.random_actions = list(range(0,self.action_dim))
        # temp_act = 0
        if random.random() <= eps:
            # action = np.zeros(self.action_dim)
            # action[act] = 1
            action = act
            # while (self.verify(action, env) == False):
            #     act = random.choice(self.random_actions)
            #     # act = random.randint(0, self.action_dim - 1)
            #     # act = temp_act
            #     action = np.zeros(self.action_dim)
            #     action[act] = 1
            #     # temp_act += 1
            #     # print(act)
            #     self.random_actions.remove(act)
        else:
            action = np.argmax(Q_values)
            # act = np.argmax(Q_values) 
            # action = np.zeros(self.action_dim)
            # action[act] = 1
            if self.verify(action, env)  == False:
                action = random.choice(legal_actions)
                return action
            # while (self.verify(action, env) == False):
            #     act = random.choice(self.random_actions)
            #     # act = random.randint(0, self.action_dim - 1)
            #     action = np.zeros(self.action_dim)
            #     action[act] = 1
            #     self.random_actions.remove(act)
                
        return action
    
    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())
        
    def add_experience(self, state, action,reward, next_state, terminal):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.append([state, action, reward, next_state, terminal])   
        
    def learn(self, batch_size, gamma, time_step):
        """Sample a batch and use it to improve the DQN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            time_step: Global frame number (used for calculating importances)
        Returns:
            The loss between the predicted and target Q as a float
        """
        states, actions, rewards, new_states, terminal_flags = self.get_replay_minibatch(batch_size=self.batch_size)
        

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.DQN.predict(states)
        # predict best action in ending state using the main network
        target_next = self.DQN.predict(new_states)
        # predict Q-values for ending state using the target network
        target_val = self.target_dqn.predict(new_states)

        for i in range(len(states)):
            # correction on the Q value for the action used
            if terminal_flags[i]:
                target[i][actions[i]] = rewards[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    # print(actions[i])
                    target[i][actions[i]] = rewards[i] + gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][actions[i]] = rewards[i] + gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.DQN.fit(states, target, batch_size=self.batch_size, verbose=0)

        # return float(loss.numpy()), error
    
    def save(self, folder_name, vec2action, action_dict, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')
        
        # Save replay buffer Object
        with open(folder_name + '/replaybuffer.pkl', 'wb') as f:
            pickle.dump(self.replay_buffer, f)
            
        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**kwargs}))  # save replay_buffer information and any other information
            
        np.save(folder_name + '/vec2action.npy', vec2action)
        np.save(folder_name + '/action_dict.npy', action_dict)

    def load(self, folder_name, load_replay_buffer=True, load_meta=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5', custom_objects={'tf': tf})
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5', custom_objects={'tf': tf})
        self.optimizer = self.DQN.optimizer
        
        vec2action = np.load(folder_name + "/vec2action.npy", allow_pickle=True)
        action_dict = np.load(folder_name + "/action_dict.npy", allow_pickle=True)
        
        if load_replay_buffer:
            with open(folder_name + '/replaybuffer.pkl', 'rb') as f:
                self.replay_buffer = pickle.load(f)
        
        if load_meta:        
            # Load meta
            with open(folder_name + '/meta.json', 'r') as f:
                meta = json.load(f)
                
            return vec2action, action_dict, meta

    def predict(self, state, env):
        Q_values = self.DQN.predict(state)
        act = np.argmax(Q_values) 
        action = np.zeros(self.action_dim)
        action[act] = 1

        return action

    def verify(self, action, env):
        psedo_env = env
        try:
            a, b, c, is_valid, _ = psedo_env.step(action, verify=True) 
            return is_valid
        except:
            #print(is_valid)
            return False
        
from ENV2 import SplendorEnv

# Hyper Parameters
EPISODE = 10000 # Episode limitation
STEP = 1000 # Step limitation in an episode
# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = 'Splendor-saves/save-00009222'
# LOAD_FROM = None
SAVE_PATH = 'Splendor-saves'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = "tensorboard/"

DISCOUNT_FACTOR = 0.9            # Gamma, how much to discount future rewards
MEM_SIZE = 10000                # The maximum size of the replay buffer

UPDATE_FREQ = 3                   # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 5         # Number of actions between when the target network is updated

BATCH_SIZE = 256                   # Number of samples the agent learns from at once
LEARNING_RATE = 0.0001

def main():
    EVAL_LENGTH = 64               # Number of Episodes to evaluate for
    SAVE_PATH = 'Splendor-saves'
    # Create environment
    env = SplendorEnv()
    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Build main and target networks
    MAIN_DQN = build_q_network(ACTION_DIM, LEARNING_RATE, STATE_DIM)
    TARGET_DQN = build_q_network(ACTION_DIM, STATE_DIM)
    
    agent = DQNAgent(env, MAIN_DQN, TARGET_DQN, action_dim=ACTION_DIM, state_dim=STATE_DIM, batch_size=BATCH_SIZE)

    # Training and evaluation
    if LOAD_FROM is None:
        agent.time_step = 0
        rewards = []
        # loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        vec2action, action_dict, meta = agent.load(LOAD_FROM)
        print(type(vec2action))

        # Apply information loaded from meta
        agent.time_step = meta['frame_number']
        rewards = meta['rewards']
        agent.legal_action_count = meta['legal_action_count']
        agent.vec2action = vec2action.item()
        agent.action_dict = action_dict.item()
        # loss_list = meta['loss_list']

        print('Loaded')
    # Main loop
    try:
        with writer.as_default():
            for episode in range(EPISODE):
                # Training
                start_time = time.time()
                state = env.reset()
                state = np.reshape(state, [1, STATE_DIM])
                
                episode_reward_sum = 0
                terminal = False

                while not terminal:
                    # Get action
                    legal_actions = env.get_legalActions()
                    for action in legal_actions:
                        if str(action) not in agent.action_dict.keys():
                            agent.action_dict[str(action)] = agent.legal_action_count
                            agent.legal_action_count += 1
                            
                            agent.vec2action[agent.legal_action_count] = action
                            # print(agent.legal_action_count)
                            with open("actions_space.txt", "w") as f:
                                f.write(str(agent.legal_action_count))
                        
                    if agent.time_step % 500 == 0:
                        print("Training Time step: ", agent.time_step)
                    
                    action = agent.get_action(agent.time_step, state, legal_actions, env=env)
                    # print(action)
                    

                    # Take step
                    next_state, reward, terminal, valid, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, STATE_DIM])

                    agent.time_step += 1
                    episode_reward_sum += reward
                    if type(action) == dict:
                        action = agent.action_dict[str(action)]

                    # Add experience to replay memory
                    agent.add_experience(state=state,
                                        action=action,
                                        reward=reward,
                                        next_state=next_state,
                                        terminal=terminal)

                    # Update agent
                    if len(agent.replay_buffer) > MEM_SIZE:
                        # remove when overflow
                        agent.replay_buffer.popleft()                     
                    if agent.time_step % UPDATE_FREQ == 0 and len(agent.replay_buffer) > BATCH_SIZE:
                        agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, time_step=agent.time_step)
                        # loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, time_step=frame_number)
                        # loss_list.append(loss)

                    # Update target network
                    if agent.time_step % TARGET_UPDATE_FREQ == 0 and len(agent.replay_buffer) > BATCH_SIZE:
                        agent.update_target_network()

                    state = next_state
                    
                    # Break the loop when the game is over
                    # print(terminal)
                    if terminal:
                        # terminal = False
                        break

                rewards.append(episode_reward_sum)
                # print(rewards)
                prev_steps = agent.time_step - agent.prev_step
                agent.prev_step = agent.time_step
                print(f'Game number: {str(len(rewards)).zfill(6)}  Time step: {str(agent.time_step).zfill(8)}  '
                      f'Reward: {np.mean(rewards[-1:]):0.1f} Steps taken: {str(prev_steps)} Time taken: {(time.time() - start_time):.1f}s')


                # Output the progress every 10 games
                if len(rewards) % 5 == 0:
                    # Write to TensorBoard
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar('Reward', np.mean(rewards[-5:]), agent.time_step)
                        # tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                        writer.flush()

                    print(f'Game number: {str(len(rewards)).zfill(6)}  Time step: {str(agent.time_step).zfill(8)}  '  
                          f'Average reward: {np.mean(rewards[-5:]):0.1f} Steps taken: {str(prev_steps)} Time taken: {(time.time() - start_time):.1f}s')

                # Save model every 1000 time steps
                if agent.time_step % 1000 == 0 and SAVE_PATH is not None:
                    print('Saving at time step {}...'.format(agent.time_step))
                    # agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
                    agent.save(f'{SAVE_PATH}/save-{str(agent.time_step).zfill(8)}', vec2action=agent.vec2action, action_dict=agent.action_dict, 
                               frame_number=agent.time_step, rewards=rewards, legal_action_count=agent.legal_action_count)
                    print('Saved.')

                
                # Evaluate every 20 games
                if episode % 50 == 0  and episode>=50:

                    for i in range(EVAL_LENGTH):
                        total_score = 0
                        game_number = 0
                        state = env.reset()
                        state = np.reshape(state, [1, STATE_DIM])
                        episode_reward_sum = 0
                        terminal = False
                        while not terminal:
                            legal_actions = env.get_legalActions()
                            action = agent.get_action(agent.time_step, state, legal_actions, env=env, evaluation=True)
                            
                            if type(action) != dict:
                                action = agent.vec2action[action]

                            # Step action
                            next_state, reward, terminal, valid, _ = env.step(action)
                            total_score += reward
                            next_state = np.reshape(next_state, [1, STATE_DIM])
                            state = next_state

                            # On game-over
                            if terminal:
                                game_number += 1
                                break
                        
                    final_score = total_score / EVAL_LENGTH
                    if EVAL_LENGTH < 1024:
                        EVAL_LENGTH *= 2
                    # Print score and write to tensorboard
                    print('Evaluation score:', final_score)
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar('Evaluation score', final_score, game_number)
                        writer.flush()

                    # Save model
                    if len(rewards) >= 50 and SAVE_PATH is not None:
                        print('Saving at time step {}...'.format(agent.time_step))
                        # agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
                        agent.save(f'{SAVE_PATH}/save-{str(agent.time_step).zfill(8)}', vec2action=agent.vec2action, action_dict=agent.action_dict, 
                                frame_number=agent.time_step, rewards=rewards, legal_action_count=agent.legal_action_count)
                        print('Saved.')
    
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            # agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
            agent.save(f'{SAVE_PATH}/save-{str(agent.time_step).zfill(8)}', vec2action=agent.vec2action, action_dict=agent.action_dict, 
                        frame_number=agent.time_step, rewards=rewards, legal_action_count=agent.legal_action_count)
            print('Saved.')

if __name__ == '__main__':
  	main()