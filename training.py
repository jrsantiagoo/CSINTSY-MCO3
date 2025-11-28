import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################
import matplotlib.pyplot as plt

def reward(new_state: int, old_state: int):
    agent_row = new_state // 1000
    agent_col = (new_state // 100) % 10
    cat_row = (new_state // 10) % 10
    cat_col = new_state % 10

    old_agent_row = old_state // 1000
    old_agent_col = (old_state // 100) % 10
    old_cat_row = (old_state // 10) % 10
    old_cat_col = old_state % 10

    new_dist = abs(agent_row - cat_row) + abs(agent_col - cat_col)
    old_dist = abs(old_agent_row - old_cat_row) + abs(old_agent_col - old_cat_col)

    if agent_row == cat_row and agent_col == cat_col:
       return 1000
    
    if new_dist < old_dist:
        return 5
    else:
        return -5
    
    return 0

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    learning_rate_a = 0.5
    discount_factor_g = 0.95

    epsilon = 1
    epsilon_decay_rate = 0.0004
    rng = np.random.default_rng()

    rewards_per_ep = np.zeros(episodes + 1)
    moves_per_ep = np.zeros(episodes + 1)

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        state = env.reset()[0]
        done = False
        moves = 0
        ep_reward = 0

        while not done and moves < 60:
            if rng.random() < epsilon:
                action = random.randint(0,3)
            else: 
                action = np.argmax(q_table[state])

            new_state, _, done, _, _ = env.step(action)

            r = reward(new_state, state)

            if state == new_state:
                r = -8

            q_table[state][action] = q_table[state][action] + learning_rate_a * (
                r + discount_factor_g * np.max(q_table[new_state]) - q_table[state][action]
            )

            moves += 1
            ep_reward += r
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001
        
        rewards_per_ep[ep] = ep_reward
        moves_per_ep[ep] = moves

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_ep[max(0,t-100):(t+1)])
    mean_moves = np.zeros(episodes)
    for t in range(episodes):
        mean_moves[t] = np.mean(moves_per_ep[max(0,t-100):(t+1)])

    print("Average move of last 100 runs: ", int(mean_moves[4999]))

    plt.plot(sum_rewards)
    plt.savefig('plot.png')

    plt.clf()

    plt.plot(mean_moves)
    plt.savefig('plot_moves.png')

    return q_table