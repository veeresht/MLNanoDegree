import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import math
import pprint
from collections import defaultdict
from itertools import izip

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Initialize q-table as a default dictionary
        self.q_table = defaultdict(float)
        # Initialize variables for state, previous state, previous action, previous reward
        self.state = None
        self.prev_state = ('red', 'safe', 'safe', 'forward') # chosen randomly
        self.prev_action = 'None' # chosen randomly
        self.prev_reward = 0
        # Initialize Q-learning parameters
        # The probability of taking a random action (exploration)
        self.epsilon = 0.1
        # The learning rate
        self.alpha = 0.9
        # The discount
        self.gamma = 0.4
       
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
    def create_state_tuple(self, inputs, next_waypoint):

        # Merge oncoming traffic inputs 'None' and 'right' into a single state 'safe'
        if inputs['oncoming'] == None or inputs['oncoming'] == 'right':
            oncoming_state_value = 'safe'
        else:
            oncoming_state_value = inputs['oncoming']

        # Merge from-left traffic inputs 'None', 'left' and 'right' into a single state 'safe'
        if inputs['left'] == None or inputs['left'] == 'left' or inputs['left'] == 'right':
            left_state_value = 'safe'
        else:
            left_state_value = inputs['left']

        # Create the state tuple from the inputs and return
        return (inputs['light'], oncoming_state_value, left_state_value, next_waypoint)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.create_state_tuple(inputs, self.next_waypoint)

        # TODO: Learn policy based on state, action, reward
        # Compute the maximum Q-value for the next state over all possible actions
        max_next_state_q_value = max([self.q_table[(self.state, a)] for a in self.env.valid_actions])

        # Q-Learning Update 
        self.q_table[(self.prev_state, self.prev_action)] = (1 - self.alpha)*self.q_table[(self.prev_state, self.prev_action)] + self.alpha*(self.prev_reward + self.gamma*max_next_state_q_value)

        # TODO: Select action according to your policy
        if (random.uniform(0, 1) < self.epsilon):
            # Choose an action randomly
            action = random.choice(self.env.valid_actions)
        else:
            # Choose action using policy from Q-Learning
            q_values_for_state = [self.q_table[(self.state, a)] for a in self.env.valid_actions]
            argmax_function = lambda array: max(izip(array, xrange(len(array))))[1]
            best_action_idx = argmax_function(q_values_for_state)
            action = self.env.valid_actions[best_action_idx]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Store previous state, action, reward
        self.prev_reward = reward
        self.prev_state = self.state
        self.prev_action = action

        print "LearningAgent.update(): deadline = {}, inputs = {}, waypoint = {},action = {}, reward = {}".format(deadline, inputs, self.next_waypoint, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit
 
if __name__ == '__main__':
    run()
