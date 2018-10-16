"""Multi-agent search agents.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

Author: Dylan Blanchard, Sloan Anderson, and Stephen Johnson
Class: CSI-480-01
Assignment: PA 3 -- Multiagent
Due Date: October 15, 2018 11:59 PM

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


import random
import util
from game import Agent


class ReflexAgent(Agent):
    """An agent that chooses an action reflexively based on a state evaluation.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """Choose among the best options according to the evaluation function.

        You do not need to change this method, but you're welcome to.

        Just like in the previous project, get_action takes a GameState and
        returns some Directions.X for some X in the set
        {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action)
                  for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores))
                        if scores[index] == best_score]
        # Pick randomly among the best actions
        chosen_index = random.choice(best_indices)

        # *** Add more of your code here if you want to ***

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """Return evaluation (number) based on game state and proposed action.

        *** Design a better evaluation function here. ***

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code samples below extracts some useful information from the state,
        like the remaining food (new_food) and Pacman position after moving
        (new_pos). new_scared_times holds the number of moves that each ghost
        will remain scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state =\
            current_game_state.generate_pacman_successor(action)

        new_pos = successor_game_state.get_pacman_position()

        new_food = successor_game_state.get_food()

        new_ghost_states = successor_game_state.get_ghost_states()

        "*** YOUR CODE HERE ***"

        # Its the goal just do it!!!
        if successor_game_state.is_win():
            return float('inf')

        # Base utility
        utility = successor_game_state.get_score()

        # Check how close a gost is to you and move accordingly
        for ghost_state in new_ghost_states:
            if ghost_state.scared_timer == 0:
                pos = ghost_state.get_position()
                distance_to_ghost = util.manhattan_distance(pos, new_pos)
                if distance_to_ghost <= 10:
                    utility -= 10 - distance_to_ghost
                elif distance_to_ghost <= 2:
                    utility -= 100

        # If the next game state has you eating a food try and go there
        current_num_food = current_game_state.get_num_food()
        next_num_food = successor_game_state.get_num_food()
        if current_num_food > next_num_food:
            utility += 25

        # Try and go closer to nearest food
        utility -= min([util.manhattan_distance(new_pos, pos) for pos in
                        [(i, j) for i, lst in enumerate(new_food)
                            for j, val in enumerate(lst) if val]])

        return utility


def score_evaluation_function(current_game_state):
    """Return the score of the current game state.

    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """Common elements to all multi-agent searchers.

    Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        """Create agent given an evaluation function and search depth."""
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Your minimax agent (question 2)."""

    def get_action(self, game_state):
        """Return the minimax action from the given game_state.

        Run minimax to max depth self.depth and evaluate "leaf" nodes using
        self.evaluation_function.

        Here are some method calls that might be useful when implementing
        minimax:
            game_state.get_legal_actions(agent_index):
                Returns a list of legal actions for an agent
                agent_index=0 means Pacman, ghosts are >= 1

            game_state.generate_successor(agent_index, action):
                Returns the successor game state after an agent takes action.

            game_state.get_num_agents():
                Returns the total number of agents in the game.
        """
        def min_value(current_game_state, ghost, current_depth):
            """Calculate the minimum value for the ghost agent.

            The Best choice for the ghost and returns that value.
            """
            current_value = float('inf')

            for new_action in current_game_state.get_legal_actions(ghost):
                new_state = \
                    current_game_state.generate_successor(ghost, new_action)

                new_value, current_action = \
                    minimax_decision(new_state, ghost + 1, current_depth)

                if new_value < current_value:
                    current_value = new_value

            return current_value

        def max_value(current_game_state, current_depth):
            """Calculate the maximum value for the pacman agent.

            The Best choice for pacman and returns that action
            """
            current_value = float('-inf')
            best_action = None

            for new_action in current_game_state.get_legal_actions(0):
                new_state = \
                    current_game_state.generate_successor(0, new_action)
                new_value, _ = minimax_decision(new_state, 1, current_depth)
                if new_value > current_value:
                    current_value = new_value
                    best_action = new_action

            return current_value, best_action

        def minimax_decision(current_game_state, agent, current_depth):
            """Use to determine the depth and win/lose.

            Also, what function gets called.
            """
            if agent >= current_game_state.get_num_agents():
                agent = 0
                current_depth += 1

            if current_game_state.is_win() or current_game_state.is_lose() or \
                    self.depth < current_depth:
                return self.evaluation_function(current_game_state), ''

            if 0 == agent:
                return max_value(current_game_state, current_depth)
            else:
                return min_value(current_game_state, agent, current_depth), ''

        depth = 1
        first_agent = 0
        value, action = minimax_decision(game_state, first_agent, depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Your minimax agent with alpha-beta pruning (question 3)."""

    def get_action(self, game_state):
        """Return the minimax action from the given game_state.

        Run minimax with alpha-beta pruning to max depth self.depth and
        evaluate "leaf" nodes using self.evaluation_function.
        """
        # *** YOUR CODE HERE ***
        def max_value(current_game_state, current_depth, alpha, beta):
            current_value = float('-inf')
            best_action = 'Stop'
            for new_action in current_game_state.get_legal_actions(0):
                if current_value > beta:
                    return current_value, best_action

                new_state = \
                    current_game_state.generate_successor(0, new_action)
                new_value, _ = \
                    minimax_decision(new_state, 1, current_depth, alpha, beta)

                if new_value > current_value:
                    current_value = new_value
                    best_action = new_action
                    if current_value > alpha:
                        alpha = current_value

            return current_value, best_action

        def min_value(current_game_state, ghost, current_depth, alpha, beta):
            current_value = float('inf')

            for new_action in current_game_state.get_legal_actions(ghost):
                if current_value < alpha:
                    return current_value
                new_state = \
                    current_game_state.generate_successor(ghost, new_action)

                new_value, _ = minimax_decision(
                        new_state, ghost + 1, current_depth, alpha, beta)

                if new_value < current_value:
                    current_value = new_value
                    if new_value < beta:
                        beta = current_value

            return current_value

        def minimax_decision(
                current_game_state, agent, current_depth, alpha, beta):

            if agent >= current_game_state.get_num_agents():
                agent = 0
                current_depth += 1

            if current_game_state.is_win() or current_game_state.is_lose() or \
                    self.depth < current_depth:
                return self.evaluation_function(current_game_state), ''

            if 0 == agent:
                return max_value(
                    current_game_state, current_depth, alpha, beta)
            else:
                return min_value(
                    current_game_state, agent, current_depth, alpha, beta), ''

        depth = 1
        first_agent = 0
        alpha = float('-inf')
        beta = float('inf')
        value, action = \
            minimax_decision(game_state, first_agent, depth, alpha, beta)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """Your expectimax agent (question 4)."""

    def get_action(self, game_state):
        """Return the expectimax action from the given game_state.

        Run expectimax to max depth self.depth and evaluate "leaf" nodes using
        self.evaluation_function.

        All ghosts are modeled as choosing uniformly at random from their
        legal moves.
        """
        # *** YOUR CODE HERE ***
        def exp_val(current_game_state, agent, current_depth):
            current_value = 0

            legal_actions = current_game_state.get_legal_actions(agent)

            for new_action in legal_actions:
                new_state = \
                    current_game_state.generate_successor(agent, new_action)
                new_value, _ = exp_max_val(new_state, agent + 1, current_depth)
                current_value += new_value

            return current_value / len(legal_actions)

        # Maximum evaluation function
        def max_val(current_game_state, current_depth):
            # Create value with worst case
            current_value = float('-inf')

            # Create base action if no action is chosen
            best_action = 'Stop'

            # Loop through all possible actions
            for new_action in current_game_state.get_legal_actions(0):

                # Get new state from action
                new_state = \
                    current_game_state.generate_successor(0, new_action)

                # Run expectimax decision on new state
                new_value, _ = exp_max_val(new_state, 1, current_depth)

                # Check if value for action is larger then current value
                if new_value > current_value:

                    # Set new value to best current value
                    current_value = new_value

                    # Set new action as best current acton
                    best_action = new_action

            # Return the best value and action from the list actions
            return current_value, best_action

        # Expectimax decision function
        def exp_max_val(current_game_state, current_agent, current_depth):
            # Check if you have moved to the next depth
            if current_agent >= current_game_state.get_num_agents():

                # Set current agent back to Pacman
                current_agent = 0

                # Increment depth by 1
                current_depth += 1

            # Check if you are at a terminal state or leaf node
            if current_game_state.is_win() or current_game_state.is_lose() or \
                    self.depth < current_depth:
                return self.evaluation_function(current_game_state), ''

            # Run max evaluation function for Pacman
            if current_agent == 0:
                return max_val(current_game_state, current_depth)

            # Run expectation evaluation function for ghost
            else:
                return exp_val(current_game_state,
                               current_agent, current_depth), ''

        # Set starting depth
        depth = 1

        # Set to Pacman for first agent
        first_agent = 0

        # Run expectimax evaluation
        value, action = exp_max_val(game_state, first_agent, depth)

        # Return best expected action
        return action


def better_evaluation_function(current_game_state):
    """Your awesome evaluation function (question 5).

    Description: This function works off the simpler evaluation function above
        its primary deference is that it works off of the current game state
        and also evaluations power pellets and some small changes to the math
        when evaluating ghost and food.
    """
    # Set base utility to current score
    utility = current_game_state.get_score()

    # Return max utility if you will win
    if current_game_state.is_win():
        return float('inf')

    # Get pacmans current position
    current_position = current_game_state.get_pacman_position()

    # Try and go closer to nearest food
    food_dist = [util.manhattan_distance(current_position, pos)
                 for pos in
                 [(i, j)
                  for i, lst in
                  enumerate(current_game_state.get_food())
                  for j, val in
                  enumerate(lst)
                  if val]]
    utility -= min(food_dist) / len(food_dist)

    # Get distance to each ghost that is not scared
    dist = [util.manhattan_distance(current_position, ghost.get_position())
            for ghost in current_game_state.get_ghost_states()
            if ghost.scared_timer == 0]

    # Try and stay away from all ghosts
    for d in dist:
        if d < 5:
            utility -= (10 - d)
        if d <= 1:
            utility -= 300

    # Try and go closer to power capsules
    dist_capsules = [util.manhattan_distance(current_position, capsule)
                     for capsule in current_game_state.get_capsules()]
    for c_dist in dist_capsules:
        utility -= c_dist

    # Return utility
    return utility


# Abbreviation
better = better_evaluation_function
