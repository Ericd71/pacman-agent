# my_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# my_team.py (A-star project)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import heapq
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='Attacker', second='Defender', num_training=0):
    """
    Creates a team of agents using A* pathfinding algorithm.
    The Attacker focuses on food collection while avoiding ghosts.
    The Defender focuses on intercepting enemy invaders.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class AStarAgent(CaptureAgent):
    """
    Base class for agents using A* pathfinding algorithm.
    Provides the general pathfinding functionality and action selection.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.current_path = []  # List of actions to execute
        self.goal_position = None
        self.last_state = None
        self.path_recalc_threshold = 3  # Recalculate path after this many steps
        self.steps_since_recalc = 0

    def register_initial_state(self, game_state):
        """
        Initializes agent with starting position and game state.
        """
        self.start = game_state.get_agent_position(self.index)
        self.last_state = game_state
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Selects next action using A* pathfinding.
        Recalculates path when needed or when goal is reached.
        """
        self.steps_since_recalc += 1
        
        # Recalculate path if empty, reached goal, or threshold exceeded
        if (not self.current_path or 
            self.steps_since_recalc >= self.path_recalc_threshold or
            self.reached_goal(game_state)):
            self.recalculate_path(game_state)
            self.steps_since_recalc = 0
        
        # Execute next action from path or choose randomly if no path
        if self.current_path:
            return self.current_path.pop(0)
        return random.choice(game_state.get_legal_actions(self.index))

    def reached_goal(self, game_state):
        """
        Checks if agent has reached its current goal position.
        """
        if not self.goal_position:
            return True
        
        my_pos = game_state.get_agent_state(self.index).get_position()
        return self.get_maze_distance(my_pos, self.goal_position) < 1.0

    def recalculate_path(self, game_state):
        """
        Computes new path using A* algorithm to reach goal position.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Get goal position from subclass
        goal_pos = self.get_goal_position(game_state)
        self.goal_position = goal_pos
        
        # If no goal or already there, clear path
        if not goal_pos or my_pos == goal_pos:
            self.current_path = []
            return
        
        # Get positions to avoid (ghosts, walls, etc.)
        avoid_positions = self.get_avoid_positions(game_state)
        
        # Calculate path using A*
        self.current_path = self.astar_path(game_state, my_pos, goal_pos, avoid_positions)

  
    def astar_path(self, game_state, start_pos, goal_pos, avoid_positions=None):
        """
        A* algorithm (done in class) to find optimal path from start (S) to goal.
        """
        if avoid_positions is None:
            avoid_positions = []
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_pos, []))
        
        # Track visited positions and their costs
        closed_set = set()
        g_scores = {start_pos: 0}
        
        layout = game_state.data.layout
        width, height = layout.width, layout.height
        
        while open_set:
            f_score, g_score, current_pos, path = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
            
            # Check if it has reached goal
            if self.get_maze_distance(current_pos, goal_pos) < 1.0:
                return path
            
            closed_set.add(current_pos)
            
            #Try all four cardinal directions (N,S,E,W)
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                # Calculate next position
                dx, dy = Actions.direction_to_vector(action)
                next_pos = (int(current_pos[0] + dx), int(current_pos[1] + dy))

              
                # Check bounds and walls
                if (next_pos[0] < 0 or next_pos[0] >= width or 
                    next_pos[1] < 0 or next_pos[1] >= height or
                    game_state.has_wall(next_pos[0], next_pos[1])):
                    continue
                
                # Check if said position should be avoided (too close!)
                skip_position = False
                for avoid_pos in avoid_positions:
                    if self.get_maze_distance(next_pos, avoid_pos) < 2:
                        skip_position = True
                        break
                
                if skip_position:
                    continue
                
                # Calculate new cost
                new_g = g_score + 1
                new_h = self.heuristic(next_pos, goal_pos)
                new_f = new_g + new_h
                
                # If better path found or first time seeing this position
                if next_pos not in g_scores or new_g < g_scores[next_pos]:
                    g_scores[next_pos] = new_g
                    new_path = path + [action]
                    heapq.heappush(open_set, (new_f, new_g, next_pos, new_path))
        
        return []  # No path found, return empty

    def heuristic(self, pos1, pos2):
        """
        Heuristic for A* algo - uses maze distance for accuracy
        """
        return self.get_maze_distance(pos1, pos2)

    def get_avoid_positions(self, game_state):
        """
        Returns positions that should be avoided during path planning.
        Can be overridden by subclasses for custom behavior.
        """
        return []


class Attacker(AStarAgent):
    """
    Offensive agent that uses A* to collect food efficiently while avoiding ghosts, return to home as soon as possible when carrying food.
    Try not to lose food in opponent's base!
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_carrying_threshold = 3  # Return home after carrying this many pellets
        self.danger_distance = 4  # Distance to ghosts that triggers immediate return
    
    def get_goal_position(self, game_state):
        """
        Determines goal position (action) for attacker - either food collection or returning home.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_carrying = my_state.num_carrying
        
        # Get ghost positions and distances
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
        min_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')
        
        # Return home if carrying food and in danger or reached threshold
        if food_carrying > 0:
            # Immediate return if ghosts are very close
            if min_ghost_distance < self.danger_distance:
                return self.start
            
            # Return home if carrying enough food
            if food_carrying >= self.food_carrying_threshold:
                return self.start
        
        # Otherwise, collect food (more cautious though!)
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.start  # No food left, return home
        
        # Find closest food, but prefer safer food if carrying pellets
        if food_carrying > 0:
            # When carrying food, prefer food that's closer to home territory
            home_distances = {food: self.get_maze_distance(food, self.start) for food in food_list}
            closest_food = min(food_list, key=lambda food: home_distances[food])
        else:
            # When not carrying food, go for closest food
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
        
        return closest_food

    def get_avoid_positions(self, game_state):
        """
        Returns ghost positions to avoid during path planning, with stronger avoidance when carrying food.
        """
        avoid_positions = []
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_carrying = my_state.num_carrying
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        for ghost in ghosts:
            ghost_pos = ghost.get_position()
            ghost_distance = self.get_maze_distance(my_pos, ghost_pos)
            
            # Stronger avoidance when carrying food or ghost is close
            if food_carrying > 0:
                # Avoid ghosts more aggressively when carrying food
                if ghost_distance < 8:  # Wider avoidance radius when carrying food
                    avoid_positions.append(ghost_pos)
            else:
                # Normal avoidance when not carrying food
                if ghost_distance < 5 and ghost.scared_timer == 0:
                    avoid_positions.append(ghost_pos)
        
        return avoid_positions


class Defender(AStarAgent):
    """
    Defensive agent that uses A* to intercept enemy invaders.
    Patrols territory when no invaders are present. ("Police behavoiur")
    """

    def get_goal_position(self, game_state):
        """
        Determines goal position for defender - closest invader or patrol point.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Find enemy invaders if any
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        if invaders:
            # Chase closest invader
            closest_invader = min(invaders, 
                                key=lambda invader: self.get_maze_distance(my_pos, invader.get_position()))
            return closest_invader.get_position()
        
        # No invaders - patrol midline
        layout = game_state.data.layout
        mid_x = layout.width // 2
        patrol_x = mid_x - 1 if self.red else mid_x + 1
        
        # Find best patrol position on our side
        patrol_points = []
        for y in range(1, layout.height - 1):
            if not game_state.has_wall(patrol_x, y):
                patrol_points.append((patrol_x, y))
        
        if patrol_points:
            return min(patrol_points, 
                      key=lambda p: self.get_maze_distance(my_pos, p))
        
        return self.start  # Fallback to starting position

    def get_avoid_positions(self, game_state):
        """
        Defender typically doesn't need to avoid positions unless it trying to catch vulnerable ghosts.
        """
        avoid_positions = []
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Only avoid if we're scared
        if my_state.scared_timer > 0:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            for enemy in enemies:
                if enemy.get_position() is not None:
                    avoid_positions.append(enemy.get_position())
        
        return avoid_positions


class Actions:
    """
    Directions for the pacmans, needed for A* movement calculations.
    """
    @staticmethod
    def direction_to_vector(direction):
        """
        Converts direction to (dx, dy) vector.
        """
        if direction == Directions.NORTH:
            return (0, 1)
        elif direction == Directions.SOUTH:
            return (0, -1)
        elif direction == Directions.EAST:
            return (1, 0)
        elif direction == Directions.WEST:
            return (-1, 0)
        return (0, 0)
