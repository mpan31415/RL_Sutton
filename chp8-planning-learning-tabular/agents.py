from maze_engine import Maze
import numpy as np
from random import choice
from scipy.stats import bernoulli
from time import sleep


##########################################  
class QAgent:
    
    def __init__(self, alpha, gamma, epsilon, epsilon_scaling, num_episodes) -> None:
        
        self.maze = Maze()
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)]    # up, right, down, left
        self.action_values = np.zeros((self.maze.height, self.maze.width, len(self.actions)))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.curr_epsilon = 0.5
        self.epsilon_scaling = epsilon_scaling
        self.target_epsilon = epsilon
        
        self.num_episodes = num_episodes
        
    
    def learn(self):
        for episode_num in range(1, self.num_episodes+1):
            self.episode(episode_num)
        print()
        self.print_action_values()
        
        
    def act(self):
        self.curr_epsilon = 0
        curr_pos = (0, 0)
        self.print_action_values(show_curr_pos=True, curr_pos=curr_pos)
        while curr_pos != self.maze.goal_pos:
            # select action according to e-greedy policy
            action_idx, action = self.get_action(curr_pos)
            # get reward and next state from environment
            reward, next_pos = self.maze.step(curr_pos, action)
            # advance to next position
            curr_pos = next_pos
            
            # delay and print current board
            sleep(1)
            self.print_action_values(show_curr_pos=True, curr_pos=curr_pos)
            print("="*50)
        
        print("="*50)
        print("Finished this round!")
        
        
    def episode(self, episode_num):
        curr_pos = (0, 0)
        step_counter = 0
        while curr_pos != self.maze.goal_pos:
            # select action according to e-greedy policy
            action_idx, action = self.get_action(curr_pos)
            # get reward and next state from environment
            reward, next_pos = self.maze.step(curr_pos, action)
            # perform q-learning update
            err = reward + self.gamma*np.max(self.action_values[next_pos[0],next_pos[1],:]) - self.action_values[curr_pos[0],curr_pos[1],action_idx]
            self.action_values[curr_pos[0],curr_pos[1],action_idx] += self.alpha * err
            # advance to next position
            curr_pos = next_pos
            
            step_counter += 1
            # if step_counter % 10000 == 0:
            #     print("step = %d" % step_counter)
        
        # print("="*50)
        # print("Finished episode #%d" % episode_num)
        # self.print_action_values()
        print("|", end="")
        
        # scale-down exploration constant epsilon
        if self.curr_epsilon * self.epsilon_scaling > self.target_epsilon:
            self.curr_epsilon *= self.epsilon_scaling
            # print("scaling down epsilon = %.3f" % self.curr_epsilon)
        else:
            self.curr_epsilon = self.target_epsilon
            # print("maintaining current epsilon = %.3f" % self.curr_epsilon)
            
        
    def get_action(self, s):
        if bernoulli.rvs(self.curr_epsilon)==1:
            # random action
            action_idx = choice([0,1,2,3])
            return action_idx, self.actions[action_idx]
        else:
            # greedy action based on current action value estimate
            action_idx = np.argmax(self.action_values[s[0], s[1], :])
            return action_idx, self.actions[action_idx]
        
        
    def print_action_values(self, show_curr_pos=False, curr_pos=None):
        c = 65
        # First row
        print(f"  ", end='')
        for j in range(self.maze.width):
            print(f"|   {j}  ", end='')
        print("| ")
        print((self.maze.width*4+4)*"-")

        # Other rows
        for i in range(self.maze.height):
            # print(f"{chr(c+i)} ", end='')
            print(f"{i} ", end='')
            for j in range(self.maze.width):
                if self.maze.board[i, j] == -1:
                    print("| {:>4} ".format("X"), end='')
                else:
                    if show_curr_pos and (i, j)==curr_pos:
                        print("| {:>4} ".format("O"), end='')
                    else:
                        if show_curr_pos:
                            val = np.max(self.action_values[i, j, :])
                            print("|      ", end='')
                        else:
                            val = np.max(self.action_values[i, j, :])
                            print("| {:>4} ".format(val.round(1)), end='')
            print("| ")
            print((self.maze.width*4+4)*"-")
            
            
            
##########################################  
class DynaQ:
    
    def __init__(self, alpha, gamma, epsilon, epsilon_scaling, num_episode, num_planning_iters) -> None:
        
        self.maze = Maze()
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)]    # up, right, down, left
        self.action_values = np.zeros((self.maze.height, self.maze.width, len(self.actions)))
        
        # for each state, for each action, the model maps to [reward, next_pos[0], next_pos[1]]
        self.model = np.zeros((self.maze.height, self.maze.width, len(self.actions), 3))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.curr_epsilon = 0.5
        self.epsilon_scaling = epsilon_scaling
        self.target_epsilon = epsilon
        
        self.num_episode = num_episode
        self.num_planning_iters = num_planning_iters
        
        self.observed_state_actions = []       # list of tuples = (pos[0], pos[1], action_idx)
        
    
    def learn(self):
        # self.curr_pos = (0, 0)
        for episode_num in range(1, self.num_episode+1):
            self.episode(episode_num)
        print()
        self.print_action_values()
        
        
        
    def episode(self, outer_iter_num):
        step_counter = 0
        # initialize curr_pos to global current position (self.curr_pos)
        curr_pos = (0, 0)
        while curr_pos != self.maze.goal_pos:
            
            # select action according to e-greedy policy
            action_idx, action = self.get_action(curr_pos)
            # get reward and next state from environment
            reward, next_pos = self.maze.step(curr_pos, action)
            # perform q-learning update
            err = reward + self.gamma*np.max(self.action_values[next_pos[0],next_pos[1],:]) - self.action_values[curr_pos[0],curr_pos[1],action_idx]
            self.action_values[curr_pos[0],curr_pos[1],action_idx] += self.alpha * err
            
            # planning
            # update model
            self.model[curr_pos[0], curr_pos[1], action_idx] = [reward, next_pos[0], next_pos[1]]
            # loop planning
            for planning_iter in range(1, self.num_planning_iters+1):
                pass
            
            
            # advance to next position
            curr_pos = next_pos
            
            
            step_counter += 1
            if step_counter % 1000 == 0:
                print("step = %d" % step_counter)
        
        # print("="*50)
        # print("Finished episode #%d" % episode_num)
        # self.print_action_values()
        print("|", end="")
        
        # scale-down exploration constant epsilon
        if self.curr_epsilon * self.epsilon_scaling > self.target_epsilon:
            self.curr_epsilon *= self.epsilon_scaling
            # print("scaling down epsilon = %.3f" % self.curr_epsilon)
        else:
            self.curr_epsilon = self.target_epsilon
            # print("maintaining current epsilon = %.3f" % self.curr_epsilon)
            
        
    def get_action(self, s):
        if bernoulli.rvs(self.curr_epsilon)==1:
            # random action
            action_idx = choice([0,1,2,3])
            return action_idx, self.actions[action_idx]
        else:
            # greedy action based on current action value estimate
            action_idx = np.argmax(self.action_values[s[0], s[1], :])
            return action_idx, self.actions[action_idx]
        
        
    def print_action_values(self, show_curr_pos=False, curr_pos=None):
        c = 65
        # First row
        print(f"  ", end='')
        for j in range(self.maze.width):
            print(f"|   {j}  ", end='')
        print("| ")
        print((self.maze.width*4+4)*"-")

        # Other rows
        for i in range(self.maze.height):
            # print(f"{chr(c+i)} ", end='')
            print(f"{i} ", end='')
            for j in range(self.maze.width):
                if self.maze.board[i, j] == -1:
                    print("| {:>4} ".format("X"), end='')
                else:
                    if show_curr_pos and (i, j)==curr_pos:
                        print("| {:>4} ".format("O"), end='')
                    else:
                        if show_curr_pos:
                            val = np.max(self.action_values[i, j, :])
                            print("|      ", end='')
                        else:
                            val = np.max(self.action_values[i, j, :])
                            print("| {:>4} ".format(val.round(1)), end='')
            print("| ")
            print((self.maze.width*4+4)*"-")
            
            
            
    
        
##########################################
if __name__ == "__main__":
    
    alpha = 0.1
    gamma = 1
    
    epsilon = 0.1
    epsilon_scaling = 0.95
    
    num_episodes = 50
    
    # q_agent = QAgent(alpha, gamma, epsilon, epsilon_scaling, num_episodes)
    # q_agent.learn()
    
    dynaq_agent = DynaQ(alpha, gamma, epsilon, epsilon_scaling, num_episodes)
    dynaq_agent.learn()