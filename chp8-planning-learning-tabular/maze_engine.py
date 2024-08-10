import numpy as np

##########################################
class Maze:
    
    def __init__(self) -> None:
        
        # maze dimensions
        self.height = 5
        self.width = 8
        
        # initialize maze and set start and goal positions
        self.board = np.ones((self.height, self.width))
        self.start_pos = (0, 0)
        self.goal_pos = (self.height-1, self.width-1)
        
        # set obstacle cells
        self.board[1, :4] = -1
        self.board[3, 4:] = -1
        self.obstacles = self.get_obstacle_positions()
        
        
    def step(self, curr_pos, action):
        # compute next position
        next_pos = self.add_tuples(curr_pos, action)
        # compute reward and return reward and next state
        if self.is_feasible_pos(next_pos):
            # check goal state
            if next_pos == self.goal_pos:
                return 1, next_pos
            # normal position
            return 0, next_pos
        else:
            return -1, curr_pos
        
    
    ###### helper functions ######
    def is_feasible_pos(self, pos):
        feasible = False
        # check bounds
        if -1<pos[0]<self.height and -1<pos[1]<self.width:
            # check obstacles
            if pos not in self.obstacles:
                feasible = True
        return feasible
    
    def get_obstacle_positions(self):
        obstacles = []
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] == -1:
                    obstacles.append((i, j))
        return obstacles
    
    def add_tuples(self, a:tuple, b:tuple):
        return (a[0]+b[0], a[1]+b[1])
        
    def display(self):
        c = 65
        # First row
        print(f"  ", end='')
        for j in range(self.width):
            print(f"| {j} ", end='')
        print("| ")
        print((self.width*4+4)*"-")

        # Other rows
        for i in range(self.height):
            # print(f"{chr(c+i)} ", end='')
            print(f"{i} ", end='')
            for j in range(self.width):
                if self.board[i, j] == -1:
                    print(f"| X ", end='')
                elif i==0 and j==0:
                    print(f"| S ", end='')
                elif i==self.height-1 and j==self.width-1:
                    print(f"| G ", end='')
                else:
                    print(f"|   ", end='')
            print("| ")
            print((self.width*4+4)*"-")